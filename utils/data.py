"""Stable seed splits, experiment fingerprints, atomic caches and rank records."""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def request_for(sample_id, task, request_seed):
    return 1 if int(stable_id(sample_id, task, request_seed), 16) % 2 else -1


def read_json(path):
    return json.loads(Path(path).read_text())


def _atomic(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, obj):
    _atomic(path, lambda f: json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False))


def atomic_jsonl(path, rows):
    def write(f):
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _atomic(path, write)


def atomic_torch(path, obj):
    import torch
    _atomic(path, lambda f: torch.save(obj, f), binary=True)


def load_tensor(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def make_samples(args):
    prompts = read_jsonl(args.prompts_path)
    if not prompts:
        raise ValueError("Prompt bank is empty")
    if len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt IDs")
    for prompt in prompts:
        if not all(isinstance(prompt.get(k), str) and prompt[k].strip() for k in ("prompt_id", "text", "object")):
            raise ValueError("Every prompt needs nonempty prompt_id, text, object")
    # num_samples counts root seeds; a root's prompt/stage/edited variants share its split.
    seeds = [args.seed + i for i in range(args.num_samples)]
    ordered = sorted(seeds, key=lambda s: stable_id(args.split_seed, s))
    train_end = int(len(seeds) * args.train_fraction)
    val_end = train_end + int(len(seeds) * args.validation_fraction)
    split = {s: ("train" if i < train_end else "validation" if i < val_end else "test") for i, s in enumerate(ordered)}
    samples = []
    for i, seed in enumerate(seeds):
        p = prompts[i % len(prompts)]
        samples.append({"sample_id": stable_id(seed, p["prompt_id"]), "root_seed": seed,
                        "seed": seed, "prompt_id": p["prompt_id"], "prompt": p["text"],
                        "object": p["object"], "prompt_index": i % len(prompts), "split": split[seed]})
    return samples, prompts


# Operational/analysis knobs cannot change a sample's scientific identity.
NON_SCIENTIFIC = {"config", "run_id", "logs_dir", "ckpts_dir", "figs_dir", "resume", "overwrite", "dry_run",
                  "local_files_only", "device", "distributed_timeout", "torch_threads", "audit_labels",
                  "plot_dpi", "plot_width", "plot_height", "plot_splits", "audit_count",
                  "raw_display_scale", "bootstrap_seed", "bootstrap_count", "confidence_level", "useful_success",
                  "prompts_path", "tasks_path", "progress", "progress_mininterval"}


def scientific_fingerprint(args):
    config = {k: v for k, v in vars(args).items() if k not in NON_SCIENTIFIC}
    config["prompt_bank"] = read_jsonl(args.prompts_path)
    config["task_definitions"] = read_json(args.tasks_path)
    config["schema_version"] = 1
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def software_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "diffusers", "transformers", "accelerate", "numpy", "matplotlib", "pytest", "Pillow", "sentencepiece", "protobuf", "tqdm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def base_row(sample, stage=None, task=None, request=None, **kwargs):
    row = {k: sample[k] for k in ("sample_id", "root_seed", "prompt_id", "prompt", "object", "split")}
    row.update(stage=stage, native_timestep=None, task=task, request=request,
               observation=None, controller=None, coordinate="native", budget=None,
               status="ok", failure_reason=None)
    row.update(kwargs)
    return row


def merge_rows(rows, expected_keys=None):
    by_key = {}
    for row in rows:
        key = row["key"]
        if key in by_key:
            raise ValueError(f"Duplicate result key: {key}")
        by_key[key] = row
    if expected_keys is not None:
        missing, extra = set(expected_keys) - set(by_key), set(by_key) - set(expected_keys)
        if missing or extra:
            raise ValueError(f"Result key mismatch: {len(missing)} missing, {len(extra)} extra; examples {sorted(missing)[:3]}, {sorted(extra)[:3]}")
    return [by_key[k] for k in sorted(by_key)]


class Run:
    def __init__(self, args, dist, command):
        self.args, self.dist, self.command = args, dist, command
        self.path = Path(args.logs_dir) / args.run_id
        self.ckpt_path = Path(args.ckpts_dir) / args.run_id
        self.fig_path = Path(args.figs_dir) / args.run_id
        self.samples, self.prompts = make_samples(args)
        self.prompt_index = {p["prompt_id"]: i for i, p in enumerate(self.prompts)}
        self.fingerprint = scientific_fingerprint(args)
        self._rank_rows, self._completed = {}, {}
        from .distributed import main_process
        # Validate before any destructive operation. Only collect --overwrite
        # explicitly replaces the scientific identity of the whole run.
        manifest = self.path / "manifest.json"
        replace_manifest = getattr(args, "overwrite", False) and command == "collect"
        if manifest.exists() and not replace_manifest:
            saved = read_json(manifest)
            if saved["fingerprint"] != self.fingerprint:
                raise ValueError("Incompatible scientific cache fingerprint: choose another run-id or restore the original settings")
            if saved["samples"] != self.samples:
                raise ValueError("Split/sample manifest differs from the saved run")
        if getattr(args, "overwrite", False):
            from .cache import reset_stage
            main_process(dist, lambda: reset_stage(args, command))
        def initialize_files():
            import torch
            self.path.mkdir(parents=True, exist_ok=True)
            self.ckpt_path.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                atomic_json(manifest, {"fingerprint": self.fingerprint, "samples": self.samples, "prompts": self.prompts,
                                      "config": vars(args), "versions": software_versions()})
            hardware = {"device": str(dist.device), "world_size": dist.world_size,
                        "precision": args.precision, "gpu_name": torch.cuda.get_device_name(dist.device) if dist.device.type == "cuda" else None}
            atomic_json(self.path / f"invocation_{command}.json", {"config": vars(args), "command": sys.argv,
                                                                 "versions": software_versions(), "hardware": hardware})
        from .distributed import main_process
        main_process(dist, initialize_files)

    def rows(self, command):
        from .cache import cached_rows
        return cached_rows(self, command)

    def completed(self, command):
        if command not in self._completed:
            from .cache import clean_records
            from .distributed import main_process
            main_process(self.dist, lambda: clean_records(self, command))
            self._rank_rows.pop(command, None)
            self._completed[command] = {row["key"] for row in self.rows(command)}
        return set(self._completed[command])

    def invalidate(self, command, keys):
        """Remove only requested stale rows, synchronously before new trial work."""
        from .cache import discard_rows
        from .distributed import main_process
        main_process(self.dist, lambda: discard_rows(self, command, keys))
        self._rank_rows.pop(command, None)
        self._completed.pop(command, None)

    def write_row(self, command, row):
        if "key" not in row:
            raise ValueError("Every result row requires a stable key")
        path = self.path / "ranks" / f"{command}.rank{self.dist.rank:05d}.jsonl"
        if command not in self._rank_rows:
            self._rank_rows[command] = read_jsonl(path)
        if any(r["key"] == row["key"] for r in self._rank_rows[command]):
            raise ValueError(f"Duplicate write for {row['key']}")
        self._rank_rows[command].append(row)
        atomic_jsonl(path, self._rank_rows[command])

    def write_tensor(self, relative, obj):
        atomic_torch(self.path / relative, obj)
        return str(relative)

    def write_json(self, relative, obj):
        atomic_json(self.path / relative, obj)
        return str(relative)

    def load_trajectory(self, sample_id):
        obj = load_tensor(self.path / "trajectories" / f"{sample_id}.pt")
        if obj["fingerprint"] != self.fingerprint:
            raise ValueError(f"Incompatible trajectory cache for {sample_id}")
        component_path = self.path / "components.json"
        if component_path.exists() and obj.get("component_fingerprint") != read_json(component_path)["fingerprint"]:
            raise ValueError(f"Incompatible component identities in trajectory {sample_id}")
        return obj

    def finish(self, command, expected_keys=None):
        from .progress import status
        status(self.args, f"{command}: waiting for workers and merging saved records", self.dist)
        self.dist.barrier()
        result = [None]
        if self.dist.is_main:
            try:
                rows = merge_rows(self.rows(command), expected_keys)
                atomic_jsonl(self.path / f"{command}.jsonl", rows)
                result[0] = {"count": len(rows)}
            except (ValueError, OSError) as exc:
                result[0] = {"error": str(exc)}
        if self.dist.world_size > 1:
            import torch.distributed as distributed
            distributed.broadcast_object_list(result, src=0, device=self.dist.device)
        if "error" in result[0]:
            raise ValueError(result[0]["error"])
        status(self.args, f"{command}: complete ({result[0]['count']} saved records)", self.dist)
        return result[0]


def model_identity(identifier, revision, config_name):
    """Resolve cached Hub commits, or hash local component contents; never network."""
    path = Path(identifier).expanduser()
    if path.is_dir():
        digest = hashlib.sha256()
        suffixes = {'.json', '.txt', '.model', '.bin', '.safetensors', '.pt', '.pth', '.tiktoken'}
        files = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in suffixes and '.cache' not in p.parts)
        if not files:
            raise ValueError(f'Local model directory has no component files: {identifier}')
        for file in files:
            digest.update(str(file.relative_to(path)).encode())
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
        return {'source': 'local', 'content_sha256': digest.hexdigest()}
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(identifier, config_name, revision=revision)
    if isinstance(cached, str):
        parts = Path(cached).parts
        if 'snapshots' in parts:
            return {'source': 'hub', 'repo': identifier, 'requested_revision': revision,
                    'resolved_commit': parts[parts.index('snapshots') + 1]}
    raise ValueError(f'Could not determine the resolved local cache commit for {identifier}; supply a local directory')


def register_components(run, sampler, scorers):
    """Bind all cached trajectories and later continuations to loaded components."""
    identities = {'generator': getattr(sampler, 'weights_identity', {'local_fixture': sampler.model_id}),
                  'evaluators': {r: getattr(s, 'weights_identity', {'local_fixture': type(s).__name__}) for r,s in scorers.items()}}
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    path = run.path / 'components.json'
    if path.exists() and read_json(path)['fingerprint'] != digest:
        raise ValueError('Loaded component identities differ from this cache (changed Hub commit or local weights)')
    from .distributed import main_process
    main_process(run.dist, lambda: atomic_json(path, {'fingerprint': digest, 'identities': identities}) if not path.exists() else None)
    return digest
