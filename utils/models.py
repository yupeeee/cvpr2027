"""Prepare frozen-model files before distributed work, without loading any weights."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Iterator

from .progress import progress


# Only files used by the supported Transformers CLIP/SigLIP processors.
_PROCESSOR_FILES = [
    "config.json", "preprocessor_config.json", "processor_config.json",
    "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
    "added_tokens.json", "merges.txt", "vocab.json", "vocab.txt",
    "spiece.model", "tokenizer.model",
]


def _offline_environment() -> list[str]:
    return [name for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
            if os.environ.get(name, "").upper() in {"1", "ON", "YES", "TRUE"}]


@contextmanager
def model_access(role: str, identifier: str, revision: str | None,
                 local_files_only: bool) -> Iterator[None]:
    """Add actionable context to filesystem/cache errors, preserving their cause."""
    try:
        yield
    except OSError as exc:
        option = "model" if role == "model" else role.replace("_", "-")
        offline = _offline_environment()
        message = (f"Cannot access {role} {identifier!r} at revision {revision or 'main'!r}. "
                   f"Supply a complete local directory with --{option}-id, or prepare the "
                   "models with `python -m exps.prepare --config YOUR_CONFIG "
                   "--no-local-files-only --overwrite` (model preparation only; "
                   "this command does not remove experiment outputs). ")
        if local_files_only:
            message += "--local-files-only forbids downloading missing files. "
        if offline:
            message += (f"{', '.join(offline)} is enabled; unset it to allow downloads "
                        "(--no-local-files-only does not override offline environment variables). ")
        message += f"Original error: {exc}"
        raise RuntimeError(message) from exc


def _directory(identifier: str) -> Path | None:
    path = Path(identifier).expanduser()
    if path.is_dir():
        return path
    if path.exists() or identifier.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Local model directory does not exist: {identifier}")
    return None


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")


def _check_weights(folder: Path, stem: str = "model") -> None:
    """Check filenames and shard indexes only; do not deserialize checkpoint tensors."""
    bin_stem = "pytorch_model" if stem == "model" else stem
    for filename in (f"{stem}.safetensors", f"{stem}.safetensors.index.json",
                     f"{bin_stem}.bin", f"{bin_stem}.bin.index.json"):
        file = folder / filename
        if not file.is_file():
            continue
        if filename.endswith(".index.json"):
            mapping = json.loads(file.read_text(encoding="utf-8")).get("weight_map", {})
            if not mapping:
                raise ValueError(f"Empty checkpoint shard map: {file}")
            for shard in set(mapping.values()):
                _require(folder / shard)
        return
    raise FileNotFoundError(f"Missing PyTorch/safetensors weights in {folder}")


def _check_tokenizer(folder: Path) -> None:
    if any((folder / name).is_file() for name in (
            "tokenizer.json", "spiece.model", "tokenizer.model", "vocab.txt")):
        return
    if (folder / "vocab.json").is_file() and (folder / "merges.txt").is_file():
        return
    raise FileNotFoundError(f"Missing tokenizer vocabulary in {folder}")


def _prepare_generator(identifier: str, revision: str | None, local_only: bool) -> Path:
    folder = _directory(identifier)
    if folder is None:
        if local_only:
            from huggingface_hub import snapshot_download
            folder = Path(snapshot_download(identifier, revision=revision, local_files_only=True))
        else:
            from diffusers import StableDiffusionPipeline
            # Diffusers selects the actual components and preferred weight format;
            # snapshotting the entire repository fetches unrelated multi-GB checkpoints.
            folder = Path(StableDiffusionPipeline.download(
                identifier, revision=revision, local_files_only=False,
                safety_checker=None, requires_safety_checker=False))
    _require(folder / "model_index.json")
    _require(folder / "scheduler" / "scheduler_config.json")
    for component, stem in (("unet", "diffusion_pytorch_model"),
                            ("vae", "diffusion_pytorch_model"), ("text_encoder", "model")):
        _require(folder / component / "config.json")
        _check_weights(folder / component, stem)
    _check_tokenizer(folder / "tokenizer")
    return folder


def _prepare_evaluator(identifier: str, revision: str | None, local_only: bool) -> Path:
    folder = _directory(identifier)
    if folder is None:
        from huggingface_hub import list_repo_files, snapshot_download
        from huggingface_hub.constants import HF_HUB_OFFLINE
        # Respect both explicit local-only and process-wide Hub offline configuration.
        local_only = local_only or HF_HUB_OFFLINE
        weights = ["model.safetensors", "model.safetensors.index.json", "model-*.safetensors",
                   "pytorch_model.bin", "pytorch_model.bin.index.json", "pytorch_model-*.bin"]
        if not local_only:
            files = set(list_repo_files(identifier, revision=revision))
            if "model.safetensors" in files:
                weights = ["model.safetensors"]
            elif "model.safetensors.index.json" in files:
                weights = ["model.safetensors.index.json", "model-*.safetensors"]
            elif "pytorch_model.bin" in files:
                weights = ["pytorch_model.bin"]
            elif "pytorch_model.bin.index.json" in files:
                weights = ["pytorch_model.bin.index.json", "pytorch_model-*.bin"]
            else:
                raise FileNotFoundError(f"No supported CLIP/SigLIP weights in {identifier}")
        folder = Path(snapshot_download(identifier, revision=revision,
                                       local_files_only=local_only,
                                       allow_patterns=_PROCESSOR_FILES + weights))
    _require(folder / "config.json")
    _require(folder / "preprocessor_config.json")
    _check_tokenizer(folder)
    _check_weights(folder)
    return folder


def _prepare(args: Any, role: str) -> str:
    identifier, revision = getattr(args, f"{role}_id"), getattr(args, f"{role}_revision")
    with model_access(role, identifier, revision, args.local_files_only):
        prepare = _prepare_generator if role == "model" else _prepare_evaluator
        local_only = args.local_files_only or bool(_offline_environment())
        # Existing complete snapshots are reusable without Hub metadata requests.
        # Only an explicitly permitted online run may fill a missing/incomplete cache.
        if not local_only and _directory(identifier) is None:
            try:
                return str(prepare(identifier, revision, True))
            except FileNotFoundError:
                pass
        return str(prepare(identifier, revision, local_only))


def prepare_generator(args: Any) -> str:
    """Return a complete local generator path, acquiring missing files if allowed."""
    return _prepare(args, "model")


def prepare_evaluator(args: Any, role: str) -> str:
    """Return a complete local evaluator path, acquiring missing files if allowed."""
    if role not in {"a", "b"}:
        raise ValueError("Evaluator role must be a or b.")
    return _prepare(args, f"evaluator_{role}")


@contextmanager
def _tokenizer_dependencies(model_path: str) -> Iterator[None]:
    """Explain optional tokenizer dependencies without relabelling cache failures."""
    try:
        yield
    except ImportError as exc:
        command = shlex.join([sys.executable, "-m", "pip", "install", "--no-deps",
                              "--only-binary=:all:", "sentencepiece>=0.2,<0.3",
                              "protobuf>=4.25,<7"])
        raise ImportError(
            f"Cannot initialize the tokenizer/processor in {model_path!r}. "
            f"SigLIP requires SentencePiece and protobuf in this Python environment. "
            f"Install them with `{command}`, then restart the pilot. Original error: {exc}"
        ) from exc


def load_evaluator_processor(model_path: str) -> Any:
    """Load only the local processor/tokenizer; this does not load neural weights."""
    with _tokenizer_dependencies(model_path):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(model_path, local_files_only=True)


def validate_preprocessors(paths: dict[str, str]) -> None:
    """Construct local tokenizers/processors before allocating model replicas."""
    tokenizer_path = str(Path(paths["model"]) / "tokenizer")
    with _tokenizer_dependencies(tokenizer_path):
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    for role in ("evaluator_a", "evaluator_b"):
        load_evaluator_processor(paths[role])


def prepare_models(args: Any) -> dict[str, str]:
    """Check files and tokenizer dependencies before workers; never load weights.

    IDs and revisions in ``args`` remain unchanged for experiment identity tracking.
    Local-only mode and offline environment variables never trigger an online retry.
    """
    from .progress import quiet_library_progress
    paths = {}
    with progress(args, total=4, desc="prepare: initializing model file checks", unit="check") as bar:
        # Importing model libraries can take time; display the bar beforehand.
        quiet_library_progress()
        bar.set_description("prepare: resolving/checking generator files")
        paths["model"] = prepare_generator(args)
        bar.update()
        for role in ("a", "b"):
            bar.set_description(f"prepare: resolving/checking evaluator {role.upper()} files")
            paths[f"evaluator_{role}"] = prepare_evaluator(args, role)
            bar.update()
        bar.set_description("prepare: checking local SD/CLIP/SigLIP tokenizers and processors")
        validate_preprocessors(paths)
        bar.update()
    return paths


def prepare_for_inference(args: Any) -> None:
    """Check assets/dependencies before automatic/external distributed initialization."""
    if not getattr(args, "dry_run", False):
        prepare_models(args)


def load_frozen_models(args, dist):
    """Load each worker's replica with one aggregate device progress display."""
    from .progress import quiet_library_progress
    from .sampler import Sampler
    from .tasks import Scorer
    quiet_library_progress()
    with progress(args, total=3, desc='Load frozen model replicas', unit='model', dist=dist) as bar:
        bar.set_postfix(model='Stable Diffusion')
        sampler = Sampler.load(args, dist.device)
        bar.update()
        scorers = {}
        for role in ('a', 'b'):
            bar.set_postfix(model=f'evaluator {role.upper()}')
            scorers[role] = Scorer.load(args, role, dist.device)
            bar.update()
    return sampler, scorers
