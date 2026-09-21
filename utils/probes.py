"""Spatially pooled endpoint regressors with training-only preprocessing."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from utils.controls import coordinate
from utils.distributed import main_process
from utils.progress import progress

FIELDS = ("appearance", "composition", "category")


def features(latent: torch.Tensor, prompt_index, prompt_count: int,
             grid: int) -> torch.Tensor:
    """Retain a spatial grid and give every observer the same prompt one-hot."""
    if latent.ndim != 4 or grid < 1 or prompt_count < 1:
        raise ValueError("Expected BCHW latents, positive pooling grid/prompt count")
    indices = torch.as_tensor(prompt_index, device=latent.device, dtype=torch.long)
    if indices.ndim == 0:
        indices = indices.expand(latent.shape[0])
    if indices.shape != (latent.shape[0],) or not ((indices >= 0) & (indices < prompt_count)).all():
        raise ValueError("Invalid prompt indices")
    pooled = F.adaptive_avg_pool2d(latent.float(), (grid, grid)).flatten(1)
    return torch.cat((pooled, F.one_hot(indices, prompt_count).float()), dim=1)


def make_head(input_dim: int, head: str) -> nn.Module:
    if head == "linear":
        return nn.Linear(input_dim, len(FIELDS))
    if head.startswith("mlp") and head[3:].isdigit() and int(head[3:]) > 0:
        return nn.Sequential(nn.Linear(input_dim, int(head[3:])), nn.ReLU(),
                             nn.Linear(int(head[3:]), len(FIELDS)))
    raise ValueError(f"Unsupported probe head: {head}")


def head_names(args) -> list[str]:
    result = []
    for name in args.probe_heads:
        if name == "linear":
            result.append(name)
        elif name == "mlp":
            result.extend(f"mlp{width}" for width in args.probe_hidden_sizes)
        else:
            raise ValueError(f"Unsupported probe family: {name}")
    return result


class Probe(nn.Module):
    """A frozen learned readout; derivatives with respect to input remain live."""
    def __init__(self, checkpoint: dict):
        super().__init__()
        self.metadata = checkpoint["metadata"]
        self.head = make_head(self.metadata["input_dim"], self.metadata["head"])
        self.head.load_state_dict(checkpoint["state_dict"])
        for name in ("feature_mean", "feature_scale", "label_mean", "label_scale"):
            self.register_buffer(name, checkpoint[name].float())
        self.requires_grad_(False).eval()

    def predict(self, latent: torch.Tensor, prompt_index, cost=None) -> torch.Tensor:
        f = features(latent, prompt_index, self.metadata["prompt_count"],
                     self.metadata["pool_grid"])
        if cost is not None:
            cost.add("probe_calls", latent.shape[0])
            cost.add("probe_parameter_work", latent.shape[0] * self.metadata["parameter_count"])
        return self.head((f - self.feature_mean) / self.feature_scale) * self.label_scale + self.label_mean

    def bind(self, prompt_index, cost=None):
        return lambda latent: self.predict(latent, prompt_index, cost)


def probe_path(run, stage: int, access: str, head: str,
               variant: str = "native") -> Path:
    return Path(run.ckpt_path) / variant / f"stage_{stage:03d}_{access}_{head}.pt"


def load_probe(path, device) -> Probe:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return Probe(checkpoint).to(device)


def component_fingerprint(run):
    """None is reserved for standalone offline fixtures with no component manifest."""
    path = Path(run.path) / "components.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["fingerprint"]


def _validate_saved(saved: dict, run, description: str):
    if saved["fingerprint"] != run.fingerprint:
        raise ValueError(f"{description} has an incompatible scientific fingerprint")
    if saved.get("component_fingerprint") != component_fingerprint(run):
        raise ValueError(f"{description} has incompatible resolved component identities")


def validate_probe(probe: Probe, run) -> Probe:
    """Reject copied heads trained against another generator or evaluator revision."""
    _validate_saved(probe.metadata, run, "Probe checkpoint")
    return probe


def load_stage_scales(run) -> dict[int, float]:
    with open(Path(run.ckpt_path) / "stage_scales.json", encoding="utf-8") as handle:
        stored = json.load(handle)
    _validate_saved(stored, run, "Stage RMS checkpoint")
    return {int(k): float(v) for k, v in stored["scales"].items()}


def _scores(trajectory, role: str) -> torch.Tensor:
    return torch.tensor([float("nan") if trajectory["scores"][role][name] is None
                         else trajectory["scores"][role][name] for name in FIELDS], dtype=torch.float32)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dataset(args, run, samples, stage, access, variant, dist=None, split=""):
    xs, ys = [], []
    with progress(args, samples, desc=f"Load {split} features: stage {stage} {access}",
                  unit="sample", dist=dist, position=1, leave=False) as bar:
        for sample in bar:
            trajectory = run.load_trajectory(sample["sample_id"])
            latent = trajectory["states" if access == "raw" else "clean"][stage]
            if variant == "wrapped":
                if access != "raw":
                    raise ValueError("Coordinate readouts only accept raw states")
                latent = coordinate(latent, stage, args.num_steps, args.coordinate_amplitude)
            xs.append(features(latent, sample["prompt_index"], len(run.prompts), args.pool_grid).squeeze(0))
            ys.append(_scores(trajectory, "a"))
    if not xs:
        raise ValueError("Probe fitting requires nonempty training and validation splits")
    return torch.stack(xs), torch.stack(ys)


def optimize_batch(model, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """The same explicit backward/optimizer path for DDP and single-rank fitting."""
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x), y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite probe training loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _fit_one(args, dist, run, stage, access, head, variant, split_samples):
    """One model at a time; DDP backward counts are equal via sampler padding."""
    from utils.data import atomic_torch, stable_id

    path = probe_path(run, stage, access, head, variant)
    from utils.cache import checkpoint_valid
    if checkpoint_valid(args, stage, access, head, variant, run=run):
        return
    _sync(dist.device)
    if dist.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dist.device)
    started = time.perf_counter()
    train_x, train_y = _dataset(args, run, split_samples["train"], stage, access, variant,
                                dist=dist, split="train")
    val_x, val_y = _dataset(args, run, split_samples["validation"], stage, access, variant,
                            dist=dist, split="validation")
    feature_mean = train_x.mean(0)
    feature_std = train_x.std(0, unbiased=False)
    # A prompt absent from training must not be amplified by 1/epsilon.
    feature_scale = torch.where(feature_std == 0, torch.ones_like(feature_std),
                                feature_std.clamp_min(args.normalization_eps))
    label_mean = train_y.mean(0)
    label_scale = train_y.std(0, unbiased=False).clamp_min(args.normalization_eps)
    train = TensorDataset((train_x - feature_mean) / feature_scale,
                          (train_y - label_mean) / label_scale)
    val_x = (val_x - feature_mean) / feature_scale
    val_y = (val_y - label_mean) / label_scale
    # Explicit identical initialization independent of Python hash/random rank.
    seed = int(stable_id(args.probe_seed, stage, access, head, variant), 16) % (2 ** 31)
    torch.manual_seed(seed)
    model = make_head(train_x.shape[1], head).to(dist.device)
    raw_model = model
    if dist.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[dist.local_rank]
                                        if dist.device.type == "cuda" else None)
    sampler = DistributedSampler(train, num_replicas=dist.world_size, rank=dist.rank,
                                 shuffle=True, seed=seed, drop_last=False)
    loader = DataLoader(train, batch_size=args.probe_batch_size, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_lr,
                                  weight_decay=args.probe_weight_decay)
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    train_examples = backward_steps = val_examples = 0
    local_indices = list(range(dist.rank, len(val_x), dist.world_size))
    with progress(args, range(args.probe_epochs),
                  desc=f"Train {variant}: stage {stage} {access} {head}",
                  unit="epoch", dist=dist, position=1, leave=False) as epoch_bar:
        for epoch in epoch_bar:
            sampler.set_epoch(epoch)
            model.train()
            train_sum = 0.0
            train_count = 0
            for x, y in loader:
                x, y = x.to(dist.device), y.to(dist.device)
                loss = optimize_batch(model, optimizer, x, y)
                train_sum += loss * len(x)
                train_count += len(x)
                train_examples += len(x)
                backward_steps += 1
            # Unwrapped evaluation avoids DDP collectives on uneven shards.
            raw_model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for start in range(0, len(local_indices), args.probe_batch_size):
                    indices = local_indices[start:start + args.probe_batch_size]
                    x, y = val_x[indices].to(dist.device), val_y[indices].to(dist.device)
                    val_sum += float(F.mse_loss(raw_model(x), y, reduction="sum")) / len(FIELDS)
                    val_count += len(indices)
                    val_examples += len(indices)
            val_sum, val_count = dist.sum_count(val_sum, val_count)
            train_sum, train_count = dist.sum_count(train_sum, train_count)
            val_loss = val_sum / val_count
            if not math.isfinite(val_loss):
                raise FloatingPointError("Non-finite validation loss; no checkpoint selected")
            history.append({"epoch": epoch, "train_normalized_mse": train_sum / train_count,
                            "validation_normalized_mse": val_loss})
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in raw_model.state_dict().items()}
            epoch_bar.set_postfix(train=f"{train_sum / train_count:.4g}",
                                  validation=f"{val_loss:.4g}", best=f"{best_loss:.4g}")
    train_examples, backward_steps = dist.sum_count(train_examples, backward_steps)
    val_examples, _ = dist.sum_count(val_examples, 0)
    _sync(dist.device)
    elapsed = time.perf_counter() - started
    elapsed_gpu_total, _ = dist.sum_count(elapsed, 0)
    peak_memory = torch.cuda.max_memory_allocated(dist.device) if dist.device.type == "cuda" else 0
    memory_by_rank = [peak_memory]
    if dist.world_size > 1:
        import torch.distributed as distributed
        memory_by_rank = [None] * dist.world_size
        distributed.all_gather_object(memory_by_rank, peak_memory)
    parameter_count = sum(p.numel() for p in raw_model.parameters())
    metadata = {"fingerprint": run.fingerprint,
                "component_fingerprint": component_fingerprint(run),
                "stage": stage, "access": access,
                "head": head, "variant": variant, "input_dim": train_x.shape[1],
                "pool_grid": args.pool_grid, "prompt_count": len(run.prompts),
                "parameter_count": parameter_count,
                "best_epoch": best_epoch, "validation_normalized_mse": best_loss,
                "split_ids": {key: [s["sample_id"] for s in items] for key, items in split_samples.items()},
                "training_config": vars(args), "history": history,
                "training_padding": "DistributedSampler(drop_last=False): repeat at most world_size-1 examples per epoch",
                "costs": {"training_examples": train_examples,
                          "backward_passes": backward_steps,
                          "calibration_examples": val_examples,
                          "probe_parameter_work": (train_examples + val_examples) * parameter_count,
                          "feature_extraction_examples_per_rank": len(train_x) + len(val_x),
                          "peak_memory_bytes": max(memory_by_rank),
                          "peak_memory_bytes_by_rank": memory_by_rank,
                          "rank_zero_elapsed_seconds": elapsed,
                          "summed_rank_seconds": elapsed_gpu_total}}
    def save_checkpoint():
        atomic_torch(path, {"metadata": metadata, "state_dict": best_state,
                            "feature_mean": feature_mean, "feature_scale": feature_scale,
                            "label_mean": label_mean, "label_scale": label_scale})
    main_process(dist, save_checkpoint)


def train_probes(args, dist, run, variant: str = "native", evaluate: bool = True, on_progress=None):
    from utils.data import atomic_json, atomic_torch
    from utils.cache import checkpoint_valid, training_statistics_valid

    if variant not in ("native", "wrapped"):
        raise ValueError("Only native or wrapped probes can be fitted")
    completed = run.completed("fit") if evaluate else set()
    split_samples = {split: [s for s in run.samples if s["split"] == split]
                     for split in ("train", "validation", "test")}
    if not split_samples["train"] or not split_samples["validation"]:
        raise ValueError("Probe fitting requires nonempty training and validation seeds")
    accesses = ("raw", "clean") if variant == "native" else ("raw",)
    jobs = [(stage, access, head) for stage in args.stages
            for access in accesses for head in head_names(args)]
    cached_jobs = {job for job in jobs if checkpoint_valid(args, *job, variant, run=run)}
    statistics_ready = training_statistics_valid(args, run=run)
    # Check on every rank before a barrier: an invalid training target must not
    # strand the other ranks or be silently removed from the fitting cohort.
    if len(cached_jobs) != len(jobs) or not statistics_ready:
        invalid = []
        with progress(args, split_samples["train"] + split_samples["validation"],
                      desc=f"Validate {variant} cached trajectories", unit="sample", dist=dist) as bar:
            for sample in bar:
                trajectory = run.load_trajectory(sample["sample_id"])
                valid = trajectory.get("status", "ok") == "ok"
                if valid:
                    valid = bool(torch.isfinite(_scores(trajectory, "a")).all())
                    for stage in args.stages:
                        valid = valid and bool(torch.isfinite(trajectory["states"][stage]).all())
                        valid = valid and bool(torch.isfinite(trajectory["clean"][stage]).all())
                if not valid:
                    invalid.append(sample)
        if invalid:
            if dist.is_main:
                from utils.data import base_row
                existing = {row["key"] for row in run.rows("fit_failures")}
                for sample in invalid:
                    key = f"{variant}/{sample['sample_id']}"
                    if key not in existing:
                        run.write_row("fit_failures", base_row(sample, key=key,
                            command="fit", status="malformed", coordinate=variant,
                            failure_reason="Malformed training/validation trajectory; fitting stopped"))
            raise ValueError("Malformed training/validation trajectories recorded in fit_failures; fitting stopped")
    def save_training_statistics():
        scales = {}
        prompt_labels = {index: [] for index in range(len(run.prompts))}
        with progress(args, total=len(args.stages) * len(split_samples["train"]),
                      desc="Compute training statistics", unit="sample") as bar:
            for stage in args.stages:
                bar.set_postfix(stage=stage)
                square_sum = size = 0
                for sample in split_samples["train"]:
                    trajectory = run.load_trajectory(sample["sample_id"])
                    latent = trajectory["states"][stage].double()
                    square_sum += float(latent.square().sum())
                    size += latent.numel()
                    if stage == args.stages[0]:
                        prompt_labels[sample["prompt_index"]].append(_scores(trajectory, "a"))
                    bar.update(1)
                scales[str(stage)] = max((square_sum / size) ** 0.5, args.normalization_eps)
        atomic_json(Path(run.ckpt_path) / "stage_scales.json",
                    {"fingerprint": run.fingerprint,
                     "component_fingerprint": component_fingerprint(run), "scales": scales,
                     "training_ids": [s["sample_id"] for s in split_samples["train"]]})
        labels = torch.stack([label for group in prompt_labels.values() for label in group])
        means = torch.stack([torch.stack(group).mean(0) if group else labels.mean(0)
                             for group in prompt_labels.values()])
        atomic_torch(Path(run.ckpt_path) / "prompt_mean.pt",
                     {"means": means, "fingerprint": run.fingerprint,
                      "component_fingerprint": component_fingerprint(run),
                      "counts": [len(group) for group in prompt_labels.values()],
                      "training_ids": [s["sample_id"] for s in split_samples["train"]]})
    # Decide on rank zero inside the synchronized operation: a late worker
    # may already see the newly written files and must still join the broadcast.
    main_process(dist, lambda: None if training_statistics_valid(args, run=run)
                 else save_training_statistics())
    reused = 0
    with progress(args, jobs, desc=f"Fit {variant} probes", unit="head", dist=dist) as bar:
        for stage, access, head in bar:
            cached = (stage, access, head) in cached_jobs
            bar.set_postfix(stage=stage, access=access, head=head, reused=reused)
            if not cached:
                _fit_one(args, dist, run, stage, access, head, variant, split_samples)
            if on_progress is not None:
                on_progress(stage, access, head, cached)
            reused += int(cached)
            bar.set_postfix(stage=stage, access=access, head=head, reused=reused)
    if evaluate:
        for row in readout_rows(args, dist, run, variant=variant, completed=completed):
            if row["key"] not in completed:
                run.write_row("fit", row)
        expected = [row["key"] for row in _row_keys(args, run, variant)]
        run.finish("fit", expected)


def _row_keys(args, run, variant):
    accesses = ("raw", "clean", "prompt") if variant == "native" else ("raw",)
    for sample in run.samples:
        if sample["split"] not in ("validation", "test"):
            continue
        for stage in args.stages:
            for access in accesses:
                for head in (["prompt_mean"] if access == "prompt" else head_names(args)):
                    for task in FIELDS:
                        yield {"key": f"{variant}/{sample['sample_id']}/{stage}/{access}/{head}/{task}"}


def _invalid_readout_rows(sample, trajectory, stage, access, head, variant, num_steps, reason, completed=None):
    from utils.data import base_row
    for task in FIELDS:
        if f"{variant}/{sample['sample_id']}/{stage}/{access}/{head}/{task}" in (completed or set()):
            continue
        references = {}
        for role in ("a", "b"):
            value = trajectory["scores"][role].get(task)
            references[role] = value if value is not None and math.isfinite(float(value)) else None
        yield base_row(sample, stage=stage, task=task,
            key=f"{variant}/{sample['sample_id']}/{stage}/{access}/{head}/{task}",
            command="fit", access=access, observation=access, head=head,
            native_timestep=trajectory["metadata"]["timesteps"][stage] if stage < num_steps else None,
            coordinate=variant, coordinate_variant=variant, status="malformed", failure_reason=reason,
            prediction_a=None, reference_a=references["a"], reference_b=references["b"],
            squared_error=None, absolute_error=None, predicted_class=None, reference_class=None,
            correct=None, correct_b=None, evaluator_agreement=None, generation_work=None,
            observation_work=None, parameter_count=None, costs={})


def readout_rows(args, dist, run, variant: str = "native", transported: bool = False,
                 raw_only: bool = False, completed=None):
    """Evaluate nonpadded held-out shards, with transported inverse BEFORE pooling."""
    from utils.tasks import threshold
    from utils.data import read_json
    from utils.metrics import Cost, add_costs

    if variant == "transported":
        transported, variant = True, "native"
    output_variant = "transported" if transported else variant
    completed = completed or set()
    task_config = read_json(args.tasks_path)
    accesses = ("raw",) if raw_only or transported or variant == "wrapped" else ("raw", "clean", "prompt")
    jobs = [(stage, access, head) for stage in args.stages for access in accesses
            for head in (["prompt_mean"] if access == "prompt" else head_names(args))]
    samples = [sample for sample in dist.shard(run.samples)
               if sample["split"] in ("validation", "test")]
    with progress(args, jobs, desc=f"Evaluate {output_variant} readouts", unit="head", dist=dist) as bar:
        for stage, access, head in bar:
            pending_samples = [sample for sample in samples if any(
                f"{output_variant}/{sample['sample_id']}/{stage}/{access}/{head}/{task}" not in completed
                for task in FIELDS)]
            if not pending_samples:
                continue
            bar.set_postfix(stage=stage, access=access, head=head)
            probe = None if access == "prompt" else load_probe(probe_path(run, stage, access, head, variant), dist.device)
            if probe is not None:
                validate_probe(probe, run)
            baseline = None
            if access == "prompt":
                baseline = torch.load(Path(run.ckpt_path) / "prompt_mean.pt", map_location="cpu", weights_only=True)
                _validate_saved(baseline, run, "Prompt baseline")
            with progress(args, pending_samples,
                          desc=f"Readout: stage {stage} {access} {head}",
                          unit="sample", dist=dist, position=1, leave=False) as sample_bar:
                for sample in sample_bar:
                    trajectory = run.load_trajectory(sample["sample_id"])
                    if trajectory.get("status", "ok") != "ok" or not all(
                            bool(torch.isfinite(_scores(trajectory, role)).all()) for role in ("a", "b")):
                        yield from _invalid_readout_rows(sample, trajectory, stage, access, head,
                            output_variant, args.num_steps, "Malformed collected trajectory or scores", completed)
                        continue
                    a, b = _scores(trajectory, "a"), _scores(trajectory, "b")
                    inference_cost, coordinate_preparation = Cost(), Cost()
                    if access == "prompt":
                        prediction = baseline["means"][sample["prompt_index"]]
                    else:
                        latent = trajectory["states" if access == "raw" else "clean"][stage].to(dist.device)
                        if not torch.isfinite(latent).all():
                            yield from _invalid_readout_rows(sample, trajectory, stage, access, head,
                                output_variant, args.num_steps, "Nonfinite current observation", completed)
                            continue
                        if variant == "wrapped" or transported:
                            with coordinate_preparation.measure(dist.device), torch.no_grad():
                                latent = coordinate(latent, stage, args.num_steps, args.coordinate_amplitude)
                                coordinate_preparation.add("coordinate_forward_calls")
                                coordinate_preparation.add("coordinate_elements", latent.numel())
                        with inference_cost.measure(dist.device), torch.no_grad():
                            if transported:
                                latent = coordinate(latent, stage, args.num_steps, args.coordinate_amplitude, inverse=True)
                                inference_cost.add("coordinate_inverse_calls")
                                inference_cost.add("coordinate_elements", latent.numel())
                            prediction = probe.predict(latent, sample["prompt_index"], inference_cost)[0].cpu()
                    if not torch.isfinite(prediction).all():
                        yield from _invalid_readout_rows(sample, trajectory, stage, access, head,
                            output_variant, args.num_steps, "Nonfinite readout prediction", completed)
                        continue
                    work = trajectory["stage_costs"][stage]
                    preview_work = work.get("prediction", work["preview"]) if access == "clean" else {}
                    generation_work = work["generation"] if access != "prompt" else {}
                    parameter_count = 0 if probe is None else probe.metadata["parameter_count"]
                    for index, task in enumerate(FIELDS):
                        if f"{output_variant}/{sample['sample_id']}/{stage}/{access}/{head}/{task}" in completed:
                            continue
                        row = {"key": f"{output_variant}/{sample['sample_id']}/{stage}/{access}/{head}/{task}",
                               "command": "fit", "failure_reason": None,
                               "sample_id": sample["sample_id"], "root_seed": sample["root_seed"],
                               "prompt": sample["prompt"], "object": sample["object"],
                               "prompt_id": sample["prompt_id"], "split": sample["split"],
                               "task": task, "request": None, "stage": stage,
                               "native_timestep": trajectory["metadata"]["timesteps"][stage] if stage < args.num_steps else None,
                               "access": access, "observation": access, "head": head,
                               "controller": None, "coordinate": output_variant,
                               "coordinate_variant": output_variant, "budget": None,
                               "status": "ok", "prediction_a": float(prediction[index]),
                               "reference_a": float(a[index]), "reference_b": float(b[index]),
                               "squared_error": float((prediction[index].double() - a[index].double()).square()),
                               "absolute_error": float((prediction[index].double() - a[index].double()).abs()),
                               "parameter_count": parameter_count,
                               "generation_work": generation_work.get("unet_calls", 0),
                               "observation_work": preview_work.get("unet_calls", 0),
                               "costs": {"generation": generation_work,
                                         "observation": add_costs(preview_work, inference_cost.to_dict()),
                                         "cached_prediction_extraction": preview_work,
                                         "measured_cached_readout": inference_cost.to_dict(),
                                         "offline_coordinate_preparation": coordinate_preparation.to_dict()},
                               "timing_note": "Prediction/generation timings from original collection; readout measured on cached state; not online wall-clock speedup"}
                        if task != "category":
                            da = threshold(task_config, task, "a", "decision")
                            db = threshold(task_config, task, "b", "decision")
                            truth, other = bool(a[index] >= da), bool(b[index] >= db)
                            predicted = bool(prediction[index] >= da)
                            row.update(predicted_class=int(predicted), reference_class=int(truth),
                                       correct=predicted == truth, correct_b=predicted == other,
                                       evaluator_agreement=truth == other)
                        else:
                            row.update(predicted_class=None, reference_class=None, correct=None,
                                       correct_b=None, evaluator_agreement=None)
                        yield row
