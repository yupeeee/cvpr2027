"""One source of defaults; CLI overrides JSON, which overrides these values."""
import argparse
import json
import math
import re
from pathlib import Path


DEFAULTS = {
    "run_id": "demo", "logs_dir": "logs", "ckpts_dir": "ckpts", "figs_dir": "figs",
    "prompts_path": "configs/prompts.jsonl", "tasks_path": "configs/tasks.json",
    "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5", "model_revision": "main",
    "evaluator_a_id": "openai/clip-vit-base-patch32", "evaluator_a_revision": "main",
    "evaluator_b_id": "google/siglip-base-patch16-224", "evaluator_b_revision": "main",
    "local_files_only": True, "resume": True, "overwrite": False, "dry_run": False,
    "device": "auto", "precision": "float32", "evaluator_precision": "float32",
    "distributed_timeout": 180, "torch_threads": 1,
    "progress": True, "progress_mininterval": 1.0,
    "seed": 2027, "split_seed": 29, "request_seed": 41, "num_samples": 128,
    "train_fraction": 0.6, "validation_fraction": 0.2,
    "height": 512, "width": 512, "num_steps": 30, "stages": [0, 5, 15, 25, 30],
    "negative_prompt": "", "guidance_scale": 7.5, "scheduler": "ddim", "ddim_eta": 0.0,
    "ddim_timestep_spacing": "leading", "ddim_steps_offset": 1,
    "ddim_clip_sample": False, "ddim_set_alpha_to_one": False,
    "scheduler_prediction_type": "native",
    "evaluator_a_resize_mode": "native", "evaluator_b_resize_mode": "native",
    "evaluator_a_size": 0, "evaluator_b_size": 0,
    "evaluator_a_mean": [], "evaluator_a_std": [],
    "evaluator_b_mean": [], "evaluator_b_std": [],
    "evaluator_interpolation": "bicubic", "evaluator_antialias": True,
    "pool_grid": 4, "probe_heads": ["linear", "mlp"], "probe_hidden_sizes": [32, 128],
    "probe_epochs": 50, "probe_batch_size": 32, "probe_lr": 0.001,
    "probe_weight_decay": 0.0, "probe_seed": 23, "normalization_eps": 1e-6,
    "edit_grid": 8, "edit_steps": 10, "edit_lr": 0.25, "edit_margin": None,
    "protected_weight": 1.0, "edit_penalty": 0.01, "edit_eps": 1e-8,
    "edit_budgets": [0.05, 0.1], "editors": ["noop", "random", "probe", "preview"],
    "random_match_editor": "probe", "editor_head": "linear", "budget_tolerance": 1e-6,
    "tasks": ["appearance", "composition"], "eval_split": "test",
    "intervention_limit": 16, "diagnostic_limit": 4, "policy_limit": 8,
    "intervention_stages": None, "diagnostic_stages": None, "policy_stages": None,
    "control_limit": 4, "control_stages": None, "control_editors": ["noop", "probe"],
    "coordinate_amplitude": 2.0, "clock_power": 2.0,
    "endpoint_atol": 1e-4, "endpoint_rtol": 1e-4,
    "policy_editor": "preview", "policy_head": "linear",
    "policy_thresholds": [0.0, 0.25, 0.5, 1.0], "policy_deadline": None,
    "policy_fallback": "predicted", "policy_calibration_limit": 8, "policy_time_only_default": False,
    "policy_variants": ["never", "always", "time_only", "fixed", "adaptive", "endpoint", "single_stage"],
    "bootstrap_seed": 53, "bootstrap_count": 1000, "confidence_level": 0.95,
    "useful_success": 0.9, "audit_count": 8, "audit_labels": None,
    "plot_dpi": 150, "plot_width": 8.0, "plot_height": 4.5,
    "raw_display_scale": 3.0, "plot_splits": ["test"],
}
LIST_TYPES = {
    "stages": int, "probe_heads": str, "probe_hidden_sizes": int,
    "edit_budgets": float, "editors": str, "tasks": str,
    "intervention_stages": int, "diagnostic_stages": int, "policy_stages": int,
    "policy_thresholds": float, "policy_variants": str, "plot_splits": str,
    "control_stages": int, "control_editors": str,
    "evaluator_a_mean": float, "evaluator_a_std": float,
    "evaluator_b_mean": float, "evaluator_b_std": float,
}
OPTIONAL_TYPES = {"edit_margin": float, "policy_deadline": int, "audit_labels": str}
CHOICES = {
    "device": ["auto", "cpu", "cuda"], "precision": ["float32", "float16", "bfloat16"],
    "evaluator_precision": ["float32", "float16", "bfloat16"], "scheduler": ["ddim"],
    "ddim_timestep_spacing": ["leading", "linspace", "trailing"],
    "scheduler_prediction_type": ["native", "epsilon", "v_prediction", "sample"],
    "evaluator_a_resize_mode": ["native", "shortest", "square"],
    "evaluator_b_resize_mode": ["native", "shortest", "square"],
    "evaluator_interpolation": ["bicubic", "bilinear"],
    "random_match_editor": ["probe", "preview"], "policy_editor": ["probe", "preview"],
    "eval_split": ["validation", "test"], "policy_fallback": ["edit", "skip", "predicted"],
}


def parse_args(command: str, argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str)
    known, _ = pre.parse_known_args(argv)
    defaults = DEFAULTS.copy()
    if known.config:
        loaded = json.loads(Path(known.config).read_text())
        if not isinstance(loaded, dict):
            pre.error("JSON config must be an object")
        unknown = set(loaded) - set(defaults)
        if unknown:
            pre.error(f"Unknown config keys: {sorted(unknown)}")
        defaults.update(loaded)
    parser = argparse.ArgumentParser(description=f"Beyond the Endpoint: {command}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Optional JSON configuration")
    for key, default in DEFAULTS.items():
        kw = {"default": defaults[key], "help": key.replace("_", " ")}
        if isinstance(default, bool):
            kw["action"] = argparse.BooleanOptionalAction
        elif key in LIST_TYPES:
            kw.update(type=LIST_TYPES[key], nargs="+")
        else:
            kw["type"] = OPTIONAL_TYPES.get(key, type(default))
        if key == "device":
            kw["help"] = "auto: all usable visible GPUs (CPU fallback); cuda: explicit rank device; cpu: CPU only"
        elif key == "local_files_only":
            kw["help"] = "require cached/local weights; explicit --no-local-files-only permits missing-weight downloads when you run an experiment"
        elif key == "resume":
            kw["help"] = "legacy compatibility flag; valid saved work is always reused unless --overwrite"
        elif key == "overwrite":
            kw["help"] = "recompute this stage and invalidate its dependent outputs; the launcher recomputes the whole workflow"
        elif key == "progress":
            kw["help"] = "show one combined tqdm bar with every device's counts, work, elapsed time and ETA"
        elif key == "progress_mininterval":
            kw["help"] = "minimum seconds between progress refreshes; combined display polls at most 10 times per second"
        if key in CHOICES:
            kw["choices"] = CHOICES[key]
        parser.add_argument("--" + key.replace("_", "-"), **kw)
    args = parser.parse_args(argv)
    try:
        validate(args)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    return args


def validate(a):
    # Reject NaN/Inf accepted by Python JSON and argparse floats.
    for key, value in vars(a).items():
        values = value if isinstance(value, list) else [value]
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain finite values")
    for key in ("probe_heads", "probe_hidden_sizes", "editors", "edit_budgets", "tasks", "policy_thresholds", "policy_variants", "control_editors", "plot_splits"):
        value = getattr(a, key)
        if isinstance(value, list) and len(set(value)) != len(value):
            raise ValueError(f"{key} must not contain duplicates")
    # Validate JSON types too: argparse does not validate non-string defaults.
    for key, default in DEFAULTS.items():
        value = getattr(a, key)
        if key in LIST_TYPES:
            if value is None and default is None:
                continue
            typ = LIST_TYPES[key]
            if not isinstance(value, list) or any(isinstance(v, bool) or not isinstance(v, (int, float) if typ is float else typ) for v in value):
                raise ValueError(f"{key} must be a list of {typ.__name__}")
        elif value is None and default is None:
            continue
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean")
        else:
            typ = OPTIONAL_TYPES.get(key, type(default))
            if isinstance(value, bool) or not isinstance(value, (int, float) if typ is float else typ):
                raise ValueError(f"{key} must be {typ.__name__}")
        if key in CHOICES and value not in CHOICES[key]:
            raise ValueError(f"Unsupported {key}: {value}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.run_id):
        raise ValueError("run_id must be a simple directory name")
    for key in ("num_samples", "num_steps", "height", "width", "pool_grid", "probe_epochs", "probe_batch_size", "edit_grid", "edit_steps", "distributed_timeout", "torch_threads", "bootstrap_count", "plot_dpi"):
        if getattr(a, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if a.ddim_eta != 0:
        raise ValueError("Only deterministic DDIM (eta=0) is implemented")
    if not 0 < a.train_fraction < 1 or not 0 < a.validation_fraction < 1 or a.train_fraction + a.validation_fraction >= 1:
        raise ValueError("train and validation fractions must be positive with sum < 1")
    if not 0 < a.confidence_level < 1 or not 0 < a.useful_success <= 1:
        raise ValueError("confidence_level must be in (0,1); useful_success in (0,1]")
    if a.stages != sorted(set(a.stages)) or not a.stages or a.stages[0] != 0 or a.stages[-1] != a.num_steps:
        raise ValueError("stages must be sorted, unique, include 0 and num_steps")
    for key in ("intervention_stages", "diagnostic_stages", "policy_stages", "control_stages"):
        if getattr(a, key) is None:
            setattr(a, key, list(a.stages))
        v = getattr(a, key)
        if not v or v != sorted(set(v)) or not set(v) <= set(a.stages):
            raise ValueError(f"{key} must be an ordered subset of collected stages")
    if a.policy_deadline is None:
        a.policy_deadline = a.policy_stages[-1]
    if a.policy_deadline not in a.policy_stages:
        raise ValueError("policy_deadline must be a selected policy stage")
    for key in ("intervention_limit", "diagnostic_limit", "policy_limit", "control_limit", "policy_calibration_limit", "audit_count"):
        if getattr(a, key) < 0:
            raise ValueError(f"{key} must be >= 0")
    if a.edit_margin is not None and a.edit_margin < 0:
        raise ValueError("edit_margin must be nonnegative")
    if not a.edit_budgets or any(b < 0 for b in a.edit_budgets):
        raise ValueError("edit_budgets must be nonempty and nonnegative")
    if not set(a.editors) <= {"noop", "random", "probe", "preview"}:
        raise ValueError("Unsupported editor")
    if not set(a.control_editors) <= {"noop", "random", "probe", "preview"}:
        raise ValueError("Unsupported control editor")
    if not set(a.probe_heads) <= {"linear", "mlp"} or not a.probe_heads or any(n <= 0 for n in a.probe_hidden_sizes):
        raise ValueError("probe_heads must contain linear/mlp; hidden sizes must be positive")
    if "mlp" in a.probe_heads and not a.probe_hidden_sizes:
        raise ValueError("MLP needs at least one hidden size")
    heads = ({"linear"} if "linear" in a.probe_heads else set()) | ({f"mlp{n}" for n in a.probe_hidden_sizes} if "mlp" in a.probe_heads else set())
    if a.editor_head not in heads or a.policy_head not in heads:
        raise ValueError("editor_head/policy_head must name a configured probe head")
    if not set(a.tasks) <= {"appearance", "composition"} or not a.tasks:
        raise ValueError("tasks must select appearance and/or composition")
    if not a.plot_splits or not set(a.plot_splits) <= {"train", "validation", "test"}:
        raise ValueError("plot_splits must select train, validation, and/or test")
    if not a.policy_variants or not set(a.policy_variants) <= {"never", "always", "time_only", "fixed", "adaptive", "endpoint", "single_stage"}:
        raise ValueError("Unknown policy variant")
    for key in ("probe_lr", "edit_lr", "normalization_eps", "edit_eps", "clock_power", "raw_display_scale", "plot_width", "plot_height"):
        if getattr(a, key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("protected_weight", "edit_penalty", "probe_weight_decay", "endpoint_atol", "endpoint_rtol", "budget_tolerance", "guidance_scale", "progress_mininterval"):
        if getattr(a, key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    for role in ("a", "b"):
        for field in ("mean", "std"):
            values = getattr(a, f"evaluator_{role}_{field}")
            if values and len(values) != 3:
                raise ValueError(f"evaluator_{role}_{field} needs 3 values")
            if field == "std" and any(v <= 0 for v in values):
                raise ValueError("normalization standard deviations must be positive")
    if not a.policy_thresholds or any(t < 0 for t in a.policy_thresholds):
        raise ValueError("policy_thresholds must be nonempty and nonnegative")


def dry_run(args, command):
    if not args.dry_run:
        return False
    from .data import make_samples, scientific_fingerprint
    samples, _ = make_samples(args)
    print(json.dumps({"command": command, "config": vars(args),
                      "fingerprint": scientific_fingerprint(args),
                      "split_counts": {s: sum(x["split"] == s for x in samples) for s in ("train", "validation", "test")},
                      "message": "Configuration only; no models, CUDA, or experiment outputs."}, indent=2))
    return True
