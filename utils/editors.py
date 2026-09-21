"""Local bounded controllers; no endpoint labels or continuation gradients."""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch.nn import functional as F

FIELDS = ("appearance", "composition", "category")


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().square().mean().sqrt()


def field(u: torch.Tensor, shape) -> torch.Tensor:
    """A fixed bilinear intervention subspace shared by all editors."""
    return F.interpolate(u, size=shape[-2:], mode="bilinear", align_corners=False)


def project(u: torch.Tensor, shape, radius: float, eps: float) -> torch.Tensor:
    """Project by radial scaling of the actual upsampled displacement."""
    scale = (radius / rms(field(u, shape)).clamp_min(eps)).clamp(max=1)
    return u * scale


def edit(x: torch.Tensor, k: int, conditioning, request: int, task: str,
         editor: str, budget: float, stage_rms: float, sampler, scorer,
         probe: Callable, args, cost, seed: int, object_name: str,
         task_config: dict) -> tuple[torch.Tensor, dict]:
    """Optimize a low-resolution latent field using only current observations.

    `probe` is a prompt-bound raw-state head returning Bx3 raw-scale scores.
    Random editing runs the nominated gradient controller to determine its
    norm, and charges that matching work explicitly. It randomizes direction
    within exactly the same low-resolution field family.
    """
    from utils.tasks import protected_fields, threshold

    if request not in (-1, 1) or task not in FIELDS[:2]:
        raise ValueError("An edit needs a binary request and declared task")
    if budget < 0 or not math.isfinite(budget) or stage_rms <= 0 or not math.isfinite(stage_rms):
        raise ValueError("Budget must be finite/nonnegative and training RMS positive")
    if x.shape[0] != 1:
        raise ValueError("Independent edits require a one-sample latent")
    if editor not in ("noop", "random", "probe", "preview"):
        raise ValueError(f"Unsupported editor {editor!r}")
    radius = budget * stage_rms
    base = x.detach().clone()
    target = FIELDS.index(task)
    protected = [FIELDS.index(name) for name in protected_fields(task_config, task)]
    decision = threshold(task_config, task, "a", "decision")
    margin = (threshold(task_config, task, "a", "margin")
              if args.edit_margin is None else args.edit_margin)
    info = {"editor": editor, "edit_iterations": 0, "surrogate_initial": None,
            "surrogate_final": None, "surrogate_target_initial": None,
            "surrogate_target_final": None, "random_match_editor": None, "status": "ok", "failure_reason": None}

    def failed(reason, input_valid=True):
        for name, value in info.items():
            if isinstance(value, float) and not math.isfinite(value):
                info[name] = None
        displacement = 0.0 if input_valid else None
        info.update(status="failed", failure_reason=reason, delta_rms=displacement,
                    delta_max=displacement, relative_rms=displacement,
                    training_stage_rms=stage_rms, budget_respected=input_valid)
        return base, info

    if not torch.isfinite(base).all():
        return failed("nonfinite_input_latent", input_valid=False)

    def observe(z: torch.Tensor) -> torch.Tensor:
        if editor == "probe":
            if probe is None:
                raise ValueError("Probe editing requires a fitted raw-state probe")
            return probe(z)
        clean = sampler.preview(z, k, conditioning, cost)
        image = sampler.decode(clean, cost)
        scores = scorer.score(image, object_name, cost)
        return torch.stack([scores[name] for name in FIELDS], dim=-1).float()

    if editor == "noop" or radius == 0:
        delta = torch.zeros_like(base)
    elif editor == "random":
        matched, matched_info = edit(
            base, k, conditioning, request, task, args.random_match_editor,
            budget, stage_rms, sampler, scorer, probe, args, cost, seed,
            object_name, task_config)
        if matched_info.get("status") == "failed":
            return failed("random_matching_controller_failed")
        matched_rms = float(rms(matched - base))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        u = torch.randn((1, base.shape[1], args.edit_grid, args.edit_grid),
                        generator=generator, dtype=torch.float32).to(base.device)
        direction = field(u, base.shape)
        delta = direction * (matched_rms / rms(direction).clamp_min(args.edit_eps))
        info.update(random_match_editor=args.random_match_editor,
                    matched_rms=matched_rms,
                    matching_edit_iterations=matched_info["edit_iterations"])
    else:
        # Clone outside inference_mode in callers; collection tensors never
        # become optimization variables. All frozen weights remain frozen.
        with torch.no_grad():
            reference = observe(base).detach()
        if not torch.isfinite(reference).all():
            return failed("nonfinite_controller_reference")
        u = torch.zeros((1, base.shape[1], args.edit_grid, args.edit_grid),
                        device=base.device, dtype=torch.float32, requires_grad=True)

        def objective(scores, displacement):
            hinge = F.relu(margin - request * (scores[:, target] - decision)).mean()
            preserve = ((scores[:, protected] - reference[:, protected]).square().mean()
                        if protected else scores.new_zeros(()))
            penalty = displacement.float().square().mean() / (stage_rms ** 2)
            return hinge + args.protected_weight * preserve + args.edit_penalty * penalty

        with torch.no_grad():
            info["surrogate_initial"] = float(objective(reference, field(u, base.shape)))
            info["surrogate_target_initial"] = float(reference[0, target])
        for _ in range(args.edit_steps):
            with torch.enable_grad():
                displacement = field(u, base.shape)
                scores = observe(base + displacement.to(base.dtype))
                loss = objective(scores, displacement)
                gradient, = torch.autograd.grad(loss, u)
            cost.add("edit_iterations", 1)
            cost.add("backward_passes", 1)
            info["edit_iterations"] += 1
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                return failed("nonfinite_controller_loss_or_gradient")
            # Relative-RMS step size makes the budget meaningful across stages.
            with torch.no_grad():
                gradient = gradient / rms(gradient).clamp_min(args.edit_eps)
                u.add_(gradient, alpha=-args.edit_lr * radius)
                u.copy_(project(u, base.shape, radius, args.edit_eps))
        delta = field(u.detach(), base.shape)
        with torch.no_grad():
            final = observe(base + delta.to(base.dtype))
            if not torch.isfinite(final).all():
                return failed("nonfinite_controller_final")
            final_loss = objective(final, delta)
            if not torch.isfinite(final_loss):
                return failed("nonfinite_controller_final_loss")
            info["surrogate_final"] = float(final_loss)
            info["surrogate_target_final"] = float(final[0, target])

    # Measure the displacement actually representable at the sampler dtype.
    edited = base + delta.to(base.dtype)
    actual = edited - base
    # Finite precision addition may push a projected field just over its radius.
    # Find a feasible representable point on the same field ray if necessary.
    if float(rms(actual)) > radius:
        lower, upper = 0.0, 1.0
        while upper - lower > torch.finfo(torch.float32).eps:
            middle = (lower + upper) / 2
            candidate = base + (middle * delta).to(base.dtype)
            if float(rms(candidate - base)) <= radius:
                lower = middle
            else:
                upper = middle
        edited = base + (lower * delta).to(base.dtype)
        actual = edited - base
    actual_rms = float(rms(actual))
    relative = actual_rms / stage_rms
    info.update(delta_rms=actual_rms, delta_max=float(actual.float().abs().max()),
                relative_rms=relative, training_stage_rms=stage_rms,
                budget_respected=relative <= budget + args.budget_tolerance)
    return edited.detach(), info


def needs_intervention(prediction: float, request: int, decision: float,
                       margin: float) -> bool:
    """A declared request requires the full task margin, not just its sign."""
    return request * (prediction - decision) < margin


def confident_violation(prediction: float, request: int, decision: float,
                        margin: float, confidence: float) -> bool:
    return request * (prediction - decision) < margin - confidence
