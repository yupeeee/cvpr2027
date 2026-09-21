"""Exact coordinate conjugations; all generators receive native coordinates."""
from __future__ import annotations

import math
from typing import Callable

import torch


def coordinate(x: torch.Tensor, k: int, n: int, amplitude: float,
               inverse: bool = False) -> torch.Tensor:
    """Apply g_k (or its inverse) before any spatial pooling.

    Equal channel halves are required. Endpoints are identities explicitly,
    avoiding the floating point value of sin(pi).
    """
    if x.ndim != 4 or x.shape[1] < 2 or x.shape[1] % 2:
        raise ValueError("Coordinate coupling requires BCHW with even channels")
    if n < 1 or not 0 <= k <= n or not math.isfinite(amplitude):
        raise ValueError("Invalid coordinate stage, schedule length, or amplitude")
    if k == 0 or k == n:
        return x.clone()
    first, second = x.chunk(2, dim=1)
    a = amplitude * math.sin(math.pi * k / n)
    return torch.cat((first, second + (-a if inverse else a) * first.tanh()), dim=1)


def wrapped_step(sampler, w: torch.Tensor, k: int, conditioning, amplitude: float,
                 cost=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return wrapped next state and native predicted-clean latent."""
    if cost is not None:
        cost.add("coordinate_inverse_calls")
        cost.add("coordinate_forward_calls")
        cost.add("coordinate_elements", 2 * w.numel())
    native = coordinate(w, k, sampler.num_steps, amplitude, inverse=True)
    next_native, clean = sampler.step(native, k, conditioning, cost)
    return coordinate(next_native, k + 1, sampler.num_steps, amplitude), clean


def conjugate_editor(w: torch.Tensor, k: int, n: int, amplitude: float,
                     native_editor: Callable, cost=None):
    """Compute g_k I_k g_k^-1; editor budgets remain in native coordinates."""
    if cost is not None:
        cost.add("coordinate_inverse_calls")
        cost.add("coordinate_forward_calls")
        cost.add("coordinate_elements", 2 * w.numel())
    result = native_editor(coordinate(w, k, n, amplitude, inverse=True))
    if isinstance(result, tuple):
        edited, info = result
        return coordinate(edited, k, n, amplitude), info
    return coordinate(result, k, n, amplitude)


def clock_label(k: int, n: int, exponent: float) -> float:
    """Relabel unchanged states/work; no sampler update is performed."""
    if n <= 0 or not 0 <= k <= n or exponent <= 0:
        raise ValueError("Invalid clock parameters")
    return (k / n) ** exponent
