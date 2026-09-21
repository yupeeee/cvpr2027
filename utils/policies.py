"""Prospective decision rules. Inputs contain present observations, never endpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PolicySpec:
    name: str
    stage: int
    stages: tuple[int, ...]
    threshold: float
    deadline: int
    fallback: str
    time_only_action: bool = True


def decide(spec: PolicySpec, stage: int, violation_margin: float | None) -> tuple[bool, bool]:
    """Return (decision_is_final, act); a deferral is not a no-need assessment.

    A positive margin means the current observer predicts that the requested
    success margin is unmet. ``threshold`` is a native matching-score distance,
    not a probability. At the deadline ``predicted`` uses its sign alone.
    """
    if spec.name == 'never':
        return True, False
    if spec.name in {'always', 'time_only'}:
        return stage == spec.stage, spec.name == 'always' or spec.time_only_action
    if spec.name in {'fixed', 'single_stage', 'endpoint'}:
        if stage != spec.stage:
            return False, False
        if violation_margin is None:
            raise ValueError('A scored observation is required for this decision')
        return True, violation_margin > 0
    if spec.name != 'adaptive':
        raise ValueError(f'Unknown policy: {spec.name}')
    if stage not in spec.stages and stage != spec.deadline:
        return False, False
    if violation_margin is None:
        raise ValueError('Adaptive inspection needs a present-state score')
    if violation_margin > spec.threshold:
        return True, True
    if stage >= spec.deadline:
        if spec.fallback == 'edit':
            return True, True
        if spec.fallback == 'skip':
            return True, False
        if spec.fallback == 'predicted':
            return True, violation_margin > 0
        raise ValueError(f'Unknown policy fallback: {spec.fallback}')
    return False, False


def inspection_stages(spec: PolicySpec, total_steps: int) -> tuple[int, ...]:
    if spec.name == 'never':
        return (0,)
    if spec.name == 'adaptive':
        return tuple(sorted({k for k in spec.stages if k <= spec.deadline} | {spec.deadline}))
    if spec.name == 'endpoint':
        return (total_steps,)
    return (spec.stage,)


def select_validation_settings(rows: Sequence[dict], stages: Sequence[int],
                               thresholds: Sequence[float]) -> dict:
    """Freeze choices using validation joint success, then online work tie-breaks."""
    if not rows or any(row['split'] != 'validation' for row in rows):
        raise ValueError('Policy calibration requires nonempty validation-only records')
    groups: dict[tuple[str, float], list[dict]] = {}
    necessity: dict[str, dict[str, bool]] = {}
    for row in rows:
        kind = row['calibration_kind']
        value = float(row['calibration_value'])
        groups.setdefault((kind, value), []).append(row)
        group = f"{row['task']}:{row['request']}"
        if row['a']['originally_unsatisfied'] is not None:
            necessity.setdefault(group, {})[row['sample_id']] = bool(row['a']['originally_unsatisfied'])

    def choose(kind: str, candidates: Sequence[float]) -> float:
        def objective(value: float) -> tuple[float, float, float]:
            records = groups.get((kind, float(value)), [])
            if not records:
                raise ValueError(f'Missing validation candidate {kind}={value}')
            success = sum(float(r['a']['joint_success']) for r in records) / len(records)
            work = sum(float(r['costs']['online'].get('unet_samples', 0)) for r in records) / len(records)
            return success, -work, -float(value)
        return max(candidates, key=objective)

    return {
        'stage': int(choose('stage', stages)),
        'threshold': float(choose('threshold', thresholds)),
        'time_only_actions': {key: sum(values.values()) > len(values) / 2
                              for key, values in necessity.items()},
        'validation_sample_ids': sorted({r['sample_id'] for r in rows}),
        'invalid_reference_trial_count': sum(r['a']['originally_unsatisfied'] is None for r in rows),
        'selection': 'maximum validation joint success; ties: fewer UNet calls, smaller candidate',
        'confidence_units': 'evaluator-A native matching-score distance; not probability',
    }
