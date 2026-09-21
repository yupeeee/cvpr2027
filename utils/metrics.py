"""Separate work counters and semantic proxy measurements; seed-cluster uncertainty."""
import math
import time
from contextlib import contextmanager

import numpy as np


class Cost:
    def __init__(self):
        self.counters = {}

    def add(self, name, amount=1):
        self.counters[name] = self.counters.get(name, 0) + amount

    def to_dict(self):
        return dict(self.counters)

    @contextmanager
    def measure(self, device):
        import torch
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        try:
            yield self
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                self.counters["peak_memory_bytes"] = max(self.counters.get("peak_memory_bytes", 0), torch.cuda.max_memory_allocated(device))
            self.add("seconds", time.perf_counter() - start)


def add_costs(*costs):
    result = {}
    for cost in costs:
        for key, value in cost.items():
            if isinstance(value, (float, int)):
                result[key] = max(result.get(key, 0), value) if key == "peak_memory_bytes" else result.get(key, 0) + value
    return result


def finite_scores(scores):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in scores.values())


def trial_metrics(reference, edited, task, request, tasks, role, budget_respected, assessment=None):
    """assessment is the prospective act/no-act decision, not an earlier deferral."""
    from .tasks import threshold, protected_fields
    decision = threshold(tasks, task, role, "decision")
    margin = threshold(tasks, task, role, "margin")
    valid = finite_scores(reference) and finite_scores(edited)
    old_satisfied = valid and request * (reference[task] - decision) >= margin
    target = valid and request * (edited[task] - decision) >= margin
    errors = {name: abs(edited[name] - reference[name]) if valid else None for name in protected_fields(tasks, task)}
    preserved = valid and all(error <= threshold(tasks, name, role, "protected_tolerance") for name, error in errors.items())
    correct = bool(assessment) == (not old_satisfied) if assessment is not None and valid else None
    selective = bool(target and preserved and budget_respected)
    return {"originally_unsatisfied": not old_satisfied if valid else None,
            "reference_score": reference.get(task), "final_score": edited.get(task),
            "reference_decision": int(reference[task] >= decision) if valid else None,
            "final_decision": int(edited[task] >= decision) if valid else None,
            "target_success": bool(target), "target_failure": not target,
            "protected_errors": errors, "protected_ok": bool(preserved),
            "budget_respected": bool(budget_respected), "selective_success": selective,
            "assessment_correct": correct,
            "joint_success": bool(correct and selective) if assessment is not None else None,
            "valid_scores": valid}


def regression_metrics(prediction, reference, threshold=0.0):
    p, r = np.asarray(prediction), np.asarray(reference)
    finite = np.isfinite(p) & np.isfinite(r)
    p, r = p[finite], r[finite]
    if not len(r):
        return {"n": 0, "mse": None, "accuracy": None, "balanced_accuracy": None}
    correct = (p >= threshold) == (r >= threshold)
    recalls = [float(correct[r >= threshold].mean()) if (r >= threshold).any() else None,
               float(correct[r < threshold].mean()) if (r < threshold).any() else None]
    return {"n": len(r), "mse": float(np.square(p - r).mean()), "accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.mean(recalls)) if None not in recalls else None}


def weighted_mean(sums, counts):
    count = sum(counts)
    return sum(sums) / count if count else None


def cluster_bootstrap(rows, value, seed, count, confidence, statistic=None):
    """Resample complete root-seed clusters (all their related observations)."""
    groups = {}
    missing = 0
    for row in rows:
        v = value(row)
        if v is None or not math.isfinite(float(v)):
            missing += 1
            continue
        groups.setdefault(row["root_seed"], []).append(float(v))
    clusters = list(groups.values())
    if not clusters:
        return {"mean": None, "low": None, "high": None, "n": 0, "roots": 0, "missing": missing}
    stat = statistic or np.mean
    estimate = float(stat(np.concatenate(clusters)))
    # One root does not support a population uncertainty interval.
    if len(clusters) < 2:
        low = high = None
    else:
        rng = np.random.default_rng(seed)
        draws = [float(stat(np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]))) for _ in range(count)]
        alpha = (1 - confidence) / 2
        low, high = map(float, np.quantile(draws, [alpha, 1 - alpha]))
    return {"mean": estimate, "low": low, "high": high, "n": sum(map(len, clusters)), "roots": len(clusters), "missing": missing}


def paired_bootstrap(left, right, key, value, seed, count, confidence):
    a, b = {key(r): r for r in left}, {key(r): r for r in right}
    paired = []
    for k in sorted(a.keys() & b.keys()):
        av, bv = value(a[k]), value(b[k])
        if av is not None and bv is not None:
            paired.append({"root_seed": a[k]["root_seed"], "difference": float(av) - float(bv)})
    result = cluster_bootstrap(paired, lambda r: r["difference"], seed, count, confidence)
    result.update(unmatched_left=len(a.keys() - b.keys()), unmatched_right=len(b.keys() - a.keys()))
    return result
