"""Bounded offline probe/controller checks using temporary synthetic fixtures."""
import copy
import json
from pathlib import Path

import pytest
import torch

from utils.config import parse_args
from utils.data import Run, atomic_torch, read_json
from utils.distributed import Distributed
from utils.editors import edit, field, project, rms
from utils.probes import features, load_probe, load_stage_scales, probe_path, readout_rows, train_probes
from test_sampler import Counter, tiny_sampler


def arguments(tmp_path, extra=()):
    return parse_args("test", ["--device", "cpu", "--logs-dir", str(tmp_path / "logs"),
        "--ckpts-dir", str(tmp_path / "ckpts"), "--figs-dir", str(tmp_path / "figs"),
        "--num-samples", "8", "--train-fraction", "0.5", "--validation-fraction", "0.25",
        "--num-steps", "1", "--stages", "0", "1", "--probe-heads", "linear", "mlp",
        "--probe-hidden-sizes", "2", "--probe-epochs", "1", "--probe-batch-size", "2",
        "--pool-grid", "2", "--edit-grid", "2", "--edit-steps", "2", "--edit-margin", "1",
        *extra])


def _fixture_cache(args, run):
    for index, sample in enumerate(run.samples):
        x = torch.arange(4 * 4 * 4).float().reshape(1, 4, 4, 4) / 64 + index / 10
        # Clearly separated held-out labels detect accidental all-data normalization.
        value = index / 10 if sample["split"] == "train" else 100 + index
        scores = {"appearance": value, "composition": -value, "category": value / 2}
        trajectory = {"fingerprint": run.fingerprint, "sample": sample,
            "states": {0: x, 1: x + 1}, "clean": {0: x + 1, 1: x + 1},
            "scores": {"a": scores, "b": scores.copy()}, "metadata": {"timesteps": [0]},
            "stage_costs": {k: {"generation": {"unet_calls": k},
                                 "preview": {"unet_calls": 1 - k}} for k in (0, 1)},
            "status": "ok"}
        run.write_tensor(f"trajectories/{sample['sample_id']}.pt", trajectory)


def test_spatial_pooling_prompt_and_projection():
    x = torch.zeros(1, 4, 4, 4)
    x[:, :, :2] = 1
    y = x.flip(-2)
    assert torch.equal(x.mean((-2, -1)), y.mean((-2, -1)))
    assert not torch.equal(features(x, 0, 2, 2), features(y, 0, 2, 2))
    assert features(x, 1, 2, 2)[0, -2:].tolist() == [0, 1]
    u = torch.randn(1, 4, 2, 2)
    projected = project(u, (1, 4, 9, 9), 0.12, 1e-8)
    assert float(rms(field(projected, (1, 4, 9, 9)))) <= 0.120001


def test_tiny_fit_training_only_statistics_resume_and_input_gradient(tmp_path):
    args = arguments(tmp_path)
    with Distributed(args) as dist:
        run = Run(args, dist, "fit")
        _fixture_cache(args, run)
        train_probes(args, dist, run)
        stored = torch.load(probe_path(run, 0, "raw", "linear"), weights_only=True)
        train = [s for s in run.samples if s["split"] == "train"]
        targets = torch.tensor([run.load_trajectory(s["sample_id"])["scores"]["a"]["appearance"] for s in train])
        torch.testing.assert_close(stored["label_mean"][0], targets.mean())
        assert stored["label_mean"][0] < 1  # Held-out labels are > 100.
        absent_prompts = set(range(len(run.prompts))) - {sample["prompt_index"] for sample in train}
        for index in absent_prompts:
            assert stored["feature_scale"][-len(run.prompts) + index] == 1
        assert set(stored["metadata"]["split_ids"]["train"]).isdisjoint(stored["metadata"]["split_ids"]["test"])
        scales = load_stage_scales(run)
        expected = torch.cat([run.load_trajectory(s["sample_id"])["states"][0] for s in train]).square().mean().sqrt()
        assert scales[0] == pytest.approx(float(expected))
        probe = load_probe(probe_path(run, 0, "raw", "linear"), dist.device)
        x = run.load_trajectory(train[0]["sample_id"])["states"][0].clone().requires_grad_()
        probe.predict(x, train[0]["prompt_index"]).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(p.grad is None for p in probe.parameters())
        rows = run.rows("fit")
        assert len(rows) == 4 * 2 * (4 + 1) * 3
        assert all(r["split"] in ("validation", "test") for r in rows)
        assert all(r["observation_work"] == 0 for r in rows if r["access"] != "clean")
        args.resume = True
        resumed = Run(args, dist, "fit")
        train_probes(args, dist, resumed)
        assert len(resumed.rows("fit")) == len(rows)
        malformed_sample = next(s for s in run.samples if s["split"] == "test")
        trajectory = run.load_trajectory(malformed_sample["sample_id"])
        trajectory["states"][0].fill_(float("nan"))
        run.write_tensor(f"trajectories/{malformed_sample['sample_id']}.pt", trajectory)
        malformed = [r for r in readout_rows(args, dist, run, raw_only=True)
                     if r["sample_id"] == malformed_sample["sample_id"] and r["stage"] == 0]
        assert all(r["status"] == "malformed" and r["prediction_a"] is None for r in malformed)
        json.dumps(malformed, allow_nan=False)


def test_malformed_training_is_recorded_and_stops(tmp_path):
    args = arguments(tmp_path)
    with Distributed(args) as dist:
        run = Run(args, dist, "fit")
        _fixture_cache(args, run)
        sample = next(s for s in run.samples if s["split"] == "train")
        trajectory = run.load_trajectory(sample["sample_id"])
        trajectory["status"] = "malformed"
        run.write_tensor(f"trajectories/{sample['sample_id']}.pt", trajectory)
        with pytest.raises(ValueError, match="Malformed training"):
            train_probes(args, dist, run)
        assert run.rows("fit_failures")[0]["status"] == "malformed"


def test_noop_probe_random_norm_match_and_failures(tmp_path):
    args = arguments(tmp_path)
    config = read_json(args.tasks_path)
    x = torch.zeros(1, 4, 4, 4)
    probe = lambda z: torch.stack((z[:, 0].mean((-2, -1)), z[:, 1].mean((-2, -1)), z[:, 2].mean((-2, -1))), dim=1)
    def run(kind, function=probe):
        return edit(x, 0, None, 1, "appearance", kind, 0.1, 1.0, None, None,
                    function, args, Counter(), 7, "object", config)
    noop, info = run("noop")
    assert torch.equal(noop, x) and info["delta_rms"] == 0
    edited, info = run("probe")
    assert info["budget_respected"] and info["relative_rms"] <= 0.1 + args.budget_tolerance
    assert probe(edited)[0, 0] > probe(x)[0, 0]
    random, random_info = run("random")
    assert random_info["delta_rms"] == pytest.approx(info["delta_rms"], abs=1e-7)
    assert not torch.allclose(random, edited)
    assert torch.equal(run("random")[0], random)
    failed, failed_info = run("probe", lambda z: probe(z) * float("nan"))
    assert torch.equal(failed, x) and failed_info["status"] == "failed"
    assert failed_info["surrogate_final"] is None
    bad_input = x.clone().fill_(float("nan"))
    for kind in ("noop", "probe"):
        _, bad_info = edit(bad_input, 0, None, 1, "appearance", kind, 0.1, 1.0,
            None, None, probe, args, Counter(), 7, "object", config)
        assert bad_info["delta_rms"] is None and not bad_info["budget_respected"]
        json.dumps(bad_info, allow_nan=False)
    cost = Counter()
    invalid_gradient = lambda z: probe(z).sqrt()
    _, gradient_info = edit(x, 0, None, 1, "appearance", "probe", 0.1, 1.0,
        None, None, invalid_gradient, args, cost, 7, "object", config)
    assert gradient_info["status"] == "failed" and cost["backward_passes"] == 1
    json.dumps(gradient_info, allow_nan=False)


def test_real_preview_editor_gradient_and_terminal_no_unet(tmp_path, tiny_sampler):
    args = arguments(tmp_path)
    sampler = tiny_sampler
    class DifferentiableScorer:
        def score(self, images, object_name, cost):
            cost.add("image_scorer_calls", len(images))
            return {"appearance": images[:, 0].mean((-2, -1)),
                    "composition": images[:, 1].mean((-2, -1)),
                    "category": images[:, 2].mean((-2, -1))}
    conditioning = sampler.encode("object", "", Counter())
    x = sampler.initial(19, 8, 8)
    for k in (1, sampler.num_steps):
        cost = Counter()
        result, info = edit(x, k, conditioning, 1, "appearance", "preview", 0.05,
            float(rms(x)), sampler, DifferentiableScorer(), None, args, cost, 0,
            "object", read_json(args.tasks_path))
        assert info["status"] == "ok" and info["budget_respected"]
        assert torch.isfinite(result).all() and not torch.equal(result, x)
        assert cost["backward_passes"] == args.edit_steps
        if k == sampler.num_steps:
            assert cost.get("unet_calls", 0) == 0
    assert all(p.grad is None and not p.requires_grad for p in sampler.unet.parameters())
