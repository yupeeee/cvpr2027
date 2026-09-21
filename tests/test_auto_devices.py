"""CUDA discovery and launcher checks use stubs; no real GPU or models touched."""
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from utils import distributed


@pytest.fixture(autouse=True)
def clear_launch_environment(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)


def mock_cuda(monkeypatch, count, failed=(), sync_failed=()):
    calls = []
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(distributed.torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(distributed.torch.cuda, "device", lambda index: nullcontext())

    def allocate(size, device):
        assert size == 1
        calls.append(("allocate", device.index))
        if device.index in failed:
            raise RuntimeError("device unavailable")
        return object()

    def synchronize(index):
        calls.append(("synchronize", index))
        if index in sync_failed:
            raise RuntimeError("synchronization failed")

    monkeypatch.setattr(distributed.torch, "empty", allocate)
    monkeypatch.setattr(distributed.torch.cuda, "synchronize", synchronize)
    return calls


def test_discovery_excludes_allocation_and_synchronization_failures(monkeypatch):
    calls = mock_cuda(monkeypatch, 4, failed=(1,), sync_failed=(2,))
    with pytest.warns(RuntimeWarning, match="Skipping unusable visible CUDA device") as warnings:
        assert distributed.discover_usable_cuda() == [0, 3]
    assert len(warnings) == 2
    assert calls == [("allocate", 0), ("synchronize", 0), ("allocate", 1),
                     ("allocate", 2), ("synchronize", 2), ("allocate", 3),
                     ("synchronize", 3)]


@pytest.mark.parametrize("visibility,expected", [(None, "0,2"), ("5,3,7", "5,7"),
                                                  ("GPU-a,GPU-b,GPU-c", "GPU-a,GPU-c"),
                                                  ("MIG-a,MIG-b,MIG-c", "MIG-a,MIG-c")])
def test_launcher_filters_visibility_and_preserves_cli(monkeypatch, visibility, expected):
    if visibility is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    mock_cuda(monkeypatch, 3, failed=(1,))
    launched = []
    monkeypatch.setattr(os, "execvpe", lambda *values: launched.append(values))
    argv = ["--device", "auto", "--config", "path with spaces/demo.json", "--run-id", "pilot"]
    with pytest.warns(RuntimeWarning, match="device 1"):
        assert distributed.launch_if_needed(SimpleNamespace(device="auto"), "collect", argv)
    executable, command, env = launched[0]
    assert executable == sys.executable
    assert command == [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nnodes=1", "--nproc-per-node", "2", "--module", "exps.collect", *argv]
    assert env["CUDA_VISIBLE_DEVICES"] == expected
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visibility


def test_one_usable_gpu_is_remapped_in_fresh_worker(monkeypatch):
    mock_cuda(monkeypatch, 2, failed=(0,))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bad,GPU-good")
    monkeypatch.setattr(sys, "argv", ["exps.fit", "--device", "auto"])
    launched = []
    monkeypatch.setattr(os, "execvpe", lambda *values: launched.append(values))
    with pytest.warns(RuntimeWarning, match="device 0"):
        assert distributed.launch_if_needed(SimpleNamespace(device="auto"), "exps.fit")
    _, command, env = launched[0]
    assert command[command.index("--nproc-per-node") + 1] == "1"
    assert command[-4:] == ["--module", "exps.fit", "--device", "auto"]
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-good"


@pytest.mark.parametrize("count", [0, 2])
def test_no_usable_gpu_falls_back_to_cpu_without_environment_mutation(monkeypatch, count):
    mock_cuda(monkeypatch, count, failed=(0, 1))
    before = dict(os.environ)
    args = SimpleNamespace(device="auto")
    if count:
        with pytest.warns(RuntimeWarning, match="Skipping unusable"):
            assert not distributed.launch_if_needed(args, "fit", [])
    else:
        assert not distributed.launch_if_needed(args, "fit", [])
    assert args.device == "cpu"
    assert distributed.Distributed(args).device.type == "cpu"
    assert dict(os.environ) == before


@pytest.mark.parametrize("rank_variable", ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
def test_existing_torchrun_does_not_probe_or_relaunch(monkeypatch, rank_variable):
    monkeypatch.setenv(rank_variable, "1")

    def unexpected():
        pytest.fail("existing workers must not rediscover CUDA or recursively launch")

    monkeypatch.setattr(distributed, "discover_usable_cuda", unexpected)
    assert not distributed.launch_if_needed(SimpleNamespace(device="auto"), "collect", [])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_device_does_not_launch(monkeypatch, device):
    monkeypatch.setattr(distributed, "discover_usable_cuda", lambda: pytest.fail("unexpected probe"))
    assert not distributed.launch_if_needed(SimpleNamespace(device=device), "collect", [])


def test_auto_worker_binds_local_rank_and_honors_explicit_worker_count(monkeypatch):
    mock_cuda(monkeypatch, 4)
    for name, value in {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2",
                        "LOCAL_WORLD_SIZE": "2"}.items():
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(device="auto", torch_threads=1, distributed_timeout=30, progress=False)
    assert not distributed.launch_if_needed(args, "collect", [])
    context = distributed.Distributed(args)
    assert str(context.device) == "cuda:1"
    assert context.world_size == 2
    bound = []
    monkeypatch.setattr(distributed.torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(distributed.torch.cuda, "set_device", lambda device: bound.append(str(device)))
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    with context:
        assert bound == ["cuda:1"]


def test_worker_count_exceeding_visible_gpus_fails_before_group_initialization(monkeypatch):
    mock_cuda(monkeypatch, 1)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        distributed.Distributed(SimpleNamespace(device="auto"))


@pytest.mark.parametrize("environment", [{"RANK": "2", "WORLD_SIZE": "2"},
                                          {"LOCAL_RANK": "-1"}, {"WORLD_SIZE": "0"},
                                          {"LOCAL_RANK": "2", "LOCAL_WORLD_SIZE": "2"}])
def test_invalid_worker_environment_is_rejected(monkeypatch, environment):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        distributed.Distributed(SimpleNamespace(device="cpu"))
