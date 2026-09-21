"""Bounded two-process CPU/Gloo check with one genuine DDP optimizer step."""
import json
import os
import socket
import time
from contextlib import ExitStack, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from utils.config import parse_args
from utils.distributed import Distributed, shard
from utils.probes import make_head, optimize_batch
from utils.progress import progress


def _check_cache_repairs(args, context, directory):
    """A late worker sees repaired files but must still join every collective."""
    from exps.policy import freeze_settings
    from utils.probes import train_probes

    repair_args = SimpleNamespace(**vars(args))
    repair_args.progress = False
    repair_args.stages = [0]
    trajectory = dict(status='ok', states={0: torch.ones(1, 2, 2, 2)},
                      clean={0: torch.ones(1, 2, 2, 2)},
                      scores={'a': dict(appearance=1., composition=2., category=3.)})
    run = SimpleNamespace(path=directory, ckpt_path=directory / 'probe_stats',
        fingerprint='offline_fixture', prompts=[{}],
        samples=[dict(sample_id='train', split='train', prompt_index=0),
                 dict(sample_id='validation', split='validation', prompt_index=0)],
        rows=lambda command: [], load_trajectory=lambda sample_id: trajectory)

    def wait_for_file(path):
        deadline = time.monotonic() + 5
        while not path.is_file():
            if time.monotonic() > deadline:
                raise AssertionError(f'Rank zero failed to repair {path.name}')
            time.sleep(0.01)

    def statistics_ready(*args, **kwargs):
        path = run.ckpt_path / 'prompt_mean.pt'
        if not context.is_main:
            wait_for_file(path)
        return path.is_file()

    with patch('utils.cache.checkpoint_valid', return_value=True), \
         patch('utils.cache.training_statistics_valid', side_effect=statistics_ready), \
         patch('utils.probes.head_names', return_value=['linear']):
        train_probes(repair_args, context, run, variant='wrapped', evaluate=False)
        previous = (run.ckpt_path / 'prompt_mean.pt').stat().st_mtime_ns
        train_probes(repair_args, context, run, variant='wrapped', evaluate=False)
        assert (run.ckpt_path / 'prompt_mean.pt').stat().st_mtime_ns == previous

    settings_path = directory / 'policy_settings.json'
    if not context.is_main:
        wait_for_file(settings_path)
    with patch('exps.policy.select_validation_settings',
               side_effect=lambda *args: dict(stage=0, threshold=0.0)):
        freeze_settings(repair_args, context, run)
        previous = settings_path.stat().st_mtime_ns
        freeze_settings(repair_args, context, run)
        assert settings_path.stat().st_mtime_ns == previous


def _worker(rank, port, result_dir):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), COLUMNS="80", LINES="24")
    args = parse_args("offline-ddp-test", ["--device", "cpu", "--distributed-timeout", "30",
        "--logs-dir", str(Path(result_dir) / "logs"), "--progress-mininterval", "0.01"])
    with (Path(result_dir) / f"rank{rank}.stderr").open("w") as captured, redirect_stderr(captured), Distributed(args) as context:
        torch.manual_seed(19)
        raw = make_head(3, "linear")
        model = DistributedDataParallel(raw)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
        y = x.flip(1)
        dataset = TensorDataset(x, y)
        sampler = DistributedSampler(dataset, num_replicas=2, rank=rank,
                                     shuffle=False, drop_last=False)
        sampler.set_epoch(0)
        for local_x, local_y in DataLoader(dataset, batch_size=2, sampler=sampler):
            optimize_batch(model, optimizer, local_x, local_y)
        # Non-padding inference: ranks may have unequal or zero examples.
        odd = context.shard(list(range(5)))
        tiny = context.shard([0])
        # Progress must tolerate asynchronous unequal/empty shards without
        # adding a collective call to any per-rank work loop.
        with ExitStack() as stack:
            for name in ("all_reduce", "all_gather", "all_gather_object", "barrier", "broadcast_object_list"):
                stack.enter_context(patch("torch.distributed." + name,
                    side_effect=AssertionError("Progress introduced a collective inside an uneven loop")))
            with progress(args, [], desc="Resumed inference", total=2, initial=2,
                          unit="sample", dist=context):
                pass
            with progress(args, odd, desc="Odd inference", unit="sample", dist=context) as bar:
                for _ in bar:
                    with progress(args, total=1, desc="Nested item detail", dist=context,
                                  position=1, leave=False) as detail:
                        time.sleep(0.01 if rank == 0 else 0.04)
                        detail.update()
            with progress(args, tiny, desc="Tiny inference", unit="sample", dist=context) as bar:
                for _ in bar:
                    time.sleep(0.01)
        _check_cache_repairs(args, context, Path(result_dir))
        total, count = context.sum_count(sum(odd), len(odd))
        torch.save({"state": raw.state_dict(), "odd": odd, "tiny": tiny,
                    "total": total, "count": count}, Path(result_dir) / f"rank{rank}.pt")


def test_two_process_ddp_matches_global_batch_and_inference_shards(tmp_path):
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=2, join=True)
    states = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
    torch.manual_seed(19)
    reference = make_head(3, "linear")
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
    optimize_batch(reference, optimizer, x, x.flip(1))
    for name, parameter in reference.state_dict().items():
        torch.testing.assert_close(states[0]["state"][name], states[1]["state"][name], rtol=0, atol=0)
        torch.testing.assert_close(states[0]["state"][name], parameter, rtol=1e-6, atol=1e-7)
    for key, count in (("odd", 5), ("tiny", 1)):
        merged = [item for state in states for item in state[key]]
        assert sorted(merged) == list(range(count)) and len(set(merged)) == count
    assert states[0]["total"] == 10 and states[0]["count"] == 5
    assert states[0]["total"] / states[0]["count"] == 2
    assert shard([0], 1, 2) == []
    directories = list((tmp_path / "logs" / "demo" / "progress").iterdir())
    assert len(directories) == 1
    snapshots = [json.loads((directories[0] / f"rank{rank}.json").read_text())
                 for rank in range(2)]
    assert all(snapshot["done"] for snapshot in snapshots)
    for name, expected in (("Resumed inference", 4), ("Odd inference", 5), ("Tiny inference", 1)):
        phases = [snapshot["phases"][name + "#0"] for snapshot in snapshots]
        assert sum(phase["n"] for phase in phases) == expected
        assert sum(phase["total"] for phase in phases) == expected
        assert all(phase["finished"] for phase in phases)
    assert all(len(snapshot["phases"]) == 3 for snapshot in snapshots)
    assert [snapshot["phases"]["Resumed inference#0"]["initial"] for snapshot in snapshots] == [2, 2]
    assert snapshots[1]["phases"]["Tiny inference#0"]["total"] == 0
    output = (tmp_path / "rank0.stderr").read_text()
    assert "Tiny inference" in output and "1/1" in output and "100%" in output
    assert "cpu/r0" in output and "cpu/r1" in output
    assert "<-" not in output  # Phase switches must not show a negative remaining time.
    assert (tmp_path / "rank1.stderr").read_text() == ""
