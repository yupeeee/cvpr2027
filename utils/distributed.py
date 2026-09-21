"""DDP setup for heads; non-padding rank shards for frozen inference."""
import os
import sys
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist


def discover_usable_cuda():
    """Return visible logical GPUs that can allocate and synchronize a tiny tensor.

    This checks accessibility, not whether an entire model replica fits. Call only
    for real execution; argument parsing, help, and dry runs never probe CUDA.
    """
    if not torch.cuda.is_available():
        return []
    usable = []
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                allocation = torch.empty(1, device=torch.device("cuda", index))
                torch.cuda.synchronize(index)
                del allocation
        except (RuntimeError, AssertionError) as exc:
            warnings.warn(f"Skipping unusable visible CUDA device {index}: {exc}",
                          RuntimeWarning, stacklevel=2)
        else:
            usable.append(index)
    return usable


def launch_if_needed(args, command, argv=None):
    """Exec torchrun for --device auto outside an existing distributed worker.

    Each selected GPU gets one worker, even when only one GPU is available. Exec
    releases the discovery process's CUDA contexts before model loading. Explicit
    torchrun launches retain the requested worker count. ``command`` is an
    experiment name (``collect``) or its complete module name (``exps.collect``).
    """
    if args.device != "auto" or any(name in os.environ for name in
                                    ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return False
    usable = discover_usable_cuda()
    if not usable:
        print("No usable CUDA devices found; --device auto uses CPU.", file=sys.stderr)
        args.device = "cpu"
        return False
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [value.strip() for value in visible.split(",")] if visible is not None else None
    if tokens is not None and max(usable) >= len(tokens):
        raise RuntimeError("CUDA device enumeration disagrees with CUDA_VISIBLE_DEVICES; "
                           "check the visibility setting before launching")
    selected = [tokens[index] if tokens is not None else str(index) for index in usable]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(selected))
    module = command if command.startswith("exps.") else f"exps.{command}"
    forwarded = list(sys.argv[1:] if argv is None else argv)
    launch = [sys.executable, "-m", "torch.distributed.run", "--standalone",
              "--nnodes=1", "--nproc-per-node", str(len(usable)), "--module", module,
              *forwarded]
    print(f"Launching {len(usable)} CUDA worker(s); "
          f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", file=sys.stderr)
    os.execvpe(sys.executable, launch, env)
    return True  # exec does not return; useful to injected launchers in tests.


def shard(items, rank, world_size):
    return items[rank::world_size]


class Distributed:
    def __init__(self, args):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size < 1 or not 0 <= self.rank < self.world_size or self.local_rank < 0:
            raise ValueError("RANK, WORLD_SIZE, and LOCAL_RANK describe an invalid worker")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
        if "LOCAL_WORLD_SIZE" in os.environ and not 0 <= self.local_rank < local_world_size:
            raise ValueError("LOCAL_RANK must be below LOCAL_WORLD_SIZE")
        self.is_main = self.rank == 0
        self.args = args
        use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        if use_cuda and torch.cuda.is_available():
            count = torch.cuda.device_count()
            if self.local_rank >= count or local_world_size > count:
                raise RuntimeError(f"CUDA launch needs local rank {self.local_rank} with "
                                   f"LOCAL_WORLD_SIZE={local_world_size or 'unset'}, but only "
                                   f"{count} CUDA device(s) are visible. Use --device auto "
                                   "without torchrun to select the worker count automatically.")
        self.device = torch.device("cuda", self.local_rank) if use_cuda else torch.device("cpu")
        self.owns_group = False

    def __enter__(self):
        torch.set_num_threads(self.args.torch_threads)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; use --device cpu for CPU work")
            torch.cuda.set_device(self.device)
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                    timeout=timedelta(seconds=self.args.distributed_timeout))
            self.owns_group = True
        from .progress import start_progress
        start_progress(self.args, self)
        return self

    def shard(self, items):
        return shard(items, self.rank, self.world_size)

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def sum_count(self, total, count):
        values = torch.tensor([total, count], device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(values)
        return float(values[0]), int(values[1])

    def __exit__(self, exc_type, exc, tb):
        from .progress import stop_progress
        try:
            stop_progress(exc)
        finally:
            # No failure-path barrier: another rank may already have failed.
            if self.owns_group and dist.is_initialized():
                dist.destroy_process_group()


def main_process(context, action):
    """Main-only I/O, with errors propagated before any success barrier."""
    result = [None]
    local_error = None
    if context.is_main:
        try:
            action()
        except Exception as exc:
            # Propagate, never suppress, unexpected errors to every rank.
            local_error = exc
            result[0] = f'{type(exc).__name__}: {exc}'
    if context.world_size > 1:
        dist.broadcast_object_list(result, src=0, device=context.device)
    if result[0] is not None:
        raise RuntimeError(f'Main-process operation failed: {result[0]}') from local_error
