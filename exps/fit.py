"""Fit small per-stage endpoint observers on cached training trajectories."""
from utils.config import parse_args, dry_run
from utils.data import Run
from utils.distributed import Distributed, launch_if_needed
from utils.probes import train_probes


def main(argv=None):
    args = parse_args("fit", argv)
    if dry_run(args, "fit"):
        return
    from utils.cache import skip_completed
    if skip_completed(args, 'fit'):
        return
    if launch_if_needed(args, 'fit', argv):
        return
    with Distributed(args) as dist:
        run = Run(args, dist, "fit")
        train_probes(args, dist, run)


if __name__ == "__main__":
    main()
