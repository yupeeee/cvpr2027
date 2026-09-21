"""Reproduce available figures entirely from saved records and image tensors."""
from utils.config import dry_run, parse_args


def main(argv=None):
    args = parse_args('plot', argv)
    if dry_run(args, 'plot'):
        return
    from utils.plotting import render
    figures = render(args)
    print(f'Available: {len(figures)} PNG/PDF figures with CSV tables in {args.figs_dir}/{args.run_id}')


if __name__ == '__main__':
    main()
