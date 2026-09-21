"""Prepare configured model files, honoring explicit local-only/download settings."""
import json
from pathlib import Path

from utils.config import dry_run, parse_args


def main(argv=None):
    args = parse_args('prepare', argv)
    if dry_run(args, 'prepare'):
        return
    from utils.progress import status
    if not args.overwrite and (Path(args.logs_dir) / args.run_id / 'manifest.json').is_file():
        # This is a restart hint, never proof of valid/completed output. Each
        # owning stage validates its artifacts before skipping or preparing models.
        # Repeating those tensor reads here can silently scan many gigabytes.
        status(args, 'prepare: existing run found; artifact validation and model checks '
                     'are deferred to the stages that need them')
        return
    status(args, 'prepare: checking model files and tokenizer dependencies')
    from utils.models import prepare_models
    paths = prepare_models(args)
    print(json.dumps({'status': 'models_ready', 'paths': paths}, indent=2))


if __name__ == '__main__':
    main()
