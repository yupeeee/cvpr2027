"""Validate saved work without loading models or starting CUDA workers."""
from __future__ import annotations

import itertools
import math
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

from .data import (atomic_jsonl, load_tensor, make_samples, merge_rows, read_json,
                   read_jsonl, request_for, scientific_fingerprint)
from .progress import progress

INVALID = (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError,
           EOFError, pickle.UnpicklingError)
FIELDS = ('appearance', 'composition', 'category')
_UNREAD = object()


def context(args):
    samples, prompts = make_samples(args)
    return SimpleNamespace(args=args, path=Path(args.logs_dir) / args.run_id,
                           ckpt_path=Path(args.ckpts_dir) / args.run_id,
                           samples=samples, prompts=prompts,
                           fingerprint=scientific_fingerprint(args))


def expected_keys(args, command, samples=None, settings=None):
    """Exact public record keys, independent of device count and model loading."""
    from .experiments import select_samples, trial_key
    from .probes import head_names
    samples = make_samples(args)[0] if samples is None else samples
    if command == 'collect':
        return [s['sample_id'] for s in samples]
    if command in ('fit', 'controls_reading'):
        variants = ('native',) if command == 'fit' else ('native', 'wrapped', 'transported')
        accesses = ('raw', 'clean', 'prompt') if command == 'fit' else ('raw',)
        return [f"{v}/{s['sample_id']}/{k}/{access}/{head}/{task}"
                for v in variants for s in samples if s['split'] in ('validation', 'test')
                for k in args.stages for access in accesses
                for head in (['prompt_mean'] if access == 'prompt' else head_names(args)) for task in FIELDS]
    limit = {'intervene': args.intervention_limit, 'controls': args.control_limit,
             'diagnose': args.diagnostic_limit, 'policy': args.policy_limit,
             'policy_calibration': args.policy_calibration_limit}[command]
    split = 'validation' if command == 'policy_calibration' else 'test' if command == 'policy' else args.eval_split
    samples = select_samples(samples, split, limit)
    result = []
    for sample in samples:
        if command == 'controls':
            result.append(f"{sample['sample_id']}:unmodified")
        for task, budget in itertools.product(args.tasks, args.edit_budgets):
            request = request_for(sample['sample_id'], task, args.request_seed)
            if command == 'intervene':
                jobs = [(k, e, '') for k in args.intervention_stages for e in args.editors]
            elif command == 'controls':
                jobs = [(k, e, 'coordinate') for k in args.control_stages for e in args.control_editors]
            elif command == 'diagnose':
                jobs = [(k, e, f'to{j}') for k, j in zip(args.diagnostic_stages[:-1], args.diagnostic_stages[1:])
                        for e in dict.fromkeys(['noop', *args.editors])]
            elif command == 'policy_calibration':
                jobs = [(k, 'calibration', f'stage:{k}') for k in args.policy_stages]
                jobs += [(-1, 'calibration', f'threshold:{v}') for v in args.policy_thresholds]
            else:
                if settings is None:
                    settings = read_json(Path(args.logs_dir) / args.run_id / 'policy_settings.json')
                jobs = [(k, name, '') for name in args.policy_variants
                        for k in (args.policy_stages if name == 'single_stage' else
                                  [args.num_steps if name == 'endpoint' else settings['stage']])]
            result.extend(trial_key(sample, task, request, k, budget, editor, extra)
                          for k, editor, extra in jobs)
    return result


def _component(run):
    import hashlib
    import json
    cached = getattr(run, '_validation_component', _UNREAD)
    if cached is not _UNREAD:
        return cached
    path = run.path / 'components.json'
    if not path.exists():
        return None
    saved = read_json(path)
    digest = hashlib.sha256(json.dumps(saved['identities'], sort_keys=True).encode()).hexdigest()
    if saved['fingerprint'] != digest or not saved['identities']:
        raise ValueError('Corrupt component identity manifest')
    # Only Validator's short-lived snapshot has this attribute. Never retain an
    # identity across independent checks or writes to a long-lived Run.
    if hasattr(run, '_validation_component'):
        run._validation_component = digest
    return digest


def _identity(saved, run):
    return saved['fingerprint'] == run.fingerprint and saved.get('component_fingerprint') == _component(run)


def _tensor(value, shape=None, finite=False):
    import torch
    return (torch.is_tensor(value) and value.is_floating_point() and
            (shape is None or tuple(value.shape) == tuple(shape)) and
            (not finite or bool(torch.isfinite(value).all())))


def training_statistics_valid(args, run=None):
    """Both training-only statistics are required by downstream experiments."""
    run = context(args) if run is None else run
    try:
        scales = read_json(run.ckpt_path / 'stage_scales.json')
        means = load_tensor(run.ckpt_path / 'prompt_mean.pt')
        training = [s['sample_id'] for s in run.samples if s['split'] == 'train']
        return (all(_identity(item, run) and item['training_ids'] == training for item in (scales, means))
                and set(scales['scales']) == {str(k) for k in args.stages}
                and all(math.isfinite(v) and v > 0 for v in scales['scales'].values())
                and _tensor(means['means'], (len(run.prompts), len(FIELDS)), finite=True)
                and means['counts'] == [sum(s['prompt_index'] == i for s in run.samples if s['split'] == 'train')
                                       for i in range(len(run.prompts))])
    except INVALID:
        return False


def checkpoint_valid(args, stage, access, head, variant='native', run=None):
    run = context(args) if run is None else run
    return valid_probe(args, run.ckpt_path / variant / f'stage_{stage:03d}_{access}_{head}.pt', run,
                       stage=stage, access=access, head=head, variant=variant)


def valid_probe(args, path, run=None, **expected):
    """Check the complete small checkpoint, including layer and normalizer shapes."""
    import torch
    from .probes import make_head
    run = context(args) if run is None else run
    try:
        saved = load_tensor(path)
        metadata = saved['metadata']
        if not _identity(metadata, run) or any(metadata[k] != v for k, v in expected.items()):
            return False
        dim = metadata['input_dim']
        if metadata['pool_grid'] != args.pool_grid or metadata['prompt_count'] != len(run.prompts):
            return False
        if dim <= len(run.prompts) or (dim - len(run.prompts)) % (args.pool_grid ** 2):
            return False
        split_ids = {split: [s['sample_id'] for s in run.samples if s['split'] == split]
                     for split in ('train', 'validation', 'test')}
        if metadata['split_ids'] != split_ids:
            return False
        # fork_rng restores the caller's CPU RNG state after shape construction.
        with torch.random.fork_rng(devices=[]):
            model = make_head(dim, metadata['head'])
        state = model.state_dict()
        if set(saved['state_dict']) != set(state) or metadata['parameter_count'] != sum(p.numel() for p in model.parameters()):
            return False
        if not all(_tensor(saved['state_dict'][key], value.shape, finite=True) for key, value in state.items()):
            return False
        for name, size in (('feature_mean', dim), ('feature_scale', dim), ('label_mean', len(FIELDS)), ('label_scale', len(FIELDS))):
            if not _tensor(saved[name], (size,), finite=True):
                return False
            if name.endswith('scale') and not bool((saved[name] > 0).all()):
                return False
        return True
    except INVALID:
        return False


def _safe_artifact(run, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('Missing artifact path')
    path = (run.path / relative).resolve()
    if not path.is_relative_to(run.path.resolve()):
        raise ValueError('Artifact path escapes run directory')
    return path


class Validator:
    """Memoize one validation pass and reuse successful file-hash receipts."""
    def __init__(self, run):
        self.args = run.args
        self.run = SimpleNamespace(**{name: getattr(run, name) for name in
            ('args', 'path', 'ckpt_path', 'samples', 'prompts', 'fingerprint')},
            _validation_component=_UNREAD)
        self.samples = {s['sample_id']: s for s in run.samples}
        self.cached = {}
        self.receipts = None

    def statistics(self):
        return self.once('statistics', lambda: training_statistics_valid(self.args, self.run),
            paths=lambda: [self.run.ckpt_path / 'stage_scales.json', self.run.ckpt_path / 'prompt_mean.pt'])

    def probe(self, stage, access, head, variant):
        return self.once(('probe', stage, access, head, variant),
            lambda: checkpoint_valid(self.args, stage, access, head, variant, self.run),
            paths=lambda: [self.run.ckpt_path / variant / f'stage_{stage:03d}_{access}_{head}.pt'])

    def once(self, key, check, paths=None):
        if key not in self.cached:
            try:
                if paths is None:
                    self.cached[key] = bool(check())
                else:
                    from .validation import ArtifactValidationCache
                    if self.receipts is None:
                        self.receipts = ArtifactValidationCache(self.run.path / 'validation', {
                            'schema': 1, 'fingerprint': self.run.fingerprint,
                            'component_fingerprint': _component(self.run)})
                    self.cached[key] = self.receipts.check(key, paths(), check)
            except INVALID:
                self.cached[key] = False
        return self.cached[key]

    def trajectory(self, sample, relative):
        def check():
            obj = load_tensor(_safe_artifact(self.run, relative))
            metadata = obj['metadata']
            if not _identity(obj, self.run) or obj['sample'] != sample or obj['stage_indices'] != self.args.stages:
                return False
            if obj['status'] not in ('ok', 'malformed', 'failed') or metadata['num_steps'] != self.args.num_steps:
                return False
            if metadata['prompt'] != sample['prompt'] or metadata['negative_prompt'] != self.args.negative_prompt:
                return False
            if len(metadata['timesteps']) != self.args.num_steps or metadata['scheduler'] != 'DDIMScheduler' or not isinstance(metadata['scheduler_config'], dict):
                return False
            scale, channels = metadata['vae_scale_factor'], metadata['latent_channels']
            if not isinstance(scale, int) or scale <= 0 or not isinstance(channels, int) or channels <= 0:
                return False
            if metadata['model_id'] != self.args.model_id or metadata['model_revision'] != self.args.model_revision or metadata['guidance_scale'] != self.args.guidance_scale:
                return False
            if metadata['dtype'] != f'torch.{self.args.precision}':
                return False
            shape = (1, channels, self.args.height // scale, self.args.width // scale)
            image_shape = (1, 3, self.args.height, self.args.width)
            for group in ('states', 'clean', 'previews'):
                if set(obj[group]) != set(self.args.stages) or not all(
                        _tensor(obj[group][k], image_shape if group == 'previews' else shape, finite=obj['status'] == 'ok')
                        for k in self.args.stages):
                    return False
            return (_tensor(obj['final_image'], image_shape, finite=obj['status'] == 'ok') and set(obj['stage_costs']) == set(self.args.stages)
                    and all(set(obj['scores'][r]) == set(FIELDS) for r in ('a', 'b'))
                    and all(k in obj['costs'] for k in ('generation', 'observation', 'evaluation')))
        return self.once(('trajectory', sample['sample_id'], relative), check,
                         paths=lambda: [_safe_artifact(self.run, relative)])

    def artifact(self, key, relative):
        def check():
            path = _safe_artifact(self.run, relative)
            if key == 'setup_record':
                obj = read_json(path)
                return obj['phase'] == 'shared_text_and_head_preparation' and isinstance(obj['costs'], dict)
            if key == 'settings_file':
                from .policies import select_validation_settings
                obj = read_json(path)
                expected = select_validation_settings(cached_rows(self.run, 'policy_calibration'),
                    self.args.policy_stages, self.args.policy_thresholds)
                expected.update(fingerprint=self.run.fingerprint, fallback=self.args.policy_fallback,
                    deadline=self.args.policy_deadline, time_only_default=self.args.policy_time_only_default,
                    calibration_cost_phase='training_calibration', test_settings_frozen=True)
                return obj == expected
            obj = load_tensor(path)
            # Malformed numerical outputs remain recorded failures, not retry loops.
            shape = (1, 3, self.args.height, self.args.width)
            return all(_tensor(obj[part], shape) for part in ('before', 'after'))
        paths = None if key in ('setup_record', 'settings_file') else lambda: [_safe_artifact(self.run, relative)]
        return self.once((key, relative), check, paths=paths)

    def references(self, obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ('image_path', 'image_path_first', 'image_path_second', 'setup_record', 'settings_file'):
                    if not self.artifact(key, value):
                        return False
                elif not self.references(value):
                    return False
        elif isinstance(obj, list):
            return all(self.references(value) for value in obj)
        return True

    def row(self, command, row):
        try:
            if not isinstance(row['key'], str) or row.get('status') not in ('ok', 'failed', 'malformed'):
                return False
            sample = self.samples[row['sample_id']]
            if any(row.get(key) != sample[key] for key in ('root_seed', 'prompt_id', 'prompt', 'object', 'split')):
                return False
            if command == 'collect':
                return row['key'] == sample['sample_id'] and self.trajectory(sample, row['trajectory_path'])
            if command in ('fit', 'controls_reading'):
                variant = row.get('coordinate_variant', row.get('coordinate'))
                if variant == 'transported':
                    variant = 'native'
                if row['access'] == 'prompt':
                    valid = self.statistics()
                else:
                    valid = self.probe(row['stage'], row['access'], row['head'], variant)
                return valid and all(k in row for k in ('prediction_a', 'reference_a', 'reference_b', 'costs'))
            if command in ('intervene', 'policy', 'policy_calibration'):
                if not all(k in row for k in ('image_path', 'setup_record', 'scores', 'a', 'b', 'costs', 'edit')):
                    return False
                if command.startswith('policy') and not isinstance(row['inspections'], list):
                    return False
                if command == 'policy' and 'settings_file' not in row:
                    return False
            if command == 'diagnose' and not all(k in row for k in ('first', 'second', 'image_path_first', 'image_path_second', 'residual_rms', 'costs')):
                return False
            if command == 'controls' and ('endpoint_equivalent' not in row or ('unmodified' not in row['key'] and not all(k in row for k in ('native', 'transported', 'native_edit', 'transported_edit')))):
                return False
            return self.references(row)
        except INVALID:
            return False


def row_paths(run, command):
    paths = sorted((run.path / 'ranks').glob(f'{command}.rank*.jsonl'))
    return paths or ([run.path / f'{command}.jsonl'] if (run.path / f'{command}.jsonl').exists() else [])


def cached_rows(run, command):
    return merge_rows([row for path in row_paths(run, command) for row in read_jsonl(path)])


def compatible_manifest(run):
    saved = read_json(run.path / 'manifest.json')
    if saved['fingerprint'] != run.fingerprint or saved['samples'] != run.samples or saved['prompts'] != run.prompts:
        raise ValueError('Incompatible scientific cache fingerprint or sample manifest: restore original settings, choose another run-id, or use --overwrite on collect')
    return saved


def _stage_complete(args, command, require_merged=True):
    """True only for the complete expected key set and all referenced artifacts."""
    if getattr(args, 'overwrite', False):
        return False
    run = context(args)
    if not (run.path / 'manifest.json').exists():
        return False
    compatible_manifest(run)  # Genuine scientific mismatches must never be silently replaced.
    validator = Validator(run)
    commands = _commands(command)
    try:
        if not (run.path / 'components.json').is_file():
            return False
        _component(validator.run)
        expected = {name: expected_keys(args, name, run.samples) for name in commands}
        probes = []
        if command in ('fit', 'controls'):
            from .probes import head_names
            variants = ('native', 'wrapped') if command == 'controls' else ('native',)
            probes = [(stage, access, head, variant) for variant in variants
                for stage, access, head in itertools.product(args.stages,
                    ('raw', 'clean') if variant == 'native' else ('raw',), head_names(args))]
        # Validation precedes Distributed.__enter__, so honor its CPU thread
        # setting here too. Huge default thread pools make small finite checks
        # disproportionately expensive, especially on a resumed multi-rank run.
        import torch
        torch.set_num_threads(args.torch_threads)
        total = sum(map(len, expected.values())) + len(probes) + bool(probes)
        with progress(args, desc=f'{command}: verifying saved artifacts',
                      total=total, unit='check') as bar:
            for name in commands:
                bar.set_postfix_str(f'reading {name} records')
                rows = cached_rows(run, name)
                # Preserve explicit duplicate errors rather than interpreting them as resumable work.
                if {r['key'] for r in rows} != set(expected[name]):
                    return False
                for row in rows:
                    bar.set_postfix_str(f"{name}: {row['key']}", refresh=False)
                    if not validator.row(name, row):
                        return False
                    bar.update(1)
                bar.set_postfix_str(f'checking {name} summary')
                merged = run.path / f'{name}.jsonl'
                if require_merged and (not merged.exists() or merge_rows(read_jsonl(merged)) != rows):
                    return False
            if probes:
                bar.set_postfix_str('checking training statistics')
                if not validator.statistics():
                    return False
                bar.update(1)
                for stage, access, head, variant in probes:
                    bar.set_postfix_str(f'probe: {variant}/{stage}/{access}/{head}', refresh=False)
                    if not validator.probe(stage, access, head, variant):
                        return False
                    bar.update(1)
        return True
    except INVALID as exc:
        if isinstance(exc, ValueError) and 'Duplicate result key:' in str(exc):
            raise
        return False


def _commands(command):
    return (['controls_reading', 'controls'] if command == 'controls' else
            ['controls'] if command == 'controls_endpoints' else
            ['policy_calibration', 'policy'] if command == 'policy' else [command])


def stage_complete(args, command):
    return _stage_complete(args, command)


def skip_completed(args, command):
    import os
    if (Path(args.logs_dir) / args.run_id / 'manifest.json').exists() and int(os.environ.get('RANK', '0')) == 0:
        from .progress import status
        status(args, f'{command}: checking saved artifacts')
    if _stage_complete(args, command, require_merged=False):
        if int(os.environ.get('RANK', '0')) == 0:
            run = context(args)
            for name in _commands(command):
                rows = cached_rows(run, name)
                path = run.path / f'{name}.jsonl'
                try:
                    existing = read_jsonl(path)
                except INVALID:
                    existing = None
                if existing != rows:
                    atomic_jsonl(path, rows)
            print(f'{command}: complete saved artifacts verified; skipping (use --overwrite to recompute).', flush=True)
        return True
    return False


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)



def clear_figures(args):
    """Remove generated figures while retaining the selected human-label input."""
    import os
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        raise ValueError('Unsafe run directory for --overwrite')
    directory = Path(os.path.abspath(Path(args.figs_dir) / args.run_id))
    if directory.is_symlink():
        raise ValueError('Unsafe run directory for --overwrite')
    selected = None
    if getattr(args, 'audit_labels', None):
        source = Path(os.path.abspath(args.audit_labels))
        if source.is_file():
            for candidate, root in ((source, directory), (source.resolve(), directory.resolve())):
                if candidate.is_relative_to(root):
                    selected = directory / candidate.relative_to(root)
                    break

    def remove(path):
        if selected is not None and path == selected:
            return
        if selected is not None and path in selected.parents:
            # Keep a selected path through a symlink; never traverse its target.
            if path.is_dir() and not path.is_symlink():
                for child in path.iterdir():
                    remove(child)
            return
        _remove(path)

    remove(directory)


def reset_stage(args, command):
    """Rank zero invalidates this stage and its dependents, never pretrained caches."""
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        raise ValueError('Unsafe run directory for --overwrite')
    roots = []
    for name in ('logs_dir', 'ckpts_dir', 'figs_dir'):
        base = Path(getattr(args, name)).resolve()
        path = base / args.run_id
        if path.is_symlink() or path.parent.resolve() != base or Path(args.run_id).name != args.run_id:
            raise ValueError('Unsafe run directory for --overwrite')
        roots.append(path)
    logs, ckpts, figs = roots
    if command == 'collect':
        if logs.exists():
            for path in logs.iterdir():
                if path.name != 'progress':
                    _remove(path)
        _remove(ckpts)
        clear_figures(args)
        return
    stages = {'fit': ['fit', 'fit_failures', 'intervene', 'controls_reading', 'controls', 'diagnose', 'policy_calibration', 'policy'],
              'controls': ['controls_reading', 'controls'], 'policy': ['policy_calibration', 'policy']}.get(command, [command])
    for name in stages:
        _remove(logs / f'{name}.jsonl')
        _remove(logs / f'invocation_{name}.json')
        for path in (logs / 'ranks').glob(f'{name}.rank*.jsonl'):
            _remove(path)
        for path in (logs / 'setup').glob(f'{name}.rank*.json'):
            _remove(path)
        for suffix in ('_images', '_first_images', '_second_images'):
            _remove(logs / f'{name}{suffix}')
    if command == 'fit':
        _remove(ckpts)
    elif command == 'controls':
        _remove(ckpts / 'wrapped')
    if 'policy' in stages:
        _remove(logs / 'policy_settings.json')
    clear_figures(args)


def _read_recoverable(path):
    """Atomic writes normally prevent partial lines; retain other rows if damaged."""
    import json
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get('key'), str):
            rows.append(row)
    return rows


def _rewrite_records(run, command, paths, original, retained):
    keep = {row['key'] for row in retained}
    rank_paths = [path for path in paths if path.parent.name == 'ranks']
    if rank_paths:
        for path in rank_paths:
            rows = _read_recoverable(path)
            filtered = [row for row in rows if row['key'] in keep]
            # Rewrite also repairs malformed trailing lines in the shard.
            try:
                old = read_jsonl(path)
            except INVALID:
                old = None
            if old != filtered:
                atomic_jsonl(path, filtered)
    elif paths:
        # Materialize the old merged-only cache before new ranks append work.
        atomic_jsonl(run.path / 'ranks' / f'{command}.rank00000.jsonl', retained)
    merged = run.path / f'{command}.jsonl'
    if merged.exists():
        try:
            old = read_jsonl(merged)
        except INVALID:
            old = None
        if old != retained:
            atomic_jsonl(merged, retained)


def clean_records(run, command):
    """Rank-zero-only pruning of broken rows/artifacts; all good trials survive."""
    paths = row_paths(run, command)
    if not paths:
        return
    original = merge_rows([row for path in paths for row in _read_recoverable(path)])
    validator = Validator(run)
    wanted = set(expected_keys(run.args, command, run.samples))
    retained = []
    with progress(run.args, desc=f'{command}: verifying saved artifacts for repair',
                  total=len(original), unit='record') as bar:
        for row in original:
            bar.set_postfix_str(row['key'], refresh=False)
            if row['key'] in wanted and validator.row(command, row):
                retained.append(row)
            bar.update(1)
    _rewrite_records(run, command, paths, original, retained)


def discard_rows(run, command, keys):
    keys = set(keys)
    paths = row_paths(run, command)
    original = merge_rows([row for path in paths for row in _read_recoverable(path)])
    retained = [row for row in original if row['key'] not in keys]
    _rewrite_records(run, command, paths, original, retained)
    run._rank_rows.pop(command, None)
    run._completed.pop(command, None)
