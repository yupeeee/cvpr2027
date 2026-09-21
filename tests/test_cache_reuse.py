"""Bounded local-artifact checks; never construct pretrained models or CUDA state."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.cache import (Validator, checkpoint_valid, expected_keys, reset_stage,
                         skip_completed, stage_complete, training_statistics_valid)
from utils.config import parse_args
from utils.data import Run, atomic_json, atomic_jsonl, atomic_torch, base_row, load_tensor
from utils.probes import make_head


class LocalDist:
    rank = 0
    world_size = 1
    is_main = True
    device = torch.device('cpu')
    def barrier(self):
        pass


def run_fixture(tmp_path):
    args = parse_args('collect', ['--config', 'configs/smoke.json', '--no-dry-run', '--no-progress',
        '--logs-dir', str(tmp_path / 'logs'), '--ckpts-dir', str(tmp_path / 'ckpts'),
        '--figs-dir', str(tmp_path / 'figs')])
    run = Run(args, LocalDist(), 'collect')
    identities = {'generator': {'fixture': 'tiny'}, 'evaluators': {'a': 'tiny', 'b': 'tiny'}}
    fingerprint = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    atomic_json(run.path / 'components.json', {'fingerprint': fingerprint, 'identities': identities})
    return args, run, fingerprint


def write_collection(args, run, component, sample, status='ok'):
    latent = torch.zeros(1, 4, args.height // 2, args.width // 2)
    image = torch.zeros(1, 3, args.height, args.width)
    if status == 'malformed':
        image.fill_(float('nan'))
    trajectory = dict(fingerprint=run.fingerprint, component_fingerprint=component, sample=sample,
        stage_indices=args.stages, states={k: latent for k in args.stages},
        clean={k: latent for k in args.stages}, previews={k: image for k in args.stages},
        final_image=image, scores={r: {f: 0.0 for f in ('appearance', 'composition', 'category')} for r in ('a', 'b')},
        stage_costs={k: {} for k in args.stages}, costs={k: {} for k in ('generation', 'observation', 'evaluation')},
        status=status, metadata=dict(prompt=sample['prompt'], negative_prompt=args.negative_prompt,
            num_steps=args.num_steps, timesteps=list(range(args.num_steps)), scheduler='DDIMScheduler',
            scheduler_config={}, vae_scale_factor=2, latent_channels=4, model_id=args.model_id,
            model_revision=args.model_revision, guidance_scale=args.guidance_scale, dtype=f'torch.{args.precision}'))
    path = run.write_tensor(f"trajectories/{sample['sample_id']}.pt", trajectory)
    row = base_row(sample, key=sample['sample_id'], status=status, trajectory_path=path)
    run.write_row('collect', row)
    return row


def test_complete_cache_skips_and_rebuilds_only_missing_summary(tmp_path, capsys):
    args, run, component = run_fixture(tmp_path)
    for sample in run.samples:
        write_collection(args, run, component, sample)
    assert not stage_complete(args, 'collect')  # Merged summary was never written.
    assert skip_completed(args, 'collect')
    assert 'skipping' in capsys.readouterr().out
    assert stage_complete(args, 'collect')
    args.resume = False  # Legacy flag never forces completed scientific work to rerun.
    assert skip_completed(args, 'collect')
    args.overwrite = True
    assert not skip_completed(args, 'collect')


def test_partial_cache_retains_good_and_recorded_malformed_samples(tmp_path):
    args, run, component = run_fixture(tmp_path)
    rows = [write_collection(args, run, component, sample,
             'malformed' if i == 0 else 'ok') for i, sample in enumerate(run.samples)]
    (run.path / rows[1]['trajectory_path']).unlink()
    (run.path / rows[2]['trajectory_path']).write_bytes(b'truncated tensor')
    args.resume = False
    assert run.completed('collect') == {r['key'] for r in rows} - {rows[1]['key'], rows[2]['key']}
    # Retrying a missing trial writes one row rather than creating a duplicate.
    write_collection(args, run, component, run.samples[1])
    write_collection(args, run, component, run.samples[2])
    run.finish('collect', expected_keys(args, 'collect'))
    assert stage_complete(args, 'collect')


def test_merged_only_cache_partial_repair_and_explicit_duplicate_error(tmp_path):
    args, run, component = run_fixture(tmp_path)
    rows = [write_collection(args, run, component, sample) for sample in run.samples]
    run.finish('collect', expected_keys(args, 'collect'))
    rankfile = run.path / 'ranks' / 'collect.rank00000.jsonl'
    rankfile.unlink()
    assert stage_complete(args, 'collect')
    (run.path / rows[0]['trajectory_path']).unlink()
    run = Run(args, LocalDist(), 'collect')
    assert run.completed('collect') == {r['key'] for r in rows[1:]}
    assert rankfile.exists()  # Old merged rows survive future shard writes.
    write_collection(args, run, component, run.samples[0])
    atomic_jsonl(run.path / 'ranks' / 'collect.rank00001.jsonl', [rows[1]])
    with pytest.raises(ValueError, match='Duplicate result key'):
        stage_complete(args, 'collect')


def test_artifact_references_and_recorded_failures(tmp_path):
    args, run, _ = run_fixture(tmp_path)
    sample = run.samples[0]
    image = torch.full((1, 3, args.height, args.width), float('nan'))
    setup = run.write_json('setup/intervene.rank00000.json', dict(phase='shared_text_and_head_preparation', costs={}))
    pair = run.write_tensor('intervene_images/pair.pt', {'before': image, 'after': image})
    row = base_row(sample, key='trial', status='failed', image_path=pair, setup_record=setup,
                   scores={}, a={}, b={}, costs={}, edit={})
    assert Validator(run).row('intervene', row)
    (run.path / setup).unlink()
    assert not Validator(run).row('intervene', row)
    row['setup_record'] = '../outside.json'
    assert not Validator(run).row('intervene', row)


def write_checkpoint(args, run, component):
    dim = 4 * args.pool_grid ** 2 + len(run.prompts)
    head = make_head(dim, 'linear')
    metadata = dict(fingerprint=run.fingerprint, component_fingerprint=component, stage=0,
        access='raw', head='linear', variant='native', input_dim=dim, pool_grid=args.pool_grid,
        prompt_count=len(run.prompts), parameter_count=sum(p.numel() for p in head.parameters()),
        split_ids={sp: [s['sample_id'] for s in run.samples if s['split'] == sp]
                   for sp in ('train', 'validation', 'test')})
    saved = dict(metadata=metadata, state_dict=head.state_dict(), feature_mean=torch.zeros(dim),
        feature_scale=torch.ones(dim), label_mean=torch.zeros(3), label_scale=torch.ones(3))
    path = run.ckpt_path / 'native' / 'stage_000_raw_linear.pt'
    atomic_torch(path, saved)
    return path, saved


def test_checkpoint_state_metadata_normalizers_and_rng(tmp_path):
    args, run, component = run_fixture(tmp_path)
    path, saved = write_checkpoint(args, run, component)
    before = torch.random.get_rng_state()
    assert checkpoint_valid(args, 0, 'raw', 'linear', run=run)
    assert torch.equal(before, torch.random.get_rng_state())
    saved['feature_scale'][0] = 0
    atomic_torch(path, saved)
    assert not checkpoint_valid(args, 0, 'raw', 'linear', run=run)
    saved['feature_scale'][0] = 1
    saved['state_dict']['weight'] = torch.zeros(1)
    atomic_torch(path, saved)
    assert not checkpoint_valid(args, 0, 'raw', 'linear', run=run)


def test_training_statistics_require_every_shape_and_training_id(tmp_path):
    args, run, component = run_fixture(tmp_path)
    training = [s for s in run.samples if s['split'] == 'train']
    shared = dict(fingerprint=run.fingerprint, component_fingerprint=component,
                  training_ids=[s['sample_id'] for s in training])
    scales = dict(shared, scales={str(k): 1.0 for k in args.stages})
    means = dict(shared, means=torch.zeros(len(run.prompts), 3),
                 counts=[sum(s['prompt_index'] == i for s in training) for i in range(len(run.prompts))])
    atomic_json(run.ckpt_path / 'stage_scales.json', scales)
    atomic_torch(run.ckpt_path / 'prompt_mean.pt', means)
    assert training_statistics_valid(args, run)
    means['training_ids'] = []
    atomic_torch(run.ckpt_path / 'prompt_mean.pt', means)
    assert not training_statistics_valid(args, run)


def test_overwrite_scoping_and_selective_row_invalidation(tmp_path):
    args, run, component = run_fixture(tmp_path)
    rows = [write_collection(args, run, component, sample) for sample in run.samples]
    run.invalidate('collect', {rows[0]['key']})
    assert rows[0]['key'] not in run.completed('collect')
    assert len(run.rows('collect')) == len(rows) - 1
    for name in ('native', 'wrapped'):
        (run.ckpt_path / name).mkdir()
        (run.ckpt_path / name / 'fixture.pt').write_text('fixture')
    (run.path / 'progress' / 'active').mkdir(parents=True)
    atomic_jsonl(run.path / 'intervene.jsonl', [])
    atomic_jsonl(run.path / 'controls.jsonl', [])
    reset_stage(args, 'controls')
    assert (run.ckpt_path / 'native' / 'fixture.pt').exists()
    assert not (run.ckpt_path / 'wrapped').exists()
    assert (run.path / 'intervene.jsonl').exists()
    reset_stage(args, 'fit')
    assert not (run.path / 'intervene.jsonl').exists()
    assert (run.path / 'manifest.json').exists()
    assert (run.path / rows[1]['trajectory_path']).exists()
    reset_stage(args, 'collect')
    assert not (run.path / 'manifest.json').exists()
    assert (run.path / 'progress' / 'active').exists()


def test_overwrite_collect_accepts_new_scientific_settings_only_when_explicit(tmp_path):
    args, run, _ = run_fixture(tmp_path)
    args.guidance_scale += 1
    with pytest.raises(ValueError, match='fingerprint'):
        stage_complete(args, 'collect')
    args.overwrite = True
    fresh = Run(args, LocalDist(), 'collect')
    assert fresh.fingerprint != run.fingerprint


def test_incompatible_stage_overwrite_preserves_existing_artifacts(tmp_path):
    args, run, _ = run_fixture(tmp_path)
    marker = run.ckpt_path / 'keep.pt'
    marker.write_bytes(b'unchanged')
    args.guidance_scale += 1
    args.overwrite = True
    with pytest.raises(ValueError, match='fingerprint'):
        Run(args, LocalDist(), 'fit')
    assert marker.read_bytes() == b'unchanged'


def test_corrupt_component_identity_manifest_cannot_skip(tmp_path):
    args, run, component = run_fixture(tmp_path)
    for sample in run.samples:
        write_collection(args, run, component, sample)
    assert skip_completed(args, 'collect')
    atomic_json(run.path / 'components.json', {'fingerprint': component, 'identities': {'tampered': True}})
    assert not stage_complete(args, 'collect')


def test_overwrite_rejects_parent_path_and_symlink(tmp_path):
    args, run, _ = run_fixture(tmp_path)
    args.run_id = '..'
    with pytest.raises(ValueError, match='Unsafe run directory'):
        reset_stage(args, 'collect')
    args.run_id = 'outside'
    (Path(args.logs_dir) / args.run_id).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='Unsafe run directory'):
        reset_stage(args, 'collect')


def test_deep_validation_reports_progress_before_tensor_load(tmp_path, monkeypatch, capsys):
    import utils.cache as cache
    args, run, component = run_fixture(tmp_path)
    for sample in run.samples:
        write_collection(args, run, component, sample)
    run.finish('collect', expected_keys(args, 'collect'))
    args.progress = True
    reads = []
    original = cache.load_tensor

    def checked_load(path):
        if not reads:
            assert 'collect: verifying saved artifacts' in capsys.readouterr().err
        reads.append(path)
        return original(path)

    monkeypatch.setattr(cache, 'load_tensor', checked_load)
    assert stage_complete(args, 'collect')
    assert len(reads) == len(run.samples)
    assert f'{len(run.samples)}/{len(run.samples)}' in capsys.readouterr().err


def test_validator_reads_component_manifest_only_once_per_pass(tmp_path, monkeypatch):
    import utils.cache as cache
    args, run, component = run_fixture(tmp_path)
    rows = [write_collection(args, run, component, sample) for sample in run.samples]
    original = cache.read_json
    reads = []

    def read(path):
        if Path(path).name == 'components.json':
            reads.append(path)
        return original(path)

    monkeypatch.setattr(cache, 'read_json', read)
    validator = Validator(run)
    assert all(validator.row('collect', row) for row in rows)
    assert len(reads) == 1
    atomic_json(run.path / 'components.json', {'fingerprint': component, 'identities': {'changed': True}})
    assert not Validator(run).row('collect', rows[0])
    assert len(reads) == 2


def test_row_and_required_probe_checks_share_one_read(tmp_path, monkeypatch):
    import utils.cache as cache
    args, run, component = run_fixture(tmp_path)
    path, _ = write_checkpoint(args, run, component)
    original = cache.load_tensor
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(cache, 'load_tensor', read)
    validator = Validator(run)
    row = base_row(run.samples[0], key='readout', stage=0, access='raw', head='linear',
                   coordinate_variant='native', prediction_a=0.0, reference_a=0.0,
                   reference_b=0.0, costs={})
    assert validator.row('fit', row)
    assert validator.probe(0, 'raw', 'linear', 'native')
    assert reads == [path]


def test_warm_validation_receipts_skip_tensor_reads_without_writes(tmp_path, monkeypatch):
    import utils.cache as cache
    args, run, component = run_fixture(tmp_path)
    for sample in run.samples:
        write_collection(args, run, component, sample)
    run.finish('collect', expected_keys(args, 'collect'))
    assert stage_complete(args, 'collect')
    before = {path: path.stat().st_mtime_ns for path in run.path.rglob('*') if path.is_file()}
    assert any(path.parent.name == 'validation' for path in before)
    monkeypatch.setattr(cache, 'load_tensor', lambda *_: pytest.fail('unchanged validated tensor reread'))
    assert stage_complete(args, 'collect')
    assert skip_completed(args, 'collect')
    assert before == {path: path.stat().st_mtime_ns for path in run.path.rglob('*') if path.is_file()}


def test_changed_same_size_tensor_with_restored_mtime_revalidates(tmp_path, monkeypatch):
    import os
    import utils.cache as cache
    args, run, component = run_fixture(tmp_path)
    rows = [write_collection(args, run, component, sample) for sample in run.samples]
    run.finish('collect', expected_keys(args, 'collect'))
    assert stage_complete(args, 'collect')
    target = run.path / min(rows, key=lambda row: row['key'])['trajectory_path']
    before = target.stat()
    original_bytes = target.read_bytes()
    changed = original_bytes.replace(run.fingerprint.encode(), b'0' * len(run.fingerprint))
    assert changed != original_bytes and len(changed) == len(original_bytes)
    target.write_bytes(changed)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    original_load = cache.load_tensor
    reads = []

    def load(path):
        reads.append(path)
        return original_load(path)

    monkeypatch.setattr(cache, 'load_tensor', load)
    assert not stage_complete(args, 'collect')
    assert reads == [target.resolve()]


def test_receipted_missing_probe_and_images_are_not_complete(tmp_path):
    args, run, component = run_fixture(tmp_path)
    path, _ = write_checkpoint(args, run, component)
    assert Validator(run).probe(0, 'raw', 'linear', 'native')
    path.unlink()
    assert not Validator(run).probe(0, 'raw', 'linear', 'native')
    image = torch.zeros(1, 3, args.height, args.width)
    relative = run.write_tensor('intervene_images/pair.pt', {'before': image, 'after': image})
    assert Validator(run).artifact('image_path', relative)
    (run.path / relative).unlink()
    assert not Validator(run).artifact('image_path', relative)


def test_trajectory_receipt_cannot_be_reused_for_another_sample(tmp_path):
    args, run, component = run_fixture(tmp_path)
    rows = [write_collection(args, run, component, sample) for sample in run.samples[:2]]
    validator = Validator(run)
    assert validator.row('collect', rows[0])
    mismatched = dict(rows[1], trajectory_path=rows[0]['trajectory_path'])
    assert not validator.row('collect', mismatched)
    assert not Validator(run).row('collect', mismatched)
