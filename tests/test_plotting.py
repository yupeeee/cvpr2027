"""Bounded figure/CSV and audit checks using saved toy records in temporary paths."""
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.config import DEFAULTS
from utils.data import atomic_jsonl, atomic_torch
from utils.plotting import Figures, clock_control, finite_controls, import_audit, interventions, paired_reading, policies, reading, render, saved_rows, summarize, write_csv


def settings(tmp_path):
    return SimpleNamespace(**{**DEFAULTS, 'logs_dir': str(tmp_path / 'logs'), 'figs_dir': str(tmp_path / 'figs'),
                              'run_id': 'offline', 'num_steps': 1, 'stages': [0, 1], 'tasks': ['appearance'],
                              'audit_count': 1, 'bootstrap_count': 20, 'plot_dpi': 30,
                              'plot_width': 3.0, 'plot_height': 2.0})


def test_saved_tensor_filmstrip_and_csv_only(tmp_path):
    args = settings(tmp_path)
    log_dir = Path(args.logs_dir) / args.run_id
    image = torch.full((1, 3, 4, 4), 0.5)
    atomic_torch(log_dir / 'trajectories/toy.pt', {'states': {0: torch.zeros(1, 4, 4, 4), 1: torch.ones(1, 4, 4, 4)},
                 'previews': {0: image, 1: image}, 'final_image': image, 'stage_indices': [0, 1]})
    atomic_jsonl(log_dir / 'collect.jsonl', [{'key': 'toy', 'sample_id': 'toy', 'root_seed': 12, 'split': 'test',
                                           'trajectory_path': 'trajectories/toy.pt', 'status': 'ok', 'stages': [0, 1]}])
    manifest = render(args)
    assert [r['name'] for r in manifest] == ['filmstrip_toy']
    directory = Path(args.figs_dir) / args.run_id
    for suffix in ('png', 'pdf', 'csv'):
        assert (directory / f'filmstrip_toy.{suffix}').stat().st_size > 0
    with (directory / 'filmstrip_toy.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 5
    assert {row['view'] for row in table} == {'states', 'previews', 'actual_endpoint'}
    assert (directory / 'audit_template.csv').exists()
    assert (directory / 'figures_manifest.json').exists()


def test_cluster_table_and_actual_line_heatmap_outputs(tmp_path):
    args = settings(tmp_path)
    rows = [{'root_seed': i, 'task': 'appearance', 'role': 'a', 'controller': 'noop',
             'stage': k, 'budget': 0.1, 'status': 'ok', 'success': i % 2 == 0}
            for i in range(4) for k in (0, 1)]
    table = summarize(rows, ['task', 'role', 'controller', 'stage', 'budget'], 'success', args)
    assert all(row['mean'] == 0.5 and row['n'] == 4 and row['roots'] == 4 for row in table)
    figures = Figures(tmp_path / 'output', args)
    figures.lines('line', table, 'stage', 'Toy success', ['task'], ['controller'], 'Stage', 'Offline fixture only')
    figures.heatmap('heat', table, 'Offline fixture only')
    for name in ('line', 'heat'):
        for suffix in ('png', 'pdf', 'csv'):
            assert (tmp_path / 'output' / f'{name}.{suffix}').exists()
    missing = summarize([{'root_seed': 0, 'value': None, 'status': 'malformed'}], [], 'value', args)[0]
    assert missing['mean'] is None and missing['missing'] == 1 and missing['failures'] == 1


def test_audit_retains_ambiguity_malformed_and_disagreements(tmp_path):
    records = [{'key': 'a', 'root_seed': 1, 'task': 'appearance', 'request': 1,
                'a': {'target_success': True}, 'b': {'target_success': False}}]
    write_csv(tmp_path / 'labels.csv', [
        {'key': 'a', 'valid': '1', 'target_class': '-1', 'protected_preserved': '1'},
        {'key': 'a', 'valid': '?', 'target_class': 'ambiguous', 'protected_preserved': 'broken'},
        {'key': 'missing', 'valid': '0', 'target_class': '-1', 'protected_preserved': '0'}])
    rows = import_audit(tmp_path / 'labels.csv', records)
    assert len(rows) == 3
    assert rows[0]['disagrees_with_a'] is True and rows[0]['human_target_success'] is False
    assert rows[1]['annotation_status'] == 'ambiguous_or_malformed'
    assert rows[1]['human_target_success'] is None
    assert rows[2]['annotation_status'] == 'unknown_key'


def test_saved_measurement_transforms_preserve_work_and_joint_events(tmp_path):
    args = settings(tmp_path)

    class Tables:
        def __init__(self):
            self.args = args
            self.tables = {}

        def lines(self, name, table, *unused):
            self.tables[name] = table

        def heatmap(self, name, table, *unused):
            self.tables[name] = table

    figures = Tables()
    fit = [{'key': str(i), 'sample_id': str(i), 'root_seed': i, 'task': 'appearance', 'stage': 1, 'access': 'clean', 'head': 'linear',
            'status': 'ok', 'coordinate': 'native', 'squared_error': 0.2, 'reference_class': i % 2,
            'correct': True, 'correct_b': True, 'evaluator_agreement': True,
            'costs': {'generation': {'unet_samples': 2}, 'observation': {'unet_samples': 2}}} for i in range(4)]
    reading(figures, fit)
    assert figures.tables['reading_mse_compute'][0]['compute'] == 4
    assert figures.tables['reading_balanced_accuracy_stage'][0]['mean'] == 1
    clock_control(figures, fit)
    clock_rows = figures.tables['clock_only_compute']
    assert {r['clock_variant'] for r in clock_rows} == {'native', 'relabeled'}
    assert {r['compute'] for r in clock_rows} == {4}
    assert {r['mean'] for r in clock_rows} == {0.2}
    paired_reading(figures, fit + [{**r, 'access': 'raw', 'squared_error': 0.3} for r in fit[1:]])
    paired = figures.tables['paired_raw_clean_reading'][0]
    assert paired['mean'] == pytest.approx(-0.1)
    assert paired['matched_ids'] == 3 and paired['unmatched_left'] == 1 and paired['unmatched_right'] == 0
    metrics = {'originally_unsatisfied': True, 'target_success': True, 'selective_success': True,
               'protected_ok': True, 'assessment_correct': False, 'joint_success': False,
               'protected_errors': {'composition': 0.1, 'category': 0.2}}
    trials = [{'key': str(i), 'root_seed': i, 'sample_id': str(i), 'task': 'appearance', 'stage': 1,
               'budget': 0.1, 'status': 'ok', 'controller': 'noop', 'policy': 'single_stage',
               'a': metrics, 'b': metrics, 'costs': {'online': {'seconds': 0.1, 'unet_samples': 2}}} for i in range(3)]
    interventions(figures, trials)
    policies(figures, trials)
    finite_controls(figures, [{**r, 'native': {'a': metrics, 'b': metrics}, 'transported': {'a': metrics, 'b': metrics}} for r in trials])
    assert {r['coordinate'] for r in figures.tables['coordinate_finite_control']} == {'native', 'transported'}
    assert all(row['mean'] == 0 for row in figures.tables['policy_joint_stage'])
    assert all(row['mean'] == 1 for row in figures.tables['policy_action_stage'])
    assert figures.tables['intervention_target_unsatisfied']
    with pytest.raises(ValueError, match='No saved measurements'):
        render(args)


def test_authoritative_rank_records_during_resume(tmp_path):
    atomic_jsonl(tmp_path / 'fit.jsonl', [{'key': 'old'}])
    atomic_jsonl(tmp_path / 'ranks/fit.rank00000.jsonl', [{'key': 'old'}, {'key': 'new'}])
    assert {r['key'] for r in saved_rows(tmp_path, 'fit')} == {'old', 'new'}


def cache_fixture(tmp_path, samples=2):
    args = settings(tmp_path)
    args.progress = False
    log_dir = Path(args.logs_dir) / args.run_id
    image = torch.full((1, 3, 4, 4), 0.5)
    rows = []
    for index in range(samples):
        name = f'toy{index}'
        atomic_torch(log_dir / f'trajectories/{name}.pt', {
            'states': {0: image, 1: image}, 'previews': {0: image, 1: image},
            'final_image': image, 'stage_indices': [0, 1]})
        rows.append({'key': name, 'sample_id': name, 'root_seed': index, 'split': 'test',
                     'trajectory_path': f'trajectories/{name}.pt', 'status': 'ok', 'stages': [0, 1]})
    atomic_jsonl(log_dir / 'collect.jsonl', rows)
    args.audit_count = samples
    return args, log_dir, Path(args.figs_dir) / args.run_id


def test_completed_plot_cache_skips_tensor_loads_and_preserves_files(tmp_path, monkeypatch):
    import utils.plotting as plotting
    args, _, directory = cache_fixture(tmp_path)
    first = render(args)
    before = {path.name: path.stat().st_mtime_ns for path in directory.iterdir() if path.suffix in ('.png', '.pdf', '.csv')}
    monkeypatch.setattr(plotting, 'load_tensor', lambda *a, **k: pytest.fail('Cached figures must not load image tensors'))
    assert render(args) == first
    assert {name: (directory / name).stat().st_mtime_ns for name in before} == before


def test_plot_repairs_only_damaged_figure_and_retains_valid_siblings(tmp_path, monkeypatch):
    args, _, directory = cache_fixture(tmp_path)
    render(args)
    preserved = {suffix: (directory / f'filmstrip_toy1.{suffix}').stat().st_mtime_ns for suffix in ('png', 'pdf', 'csv')}
    (directory / 'filmstrip_toy0.pdf').write_bytes(b'broken')
    saved = []
    original = Figures.save

    def track(self, figure, name, *rest):
        saved.append(name)
        return original(self, figure, name, *rest)

    monkeypatch.setattr(Figures, 'save', track)
    render(args)
    assert saved == ['filmstrip_toy0']
    assert (directory / 'filmstrip_toy0.pdf').read_bytes().startswith(b'%PDF-')
    assert all((directory / f'filmstrip_toy1.{suffix}').stat().st_mtime_ns == stamp for suffix, stamp in preserved.items())
    (directory / 'audit_template.csv').unlink()
    saved.clear()
    render(args)
    assert not saved and (directory / 'audit_template.csv').exists()


def test_plot_reuses_legacy_manifest_and_overwrite_forces_all(tmp_path, monkeypatch):
    import json
    args, _, directory = cache_fixture(tmp_path)
    render(args)
    path = directory / 'figures_manifest.json'
    manifest = json.loads(path.read_text())
    manifest.pop('cache_groups')
    path.write_text(json.dumps(manifest))
    saved = []
    original = Figures.save

    def track(self, figure, name, *rest):
        saved.append(name)
        return original(self, figure, name, *rest)

    monkeypatch.setattr(Figures, 'save', track)
    render(args)
    assert not saved
    assert json.loads(path.read_text())['cache_groups']
    args.overwrite = True
    render(args)
    assert saved == ['filmstrip_toy0', 'filmstrip_toy1']


def test_plot_detects_changed_source_images_and_display_settings(tmp_path, monkeypatch):
    args, log_dir, _ = cache_fixture(tmp_path, samples=1)
    render(args)
    saved = []
    original = Figures.save

    def track(self, figure, name, *rest):
        saved.append(name)
        return original(self, figure, name, *rest)

    monkeypatch.setattr(Figures, 'save', track)
    source = log_dir / 'trajectories/toy0.pt'
    payload = torch.load(source, weights_only=False)
    payload['final_image'] = torch.zeros_like(payload['final_image'])
    atomic_torch(source, payload)
    render(args)
    assert saved == ['filmstrip_toy0']
    saved.clear()
    args.raw_display_scale *= 2
    render(args)
    assert saved == ['filmstrip_toy0']


def test_complete_plot_groups_skip_bootstraps_but_detect_changed_records(tmp_path, monkeypatch):
    import utils.plotting as plotting
    args = settings(tmp_path)
    args.progress = False
    log_dir = Path(args.logs_dir) / args.run_id
    path = log_dir / 'fit.jsonl'
    rows = [{'key': str(index), 'sample_id': str(index), 'root_seed': index, 'split': 'test',
             'task': 'category', 'stage': 1, 'access': 'raw', 'head': 'linear', 'status': 'ok',
             'squared_error': 0.2, 'costs': {'generation': {'unet_samples': 2}, 'observation': {'unet_samples': 0}}}
            for index in range(2)]
    atomic_jsonl(path, rows)
    render(args)
    original = plotting.summarize
    monkeypatch.setattr(plotting, 'summarize', lambda *a, **k: pytest.fail('Complete cached groups must skip bootstrapping'))
    render(args)
    monkeypatch.setattr(plotting, 'summarize', original)
    rows[0]['squared_error'] = 0.8
    atomic_jsonl(path, rows)
    render(args)
    with (Path(args.figs_dir) / args.run_id / 'reading_mse_stage.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert float(table[0]['mean']) == pytest.approx(0.5)


def test_plot_audit_content_changes_and_repairs_do_not_redraw_images(tmp_path):
    import json
    args, _, directory = cache_fixture(tmp_path, samples=1)
    labels = tmp_path / 'labels.csv'
    args.audit_labels = str(labels)
    annotation = {'key': 'collect/toy0/appearance', 'valid': '1', 'target_class': '1', 'protected_preserved': '1'}
    write_csv(labels, [annotation])
    render(args)
    stamp = (directory / 'filmstrip_toy0.png').stat().st_mtime_ns
    write_csv(labels, [{**annotation, 'valid': '?'}])
    render(args)
    summary = json.loads((directory / 'audit_summary.json').read_text())
    assert summary['statuses'] == {'ambiguous_or_malformed': 1}
    assert (directory / 'filmstrip_toy0.png').stat().st_mtime_ns == stamp
    (directory / 'audit_summary.json').write_text('invalid')
    render(args)
    assert json.loads((directory / 'audit_summary.json').read_text()) == summary
    assert (directory / 'filmstrip_toy0.png').stat().st_mtime_ns == stamp


def test_plot_repairs_malformed_manifest(tmp_path):
    args, _, directory = cache_fixture(tmp_path, samples=1)
    render(args)
    (directory / 'figures_manifest.json').write_text('{"figures": [null], "config": {}}')
    assert [row['name'] for row in render(args)] == ['filmstrip_toy0']


@pytest.mark.parametrize('reset_command', [None, 'controls'])
@pytest.mark.parametrize('label_relative', ['annotations/human_labels.csv', 'audit_template.csv'])
def test_plot_overwrite_preserves_selected_labels_inside_figure_directory(tmp_path, reset_command, label_relative):
    from utils.cache import reset_stage
    args, _, directory = cache_fixture(tmp_path, samples=1)
    args.ckpts_dir = str(tmp_path / 'ckpts')
    render(args)
    labels = directory / label_relative
    write_csv(labels, [{'key': 'collect/toy0/appearance', 'valid': '1', 'target_class': '1',
                        'protected_preserved': '1'}])
    original = labels.read_bytes()
    args.audit_labels = str(labels)
    args.overwrite = True
    if reset_command:
        reset_stage(args, reset_command)
        assert labels.read_bytes() == original
        assert not (directory / 'filmstrip_toy0.png').exists()
    render(args)
    assert labels.read_bytes() == original
    assert (directory / 'filmstrip_toy0.png').exists()
    with (directory / 'audit_imported.csv').open() as stream:
        assert list(csv.DictReader(stream))[0]['annotation_status'] == 'ok'
