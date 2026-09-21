"""Saved-record figures and exact tables. No model loading or sample generation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import atomic_json, load_tensor, merge_rows, read_jsonl
from .metrics import cluster_bootstrap, paired_bootstrap
from .progress import progress, status


def csv_text(rows):
    stream = io.StringIO(newline='')
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = csv_text(rows).encode('utf-8')
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)


def file_digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(size=path.stat().st_size, sha256=digest.hexdigest())


def plot_signature(args, rows, options=(), files=(), bootstrap=True):
    keys = ('plot_dpi', 'plot_width', 'plot_height') + tuple(options)
    if bootstrap:
        keys += ('bootstrap_seed', 'bootstrap_count', 'confidence_level')
    payload = dict(version=1, config={key: getattr(args, key) for key in keys}, records=rows,
                   files={str(path): file_digest(path) for path in files})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def readable_artifact(path):
    """Legacy manifests have no checksums: verify their documented file formats."""
    try:
        if path.stat().st_size == 0:
            return False
        if path.suffix == '.png':
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
        elif path.suffix == '.pdf':
            contents = path.read_bytes()
            return contents.startswith(b'%PDF-') and contents.rstrip().endswith(b'%%EOF')
        elif path.suffix == '.csv':
            list(csv.reader(io.StringIO(path.read_text(encoding='utf-8')), strict=True))
        else:
            return False
        return True
    except (OSError, ValueError, csv.Error):
        return False


def saved_rows(log_dir, command):
    parts = sorted((log_dir / 'ranks').glob(f'{command}.rank*.jsonl'))
    if parts:
        return merge_rows([row for path in parts for row in read_jsonl(path)])
    return merge_rows(read_jsonl(log_dir / f'{command}.jsonl'))


def get(row, path):
    value = row
    for key in path.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def balance(rows):
    classes = [[r['correct'] for r in rows if r.get('reference_class') == cls and r.get('correct') is not None]
               for cls in (0, 1)]
    return float(np.mean([np.mean(values) for values in classes])) if all(classes) else None


def summarize(rows, fields, metric, args):
    """Root-seed cluster intervals; missing values remain explicitly missing."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    table = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        if metric != 'balanced_accuracy':
            estimate = cluster_bootstrap(group, lambda r: get(r, metric), args.bootstrap_seed,
                                         args.bootstrap_count, args.confidence_level)
        else:
            roots = defaultdict(list)
            for row in group:
                roots[row['root_seed']].append(row)
            clusters = list(roots.values())
            mean, draws = balance(group), []
            if len(clusters) > 1 and mean is not None:
                rng = np.random.default_rng(args.bootstrap_seed)
                for _ in range(args.bootstrap_count):
                    value = balance([r for i in rng.integers(len(clusters), size=len(clusters)) for r in clusters[i]])
                    if value is not None:
                        draws.append(value)
            alpha = (1 - args.confidence_level) / 2
            low, high = np.quantile(draws, [alpha, 1 - alpha]) if draws else (None, None)
            estimate = dict(mean=mean, low=low, high=high, n=sum(r.get('correct') is not None for r in group),
                            roots=len(clusters), missing=sum(r.get('correct') is None for r in group),
                            undefined_bootstrap_draws=args.bootstrap_count - len(draws))
        row = dict(zip(fields, key))
        row.update(estimate, metric=metric, trials=len(group), failures=sum(r.get('status') != 'ok' for r in group))
        table.append(row)
    return table


class Figures:
    def __init__(self, directory, args, previous=None, legacy=False):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt, self.path, self.args = plt, Path(directory), args
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = []
        self.previous = previous or {}
        self.previous_figures = {row['name']: row for row in self.previous.get('figures', [])}
        self.groups = {}
        self.group_previous = {}
        self.legacy = legacy
        self.reused = 0
        self.written = 0

    def reusable(self, name):
        row = self.previous_figures.get(name)
        if row is None:
            return False
        files = [f'{name}.{suffix}' for suffix in ('png', 'pdf', 'csv')]
        receipts = self.group_previous.get('files', {})
        legacy = self.legacy
        if legacy:
            try:
                with (self.path / f'{name}.csv').open(newline='', encoding='utf-8') as stream:
                    legacy = len(list(csv.DictReader(stream))) == row.get('table_rows')
            except (OSError, ValueError, csv.Error):
                legacy = False
        if all((receipts.get(filename) is not None and file_digest(self.path / filename) == receipts[filename])
               or (legacy and readable_artifact(self.path / filename)) for filename in files):
            self.manifest.append(row)
            self.reused += 1
            return True
        return False

    def group(self, name, signature, function, extras=()):
        old = self.previous.get('cache_groups', {}).get(name, {})
        self.group_previous = old if old.get('input') == signature else {}
        start = len(self.manifest)
        expected = {f'{key}.{suffix}' for key in old.get('figures', []) for suffix in ('png', 'pdf', 'csv')} | set(extras)
        if (self.group_previous and old.get('files') and set(old['files']) == expected
                and all(receipt is not None and file_digest(self.path / filename) == receipt
                        for filename, receipt in old['files'].items())
                and all(key in self.previous_figures for key in old.get('figures', []))):
            self.manifest.extend(self.previous_figures[key] for key in old['figures'])
            self.reused += len(old['figures'])
            status(self.args, f'plot: reusing complete {name} outputs')
        else:
            function()
        names = [row['name'] for row in self.manifest[start:]]
        files = [f'{key}.{suffix}' for key in names for suffix in ('png', 'pdf', 'csv')] + list(extras)
        self.groups[name] = dict(input=signature, figures=names,
                                files={filename: file_digest(self.path / filename) for filename in files})
        self.group_previous = {}

    def save(self, figure, name, table, caption):
        figure.tight_layout()
        for suffix in ('png', 'pdf'):
            figure.savefig(self.path / f'{name}.{suffix}', dpi=self.args.plot_dpi, bbox_inches='tight')
        self.plt.close(figure)
        write_csv(self.path / f'{name}.csv', table)
        self.manifest.append(dict(name=name, caption=caption, table_rows=len(table)))
        self.written += 1

    def panels(self, keys):
        columns = min(2, len(keys))
        rows = math.ceil(len(keys) / columns)
        fig, axes = self.plt.subplots(rows, columns, squeeze=False,
                                     figsize=(self.args.plot_width * columns, self.args.plot_height * rows))
        for axis in list(axes.flat)[len(keys):]:
            axis.set_visible(False)
        return fig, list(axes.flat)[:len(keys)]

    def lines(self, name, table, x, ylabel, panels, series, xlabel, caption):
        if not table or self.reusable(name):
            return
        keys = sorted({tuple(r.get(k) for k in panels) for r in table}, key=str)
        figure, axes = self.panels(keys)
        for axis, panel in zip(axes, keys):
            groups = defaultdict(list)
            for row in table:
                if tuple(row.get(k) for k in panels) == panel:
                    groups[tuple(row.get(k) for k in series)].append(row)
            for label, values in sorted(groups.items(), key=lambda item: str(item[0])):
                points = sorted([r for r in values if finite(r.get(x))], key=lambda r: float(r[x]))
                if not points:
                    continue
                xx = [r[x] for r in points]
                yy = [r['mean'] if finite(r.get('mean')) else np.nan for r in points]
                axis.plot(xx, yy, marker='o', label=', '.join(f'{k}={v}' for k, v in zip(series, label)))
                low = [r['low'] if finite(r.get('low')) else np.nan for r in points]
                high = [r['high'] if finite(r.get('high')) else np.nan for r in points]
                axis.fill_between(xx, low, high, alpha=0.12)
                useful = [r for r in points if r.get('empirical_useful_stage')]
                if useful:
                    axis.scatter([r[x] for r in useful], [r['mean'] for r in useful], facecolors='none', edgecolors='black', s=110)
            axis.set(title=', '.join(f'{k}={v}' for k, v in zip(panels, panel)), xlabel=xlabel, ylabel=ylabel)
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize='small')
        figure.suptitle(caption, fontsize=10)
        self.save(figure, name, table, caption)

    def heatmap(self, name, table, caption):
        if not table or self.reusable(name):
            return
        fields = ['task', 'role', 'controller']
        keys = sorted({tuple(r.get(k) for k in fields) for r in table}, key=str)
        figure, axes = self.panels(keys)
        for axis, panel in zip(axes, keys):
            points = [r for r in table if tuple(r.get(k) for k in fields) == panel]
            stages, budgets = sorted({r['stage'] for r in points}), sorted({r['budget'] for r in points})
            array = np.full((len(budgets), len(stages)), np.nan)
            for point in points:
                i, j = budgets.index(point['budget']), stages.index(point['stage'])
                if point['mean'] is not None:
                    array[i, j] = point['mean']
                    axis.text(j, i, f"{point['mean']:.2f}\nn={point['n']}", ha='center', va='center', fontsize=8)
            im = axis.imshow(np.ma.masked_invalid(array), vmin=0, vmax=1, aspect='auto', origin='lower')
            axis.set(xticks=range(len(stages)), xticklabels=stages, yticks=range(len(budgets)), yticklabels=budgets,
                     xlabel='Native stage (noise → endpoint)', ylabel='Relative RMS edit budget', title=', '.join(map(str, panel)))
            figure.colorbar(im, ax=axis, label='Proxy task success')
        figure.suptitle(caption, fontsize=10)
        self.save(figure, name, table, caption)


def work(row, phases, counter):
    costs = row.get('costs')
    if not isinstance(costs, dict) or any(phase not in costs for phase in phases):
        return None
    if counter == 'seconds' and any(counter not in costs[phase] for phase in phases):
        return None
    return sum(costs[phase].get(counter, 0) for phase in phases)


def reading(figures, records, coordinate=False):
    rows = []
    for original in records:
        row = dict(original)
        row['access'] = row.get('access', row.get('observation'))
        row['coordinate'] = row.get('coordinate_variant', row.get('coordinate', 'native'))
        row['compute'] = work(row, ('generation', 'observation'), 'unet_samples')
        row['clock'] = row.get('clock_relabeled', (row['stage'] / figures.args.num_steps) ** figures.args.clock_power)
        rows.append(row)
    prefix = 'coordinate_reading' if coordinate else 'reading'
    fields = ['task', 'coordinate', 'access', 'head', 'stage', 'compute', 'clock']
    metrics = [('squared_error', 'mse', 'Endpoint score MSE (A scale)'), ('correct', 'accuracy', 'Classification accuracy (A)'),
               ('balanced_accuracy', 'balanced_accuracy', 'Balanced accuracy (A)'),
               ('evaluator_agreement', 'agreement', 'Endpoint evaluator A/B agreement'),
               ('correct_b', 'accuracy_b', 'A-readout agreement with B')]
    for metric, suffix, ylabel in metrics:
        subset = rows if metric == 'squared_error' else [r for r in rows if r.get('task') != 'category']
        table = summarize(subset, fields, metric, figures.args)
        axes = [('stage', 'Native stage (noise → endpoint)'), ('compute', 'Observation-inclusive UNet sample-work (CFG included)')]
        if coordinate:
            axes.append(('clock', 'Relabeled clock; identical states and work'))
        for x, xlabel in axes:
            figures.lines(f'{prefix}_{suffix}_{x}', table, x, ylabel, ['task'], ['coordinate', 'access', 'head'], xlabel,
                          'Coordinate/interface control' if coordinate else 'Held-out proxy reading; root-seed confidence intervals')


def paired_reading(figures, records, coordinate=False):
    """Pair the same sample IDs, never replace missing partners with unpaired means."""
    fields = ['task', 'head', 'stage']
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row.get(k) for k in fields)].append(row)
    table = []
    comparisons = [('wrapped', 'native'), ('transported', 'native')] if coordinate else [('clean', 'raw')]
    for values, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        for left_name, right_name in comparisons:
            def selected(name):
                if coordinate:
                    return [r for r in rows if r.get('coordinate_variant', r.get('coordinate')) == name
                            and r.get('access', r.get('observation')) == 'raw']
                return [r for r in rows if r.get('access', r.get('observation')) == name]
            left, right = selected(left_name), selected(right_name)
            if not left and not right:
                continue
            left_ids, right_ids = {r['sample_id'] for r in left}, {r['sample_id'] for r in right}
            if len(left_ids) != len(left) or len(right_ids) != len(right):
                raise ValueError('Duplicate sample IDs in a paired reading group')
            result = paired_bootstrap(left, right, lambda r: r['sample_id'], lambda r: r.get('squared_error'),
                                      figures.args.bootstrap_seed, figures.args.bootstrap_count, figures.args.confidence_level)
            table.append({**dict(zip(fields, values)), **result, 'comparison': f'{left_name} minus {right_name}',
                          'matched_ids': len(left_ids & right_ids), 'metric': 'paired_squared_error_difference'})
    figures.lines('paired_coordinate_reading' if coordinate else 'paired_raw_clean_reading', table, 'stage',
                  'Paired MSE difference (negative favors first)', ['task'], ['head', 'comparison'],
                  'Native stage (noise → endpoint)', 'Paired root-seed intervals; unmatched and missing partners reported in CSV')


def clock_control(figures, records):
    """Duplicate the same observations/work with two labels, never recompute a model."""
    rows = []
    for record in records:
        if record.get('coordinate_variant', record.get('coordinate')) != 'native':
            continue
        for clock in ('native', 'relabeled'):
            row = dict(record)
            row['clock_variant'] = clock
            row['clock_label'] = record['stage'] / figures.args.num_steps
            if clock == 'relabeled':
                row['clock_label'] = row['clock_label'] ** figures.args.clock_power
            row['compute'] = work(row, ('generation', 'observation'), 'unet_samples')
            rows.append(row)
    table = summarize(rows, ['task', 'head', 'stage', 'clock_variant', 'clock_label', 'compute'],
                      'squared_error', figures.args)
    for x, xlabel in (('clock_label', 'Clock label (native or relabeled)'),
                      ('compute', 'Identical observation-inclusive UNet sample-work')):
        figures.lines(f'clock_only_{x}', table, x, 'Endpoint score MSE (A scale)', ['task'],
                      ['head', 'clock_variant'], xlabel,
                      'Clock-only control: identical observations and work; compute curves coincide')


def finite_controls(figures, records):
    rows = []
    for record in records:
        for variant in ('native', 'transported'):
            trial = record.get(variant)
            if isinstance(trial, dict):
                for row in trial_rows([{**record, **trial}], 'all'):
                    rows.append({**row, 'coordinate': variant})
    table = summarize(rows, ['task', 'role', 'controller', 'budget', 'stage', 'coordinate'],
                      'selective_success', figures.args)
    figures.lines('coordinate_finite_control', table, 'stage', 'Selective proxy success',
                  ['task', 'role'], ['coordinate', 'controller', 'budget'],
                  'Native stage (noise → endpoint)',
                  'Conjugated finite editors; budgets measured in original latent coordinates')


def trial_rows(records, subset):
    rows = []
    for record in records:
        for role in ('a', 'b'):
            metrics = record.get(role)
            if not isinstance(metrics, dict):
                continue
            old = metrics.get('originally_unsatisfied')
            if subset == 'unsatisfied' and old is not True or subset == 'satisfied' and old is not False:
                continue
            row = {**record, **metrics, 'role': role, 'subset': subset}
            row['online_seconds'] = work(record, ('online',), 'seconds')
            row['online_unet_samples'] = work(record, ('online',), 'unet_samples')
            rows.append(row)
    return rows


def interventions(figures, records):
    fields = ['task', 'role', 'controller', 'stage', 'budget']
    for subset in ('all', 'unsatisfied'):
        rows = trial_rows(records, subset)
        for metric, short in (('target_success', 'target'), ('selective_success', 'selective')):
            figures.heatmap(f'intervention_{short}_{subset}', summarize(rows, fields, metric, figures.args),
                            f'Proxy task {short} success: {subset} requests; failures retained')
    rows = trial_rows(records, 'all')
    success = summarize(rows, fields, 'target_success', figures.args)
    protection = {tuple(r[k] for k in fields): r for r in summarize(rows, fields, 'protected_ok', figures.args)}
    for row in success:
        preserved = protection[tuple(row[k] for k in fields)]
        row.update(preservation=preserved['mean'], preservation_low=preserved['low'], preservation_high=preserved['high'])
    figures.lines('preservation_tradeoff', success, 'preservation', 'Proxy target success', ['task', 'role'],
                  ['controller', 'stage'], 'Protected proxy preservation rate', 'Success–preservation tradeoff; points are declared budgets')
    details = [{**r, 'protected_field': field, 'protected_error': value} for r in rows
               for field, value in r.get('protected_errors', {}).items()]
    table = summarize(details, fields + ['protected_field'], 'protected_error', figures.args)
    figures.lines('protected_errors', table, 'stage', "Absolute change (evaluator's native score scale)",
                  ['task', 'role', 'protected_field'], ['controller', 'budget'], 'Native stage (noise → endpoint)',
                  'Protected semantic proxies; numerical image drift is not identity or quality')


def policies(figures, records):
    rows = [row for subset in ('all', 'unsatisfied', 'satisfied') for row in trial_rows(records, subset)]
    single = [r for r in rows if r.get('policy') == 'single_stage']
    fields = ['task', 'role', 'budget', 'subset', 'policy', 'stage']
    for metric, short in (('assessment_correct', 'reading'), ('target_success', 'action'), ('joint_success', 'joint')):
        table = summarize(single, fields, metric, figures.args)
        work_by = {tuple(r[k] for k in fields): r for r in summarize(single, fields, 'online_seconds', figures.args)}
        for row in table:
            row['online_seconds'] = work_by[tuple(row[k] for k in fields)]['mean']
            row['empirical_useful_stage'] = bool(short == 'joint' and row['low'] is not None and row['low'] >= figures.args.useful_success)
        for x, xlabel in (('stage', 'Native stage (noise → endpoint)'), ('online_seconds', 'Measured online seconds, including inspections and edits')):
            figures.lines(f'policy_{short}_{x}', table, x, f'{short.capitalize()} proxy success',
                          ['task', 'role', 'subset'], ['budget'], xlabel,
                          'Same-trial joint event; rings mark empirical useful-stage estimates' if short == 'joint'
                          else 'Same requests/seeds and evaluator as the joint event')
    others = [r for r in rows if r.get('policy') != 'single_stage']
    fields = ['task', 'role', 'budget', 'subset', 'policy']
    table = summarize(others, fields, 'joint_success', figures.args)
    for counter, label in (('online_seconds', 'Measured online seconds (all decision/generation work)'),
                           ('online_unet_samples', 'Online UNet sample-work; other counters remain in records CSV')):
        cost_by = {tuple(r[k] for k in fields): r for r in summarize(others, fields, counter, figures.args)}
        for row in table:
            row[counter] = cost_by[tuple(row[k] for k in fields)]['mean']
        figures.lines(f'policy_comparison_{counter}', table, counter, 'Same-trial joint proxy success',
                      ['task', 'role', 'subset'], ['policy'], label,
                      'Validation-frozen policies; endpoint baseline inspects its completed image')


def diagnostics(figures, records):
    rows = []
    for record in records:
        for role in ('a', 'b'):
            disagreement = get(record, f'endpoint_disagreement.{role}') or {}
            rows.append({**record, 'role': role, 'disagreement': disagreement.get('target_decision_disagrees'),
                         'failure': disagreement.get('target_failure_first'),
                         'score_disagreement': get(disagreement, f"score_absolute_difference.{record.get('task')}")})
    for metric in ('disagreement', 'failure', 'score_disagreement'):
        if not rows or figures.reusable(f'diagnostic_{metric}'):
            continue
        fields = ['task', 'role', 'stage', 'later_stage', 'budget']
        keys = sorted({tuple(r.get(k) for k in fields) for r in rows}, key=str)
        figure, axes = figures.panels(keys)
        for axis, panel in zip(axes, keys):
            selected = [r for r in rows if tuple(r.get(k) for k in fields) == panel]
            for controller in sorted({str(r.get('controller')) for r in selected}):
                points = [r for r in selected if str(r.get('controller')) == controller and finite(r.get('normalized_residual')) and finite(r.get(metric))]
                axis.scatter([r['normalized_residual'] for r in points], [r[metric] for r in points], label=controller, alpha=0.7)
            axis.set(title=f'{panel[0]}, {panel[1]}, {panel[2]}→{panel[3]}, budget={panel[4]}',
                     xlabel='Normalized segment compatibility residual', ylabel=metric.replace('_', ' '))
            axis.legend(fontsize='small')
        caption = 'Exploratory segment diagnostic; a zero no-op residual can still fail a request'
        figure.suptitle(caption, fontsize=10)
        figures.save(figure, f'diagnostic_{metric}', rows, caption)


def show_image(axis, tensor, title, raw_scale=None):
    import torch
    status = 'ok'
    if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (3, 4) or not torch.isfinite(tensor).all():
        axis.text(0.5, 0.5, 'Missing or malformed image', ha='center', va='center', wrap=True)
        status = 'missing_or_malformed'
    else:
        value = tensor[0] if tensor.ndim == 4 else tensor
        if raw_scale is not None:
            if value.shape[0] < 3:
                value = value[:1].expand(3, -1, -1)
            value = value[:3].float() / (2 * raw_scale) + 0.5
        if value.shape[0] != 3:
            axis.text(0.5, 0.5, 'Malformed channel count', ha='center', va='center')
            status = 'malformed'
        else:
            axis.imshow(value.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    axis.set_title(title, fontsize=8)
    axis.set_axis_off()
    return status


def image_grids(figures, log_dir, collected, trials):
    audits = []
    filmstrips = sorted(collected, key=lambda r: r['sample_id'])[:figures.args.audit_count]
    with progress(figures.args, filmstrips, desc="plot: audit filmstrips", unit="sample",
                  position=1, leave=False) as bar:
        for record in bar:
            path = log_dir / record['trajectory_path']
            if not figures.reusable(f"filmstrip_{record['sample_id']}"):
                trajectory = load_tensor(path) if path.exists() else {}
                stages = trajectory.get('stage_indices', record.get('stages', figures.args.stages))
                figure, axes = figures.plt.subplots(2, len(stages) + 1, squeeze=False,
                                                    figsize=(figures.args.plot_width * (len(stages) + 1) / 3, figures.args.plot_height))
                table = []
                for index, stage in enumerate(stages):
                    for row_index, (field, label) in enumerate((('states', 'Raw latent display'), ('previews', 'Clean prediction'))):
                        name = 'Terminal decode' if field == 'previews' and stage == figures.args.num_steps else label
                        status = show_image(axes[row_index, index], trajectory.get(field, {}).get(stage), f'{name}\nk={stage}',
                                            figures.args.raw_display_scale if field == 'states' else None)
                        table.append(dict(sample_id=record['sample_id'], root_seed=record['root_seed'], stage=stage,
                                          view=field, display_status=status, trajectory_path=str(path), raw_display_scale=figures.args.raw_display_scale))
                status = show_image(axes[0, -1], trajectory.get('final_image'), 'Actual endpoint')
                table.append(dict(sample_id=record['sample_id'], root_seed=record['root_seed'], stage=figures.args.num_steps,
                                  view='actual_endpoint', display_status=status, trajectory_path=str(path)))
                axes[1, -1].set_axis_off()
                caption = f"{record['sample_id']}: fixed raw scaling; clean predictions are not continuations"
                figure.suptitle(caption, fontsize=10)
                figures.save(figure, f"filmstrip_{record['sample_id']}", table, caption)
            for task in figures.args.tasks:
                audits.append(dict(key=f"collect/{record['sample_id']}/{task}", sample_id=record['sample_id'], root_seed=record['root_seed'],
                                   task=task, request=None, status=record.get('status'), image_path=str(path),
                                   valid='', target_class='', protected_preserved='', notes=''))
    selected = sorted(trials, key=lambda row: row['key'])[:figures.args.audit_count]
    if selected and not figures.reusable('before_after_audit'):
        figure, axes = figures.plt.subplots(len(selected), 2, squeeze=False,
                                            figsize=(figures.args.plot_width, figures.args.plot_height * len(selected) / 2))
        table = []
        with progress(figures.args, selected, desc="plot: before/after audit pairs", unit="pair",
                      position=1, leave=False) as bar:
            for index, record in enumerate(bar):
                relative = record.get('image_path')
                path = log_dir / relative if relative else None
                pair = load_tensor(path) if path and path.exists() else {}
                statuses = [show_image(axes[index, j], pair.get(field),
                                       f"{field}: {record.get('controller')}/{record.get('policy', '')}, k={record.get('stage')}\n"
                                       f"{record['sample_id']}, request={record.get('request')}, {record.get('status')}")
                            for j, field in enumerate(('before', 'after'))]
                table.append({**record, 'display_status': statuses})
        caption = 'Deterministic sorted-ID before/after audit; proxy task success, including failures'
        figure.suptitle(caption, fontsize=10)
        figures.save(figure, 'before_after_audit', table, caption)
    for record in selected:
        audit = {key: record.get(key) for key in ('key', 'sample_id', 'root_seed', 'task', 'request', 'status', 'image_path')}
        audit.update(valid='', target_class='', protected_preserved='', notes='')
        audits.append(audit)
    template = figures.path / 'audit_template.csv'
    selected = Path(figures.args.audit_labels).resolve() if figures.args.audit_labels else None
    if selected != template.resolve():
        write_csv(template, audits)
    return audits


def import_audit(path, records):
    """Retain unknown keys, malformed/ambiguous labels, and evaluator disagreements."""
    index = {row['key']: row for row in records}
    seen = set()
    imported = []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for annotation in csv.DictReader(stream):
            extra_columns = annotation.pop(None, None)
            key = annotation.get('key', '')
            source = index.get(key)
            result = dict(annotation)
            result['unparsed_columns'] = extra_columns
            result['duplicate_annotation'] = key in seen
            seen.add(key)
            result['annotation_status'] = 'ok' if source is not None else 'unknown_key'
            problems, parsed = [], {}
            for field, allowed in (('valid', {'0': False, '1': True}), ('target_class', {'-1': -1, '+1': 1, '1': 1}),
                                   ('protected_preserved', {'0': False, '1': True})):
                value = (annotation.get(field) or '').strip().lower()
                parsed[field] = allowed.get(value)
                if value in {'', '?', 'ambiguous', 'unknown'}:
                    problems.append(f'{field}: ambiguous/unlabeled')
                elif value not in allowed:
                    problems.append(f'{field}: malformed label {value!r}')
            result['annotation_issues'] = '; '.join(problems)
            result.update(human_valid=parsed['valid'], human_class=parsed['target_class'],
                          human_protected_preserved=parsed['protected_preserved'])
            if problems and source is not None:
                result['annotation_status'] = 'ambiguous_or_malformed'
            result['human_target_success'] = result['human_selective_success'] = None
            if source is not None:
                result.update(root_seed=source.get('root_seed'), task=source.get('task'), request=source.get('request'))
                if parsed['valid'] is False:
                    result['human_target_success'] = result['human_selective_success'] = False
                elif parsed['valid'] is True and parsed['target_class'] is not None and source.get('request') in (-1, 1):
                    result['human_target_success'] = parsed['target_class'] == source['request']
                    if parsed['protected_preserved'] is not None:
                        result['human_selective_success'] = bool(result['human_target_success'] and parsed['protected_preserved'])
                for role in ('a', 'b'):
                    predicted_class = get(source, f'{role}.final_decision')
                    result[f'class_disagrees_with_{role}'] = (2 * predicted_class - 1 != parsed['target_class']
                        if predicted_class is not None and parsed['target_class'] is not None else None)
                    proxy = get(source, f'{role}.target_success')
                    result[f'disagrees_with_{role}'] = (proxy != result['human_target_success']
                                                       if proxy is not None and result['human_target_success'] is not None else None)
            imported.append(result)
    return imported


def audit_figures(figures, imported):
    # Unknown/duplicate/ambiguous annotations remain in raw output and count tables.
    # One annotation per known trial can contribute a measured semantic event.
    selected = [r for r in imported if r.get('root_seed') is not None and not r['duplicate_annotation']]
    table = []
    for metric in ('human_target_success', 'human_selective_success'):
        rows = summarize(selected, ['task', 'request'], metric, figures.args)
        for row in rows:
            group = [r for r in imported if r.get('task') == row['task'] and r.get('request') == row['request']]
            row.update(ambiguous_or_malformed=sum(r['annotation_status'] == 'ambiguous_or_malformed' for r in group),
                       duplicate_annotations=sum(r['duplicate_annotation'] for r in group),
                       unknown_keys=sum(r['annotation_status'] == 'unknown_key' for r in group))
        table.extend(rows)
    figures.lines('human_audit_success', table, 'request', 'Human-annotated success fraction', ['task'], ['metric'],
                  'Requested class (-1 or +1)', 'Optional human audit; ambiguous labels remain missing and counted')


def legacy_compatible(previous, args, records, source_files, manifest_path, directory):
    """Migrate old manifests conservatively, without redrawing intact figures."""
    if not previous or 'cache_groups' in previous:
        return False
    keys = ('plot_dpi', 'plot_width', 'plot_height', 'bootstrap_seed', 'bootstrap_count', 'confidence_level',
            'num_steps', 'clock_power', 'useful_success', 'audit_count', 'raw_display_scale',
            'tasks', 'plot_splits', 'audit_labels', 'tasks_path')
    if any(previous.get('config', {}).get(key) != getattr(args, key) for key in keys):
        return False
    for name, rows in records.items():
        path = directory / f'records_{name}.csv'
        if rows and (not path.is_file() or path.read_bytes() != csv_text(rows).encode('utf-8')):
            return False
    return all(path.is_file() and path.stat().st_mtime_ns <= manifest_path.stat().st_mtime_ns
               for path in source_files)


def render(args):
    log_dir = Path(args.logs_dir) / args.run_id
    directory = Path(args.figs_dir) / args.run_id
    names = ('collect', 'fit', 'intervene', 'controls', 'controls_reading', 'diagnose', 'policy')
    records = {}
    with progress(args, names, desc="plot: loading saved records", unit="file group") as bar:
        for name in bar:
            bar.set_description(f"plot: loading {name} records")
            records[name] = [row for row in saved_rows(log_dir, name) if row.get('split') in args.plot_splits]
    if not any(records.values()):
        raise ValueError(f'No saved measurements for splits {args.plot_splits} in {log_dir}.')
    missing = [name for name, rows in records.items() if not rows]
    if missing:
        status(args, "plot: no saved measurements for " + ', '.join(missing) + "; skipping those figure groups")
    if getattr(args, 'overwrite', False):
        from .cache import clear_figures
        clear_figures(args)
    manifest_path = directory / 'figures_manifest.json'
    try:
        previous = json.loads(manifest_path.read_text())
        valid = isinstance(previous, dict) and isinstance(previous.get('figures'), list)
        valid = valid and isinstance(previous.get('config'), dict) and isinstance(previous.get('cache_groups', {}), dict)
        if valid:
            valid = all(isinstance(row, dict) and isinstance(row.get('name'), str)
                        and Path(row['name']).name == row['name'] and isinstance(row.get('table_rows'), int)
                        for row in previous['figures'])
            valid = valid and all(isinstance(group, dict) and isinstance(group.get('figures'), list)
                and all(isinstance(name, str) for name in group['figures']) and isinstance(group.get('files'), dict)
                and all(isinstance(name, str) and Path(name).name == name for name in group['files'])
                for group in previous.get('cache_groups', {}).values())
        if not valid:
            previous = {}
    except (OSError, ValueError):
        previous = {}
    trials = records['intervene'] + records['policy']
    selected = sorted(trials, key=lambda row: row['key'])[:args.audit_count]
    collected = sorted(records['collect'], key=lambda row: row['sample_id'])[:args.audit_count]
    images = [log_dir / row['trajectory_path'] for row in collected]
    images += [log_dir / row['image_path'] for row in selected if row.get('image_path')]
    audit_files = [Path(args.audit_labels), Path(args.tasks_path)] if args.audit_labels else []
    figures = Figures(directory, args, previous,
                      legacy_compatible(previous, args, records, images + audit_files, manifest_path, directory))

    def controls():
        coordinate = [r for r in records['controls_reading'] + records['controls'] if 'prediction_a' in r]
        if coordinate:
            native = [r for r in records['fit'] if r.get('access') == 'raw']
            native_keys = {r['key'] for r in coordinate}
            reading(figures, [r for r in native if r['key'] not in native_keys] + coordinate, coordinate=True)
            clock_control(figures, coordinate)
            paired_reading(figures, coordinate, coordinate=True)
        finite_controls(figures, records['controls'])
        drift = [row for row in records['controls'] if 'endpoint_drift_rms' in row]
        if drift:
            table = summarize(drift, ['stage', 'coordinate', 'controller', 'budget'], 'endpoint_drift_rms', args)
            figures.lines('endpoint_drift', table, 'stage', 'Paired endpoint image RMS drift', [],
                          ['coordinate', 'controller', 'budget'], 'Native stage (noise → endpoint)',
                          'Coordinate/interface endpoint control; measured numerical drift')

    def audits():
        from .tasks import load_tasks, threshold
        task_config = load_tasks(args.tasks_path)
        collection_sources = []
        for record in records['collect']:
            for task in args.tasks:
                source = dict(record, key=f"collect/{record['sample_id']}/{task}", task=task, request=None)
                for role in ('a', 'b'):
                    score = get(record, f'scores.{role}.{task}')
                    source[role] = {'final_decision': int(score >= threshold(task_config, task, role, 'decision')) if finite(score) else None}
                collection_sources.append(source)
        imported = import_audit(args.audit_labels, collection_sources + trials)
        write_csv(figures.path / 'audit_imported.csv', imported)
        audit_figures(figures, imported)
        atomic_json(figures.path / 'audit_summary.json', {
            'rows': len(imported), 'statuses': {s: sum(r['annotation_status'] == s for r in imported)
                                               for s in sorted({r['annotation_status'] for r in imported})},
            'a_disagreements': sum(r.get('disagrees_with_a') is True for r in imported),
            'b_disagreements': sum(r.get('disagrees_with_b') is True for r in imported),
            'a_class_disagreements': sum(r.get('class_disagrees_with_a') is True for r in imported),
            'b_class_disagreements': sum(r.get('class_disagrees_with_b') is True for r in imported),
            'duplicate_annotations': sum(r['duplicate_annotation'] for r in imported),
            'note': 'Ambiguous, malformed and disagreed annotations remain in audit_imported.csv.'})

    groups = [('records', plot_signature(args, records, bootstrap=False),
               lambda: [write_csv(directory / f'records_{name}.csv', rows) for name, rows in records.items() if rows],
               [f'records_{name}.csv' for name, rows in records.items() if rows])]
    if records['fit']:
        groups.append(('reading profiles', plot_signature(args, records['fit'], ('num_steps', 'clock_power')),
                       lambda: (reading(figures, records['fit']), paired_reading(figures, records['fit'])), []))
    if records['intervene']:
        groups.append(('intervention heatmaps', plot_signature(args, records['intervene']),
                       lambda: interventions(figures, records['intervene']), []))
    if records['controls'] or records['controls_reading']:
        groups.append(('coordinate and clock controls',
                       plot_signature(args, [records['fit'], records['controls'], records['controls_reading']], ('num_steps', 'clock_power')),
                       controls, []))
    if records['diagnose']:
        groups.append(('segment diagnostics', plot_signature(args, records['diagnose']),
                       lambda: diagnostics(figures, records['diagnose']), []))
    if records['policy']:
        groups.append(('policy profiles', plot_signature(args, records['policy'], ('useful_success',)),
                       lambda: policies(figures, records['policy']), []))
    groups.append(('saved-image audit grids',
                   plot_signature(args, [collected, selected], ('num_steps', 'stages', 'tasks', 'audit_count', 'raw_display_scale'), images, bootstrap=False),
                   lambda: image_grids(figures, log_dir, records['collect'], trials), ['audit_template.csv']))
    if args.audit_labels:
        groups.append(('human audits', plot_signature(args, [records['collect'], trials], ('tasks',), audit_files),
                       audits, ['audit_imported.csv', 'audit_summary.json']))
    with progress(args, groups, desc="plot: validating saved figures", unit="group") as bar:
        for name, signature, function, extras in bar:
            bar.set_description(f"plot: {name}")
            figures.group(name, signature, function, extras)
    manifest = {
        'figures': figures.manifest, 'config': vars(args), 'cache_groups': figures.groups,
        'semantics': 'Proxy task success; optional human annotations are reported separately.',
        'intervals': 'Bootstrap complete root-seed clusters; one-root intervals are undefined.',
        'missing_commands': missing}
    if manifest != previous:
        atomic_json(manifest_path, manifest)
    status(args, f'plot: wrote {figures.written} figure(s), reused {figures.reused} complete figure(s)')
    return figures.manifest
