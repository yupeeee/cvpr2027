"""Whole-controls accounting on tiny manifests, without loading any networks."""
from types import SimpleNamespace

from utils.config import parse_args


def fixture(rank=0):
    args = parse_args('controls', ['--num-steps', '2', '--stages', '0', '2',
        '--probe-heads', 'linear', '--probe-epochs', '2', '--edit-steps', '1',
        '--tasks', 'appearance', '--edit-budgets', '0.1', '--control-limit', '0',
        '--no-progress'])
    samples = [dict(sample_id=f's{index}', split=split)
               for index, split in enumerate(['train', 'validation', 'test', 'test'])]
    dist = SimpleNamespace(rank=rank, world_size=2, shard=lambda values: values[rank::2])
    records = {'controls_reading': set(), 'controls': set()}
    run = SimpleNamespace(samples=samples, completed=lambda kind: records[kind])
    return args, dist, run, records


def test_controls_total_covers_future_phases_and_accounts_for_resumed_work(monkeypatch):
    from exps import controls
    args, dist, run, records = fixture()
    monkeypatch.setattr(controls, 'checkpoint_valid', lambda *a, **kw: a[1] == 0)
    monkeypatch.setattr(controls, 'training_statistics_valid', lambda *a, **kw: True)
    local_reading = controls.reading_keys(args, dist.shard(run.samples))
    records['controls_reading'].add(local_reading[0])
    endpoint = controls.endpoint_jobs(args, dist.shard([s for s in run.samples if s['split'] == 'test']))
    cached_endpoint = next(iter(endpoint))
    records['controls'].add(cached_endpoint)
    work = controls.ControlsWork(args, dist, run)
    assert work.weights['probe/0/linear'] == args.probe_epochs
    assert set('reading/' + key for key in local_reading) <= work.weights.keys()
    assert set('endpoint/' + key for key in endpoint) <= work.weights.keys()
    assert work.initial == args.probe_epochs + 1 + endpoint[cached_endpoint]
    assert work.total == 2 + 2 * args.probe_epochs + len(local_reading) + sum(endpoint.values())
    updates = []
    bar = SimpleNamespace(update=updates.append)
    initial, total = work.initial, work.total
    for key in work.weights:
        work.complete(bar, key)
        work.complete(bar, key)  # Cached callbacks are idempotent.
    assert sum(updates) == total - initial
    assert work.total == total and work.initial == total


def test_completed_endpoints_do_not_load_models_or_trajectories(monkeypatch):
    from exps import controls
    args, dist, run, records = fixture()
    records['controls'].update(controls.endpoint_jobs(args, [s for s in run.samples if s['split'] == 'test']))
    def unexpected(*args, **kwargs):
        raise AssertionError('Completed endpoints performed model or trajectory work')
    monkeypatch.setattr(controls, 'Runtime', unexpected)
    run.load_trajectory = unexpected
    finished = []
    run.finish = lambda command, keys: finished.append((command, set(keys)))
    controls.endpoint_controls(args, dist, run)
    assert finished == [('controls', records['controls'])]
