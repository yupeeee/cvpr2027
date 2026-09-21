"""Bounded integration: three tiny offline fixture trajectories in a temp directory."""
import json
from pathlib import Path

import torch

from test_sampler import tiny_sampler  # noqa: F401


def test_entry_points_share_cache_and_resume(tmp_path, monkeypatch, tiny_sampler):
    from exps.collect import collect
    from exps.controls import endpoint_controls, reading_controls
    from exps.diagnose import main as diagnose
    from exps.intervene import main as intervene
    from exps.policy import main as policy
    from utils.config import parse_args
    from utils.data import Run, read_jsonl
    from utils.distributed import Distributed
    from utils.probes import train_probes
    from utils.sampler import Sampler
    from utils.tasks import Scorer
    from utils import models
    monkeypatch.setattr(models, 'prepare_for_inference', lambda args: None)

    # Only model factories are substituted. Generation, schedules, frozen-weight
    # gradients, caches, heads, interventions, policies and merges use real code.
    class TensorScores:
        def __init__(self, role):
            self.role = role

        def prepare_text(self, object_name, cost):
            # This deterministic test scorer has no text encoder.
            return None

        def score(self, images, object_name, cost):
            cost.add(f'scorer_{self.role}_images', images.shape[0])
            return {'appearance': images[:, 0].mean((1, 2)) - images[:, 2].mean((1, 2)),
                    'composition': images[:, :, :4].mean((1, 2, 3)) - images[:, :, 4:].mean((1, 2, 3)),
                    'category': images.mean((1, 2, 3))}

    def fixture_sampler(args, device):
        # The locally constructed sampler advertises the fixture invocation's identity.
        tiny_sampler.model_id = args.model_id
        tiny_sampler.model_revision = args.model_revision
        tiny_sampler.guidance_scale = args.guidance_scale
        return tiny_sampler

    monkeypatch.setattr(Sampler, 'load', staticmethod(fixture_sampler))
    monkeypatch.setattr(Scorer, 'load', staticmethod(lambda args, role, device: TensorScores(role)))
    prompt_path = tmp_path / 'prompts.jsonl'
    prompt_path.write_text(json.dumps({'prompt_id': 'fixture', 'text': 'object', 'object': 'object'}) + '\n')
    task_path = Path(__file__).resolve().parents[1] / 'configs' / 'tasks.json'
    argv = ['--device', 'cpu', '--num-samples', '3', '--train-fraction', '0.34',
            '--validation-fraction', '0.34', '--height', '8', '--width', '8',
            '--num-steps', '3', '--stages', '0', '1', '3', '--intervention-stages', '1',
            '--diagnostic-stages', '0', '1', '--control-stages', '1', '--control-editors', 'noop',
            '--policy-stages', '1', '--policy-deadline', '1', '--policy-thresholds', '0',
            '--policy-variants', 'never', 'endpoint', 'single_stage', '--editors', 'noop',
            '--edit-budgets', '0.05', '--tasks', 'appearance', '--probe-heads', 'linear',
            '--probe-epochs', '1', '--probe-batch-size', '1', '--pool-grid', '1',
            '--edit-steps', '1', '--edit-grid', '2', '--policy-calibration-limit', '1',
            '--intervention-limit', '1', '--diagnostic-limit', '1', '--control-limit', '1',
            '--policy-limit', '1', '--prompts-path', str(prompt_path), '--tasks-path', str(task_path),
            '--logs-dir', str(tmp_path / 'logs'), '--ckpts-dir', str(tmp_path / 'ckpts'),
            '--figs-dir', str(tmp_path / 'figs'), '--run-id', 'offline_fixture']
    args = parse_args('collect', argv)
    with Distributed(args) as dist:
        run = Run(args, dist, 'collect')
        collect(args, dist, run)
        train_probes(args, dist, run)
        reading_controls(args, dist, run)
        endpoint_controls(args, dist, run)
    intervene(argv)
    diagnose(argv)
    policy(argv)
    log = tmp_path / 'logs' / 'offline_fixture'
    assert len(read_jsonl(log / 'collect.jsonl')) == 3
    interventions = read_jsonl(log / 'intervene.jsonl')
    assert len(interventions) == 1 and interventions[0]['image_drift_rms'] == 0
    diagnostics = read_jsonl(log / 'diagnose.jsonl')
    assert len(diagnostics) == 1 and diagnostics[0]['normalized_residual'] == 0
    assert all(row['endpoint_equivalent'] for row in read_jsonl(log / 'controls.jsonl'))
    policies = read_jsonl(log / 'policy.jsonl')
    assert {row['policy'] for row in policies} == {'never', 'endpoint', 'single_stage'}
    assert len(read_jsonl(log / 'policy_calibration.jsonl')) == 2
    frozen = (log / 'policy_settings.json').read_bytes()
    policy(argv + ['--resume'])
    assert frozen == (log / 'policy_settings.json').read_bytes()
    assert read_jsonl(log / 'policy.jsonl') == policies
    assert all(not parameter.requires_grad for parameter in tiny_sampler.unet.parameters())

    # A complete restart must reuse every stage before model preparation or CUDA.
    import importlib
    import pytest
    from utils import cache, models
    stages = ('collect', 'fit', 'intervene', 'controls', 'diagnose', 'policy')
    assert all(cache.stage_complete(args, stage) for stage in stages)
    settings_path = log / 'policy_settings.json'
    damaged = json.loads(settings_path.read_text())
    del damaged['threshold']
    settings_path.write_text(json.dumps(damaged))
    assert not cache.stage_complete(args, 'policy')
    from exps.policy import freeze_settings
    with Distributed(args) as dist:
        freeze_settings(args, dist, Run(args, dist, 'policy'))
    assert settings_path.read_bytes() == frozen
    assert cache.stage_complete(args, 'policy')
    saved_times = {path: path.stat().st_mtime_ns for directory in (log, tmp_path / 'ckpts')
                   for path in directory.rglob('*') if path.is_file()}
    monkeypatch.setattr(models, 'prepare_for_inference',
                        lambda *a: pytest.fail('Complete cached stage prepared pretrained models'))
    monkeypatch.setattr(models, 'prepare_models',
                        lambda *a: pytest.fail('Complete cached workflow prepared pretrained models'))
    for name in ('prepare', *stages):
        module = importlib.import_module('exps.' + name)
        if hasattr(module, 'launch_if_needed'):
            monkeypatch.setattr(module, 'launch_if_needed',
                                lambda *a: pytest.fail('Complete cached stage discovered GPUs'))
        module.main(argv + ['--no-resume'])
    assert all(path.stat().st_mtime_ns == timestamp for path, timestamp in saved_times.items())
