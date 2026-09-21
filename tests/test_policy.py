"""Small deterministic checks of decision semantics, without research generation."""
import pytest

from utils.policies import PolicySpec, decide, inspection_stages, select_validation_settings


def test_adaptive_deferrals_and_deadline():
    spec = PolicySpec('adaptive', 2, (0, 2, 4), 0.5, 4, 'predicted')
    assert inspection_stages(spec, 6) == (0, 2, 4)
    assert decide(spec, 0, 0.2) == (False, False)
    assert decide(spec, 2, 0.6) == (True, True)
    assert decide(spec, 4, 0.2) == (True, True)
    assert decide(spec, 4, -0.2) == (True, False)
    assert decide(PolicySpec('adaptive', 2, (0, 2, 4), 0.5, 4, 'skip'), 4, 0.2) == (True, False)
    assert decide(PolicySpec('adaptive', 2, (0, 2, 4), 0.5, 4, 'edit'), 4, -0.2) == (True, True)


def test_fixed_and_endpoint_assess_full_margin():
    for name in ('single_stage', 'fixed', 'endpoint'):
        spec = PolicySpec(name, 4, (0, 2, 4), 2.0, 4, 'predicted')
        assert decide(spec, 2, 1.0) == (False, False)
        assert decide(spec, 4, 0.1) == (True, True)
        assert decide(spec, 4, 0.0) == (True, False)
    assert inspection_stages(PolicySpec('endpoint', 6, (), 0.0, 4, 'skip'), 6) == (6,)


def test_validation_only_selection_and_ties():
    rows = []
    for kind, values in [('stage', [0, 2]), ('threshold', [0.0, 0.5])]:
        for value in values:
            for sid, need in [('v1', True), ('v2', False)]:
                rows.append({'sample_id': sid, 'split': 'validation', 'task': 'appearance', 'request': 1,
                             'calibration_kind': kind, 'calibration_value': value,
                             'a': {'joint_success': True, 'originally_unsatisfied': need},
                             'costs': {'online': {'unet_samples': 6 if value == 0 else 4}}})
    result = select_validation_settings(rows, [0, 2], [0.0, 0.5])
    assert result['stage'] == 2
    assert result['threshold'] == 0.5
    assert result['time_only_actions']['appearance:1'] is False
    assert result['validation_sample_ids'] == ['v1', 'v2']
    with pytest.raises(ValueError, match='validation-only'):
        select_validation_settings([dict(rows[0], split='test')], [0], [0.0])
    with pytest.raises(ValueError, match='Missing validation candidate'):
        select_validation_settings(rows, [0, 2, 4], [0.0, 0.5])


def test_online_has_no_endpoint_cache_or_evaluator_b_access(tiny_sampler):
    from types import SimpleNamespace
    import torch
    from utils.experiments import Runtime
    from utils.probes import Probe, make_head

    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f'Prospective access to forbidden endpoint/cache/B: {name}')

    class TensorScore:
        def score(self, image, object_name, cost):
            cost.add('scorer_a_images', image.shape[0])
            value = image.mean(dim=(1, 2, 3))
            return {'appearance': value, 'composition': value * 0, 'category': value * 0}

    task_config = {'tasks': {name: {'protected': [other, 'category'], 'thresholds': {
        role: {'decision': 0.0, 'margin': 1.0, 'protected_tolerance': 1.0} for role in ('a', 'b')}}
        for name, other in [('appearance', 'composition'), ('composition', 'appearance')]},
        'category': {'thresholds': {role: {'protected_tolerance': 1.0} for role in ('a', 'b')}}}
    runtime = Runtime.__new__(Runtime)
    runtime.sampler = tiny_sampler
    runtime.run = Forbidden()
    runtime.scorers = {'a': TensorScore(), 'b': Forbidden()}
    runtime.dist = SimpleNamespace(device=torch.device('cpu'))
    runtime.args = SimpleNamespace(num_steps=3, negative_prompt='', height=8, width=8,
                                   policy_editor='preview', policy_head='linear', editor_head='linear',
                                   random_match_editor='probe', edit_grid=2, edit_steps=1,
                                   edit_lr=0.25, edit_margin=None, edit_penalty=0.01,
                                   protected_weight=1.0, edit_eps=1e-8, budget_tolerance=1e-6)
    runtime.task_config = task_config
    runtime.scales = {k: 1.0 for k in range(4)}
    head = make_head(5, 'linear')
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    probe = Probe({'metadata': {'input_dim': 5, 'head': 'linear', 'prompt_count': 1, 'pool_grid': 1,
                                'parameter_count': sum(p.numel() for p in head.parameters())},
                   'state_dict': head.state_dict(), 'feature_mean': torch.zeros(5),
                   'feature_scale': torch.ones(5), 'label_mean': torch.zeros(3), 'label_scale': torch.ones(3)})
    runtime.probes = {(k, 'linear'): probe for k in range(4)}
    sample = {'seed': 17, 'prompt': 'object', 'object': 'object', 'prompt_index': 0}
    adaptive = runtime.online(sample, 'appearance', 1, 0.05,
                              PolicySpec('adaptive', 1, (0, 1, 3), 2.0, 3, 'skip'))
    assert [row['decision'] for row in adaptive['inspections']] == ['defer', 'defer', 'no_act']
    assert [row['assessment'] for row in adaptive['inspections']] == [None, None, False]
    assert adaptive['assessment'] is False and not adaptive['acted']
    assert adaptive['cost'].to_dict()['probe_calls'] == 3
    endpoint = runtime.online(sample, 'appearance', 1, 0.05,
                              PolicySpec('endpoint', 3, (0, 1, 3), 0.0, 3, 'skip'))
    assert endpoint['acted'] and endpoint['action_stage'] == 3
    assert endpoint['inspections'][0]['interface'] == 'completed_image'
    assert endpoint['cost'].to_dict()['unet_calls'] == 3
    assert endpoint['cost'].to_dict()['backward_passes'] == 1
    assert endpoint['edit']['budget_respected']


# Reuse actual locally instantiated Diffusers components; no hub calls.
from test_sampler import tiny_sampler  # noqa: E402,F401
