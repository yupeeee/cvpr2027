"""Validation-calibrated inspect-decide-act experiments, followed by frozen test use."""
from __future__ import annotations

import itertools
import json

from utils.config import dry_run, parse_args
from utils.data import Run, base_row, request_for, atomic_json
from utils.distributed import Distributed, launch_if_needed, main_process
from utils.experiments import Runtime, select_samples, trial_key, write_inspections
from utils.policies import PolicySpec, select_validation_settings
from utils.progress import progress, status


def evaluate_policy(runtime, sample, task, request, budget, spec, key, command):
    # Do not move the cache read above online(): it is a deliberately visible
    # boundary between prospective decision making and retrospective assessment.
    output = runtime.online(sample, task, request, budget, spec)
    reference = runtime.run.load_trajectory(sample['sample_id'])
    row = base_row(sample, stage=output['decision_stage'], task=task, request=request,
                   key=key, native_timestep=runtime.timestep(output['decision_stage']),
                   observation='completed_image' if spec.name == 'endpoint' else 'raw',
                   controller='preview' if spec.name == 'endpoint' else runtime.args.policy_editor,
                   coordinate='native', budget=budget)
    row.update(runtime.evaluate(sample, task, request, output['image'], output['edit']['budget_respected'],
                                output['assessment'], output['cost'], reference))
    row.update({'policy': spec.name, 'fixed_stage': spec.stage, 'confidence_threshold': spec.threshold,
                'deadline': spec.deadline, 'fallback': spec.fallback,
                'action_stage': output['action_stage'], 'acted': output['acted'],
                'edit': output['edit'], 'inspections': output['inspections'],
                'image_path': runtime.save_pair(command, key, reference, output['image'])})
    if output['failure_reason']:
        row.update(status='malformed', failure_reason=output['failure_reason'])
        for role in ('a', 'b'):
            row[role].update(assessment_correct=False, joint_success=False)
    if output['edit'].get('status') == 'failed':
        row.update(status='failed', failure_reason=output['edit']['failure_reason'])
    return row


def calibration(runtime, samples):
    args, run, dist = runtime.args, runtime.run, runtime.dist
    completed = run.completed('policy_calibration')
    candidates = [('stage', stage) for stage in args.policy_stages]
    candidates += [('threshold', value) for value in args.policy_thresholds]
    def keys_for(sample):
        return [trial_key(sample, task, request_for(sample['sample_id'], task, args.request_seed), int(value) if kind == 'stage' else -1,
                          budget, 'calibration', f'{kind}:{value}')
                for task, budget, (kind, value) in itertools.product(args.tasks, args.edit_budgets, candidates)]
    expected = [key for sample in samples for key in keys_for(sample)]
    local_samples = dist.shard(samples)
    local_keys = {key for sample in local_samples for key in keys_for(sample)}
    with progress(args, desc='Policy: calibrate on validation trials', total=len(local_keys),
                  initial=len(local_keys & completed), unit='trial', dist=dist) as bar:
        for sample in local_samples:
            for task, budget, (kind, value) in itertools.product(args.tasks, args.edit_budgets, candidates):
                request = request_for(sample['sample_id'], task, args.request_seed)
                stage = int(value) if kind == 'stage' else args.policy_deadline
                key = trial_key(sample, task, request, stage if kind == 'stage' else -1,
                                budget, 'calibration', f'{kind}:{value}')
                if key in completed:
                    continue
                bar.set_postfix(sample=sample['sample_id'], task=task, candidate=f'{kind}:{value}',
                                budget=budget)
                spec = PolicySpec('always' if kind == 'stage' else 'adaptive', stage,
                                  tuple(args.policy_stages), 0.0 if kind == 'stage' else float(value),
                                  args.policy_deadline, args.policy_fallback)
                row = evaluate_policy(runtime, sample, task, request, budget, spec, key, 'policy_calibration')
                row.update({'calibration_kind': kind, 'calibration_value': value, 'cost_phase': 'training_calibration'})
                write_inspections(run, 'policy_calibration', row, row['inspections'])
                bar.update()
    run.finish('policy_calibration', expected)
    return freeze_settings(args, dist, run)


def freeze_settings(args, dist, run):
    status(args, 'Policy: select and freeze validation settings', dist=dist)
    settings_path = run.path / 'policy_settings.json'
    settings = select_validation_settings(run.rows('policy_calibration'), args.policy_stages,
                                          args.policy_thresholds)
    settings.update({'fingerprint': run.fingerprint, 'fallback': args.policy_fallback,
                     'deadline': args.policy_deadline, 'time_only_default': args.policy_time_only_default,
                     'calibration_cost_phase': 'training_calibration', 'test_settings_frozen': True})
    def save_settings():
        try:
            existing = json.loads(settings_path.read_text())
        except (OSError, ValueError):
            existing = None
        if existing != settings:
            # Missing or damaged settings are reconstructed solely from validation.
            atomic_json(settings_path, settings)
    # Every rank participates even if rank zero has already repaired the file.
    main_process(dist, save_settings)
    return settings



def main(argv=None):
    args = parse_args('policy', argv)
    if dry_run(args, 'policy'):
        return
    from utils.cache import skip_completed
    if skip_completed(args, 'policy'):
        return
    from utils.models import prepare_for_inference
    prepare_for_inference(args)
    if launch_if_needed(args, 'policy', argv):
        return
    with Distributed(args) as dist:
        run = Run(args, dist, 'policy')
        validation = select_samples(run.samples, 'validation', args.policy_calibration_limit)
        if not validation:
            raise ValueError('Policy calibration needs validation samples; increase num_samples or validation_fraction')
        if args.eval_split != 'test':
            raise ValueError('Policy evaluation uses held-out test seeds; calibration uses validation seeds')
        from utils.cache import expected_keys
        calibration_keys = expected_keys(args, 'policy_calibration', samples=validation)
        calibration_done = run.completed('policy_calibration')
        runtime = None
        if set(calibration_keys) <= calibration_done:
            run.finish('policy_calibration', calibration_keys)
            settings = freeze_settings(args, dist, run)
        else:
            runtime = Runtime(args, dist, run)
            settings = calibration(runtime, validation)
        samples = select_samples(run.samples, 'test', args.policy_limit)
        completed = run.completed('policy')
        configs = [(name, stage) for name in args.policy_variants
                   for stage in (args.policy_stages if name == 'single_stage' else
                                 [args.num_steps if name == 'endpoint' else settings['stage']])]
        def keys_for(sample):
            return [trial_key(sample, task, request_for(sample['sample_id'], task, args.request_seed), stage, budget, name)
                    for task, budget, (name, stage) in itertools.product(args.tasks, args.edit_budgets, configs)]
        expected = [key for sample in samples for key in keys_for(sample)]
        if set(expected) <= completed:
            run.finish('policy', expected)
            return
        if runtime is None:
            runtime = Runtime(args, dist, run)
        local_samples = dist.shard(samples)
        local_keys = {key for sample in local_samples for key in keys_for(sample)}
        with progress(args, desc='Policy: evaluate frozen policies on test trials', total=len(local_keys),
                      initial=len(local_keys & completed), unit='trial', dist=dist) as bar:
            for sample in local_samples:
                if set(keys_for(sample)) <= completed:
                    continue
                for task, budget, (name, stage) in itertools.product(args.tasks, args.edit_budgets, configs):
                    request = request_for(sample['sample_id'], task, args.request_seed)
                    key = trial_key(sample, task, request, stage, budget, name)
                    if key in completed:
                        continue
                    bar.set_postfix(sample=sample['sample_id'], task=task, policy=name,
                                    stage=stage, budget=budget)
                    group = f'{task}:{request}'
                    if group not in settings['time_only_actions']:
                        # An absent validation request uses the explicitly configured
                        # fixed prior action, frozen independently of this endpoint.
                        time_action = settings['time_only_default']
                    else:
                        time_action = settings['time_only_actions'][group]
                    spec = PolicySpec(name, stage, tuple(args.policy_stages), settings['threshold'],
                                      settings['deadline'], settings['fallback'], time_action)
                    row = evaluate_policy(runtime, sample, task, request, budget, spec, key, 'policy')
                    row['settings_file'] = 'policy_settings.json'
                    row['calibration_cost_phase'] = 'training_calibration'
                    write_inspections(run, 'policy', row, row['inspections'])
                    bar.update()
        run.finish('policy', expected)


if __name__ == '__main__':
    main()
