"""Finite edit compatibility across actual sampler segments; no Jacobians."""
from __future__ import annotations

import itertools
import math

import torch

from utils.config import dry_run, parse_args
from utils.data import Run, base_row, request_for
from utils.distributed import Distributed, launch_if_needed
from utils.experiments import Runtime, select_samples, trial_key
from utils.metrics import Cost
from utils.progress import progress
from utils.tasks import threshold


def main(argv=None):
    args = parse_args('diagnose', argv)
    if dry_run(args, 'diagnose'):
        return
    from utils.cache import skip_completed
    if skip_completed(args, 'diagnose'):
        return
    from utils.models import prepare_for_inference
    prepare_for_inference(args)
    if launch_if_needed(args, 'diagnose', argv):
        return
    with Distributed(args) as dist:
        run = Run(args, dist, 'diagnose')
        samples = select_samples(run.samples, args.eval_split, args.diagnostic_limit)
        pairs = list(zip(args.diagnostic_stages[:-1], args.diagnostic_stages[1:]))
        controllers = list(dict.fromkeys(['noop', *args.editors]))
        completed = run.completed('diagnose')
        def keys_for(sample):
            return [trial_key(sample, task, request_for(sample['sample_id'], task, args.request_seed), k, budget, controller, f'to{j}')
                    for task, (k, j), budget, controller in itertools.product(
                        args.tasks, pairs, args.edit_budgets, controllers)]
        expected = [key for sample in samples for key in keys_for(sample)]
        if set(expected) <= completed:
            run.finish('diagnose', expected)
            return
        runtime = Runtime(args, dist, run)
        local_samples = dist.shard(samples)
        local_keys = {key for sample in local_samples for key in keys_for(sample)}
        with progress(args, desc='Diagnose: compare sampler segments', total=len(local_keys),
                      initial=len(local_keys & completed), unit='trial', dist=dist) as bar:
            for sample in local_samples:
                if set(keys_for(sample)) <= completed:
                    continue
                trajectory = run.load_trajectory(sample['sample_id'])
                for task, (k, j), budget, controller in itertools.product(args.tasks, pairs, args.edit_budgets, controllers):
                    request = request_for(sample['sample_id'], task, args.request_seed)
                    key = trial_key(sample, task, request, k, budget, controller, f'to{j}')
                    if key in completed:
                        continue
                    bar.set_postfix(sample=sample['sample_id'], task=task, stages=f'{k}->{j}',
                                    editor=controller, budget=budget)
                    cost = Cost()
                    with cost.measure(dist.device):
                        with torch.no_grad():
                            conditioning = runtime.sampler.encode(sample['prompt'], args.negative_prompt, cost)
                            runtime.sampler.validate_metadata(trajectory['metadata'], conditioning)
                            x = trajectory['states'][k].to(dist.device).clone()
                        edit_k, info_k = runtime.intervention(x.clone(), k, conditioning, sample, task, request,
                                                             controller, budget, cost)
                        with torch.no_grad():
                            first_at_j = runtime.sampler.continue_from(edit_k, k, conditioning, cost, stop=j)
                            unedited_at_j = runtime.sampler.continue_from(x.clone(), k, conditioning, cost, stop=j)
                        second_at_j, info_j = runtime.intervention(unedited_at_j, j, conditioning, sample, task,
                                                                  request, controller, budget, cost)
                        residual = float((first_at_j.float() - second_at_j.float()).square().mean().sqrt())
                        with torch.no_grad():
                            first = runtime.sampler.decode(runtime.sampler.continue_from(
                                first_at_j, j, conditioning, cost), cost)
                            second = runtime.sampler.decode(runtime.sampler.continue_from(
                                second_at_j, j, conditioning, cost), cost)
                    first_result = runtime.evaluate(sample, task, request, first, info_k['budget_respected'],
                                                    None, cost, trajectory)
                    second_result = runtime.evaluate(sample, task, request, second, info_j['budget_respected'],
                                                     None, cost, trajectory)
                    disagreements = {}
                    for role in ('a', 'b'):
                        scores1, scores2 = first_result['scores'][role], second_result['scores'][role]
                        disagreements[role] = {
                            'score_absolute_difference': {field: abs(scores1[field] - scores2[field]) if scores1[field] is not None and scores2[field] is not None else None for field in scores1},
                            'target_decision_disagrees': (scores1[task] >= threshold(runtime.task_config, task, role, 'decision')) != (
                                scores2[task] >= threshold(runtime.task_config, task, role, 'decision')) if scores1[task] is not None and scores2[task] is not None else None,
                            'target_failure_first': not first_result[role]['target_success'],
                            'target_failure_second': not second_result[role]['target_success'],
                        }
                    row = base_row(sample, stage=k, task=task, request=request, key=key,
                                   native_timestep=runtime.timestep(k), controller=controller,
                                   observation='raw' if controller == 'probe' else 'predicted_clean' if controller == 'preview' else None,
                                   coordinate='native', budget=budget)
                    row.update({'later_stage': j, 'later_native_timestep': runtime.timestep(j),
                                'residual_kind': 'one_step' if j == k + 1 else 'segment_compatibility',
                                'residual_rms': residual if math.isfinite(residual) else None,
                                'normalized_residual': residual / max(runtime.scales[j], args.edit_eps) if math.isfinite(residual) else None,
                                'first': first_result, 'second': second_result,
                                'endpoint_disagreement': disagreements, 'edit_first': info_k, 'edit_second': info_j,
                                'image_path_first': runtime.save_pair('diagnose_first', key, trajectory, first),
                                'image_path_second': runtime.save_pair('diagnose_second', key, trajectory, second),
                                'costs': {'online_both_branches': cost.to_dict(),
                                          'offline_evaluation_first': first_result['costs']['offline_evaluation'],
                                          'offline_evaluation_second': second_result['costs']['offline_evaluation'],
                                          'reference_collection': trajectory.get('costs', {})}})
                    failures = [record['failure_reason'] for record in (first_result, second_result, info_k, info_j)
                                if record.get('failure_reason')]
                    if not math.isfinite(residual):
                        failures.append('nonfinite_segment_residual')
                    if failures:
                        row.update(status='failed', failure_reason=';'.join(dict.fromkeys(failures)))
                    run.write_row('diagnose', row)
                    bar.update()
        run.finish('diagnose', expected)


if __name__ == '__main__':
    main()
