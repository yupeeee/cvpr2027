"""Apply each bounded controller and evaluate the actual remaining sampler."""
from __future__ import annotations

import itertools

import torch

from utils.config import dry_run, parse_args
from utils.data import Run, base_row, request_for
from utils.distributed import Distributed, launch_if_needed
from utils.experiments import Runtime, select_samples, trial_key
from utils.metrics import Cost
from utils.progress import progress


def main(argv=None):
    args = parse_args('intervene', argv)
    if dry_run(args, 'intervene'):
        return
    from utils.cache import skip_completed
    if skip_completed(args, 'intervene'):
        return
    from utils.models import prepare_for_inference
    prepare_for_inference(args)
    if launch_if_needed(args, 'intervene', argv):
        return
    with Distributed(args) as dist:
        run = Run(args, dist, 'intervene')
        samples = select_samples(run.samples, args.eval_split, args.intervention_limit)
        completed = run.completed('intervene')
        def keys_for(sample):
            return [trial_key(sample, task, request_for(sample['sample_id'], task, args.request_seed), stage, budget, controller)
                    for task, stage, budget, controller in itertools.product(
                        args.tasks, args.intervention_stages, args.edit_budgets, args.editors)]
        expected = [key for sample in samples for key in keys_for(sample)]
        if set(expected) <= completed:
            run.finish('intervene', expected)
            return
        runtime = Runtime(args, dist, run)
        local_samples = dist.shard(samples)
        local_keys = {key for sample in local_samples for key in keys_for(sample)}
        with progress(args, desc='Intervene: edit and evaluate', total=len(local_keys),
                      initial=len(local_keys & completed), unit='trial', dist=dist) as bar:
            for sample in local_samples:
                if set(keys_for(sample)) <= completed:
                    continue
                trajectory = run.load_trajectory(sample['sample_id'])
                for task, stage, budget, controller in itertools.product(
                        args.tasks, args.intervention_stages, args.edit_budgets, args.editors):
                    request = request_for(sample['sample_id'], task, args.request_seed)
                    key = trial_key(sample, task, request, stage, budget, controller)
                    if key in completed:
                        continue
                    bar.set_postfix(sample=sample['sample_id'], task=task, stage=stage,
                                    editor=controller, budget=budget)
                    cost = Cost()
                    with cost.measure(dist.device):
                        with torch.no_grad():
                            conditioning = runtime.sampler.encode(sample['prompt'], args.negative_prompt, cost)
                            runtime.sampler.validate_metadata(trajectory['metadata'], conditioning)
                            x = trajectory['states'][stage].to(dist.device).clone()
                        edited, info = runtime.intervention(x, stage, conditioning, sample, task, request,
                                                           controller, budget, cost)
                        with torch.no_grad():
                            endpoint = runtime.sampler.continue_from(edited, stage, conditioning, cost)
                            image = runtime.sampler.decode(endpoint, cost)
                    row = base_row(sample, stage=stage, task=task, request=request,
                                   key=key, native_timestep=runtime.timestep(stage), controller=controller,
                                   observation='raw' if controller == 'probe' else 'predicted_clean' if controller == 'preview' else None,
                                   coordinate='native', budget=budget)
                    row.update(runtime.evaluate(sample, task, request, image, info['budget_respected'],
                                                None, cost, trajectory))
                    row['edit'] = info
                    if info.get('status') == 'failed':
                        row.update(status='failed', failure_reason=info['failure_reason'])
                    row['image_path'] = runtime.save_pair('intervene', key, trajectory, image)
                    row['prefix_replayed_from_cache'] = True
                    row['prefix_generation_cost'] = trajectory.get('stage_costs', {}).get(stage)
                    run.write_row('intervene', row)
                    bar.update()
        run.finish('intervene', expected)


if __name__ == '__main__':
    main()
