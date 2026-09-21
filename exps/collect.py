"""Actual DDIM trajectories; clean predictions reuse the unedited update output."""
import math

import torch

from utils.config import dry_run, parse_args
from utils.data import Run, base_row, register_components
from utils.distributed import Distributed, launch_if_needed
from utils.metrics import Cost, add_costs
from utils.models import load_frozen_models
from utils.progress import progress, status
from utils.sampler import Sampler
from utils.tasks import Scorer


def accumulate(target, source):
    target.counters = add_costs(target.to_dict(), source)


def collect(args, dist, run):
    done = run.completed('collect')
    remaining = [s for s in run.samples if s['sample_id'] not in done]
    status(args, f'collect: {len(run.samples) - len(remaining)} cached samples; {len(remaining)} remaining', dist)
    if not remaining:
        with progress(args, total=len(dist.shard(run.samples)), initial=len(dist.shard(run.samples)),
                      desc='collect: samples', unit='sample', dist=dist):
            pass
        return run.finish('collect', [s['sample_id'] for s in run.samples])
    sampler, scorers = load_frozen_models(args, dist)
    component_fingerprint = register_components(run, sampler, scorers)
    local_samples = dist.shard(run.samples)
    initial = sum(sample['sample_id'] in done for sample in local_samples)
    with progress(args, total=len(local_samples), initial=initial, desc='collect: samples',
                  unit='sample', dist=dist) as samples_bar:
        for sample in local_samples:
            if sample['sample_id'] in done:
                continue
            samples_bar.set_postfix(sample=sample['sample_id'], operation='denoising')
            generation, observation, evaluation = Cost(), Cost(), Cost()
            states, clean, previews, stage_costs = {}, {}, {}, {}
            with generation.measure(dist.device), torch.no_grad():
                conditioning = sampler.encode(sample['prompt'], args.negative_prompt, generation)
                x = sampler.initial(sample['seed'], args.height, args.width)
            metadata = sampler.state_metadata(conditioning)
            with progress(args, range(args.num_steps + 1), desc='DDIM generation and terminal decode',
                          unit='stage', dist=dist, position=1, leave=False) as steps:
                for stage in steps:
                    if stage in args.stages:
                        states[stage] = x.detach().cpu().clone()
                        generation_before = generation.to_dict()
                        prediction_cost, decode_cost = Cost(), Cost()
                        with prediction_cost.measure(dist.device), torch.no_grad():
                            if stage < args.num_steps:
                                next_x, predicted = sampler.step(x, stage, conditioning, prediction_cost)
                                prediction_cost.add('preview_unet_calls')  # Subset of existing calls, not an extra call.
                            else:
                                predicted = sampler.preview(x, stage, conditioning, prediction_cost)
                        with decode_cost.measure(dist.device), torch.no_grad():
                            preview_image = sampler.decode(predicted, decode_cost)
                        clean[stage] = predicted.detach().cpu().clone()
                        previews[stage] = preview_image.detach().cpu().clone()
                        stage_costs[stage] = {'generation': generation_before,
                            'preview': add_costs(prediction_cost.to_dict(), decode_cost.to_dict()),
                            'prediction': prediction_cost.to_dict(), 'preview_decode': decode_cost.to_dict(),
                            'prediction_reused_for_update': stage < args.num_steps}
                        if stage < args.num_steps:
                            # Only the extra decoded preview is collection overhead.
                            accumulate(observation, decode_cost.to_dict())
                            accumulate(generation, prediction_cost.to_dict())
                            x = next_x
                        else:
                            final_image = preview_image
                            accumulate(generation, decode_cost.to_dict())
                    elif stage < args.num_steps:
                        with generation.measure(dist.device), torch.no_grad():
                            x, _ = sampler.step(x, stage, conditioning, generation)
            samples_bar.set_postfix(sample=sample['sample_id'], operation='endpoint scoring and saving')
            # Stage N is required by validation; its preview is the actual endpoint.
            valid = all(bool(torch.isfinite(v).all()) for group in (states, clean, previews) for v in group.values())
            scores = {role: {field: None for field in ('appearance', 'composition', 'category')} for role in scorers}
            if valid:
                with evaluation.measure(dist.device), torch.no_grad():
                    for role, scorer in scorers.items():
                        for field, value in scorer.score(final_image, sample['object'], evaluation).items():
                            scalar = float(value[0])
                            scores[role][field] = scalar if math.isfinite(scalar) else None
                            valid = valid and math.isfinite(scalar)
            costs = {'generation': generation.to_dict(), 'observation': observation.to_dict(), 'evaluation': evaluation.to_dict()}
            trajectory = {'fingerprint': run.fingerprint, 'component_fingerprint': component_fingerprint, 'sample': sample, 'states': states,
                          'clean': clean, 'previews': previews, 'final_image': final_image.detach().cpu(),
                          'scores': scores, 'metadata': metadata, 'stage_costs': stage_costs, 'costs': costs,
                          'stage_indices': args.stages, 'status': 'ok' if valid else 'malformed'}
            path = run.write_tensor(f"trajectories/{sample['sample_id']}.pt", trajectory)
            row = base_row(sample, stage=args.num_steps, observation='endpoint', key=sample['sample_id'], command='collect',
                           trajectory_path=path, scores=scores, costs=costs, stages=args.stages,
                           status=trajectory['status'], failure_reason=None if valid else 'Nonfinite generated state, prediction, image, or evaluator score')
            run.write_row('collect', row)
            samples_bar.update()
    run.finish('collect', [s['sample_id'] for s in run.samples])


def main(argv=None):
    args = parse_args('collect', argv)
    if dry_run(args, 'collect'):
        return
    from utils.cache import skip_completed
    if skip_completed(args, 'collect'):
        return
    from utils.models import prepare_for_inference
    prepare_for_inference(args)
    if launch_if_needed(args, 'collect', argv):
        return
    with Distributed(args) as dist:
        collect(args, dist, Run(args, dist, 'collect'))


if __name__ == '__main__':
    main()
