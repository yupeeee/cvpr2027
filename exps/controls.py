"""Coordinate/interface controls; every UNet input remains in native coordinates."""
import torch

from utils.cache import checkpoint_valid, skip_completed, stage_complete, training_statistics_valid
from utils.config import dry_run, parse_args
from utils.controls import clock_label, conjugate_editor, coordinate, wrapped_step
from utils.data import Run, base_row, request_for
from utils.distributed import Distributed, launch_if_needed
from utils.experiments import Runtime, select_samples, trial_key
from utils.metrics import Cost
from utils.probes import FIELDS, head_names, readout_rows, train_probes
from utils.progress import progress


def reading_keys(args, samples):
    return [f"{variant}/{sample['sample_id']}/{stage}/raw/{head}/{task}"
            for variant in ('native', 'wrapped', 'transported')
            for sample in samples if sample['split'] in ('validation', 'test')
            for stage in args.stages for head in head_names(args) for task in FIELDS]


def endpoint_jobs(args, samples):
    """Stable trial keys with declared work estimates, including both continuations."""
    jobs = {}
    for sample in samples:
        jobs[f"{sample['sample_id']}:unmodified"] = args.num_steps + 1
        for task in args.tasks:
            request = request_for(sample['sample_id'], task, args.request_seed)
            for stage in args.control_stages:
                for budget in args.edit_budgets:
                    for editor in args.control_editors:
                        key = trial_key(sample, task, request, stage, budget, editor, 'coordinate')
                        iterations = 0 if editor == 'noop' or budget == 0 else args.edit_steps
                        jobs[key] = 2 * (args.num_steps - stage + 1 + iterations)
    return jobs


class ControlsWork:
    """One fixed work budget for the whole command, including all future phases.

    Units approximate scheduled operations: epochs/head, readout fields, paired
    continuation steps and editing iterations. Observed throughput gives an ETA
    for *all* remaining phases; heterogeneous CPU/GPU work makes it an estimate.
    Cached units enter as initial progress, never as freshly measured throughput.
    """
    def __init__(self, args, dist, run):
        reading_done, endpoint_done = run.completed('controls_reading'), run.completed('controls')
        self.weights = {'probe/setup': 1, 'endpoint/setup': 1}
        self.done = set()
        for stage in args.stages:
            for head in head_names(args):
                key = f'probe/{stage}/{head}'
                self.weights[key] = args.probe_epochs
                if checkpoint_valid(args, stage, 'raw', head, 'wrapped', run=run):
                    self.done.add(key)
        if (all(key in self.done for key in self.weights if key.startswith('probe/') and key != 'probe/setup')
                and training_statistics_valid(args, run)):
            self.done.add('probe/setup')
        for key in reading_keys(args, dist.shard(run.samples)):
            self.weights['reading/' + key] = 1
            if key in reading_done:
                self.done.add('reading/' + key)
        samples = select_samples(run.samples, args.eval_split, args.control_limit)
        for key, weight in endpoint_jobs(args, dist.shard(samples)).items():
            self.weights['endpoint/' + key] = weight
            if key in endpoint_done:
                self.done.add('endpoint/' + key)
        if set(endpoint_jobs(args, samples)) <= endpoint_done:
            self.done.add('endpoint/setup')

    @property
    def total(self):
        return sum(self.weights.values())

    @property
    def initial(self):
        return sum(self.weights[key] for key in self.done)

    def complete(self, bar, key):
        if key not in self.done:
            bar.update(self.weights[key])
            self.done.add(key)


def reading_controls(args, dist, run, complete=lambda key: None):
    train_probes(args, dist, run, variant='wrapped', evaluate=False,
                 on_progress=lambda stage, access, head, cached: complete(f'probe/{stage}/{head}'))
    complete('probe/setup')
    done = run.completed('controls_reading')
    for variant in ('native', 'wrapped', 'transported'):
        for row in readout_rows(args, dist, run, variant=variant, raw_only=True, completed=done):
            if row['access'] != 'raw':
                continue
            row.update(clock_native=row['stage'] / args.num_steps,
                       clock_relabeled=clock_label(row['stage'], args.num_steps, args.clock_power),
                       control_kind='coordinate/interface control')
            if row['key'] not in done:
                run.write_row('controls_reading', row)
                complete('reading/' + row['key'])
    expected = reading_keys(args, run.samples)
    run.finish('controls_reading', expected)


def wrapped_continue(runtime, w, start, conditioning, cost, args):
    with torch.no_grad():
        w = w.detach().clone()
        for k in range(start, args.num_steps):
            w, _ = wrapped_step(runtime.sampler, w, k, conditioning, args.coordinate_amplitude, cost)
        return w


def drift(a, b, args):
    difference = a.float() - b.float()
    finite = bool(torch.isfinite(difference).all())
    return dict(endpoint_drift_rms=float(difference.square().mean().sqrt()) if finite else None,
                endpoint_drift_max=float(difference.abs().max()) if finite else None,
                endpoint_equivalent=bool(torch.allclose(a, b, atol=args.endpoint_atol, rtol=args.endpoint_rtol)) if finite else False,
                status='ok' if finite else 'malformed', failure_reason=None if finite else 'Nonfinite endpoint')


def endpoint_controls(args, dist, run, complete=lambda key: None):
    samples = select_samples(run.samples, args.eval_split, args.control_limit)
    done = run.completed('controls')
    def key_for(s, t, k, b, e):
        return trial_key(s, t, request_for(s['sample_id'], t, args.request_seed), k, b, e, 'coordinate')
    def keys_for(sample):
        return [f"{sample['sample_id']}:unmodified"] + [
            key_for(sample, task, stage, budget, editor)
            for task in args.tasks for stage in args.control_stages
            for budget in args.edit_budgets for editor in args.control_editors]
    expected = [key for sample in samples for key in keys_for(sample)]
    local_samples = dist.shard(samples)
    local_keys = {key for sample in local_samples for key in keys_for(sample)}
    with progress(args, desc='Controls: compare native and wrapped endpoints', total=len(local_keys),
                  initial=len(local_keys & done), unit='trial', dist=dist) as bar:
        if set(expected) <= done:
            return run.finish('controls', expected)
        runtime = Runtime(args, dist, run)
        complete('endpoint/setup')
        for sample in local_samples:
            if set(keys_for(sample)) <= done:
                continue
            trajectory = run.load_trajectory(sample['sample_id'])
            preparation = Cost()
            conditioning = runtime.sampler.encode(sample['prompt'], args.negative_prompt, preparation)
            runtime.sampler.validate_metadata(trajectory['metadata'], conditioning)
            key = f"{sample['sample_id']}:unmodified"
            if key not in done:
                bar.set_postfix(sample=sample['sample_id'], control='unmodified endpoints')
                cost = Cost()
                with cost.measure(dist.device), torch.no_grad():
                    x0 = trajectory['states'][0].to(device=dist.device, dtype=runtime.sampler.dtype)
                    terminal = wrapped_continue(runtime, coordinate(x0, 0, args.num_steps, args.coordinate_amplitude), 0, conditioning, cost, args)
                    image = runtime.sampler.decode(terminal, cost)
                row = base_row(sample, stage=args.num_steps, key=key, command='controls', coordinate='wrapped',
                               control_kind='coordinate/interface control', costs={'preparation':preparation.to_dict(),'wrapped_continuation':cost.to_dict(),'reference_collection':trajectory.get('costs',{})})
                row.update(drift(image, trajectory['final_image'].to(image), args))
                run.write_row('controls', row)
                bar.update()
                complete('endpoint/' + key)
            for task in args.tasks:
                request = runtime.request(sample, task)
                for stage in args.control_stages:
                    if all(key_for(sample, task, stage, budget, editor) in done
                           for budget in args.edit_budgets for editor in args.control_editors):
                        continue
                    x = trajectory['states'][stage].to(device=dist.device, dtype=runtime.sampler.dtype)
                    w = coordinate(x, stage, args.num_steps, args.coordinate_amplitude)
                    for budget in args.edit_budgets:
                        for editor in args.control_editors:
                            key = key_for(sample, task, stage, budget, editor)
                            if key in done:
                                continue
                            bar.set_postfix(sample=sample['sample_id'], task=task, stage=stage,
                                            editor=editor, budget=budget)
                            nc, wc = Cost(), Cost()
                            with nc.measure(dist.device):
                                native, ni = runtime.intervention(x.clone(), stage, conditioning, sample, task, request, editor, budget, nc)
                                terminal = runtime.sampler.continue_from(native, stage, conditioning, nc)
                                with torch.no_grad():
                                    native_image = runtime.sampler.decode(terminal, nc)
                            with wc.measure(dist.device):
                                wrapped, wi = conjugate_editor(w.clone(), stage, args.num_steps, args.coordinate_amplitude,
                                    lambda z: runtime.intervention(z, stage, conditioning, sample, task, request, editor, budget, wc), cost=wc)
                                terminal = wrapped_continue(runtime, wrapped, stage, conditioning, wc, args)
                                with torch.no_grad():
                                    wrapped_image = runtime.sampler.decode(terminal, wc)
                            nr = runtime.evaluate(sample, task, request, native_image, ni['budget_respected'], None, nc, trajectory)
                            wr = runtime.evaluate(sample, task, request, wrapped_image, wi['budget_respected'], None, wc, trajectory)
                            row = base_row(sample, stage=stage, task=task, request=request, key=key, command='controls',
                                           native_timestep=runtime.timestep(stage), controller=editor, budget=budget,
                                           coordinate='transported', control_kind='finite conjugated editor',
                                           native=nr, transported=wr, native_edit=ni, transported_edit=wi)
                            row.update(drift(native_image, wrapped_image, args))
                            failures = [record['failure_reason'] for record in (nr, wr, ni, wi) if record.get('failure_reason')]
                            if failures:
                                row.update(status='failed', failure_reason=';'.join(dict.fromkeys(failures)))
                            run.write_row('controls', row)
                            bar.update()
                            complete('endpoint/' + key)
    run.finish('controls', expected)


def main(argv=None):
    args = parse_args('controls', argv)
    if dry_run(args, 'controls'):
        return
    if skip_completed(args, 'controls'):
        return
    from utils.models import prepare_for_inference
    if not stage_complete(args, 'controls_endpoints'):
        prepare_for_inference(args)
    if launch_if_needed(args, 'controls', argv):
        return
    with Distributed(args) as dist:
        run = Run(args, dist, 'controls')
        work = ControlsWork(args, dist, run)
        with progress(args, desc='Controls: whole command', total=work.total,
                      initial=work.initial, unit='work', dist=dist, overall=True) as bar:
            complete = lambda key: work.complete(bar, key)
            reading_controls(args, dist, run, complete)
            endpoint_controls(args, dist, run, complete)


if __name__ == '__main__':
    main()
