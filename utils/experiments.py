"""Shared finite-edit experiments and an online loop with a strict cache boundary."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import torch

from utils.data import base_row, request_for, register_components
from utils.editors import edit
from utils.metrics import Cost, trial_metrics
from utils.progress import progress
from utils.models import load_frozen_models
from utils.policies import PolicySpec, decide, inspection_stages
from utils.probes import load_probe, load_stage_scales, probe_path, validate_probe
from utils.sampler import Sampler
from utils.tasks import Scorer, load_tasks, threshold


def scalar_scores(scores: dict) -> dict[str, float | None]:
    result = {}
    for name, value in scores.items():
        number = None if value is None else (float(value.detach().float().mean())
                                            if torch.is_tensor(value) else float(value))
        result[name] = number if number is not None and math.isfinite(number) else None
    return result


def artifact_name(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def select_samples(samples: list[dict], split: str, limit: int) -> list[dict]:
    selected = [sample for sample in samples if sample['split'] == split]
    return selected[:limit] if limit > 0 else selected


def trial_key(sample: dict, task: str, request: int, stage: int, budget: float,
              controller: str, extra: str = '') -> str:
    return f"{sample['sample_id']}:{task}:{request}:{stage}:{budget!r}:{controller}:{extra}"


class Runtime:
    """One frozen replica per rank; fitted heads are loaded lazily by stage."""

    def __init__(self, args: Any, dist: Any, run: Any):
        self.args, self.dist, self.run = args, dist, run
        self.sampler, self.scorers = load_frozen_models(args, dist)
        self.component_fingerprint = register_components(run, self.sampler, self.scorers)
        self.task_config = load_tasks(args.tasks_path)
        self.scales = load_stage_scales(run)
        self.probes: dict[tuple[int, str], Any] = {}
        # Cache order must not charge text preparation only to the first policy.
        # This setup sees declared task texts/object names, never sample images.
        objects = sorted({prompt['object'] for prompt in run.prompts})
        preparation = Cost()
        heads = sorted({args.editor_head, args.policy_head})
        with preparation.measure(dist.device), progress(
                args, total=len(args.stages) * len(heads) + len(self.scorers) * len(objects),
                desc='Prepare probe heads and evaluator text', unit='item', dist=dist) as bar:
            for stage in args.stages:
                for head in heads:
                    bar.set_postfix(stage=stage, head=head)
                    loaded = self.probe(stage, head)
                    preparation.add("probe_loads")
                    preparation.add("probe_parameters_loaded", loaded.metadata["parameter_count"])
                    bar.update()
            for role, scorer in self.scorers.items():
                for object_name in objects:
                    bar.set_postfix(evaluator=role, object=object_name)
                    scorer.prepare_text(object_name, preparation)
                    bar.update()
        self.setup_record = run.write_json(
            f'setup/{run.command}.rank{dist.rank:05d}.{time.time_ns()}.json',
            {'command': run.command, 'rank': dist.rank, 'world_size': dist.world_size,
             'phase': 'shared_text_and_head_preparation', 'objects': objects,
             'costs': preparation.to_dict(),
             'accounting': 'Once per replica/runtime; shared by trials, not repeated in each online budget'})

    def probe(self, stage: int, head: str | None = None):
        head = self.args.policy_head if head is None else head
        key = (stage, head)
        if key not in self.probes:
            path = probe_path(self.run, stage, 'raw', head)
            self.probes[key] = validate_probe(load_probe(path, self.dist.device), self.run)
        return self.probes[key]

    def timestep(self, stage: int) -> int | None:
        return int(self.sampler.timesteps[stage]) if stage < len(self.sampler.timesteps) else None

    def request(self, sample: dict, task: str) -> int:
        return request_for(sample['sample_id'], task, self.args.request_seed)

    def intervention(self, x: torch.Tensor, stage: int, conditioning: Any, sample: dict,
                     task: str, request: int, controller: str, budget: float,
                     cost: Cost) -> tuple[torch.Tensor, dict]:
        probe = self.probe(stage, self.args.editor_head).bind(sample['prompt_index'], cost) if controller == 'probe' or (
            controller == 'random' and self.args.random_match_editor == 'probe') else None
        return edit(x, stage, conditioning, request, task, controller, budget,
                    self.scales[stage], self.sampler, self.scorers['a'], probe,
                    self.args, cost, sample['seed'], sample['object'], self.task_config)

    def evaluate(self, sample: dict, task: str, request: int, image: torch.Tensor,
                 budget_respected: bool, assessment: bool | None,
                 online_cost: Cost, reference: dict | None = None) -> dict:
        """Post-hoc boundary: callers invoke this only after online decisions finish."""
        reference = reference if reference is not None else self.run.load_trajectory(sample['sample_id'])
        evaluation_cost = Cost()
        image_valid = image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 3 and bool(torch.isfinite(image).all())
        with evaluation_cost.measure(self.dist.device), torch.no_grad():
            scores = {role: (scalar_scores(scorer.score(image, sample['object'], evaluation_cost)) if image_valid else
                             {field: None for field in ('appearance', 'composition', 'category')})
                      for role, scorer in self.scorers.items()}
        reference_scores = {role: scalar_scores(reference['scores'][role]) for role in ('a', 'b')}
        reference_valid = reference.get('status', 'ok') == 'ok' and bool(torch.isfinite(reference['final_image']).all())
        result = {role: trial_metrics(reference_scores[role] if reference_valid else
                                     {field: None for field in reference_scores[role]}, scores[role],
                                     task, request, self.task_config, role,
                                     budget_respected, assessment)
                  for role in ('a', 'b')}
        before = reference['final_image'].to(image.device).float()
        result.update({
            'scores': scores,
            'reference_scores': reference_scores,
            'image_drift_rms': float((image.float() - before).square().mean().sqrt()) if image_valid and reference_valid and image.shape == before.shape else None,
            'budget_respected': budget_respected,
            'assessment': assessment,
            'costs': {'online': online_cost.to_dict(), 'offline_evaluation': evaluation_cost.to_dict(),
                      'reference_collection': reference.get('costs', {})},
            'reference_cache_reused': True,
            'setup_record': self.setup_record,
        })
        if not image_valid or any(value is None for group in scores.values() for value in group.values()):
            result.update(status='malformed', failure_reason='nonfinite_or_malformed_image_or_scores')
        elif not reference_valid or any(value is None for group in result['reference_scores'].values() for value in group.values()):
            result.update(status='malformed', failure_reason='invalid_reference_scores')
        return result

    def save_pair(self, command: str, key: str, reference: dict, image: torch.Tensor) -> str:
        return self.run.write_tensor(f'{command}_images/{artifact_name(key)}.pt',
                                     {'before': reference['final_image'].detach().cpu(),
                                      'after': image.detach().cpu()})

    def online(self, sample: dict, task: str, request: int, budget: float,
               spec: PolicySpec) -> dict:
        """Generate and inspect the current state only. No trajectory/cache access.

        The endpoint policy obtains the completed image as its legitimate online
        observation. All other observed scores come from present-state probes.
        The final act/no-act assessment is retained; deferrals are explicitly null.
        """
        cost = Cost()
        inspections: list[dict] = []
        edit_info: dict = {'relative_rms': 0.0, 'delta_rms': 0.0, 'delta_max': 0.0, 'budget_respected': True}
        acted = False
        decided = False
        assessment = False
        action_stage = None
        final_decision_stage = None
        failure_reason = None
        image = None
        stages = inspection_stages(spec, self.args.num_steps)
        decision = threshold(self.task_config, task, 'a', 'decision')
        margin = threshold(self.task_config, task, 'a', 'margin')
        field_index = {'appearance': 0, 'composition': 1}[task]
        with cost.measure(self.dist.device):
            with torch.no_grad():
                conditioning = self.sampler.encode(sample['prompt'], self.args.negative_prompt, cost)
                x = self.sampler.initial(sample['seed'], self.args.height, self.args.width)
            for stage in range(self.args.num_steps + 1):
                if not decided and stage in stages:
                    value = None
                    interface = 'time_only'
                    with torch.no_grad():
                        if spec.name == 'endpoint':
                            image = self.sampler.decode(x, cost)
                            value = float(self.scorers['a'].score(image, sample['object'], cost)[task][0])
                            interface = 'completed_image'
                        elif spec.name not in {'never', 'always', 'time_only'}:
                            value = float(self.probe(stage).predict(x, sample['prompt_index'], cost)[0, field_index])
                            interface = 'raw_probe'
                    if value is not None and not math.isfinite(value):
                        value, violation, final, act = None, None, True, False
                        failure_reason = 'nonfinite_online_observation'
                    else:
                        violation = None if value is None else margin - request * (value - decision)
                        final, act = decide(spec, stage, violation)
                    inspections.append({'stage': stage, 'native_timestep': self.timestep(stage),
                                        'interface': interface, 'predicted_score': value,
                                        'violation_margin': violation, 'threshold': spec.threshold,
                                        'decision': ('act' if act else 'no_act') if final else 'defer',
                                        'assessment': bool(act) if final else None,
                                        'status': 'malformed' if failure_reason else 'ok',
                                        'costs_so_far': cost.to_dict()})
                    if final:
                        decided, assessment, final_decision_stage = True, bool(act), stage
                        if act:
                            controller = 'preview' if spec.name == 'endpoint' else self.args.policy_editor
                            x, edit_info = self.intervention(x, stage, conditioning, sample, task,
                                                            request, controller, budget, cost)
                            acted, action_stage, image = True, stage, None
                if stage < self.args.num_steps:
                    with torch.no_grad():
                        x, _ = self.sampler.step(x, stage, conditioning, cost)
            if not decided:
                raise RuntimeError('Policy reached the endpoint without a final decision')
            with torch.no_grad():
                if image is None:
                    image = self.sampler.decode(x, cost)
        return {'image': image, 'cost': cost, 'edit': edit_info, 'assessment': assessment,
                'acted': acted, 'action_stage': action_stage, 'decision_stage': final_decision_stage,
                'inspections': inspections, 'failure_reason': failure_reason}


def write_inspections(run: Any, command: str, row: dict, inspections: list[dict]) -> None:
    """Inspection records stay attached to their resumable trial, including deferrals."""
    row['inspections'] = inspections
    run.write_row(command, row)
