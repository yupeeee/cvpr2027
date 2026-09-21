"""Explicit opt-ins only; these checks never download weights."""
import copy
import os
from pathlib import Path

import pytest
import torch

from test_sampler import tiny_sampler  # noqa: F401
from utils.config import parse_args
from utils.metrics import Cost
from utils.sampler import Sampler
from utils.tasks import Scorer


@pytest.mark.skipif(os.environ.get('PILOT_CHECK_CUDA') != '1', reason='CUDA check is opt-in: PILOT_CHECK_CUDA=1; not requested')
def test_tiny_cuda_schedule_and_gradient(tiny_sampler):
    if not torch.cuda.is_available():
        pytest.skip('CUDA opt-in requested but CUDA hardware unavailable')
    sampler = Sampler(copy.deepcopy(tiny_sampler.pipe), num_steps=3, guidance_scale=2,
                      device=torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0'))), dtype=torch.float32,
                      model_id='offline-tiny-cuda', model_revision=None)
    condition = sampler.encode('object', '', Cost())
    x = sampler.initial(7, 8, 8)
    direct = sampler.continue_from(x, 0, condition, Cost())
    first, _ = sampler.step(x, 0, condition, Cost())
    replay = sampler.continue_from(first, 1, condition, Cost())
    torch.testing.assert_close(direct, replay)
    delta = torch.zeros_like(x, requires_grad=True)
    sampler.decode(sampler.preview(x + delta, 1, condition, Cost()), Cost()).mean().backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


@pytest.mark.skipif(os.environ.get('PILOT_CHECK_PRETRAINED') != '1', reason='Real-pretrained integration is opt-in and requires existing local weights; not requested')
def test_local_pretrained_suffix_and_evaluators():
    names = ['PILOT_MODEL_DIR', 'PILOT_CLIP_DIR', 'PILOT_SIGLIP_DIR']
    paths = [os.environ.get(n) for n in names]
    if not all(p and Path(p).is_dir() for p in paths):
        pytest.skip('Set PILOT_MODEL_DIR, PILOT_CLIP_DIR, PILOT_SIGLIP_DIR to existing local weight directories')
    device = os.environ.get('PILOT_CHECK_DEVICE', 'cpu')
    args = parse_args('offline-local-integration', [
        '--device', device, '--model-id', paths[0], '--evaluator-a-id', paths[1], '--evaluator-b-id', paths[2],
        '--num-steps', '2', '--stages', '0', '1', '2', '--height', '64', '--width', '64', '--local-files-only'])
    sampler = Sampler.load(args, torch.device(device))
    cond = sampler.encode('a mug on a table', '', Cost())
    x = sampler.initial(17, args.height, args.width)
    reference = sampler.continue_from(x, 0, cond, Cost())
    first, _ = sampler.step(x, 0, cond, Cost())
    torch.testing.assert_close(reference, sampler.continue_from(first, 1, cond, Cost()), atol=args.endpoint_atol, rtol=args.endpoint_rtol)
    for role in ('a','b'):
        scorer = Scorer.load(args, role, torch.device(device))
        with torch.no_grad():
            scores = scorer.score(sampler.decode(reference, Cost()), 'mug', Cost())
        assert all(torch.isfinite(v).all() for v in scores.values())
        if role == 'a':
            delta = torch.zeros_like(reference, requires_grad=True)
            scorer.score(sampler.decode(reference + delta, Cost()), 'mug', Cost())['appearance'].sum().backward()
            assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0
