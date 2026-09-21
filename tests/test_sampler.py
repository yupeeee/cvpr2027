"""Bounded offline checks using actual tiny Diffusers/Transformers components."""
import copy
import json
import os
import subprocess
import sys

import pytest
import torch

from utils.sampler import Sampler


class Counter(dict):
    def add(self, name, amount=1):
        self[name] = self.get(name, 0) + amount


@pytest.fixture(scope="module")
def tiny_sampler():
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    torch.manual_seed(123)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "object": 3},
                                 unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]",
        unk_token="[UNK]", model_max_length=8)
    text_encoder = transformers.CLIPTextModel(transformers.CLIPTextConfig(
        vocab_size=4, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=8, eos_token_id=1,
        bos_token_id=1, pad_token_id=0))
    unet = diffusers.UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",), block_out_channels=(8,),
        layers_per_block=1, cross_attention_dim=8, attention_head_dim=4,
        norm_num_groups=4)
    vae = diffusers.AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",), block_out_channels=(8,),
        latent_channels=4, norm_num_groups=4, sample_size=8, scaling_factor=0.7)
    scheduler = diffusers.DDIMScheduler(num_train_timesteps=12,
                                        clip_sample=False, steps_offset=1)
    pipe = diffusers.StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    return Sampler(pipe, num_steps=3, guidance_scale=2.0, device="cpu",
                   dtype=torch.float32, model_id="offline-random-tiny", model_revision=None)


def test_fixed_schedule_suffix_terminal_and_branch_isolation(tiny_sampler):
    sampler = tiny_sampler
    cost = Counter()
    condition = sampler.encode("object", "", cost)
    initial = sampler.initial(11, 8, 8)
    assert torch.equal(initial, sampler.initial(11, 8, 8))
    states = [initial]
    with torch.no_grad():
        for k in range(sampler.num_steps):
            state, _ = sampler.step(states[-1], k, condition, cost)
            states.append(state)
    schedule = sampler.timesteps.clone()
    for k in range(sampler.num_steps + 1):
        output = sampler.continue_from(states[k], k, condition, cost)
        torch.testing.assert_close(output, states[-1], rtol=0, atol=0)
        no_op = sampler.continue_from(states[k] + torch.zeros_like(states[k]), k, condition, cost)
        torch.testing.assert_close(no_op, output, rtol=0, atol=0)
    assert torch.equal(sampler.timesteps, schedule)
    branch = sampler.continue_from(states[1], 1, condition, cost, stop=1)
    branch.add_(1)
    assert not torch.equal(branch, states[1])
    baseline = states[1].clone()
    sampler.continue_from(states[1], 1, condition, cost)
    assert torch.equal(states[1], baseline)
    terminal_cost = Counter()
    terminal = sampler.preview(states[-1], sampler.num_steps, condition, terminal_cost)
    assert terminal is states[-1]
    image = sampler.decode(terminal, terminal_cost)
    assert image.shape == (1, 3, 8, 8)
    assert image.min() >= 0 and image.max() <= 1
    assert terminal_cost.get("unet_calls", 0) == 0
    assert terminal_cost["vae_decodes"] == 1


def test_preview_calculation_and_frozen_gradient_path(tiny_sampler):
    sampler = tiny_sampler
    condition = sampler.encode("object", "", Counter())
    base = sampler.initial(23, 8, 8)
    cost = Counter()
    clean = sampler.preview(base, 1, condition, cost)
    _, step_clean = sampler.step(base, 1, condition, Counter())
    torch.testing.assert_close(clean, step_clean, rtol=0, atol=0)
    assert cost["unet_calls"] == cost["preview_unet_calls"] == 1
    assert cost["unet_samples"] == 2  # Two CFG branches, one batched forward.
    for k in (1, sampler.num_steps):
        edit = torch.zeros_like(base, requires_grad=True)
        preview = sampler.preview(base + edit, k, condition, Counter())
        image = sampler.decode(preview, Counter())
        image.square().mean().backward()
        assert edit.grad is not None and torch.isfinite(edit.grad).all()
        assert edit.grad.abs().sum() > 0
    for module in (sampler.unet, sampler.vae, sampler.pipe.text_encoder):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())


def test_metadata_and_stage_validation(tiny_sampler):
    sampler = tiny_sampler
    condition = sampler.encode("object", "", Counter())
    metadata = sampler.state_metadata(condition)
    sampler.validate_metadata(metadata, condition)
    sampler.validate_metadata(metadata)
    broken = copy.deepcopy(metadata)
    broken["timesteps"][0] -= 1
    with pytest.raises(ValueError, match="timesteps"):
        sampler.validate_metadata(broken)
    broken = copy.deepcopy(metadata)
    broken["conditioning_identity"] = "wrong-prompt"
    with pytest.raises(ValueError, match="conditioning_identity"):
        sampler.validate_metadata(broken)
    x = sampler.initial(0, 8, 8)
    with pytest.raises(ValueError, match="Invalid stage"):
        sampler.step(x, sampler.num_steps, condition, Counter())
    with pytest.raises(ValueError, match="k <= stop"):
        sampler.continue_from(x, 2, condition, Counter(), stop=1)


def test_scheduler_default_provenance_does_not_invalidate_cache(tiny_sampler):
    condition = tiny_sampler.encode("object", "", Counter())
    metadata = tiny_sampler.state_metadata(condition)
    original = copy.deepcopy(metadata)
    defaults = metadata["scheduler_config"].get("_use_default_values", [])
    metadata["scheduler_config"]["_use_default_values"] = list(reversed(defaults))
    tiny_sampler.validate_metadata(metadata, condition)
    # Explicitly supplied defaults produce the same sampler even though this
    # constructor-origin bookkeeping differs or is absent in an older cache.
    metadata["scheduler_config"].pop("_use_default_values")
    tiny_sampler.validate_metadata(metadata, condition)
    assert tiny_sampler.state_metadata(condition) == original


def test_scheduler_cache_replays_across_process_hash_seeds(tiny_sampler):
    # Only schedulers are constructed in the children: no network, neural model,
    # latent generation, or CUDA. Diffusers' default-key list comes from a set.
    script = """
import json
from diffusers import DDIMScheduler
scheduler = DDIMScheduler(num_train_timesteps=12, clip_sample=False, steps_offset=1)
print(json.dumps(dict(scheduler.config)))
"""
    condition = tiny_sampler.encode("object", "", Counter())
    metadata = tiny_sampler.state_metadata(condition)
    for seed in ("0", "1"):
        result = subprocess.run([sys.executable, "-c", script], check=True,
                                capture_output=True, text=True, timeout=60,
                                env={**os.environ, "PYTHONHASHSEED": seed,
                                     "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                                     "TRANSFORMERS_OFFLINE": "1"})
        metadata["scheduler_config"] = json.loads(result.stdout)
        tiny_sampler.validate_metadata(metadata, condition)


@pytest.mark.parametrize("field,value", [
    ("prediction_type", "v_prediction"), ("beta_start", 0.002),
    ("num_train_timesteps", 15), ("steps_offset", 0),
    ("clip_sample", True), ("set_alpha_to_one", False),
    ("timestep_spacing", "trailing"),
])
def test_scheduler_effective_config_changes_remain_incompatible(tiny_sampler, field, value):
    condition = tiny_sampler.encode("object", "", Counter())
    metadata = tiny_sampler.state_metadata(condition)
    metadata["scheduler_config"][field] = value
    with pytest.raises(ValueError, match=f"scheduler_config.*{field}"):
        tiny_sampler.validate_metadata(metadata, condition)


def test_scheduler_missing_effective_config_is_rejected(tiny_sampler):
    condition = tiny_sampler.encode("object", "", Counter())
    metadata = tiny_sampler.state_metadata(condition)
    metadata["scheduler_config"].pop("beta_start")
    with pytest.raises(ValueError, match="scheduler_config.*beta_start"):
        tiny_sampler.validate_metadata(metadata, condition)
