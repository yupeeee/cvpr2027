"""Frozen Stable Diffusion components with a fixed, exactly replayable DDIM schedule."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch


@dataclass(frozen=True)
class Conditioning:
    embeddings: torch.Tensor
    prompt: str
    negative_prompt: str
    identity: str


def _plain(value: Any) -> Any:
    """Diffusers FrozenDict configuration, converted to JSON-compatible values."""
    return json.loads(json.dumps(dict(value), default=lambda x: x.tolist()))


class Sampler:
    def __init__(self, pipe: Any, *, num_steps: int, guidance_scale: float,
                 device: torch.device | str, dtype: torch.dtype,
                 model_id: str, model_revision: str | None):
        self.pipe = pipe
        self.unet, self.vae = pipe.unet, pipe.vae
        self.scheduler = pipe.scheduler
        self.device, self.dtype = torch.device(device), dtype
        self.num_steps, self.guidance_scale = num_steps, guidance_scale
        self.model_id, self.model_revision = model_id, model_revision
        for component in (self.unet, self.vae, pipe.text_encoder):
            component.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        self.scheduler.set_timesteps(num_steps, device=self.device)
        self.timesteps = self.scheduler.timesteps.clone()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_channels = int(self.unet.config.in_channels)
        if self.latent_channels != self.vae.config.latent_channels:
            raise ValueError("Only ordinary text-to-image SD latents are supported (no inpainting UNet).")
        if self.unet.config.time_cond_proj_dim is not None:
            raise ValueError("Guidance-embedding UNets are not supported by this SD/DDIM backend.")

    @classmethod
    def load(cls, args: Any, device: torch.device | str) -> "Sampler":
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        if args.scheduler != "ddim" or args.ddim_eta != 0:
            raise ValueError("This pilot supports deterministic DDIM only (scheduler=ddim, eta=0).")
        dtype = getattr(torch, args.precision)
        from .models import model_access, prepare_generator
        model_path = prepare_generator(args)
        with model_access("model", args.model_id, args.model_revision, args.local_files_only):
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path, torch_dtype=dtype,
                local_files_only=True, safety_checker=None,
                requires_safety_checker=False,
            )
        overrides = dict(timestep_spacing=args.ddim_timestep_spacing,
                         steps_offset=args.ddim_steps_offset,
                         clip_sample=args.ddim_clip_sample,
                         set_alpha_to_one=args.ddim_set_alpha_to_one)
        if args.scheduler_prediction_type != "native":
            overrides["prediction_type"] = args.scheduler_prediction_type
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, **overrides)
        result = cls(pipe, num_steps=args.num_steps, guidance_scale=args.guidance_scale,
                   device=device, dtype=dtype, model_id=args.model_id,
                   model_revision=args.model_revision)
        from .data import model_identity
        result.weights_identity = model_identity(args.model_id, args.model_revision, "model_index.json")
        return result

    @torch.no_grad()
    def encode(self, prompt: str, negative_prompt: str, cost: Any) -> Conditioning:
        cfg = self.guidance_scale > 1
        positive, negative = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt, device=self.device,
            num_images_per_prompt=1, do_classifier_free_guidance=cfg,
        )
        cost.add("text_encoder_calls", 1 + int(cfg))
        cost.add("text_encoder_samples", 1 + int(cfg))
        embeddings = torch.cat((negative, positive)) if cfg else positive
        identity = hashlib.sha256(json.dumps(
            [prompt, negative_prompt, self.guidance_scale, self.model_id, self.model_revision],
            ensure_ascii=False).encode()).hexdigest()
        return Conditioning(embeddings, prompt, negative_prompt, identity)

    def initial(self, seed: int, height: int, width: int) -> torch.Tensor:
        if height % self.vae_scale_factor or width % self.vae_scale_factor:
            raise ValueError("Image dimensions must be divisible by the VAE scale factor.")
        # CPU RNG is independent of the inference rank/device and batch boundaries.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = torch.randn((1, self.latent_channels, height // self.vae_scale_factor,
                              width // self.vae_scale_factor), generator=generator,
                             dtype=torch.float32).to(device=self.device, dtype=self.dtype)
        return latent * self.scheduler.init_noise_sigma

    def _check_stage(self, k: int, terminal: bool = False) -> None:
        if not isinstance(k, int) or not 0 <= k < self.num_steps + int(terminal):
            raise ValueError(f"Invalid stage {k}; expected 0..{self.num_steps - 1 + int(terminal)}.")

    def _update(self, x: torch.Tensor, k: int, conditioning: Conditioning,
                cost: Any, *, preview: bool) -> Any:
        self._check_stage(k)
        t = self.timesteps[k]
        cfg = self.guidance_scale > 1
        model_input = torch.cat((x, x)) if cfg else x
        model_input = self.scheduler.scale_model_input(model_input, t)
        embeddings = conditioning.embeddings
        if x.shape[0] != 1:
            # A conditioning object represents one prompt, repeat across its branches.
            embeddings = embeddings.repeat_interleave(x.shape[0], dim=0)
        output = self.unet(model_input, t, encoder_hidden_states=embeddings).sample
        cost.add("unet_calls")
        cost.add("unet_samples", model_input.shape[0])
        if preview:
            cost.add("preview_unet_calls")
        if cfg:
            unconditional, conditional = output.chunk(2)
            output = unconditional + self.guidance_scale * (conditional - unconditional)
        return self.scheduler.step(output, t, x, eta=0, return_dict=True)

    def step(self, x: torch.Tensor, k: int, conditioning: Conditioning,
             cost: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Psi_k(x_k), plus its predicted clean latent; one denoiser evaluation."""
        result = self._update(x, k, conditioning, cost, preview=False)
        return result.prev_sample, result.pred_original_sample

    def preview(self, x: torch.Tensor, k: int, conditioning: Conditioning,
                cost: Any) -> torch.Tensor:
        """Differentiable predicted clean latent; terminal stage needs no UNet call."""
        self._check_stage(k, terminal=True)
        if k == self.num_steps:
            return x
        return self._update(x, k, conditioning, cost, preview=True).pred_original_sample

    @torch.no_grad()
    def continue_from(self, x: torch.Tensor, k: int, conditioning: Conditioning,
                      cost: Any, stop: int | None = None) -> torch.Tensor:
        """Replay the original schedule suffix/segment, never a freshly shortened schedule."""
        self._check_stage(k, terminal=True)
        stop = self.num_steps if stop is None else stop
        self._check_stage(stop, terminal=True)
        if stop < k:
            raise ValueError("A continuation segment must satisfy k <= stop <= num_steps.")
        result = x.detach().clone()
        for j in range(k, stop):
            result, _ = self.step(result, j, conditioning, cost)
        return result

    def decode(self, x: torch.Tensor, cost: Any) -> torch.Tensor:
        """Differentiable VAE decode in [0,1]; no PIL/NumPy on the edit path."""
        latent = x / self.vae.config.scaling_factor
        shift = getattr(self.vae.config, "shift_factor", None)
        if shift is not None:
            latent = latent + shift
        image = self.vae.decode(latent).sample
        cost.add("vae_decodes")
        cost.add("vae_samples", x.shape[0])
        return (image / 2 + 0.5).clamp(0, 1)

    def state_metadata(self, conditioning: Conditioning) -> dict[str, Any]:
        return {"model_id": self.model_id, "model_revision": self.model_revision,
                "weights_identity": getattr(self, "weights_identity", {"local_fixture": self.model_id}),
                "scheduler": "DDIMScheduler", "scheduler_config": _plain(self.scheduler.config),
                "timesteps": self.timesteps.cpu().tolist(), "num_steps": self.num_steps,
                "guidance_scale": self.guidance_scale, "dtype": str(self.dtype),
                "conditioning_identity": conditioning.identity,
                "prompt": conditioning.prompt, "negative_prompt": conditioning.negative_prompt,
                "latent_channels": self.latent_channels,
                "vae_scale_factor": self.vae_scale_factor,
                "vae_scaling_factor": self.vae.config.scaling_factor}

    def validate_metadata(self, metadata: dict[str, Any],
                          conditioning: Conditioning | None = None) -> None:
        identity = hashlib.sha256(json.dumps(
            [metadata["prompt"], metadata["negative_prompt"], self.guidance_scale,
             self.model_id, self.model_revision], ensure_ascii=False).encode()).hexdigest()
        context = Conditioning(torch.empty(0), metadata["prompt"],
                               metadata["negative_prompt"], identity)
        expected = self.state_metadata(conditioning or context)
        for key, value in expected.items():
            cached = metadata.get(key)
            if key == "scheduler_config" and isinstance(cached, dict):
                # Diffusers stores this as list(set(...)), whose order changes
                # across processes. It tracks which constructor arguments were
                # omitted, not their effective values. Keep the complete config
                # in saved records, but compare every actual setting strictly.
                cached = {name: setting for name, setting in cached.items()
                          if name != "_use_default_values"}
                value = {name: setting for name, setting in value.items()
                         if name != "_use_default_values"}
            if cached != value:
                detail = ""
                if key == "scheduler_config" and isinstance(cached, dict):
                    changed = sorted(name for name in cached.keys() | value.keys()
                                     if name not in cached or name not in value or cached[name] != value[name])
                    detail = f" ({', '.join(changed)})"
                raise ValueError(f"Cached sampler state has incompatible {key}{detail}.")
