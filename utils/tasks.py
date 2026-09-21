"""Explicit binary proxy tasks and frozen, differentiable CLIP/SigLIP scorers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


def load_tasks(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        tasks = json.load(stream)
    if set(tasks["tasks"]) != {"appearance", "composition"}:
        raise ValueError("The pilot requires exactly appearance and composition tasks.")
    for name, definition in {**tasks["tasks"], "category": tasks["category"]}.items():
        for field in ("positive", "negative"):
            definition[field].format(object="object")  # Validate the configurable templates.
        for role in ("a", "b"):
            required = ("protected_tolerance",) if name == "category" else (
                "decision", "margin", "protected_tolerance")
            for field in required:
                value = threshold(tasks, name, role, field)
                if not math.isfinite(value) or (field != "decision" and value < 0):
                    raise ValueError(f"Invalid {role}/{name}/{field} threshold.")
        if name != "category":
            protected = protected_fields(tasks, name)
            if set(protected) != ({"appearance", "composition", "category"} - {name}):
                raise ValueError(f"{name} must protect the other task and object category.")
    return tasks


def threshold(tasks: dict[str, Any], field: str, role: str, kind: str) -> float:
    definition = tasks["category"] if field == "category" else tasks["tasks"][field]
    return float(definition["thresholds"][role][kind])


def protected_fields(tasks: dict[str, Any], target: str) -> list[str]:
    return list(tasks["tasks"][target]["protected"])


def _features(output: Any) -> torch.Tensor:
    # Transformers 4 returned tensors; Transformers 5 returns a pooled model output.
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, tuple) and len(output) > 1:
        return output[1]
    raise TypeError("Unsupported Transformers feature output; expected tensor or pooler_output.")


class Scorer:
    def __init__(self, model: Any, processor: Any, tokenizer: Any, *,
                 tasks: dict[str, Any], role: str, device: torch.device | str,
                 dtype: torch.dtype, resize_mode: str, size: int,
                 mean: list[float], std: list[float], interpolation: str,
                 antialias: bool):
        if role not in {"a", "b"}:
            raise ValueError("Evaluator role must be a or b.")
        if model.config.model_type not in {"clip", "siglip"}:
            raise ValueError("Only CLIP and SigLIP evaluator architectures are supported.")
        self.model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.processor, self.tokenizer = processor, tokenizer
        self.tasks, self.role = tasks, role
        self.device, self.dtype = torch.device(device), dtype
        self.resize_mode, self.size = resize_mode, size
        self.mean = mean or list(processor.image_mean)
        self.std = std or list(processor.image_std)
        if len(self.mean) != 3 or len(self.std) != 3 or min(self.std) <= 0:
            raise ValueError("Evaluator mean/std require three channels and positive std.")
        self.interpolation, self.antialias = interpolation, antialias
        if interpolation not in {"bilinear", "bicubic"}:
            raise ValueError("Differentiable preprocessing supports bilinear or bicubic resizing.")
        if resize_mode not in {"native", "shortest", "square"} or size < 0:
            raise ValueError("Invalid evaluator resize settings.")
        self.text_cache: dict[tuple[str, ...], torch.Tensor] = {}

    @classmethod
    def load(cls, args: Any, role: str, device: torch.device | str) -> "Scorer":
        from transformers import AutoConfig, CLIPModel, SiglipModel

        identifier = getattr(args, f"evaluator_{role}_id")
        kwargs = dict(revision=getattr(args, f"evaluator_{role}_revision"),
                      local_files_only=args.local_files_only)
        from .models import load_evaluator_processor, model_access, prepare_evaluator
        model_path = prepare_evaluator(args, role)
        load_kwargs = dict(local_files_only=True)
        with model_access(f"evaluator_{role}", identifier, kwargs["revision"], args.local_files_only):
            config = AutoConfig.from_pretrained(model_path, **load_kwargs)
            if config.model_type not in {"clip", "siglip"}:
                raise ValueError(f"Unsupported evaluator architecture {config.model_type!r}.")
            processor = load_evaluator_processor(model_path)
            model_class = CLIPModel if config.model_type == "clip" else SiglipModel
            model = model_class.from_pretrained(model_path, **load_kwargs)
        result = cls(model, processor.image_processor, processor.tokenizer,
                   tasks=load_tasks(args.tasks_path), role=role, device=device,
                   dtype=getattr(torch, args.evaluator_precision),
                   resize_mode=getattr(args, f"evaluator_{role}_resize_mode"),
                   size=getattr(args, f"evaluator_{role}_size"),
                   mean=getattr(args, f"evaluator_{role}_mean"),
                   std=getattr(args, f"evaluator_{role}_std"),
                   interpolation=args.evaluator_interpolation,
                   antialias=args.evaluator_antialias)
        from .data import model_identity
        result.weights_identity = model_identity(identifier, kwargs["revision"], "config.json")
        return result

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Native resize/crop geometry and normalization, with differentiable tensors."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Scorer input must have shape B,3,H,W and values in [0,1].")
        value = images.to(device=self.device, dtype=torch.float32)
        height, width = value.shape[-2:]
        native_size = self.processor.size
        native_shortest = "shortest_edge" in native_size
        mode = self.resize_mode
        if mode == "native":
            mode = "shortest" if native_shortest else "square"
        do_resize = self.resize_mode != "native" or self.size > 0 or self.processor.do_resize
        if do_resize:
            if mode == "shortest":
                edge = self.size or native_size.get("shortest_edge", native_size.get("height"))
                target = (edge, int(width * edge / height)) if height <= width else (
                    int(height * edge / width), edge)
            else:
                target = (self.size, self.size) if self.size else (
                    native_size.get("height", native_size.get("shortest_edge")),
                    native_size.get("width", native_size.get("shortest_edge")))
            if tuple(target) != (height, width):
                value = F.interpolate(value, size=target, mode=self.interpolation,
                                      align_corners=False, antialias=self.antialias)
        if self.resize_mode != "square" and getattr(self.processor, "do_center_crop", False):
            crop = self.processor.crop_size
            crop_h, crop_w = (self.size, self.size) if self.size else (crop["height"], crop["width"])
            h, w = value.shape[-2:]
            pad_h, pad_w = max(0, crop_h - h), max(0, crop_w - w)
            if pad_h or pad_w:
                value = F.pad(value, (pad_w // 2, pad_w - pad_w // 2,
                                      pad_h // 2, pad_h - pad_h // 2))
            h, w = value.shape[-2:]
            top, left = (h - crop_h) // 2, (w - crop_w) // 2
            value = value[:, :, top:top + crop_h, left:left + crop_w]
        if getattr(self.processor, "do_normalize", True):
            mean = value.new_tensor(self.mean)[None, :, None, None]
            std = value.new_tensor(self.std)[None, :, None, None]
            value = (value - mean) / std
        return value.to(dtype=self.dtype)

    @torch.no_grad()
    def _text_features(self, texts: tuple[str, ...], cost: Any) -> torch.Tensor:
        if texts not in self.text_cache:
            # SigLIP was trained with fixed max-length padding, unlike CLIP's longest padding.
            options = dict(padding="max_length" if self.model.config.model_type == "siglip" else True,
                           truncation=True, return_tensors="pt",
                           max_length=self.model.config.text_config.max_position_embeddings)
            encoded = self.tokenizer(list(texts), **options)
            inputs = {key: tensor.to(self.device) for key, tensor in encoded.items()
                      if key in {"input_ids", "attention_mask", "position_ids"}}
            feature = _features(self.model.get_text_features(**inputs))
            self.text_cache[texts] = F.normalize(feature, dim=-1).detach()
            cost.add(f"evaluator_{self.role}_text_calls")
            cost.add(f"evaluator_{self.role}_text_samples", len(texts))
        return self.text_cache[texts]

    def prepare_text(self, object_name: str, cost: Any) -> torch.Tensor:
        """Cache task texts using the declared object bank, without inspecting images."""
        definitions = {**self.tasks["tasks"], "category": self.tasks["category"]}
        texts = tuple(definitions[name][side].format(object=object_name)
                      for name in ("appearance", "composition", "category")
                      for side in ("positive", "negative"))
        return self._text_features(texts, cost)

    def score(self, images: torch.Tensor, object_name: str, cost: Any) -> dict[str, torch.Tensor]:
        names = ("appearance", "composition", "category")
        text = self.prepare_text(object_name, cost)
        pixels = self.preprocess(images)
        size = self.model.config.vision_config.image_size
        feature = _features(self.model.get_image_features(
            pixel_values=pixels, interpolate_pos_encoding=tuple(pixels.shape[-2:]) != (size, size)))
        feature = F.normalize(feature, dim=-1)
        logits = feature @ text.T * self.model.logit_scale.exp()
        if self.model.config.model_type == "siglip":
            logits = logits + self.model.logit_bias
        cost.add(f"evaluator_{self.role}_image_calls")
        cost.add(f"evaluator_{self.role}_image_samples", images.shape[0])
        # Differences are in each model's own raw matching-logit scale, never probabilities.
        return {name: logits[:, 2 * index] - logits[:, 2 * index + 1]
                for index, name in enumerate(names)}
