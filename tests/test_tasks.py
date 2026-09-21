"""Native-logit and differentiable preprocessing checks with tiny local scorers."""
from pathlib import Path

import pytest
import torch

from utils.tasks import Scorer, load_tasks, protected_fields, threshold


class Counter(dict):
    def add(self, name, amount=1):
        self[name] = self.get(name, 0) + amount


@pytest.fixture(params=["clip", "siglip"])
def tiny_scorer(request):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    torch.manual_seed(71)
    vocab = {word: i for i, word in enumerate([
        "[PAD]", "[EOS]", "[UNK]", "a", "photo", "of", "red", "blue", "object",
        "close-up", "part", "showing", "the", "entire", "without"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]",
        unk_token="[UNK]", model_max_length=12)
    text_config = dict(vocab_size=len(vocab), hidden_size=8, intermediate_size=16,
                       num_hidden_layers=1, num_attention_heads=2,
                       max_position_embeddings=12, eos_token_id=1, bos_token_id=1, pad_token_id=0)
    vision_config = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, image_size=8, patch_size=4)
    if request.param == "clip":
        config = transformers.CLIPConfig(text_config=text_config, vision_config=vision_config,
                                         projection_dim=8)
        model = transformers.CLIPModel(config)
        processor = transformers.CLIPImageProcessor(
            size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8})
        role = "a"
    else:
        config = transformers.SiglipConfig(text_config=text_config, vision_config=vision_config)
        model = transformers.SiglipModel(config)
        processor = transformers.SiglipImageProcessor(size={"height": 8, "width": 8})
        role = "b"
    task_config = load_tasks(Path(__file__).parents[1] / "configs/tasks.json")
    return Scorer(model, processor, tokenizer, tasks=task_config, role=role, device="cpu",
                  dtype=torch.float32, resize_mode="native", size=0, mean=[], std=[],
                  interpolation="bicubic", antialias=True)


def test_native_logits_and_text_cache(tiny_scorer):
    scorer = tiny_scorer
    image = torch.rand(1, 3, 8, 8, requires_grad=True)
    cost = Counter()
    scores = scorer.score(image, "object", cost)
    assert list(scores) == ["appearance", "composition", "category"]
    texts = [scorer.tasks["tasks"]["appearance"][side].format(object="object")
             for side in ("positive", "negative")]
    encoded = scorer.tokenizer(texts, padding="max_length" if scorer.role == "b" else True,
                               truncation=True, max_length=12, return_tensors="pt")
    direct = scorer.model(pixel_values=scorer.preprocess(image), **encoded).logits_per_image
    torch.testing.assert_close(scores["appearance"], direct[:, 0] - direct[:, 1], rtol=1e-4, atol=1e-5)
    scores["appearance"].sum().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in scorer.model.parameters())
    scorer.score(image.detach(), "object", cost)
    assert cost[f"evaluator_{scorer.role}_text_calls"] == 1
    assert cost[f"evaluator_{scorer.role}_image_calls"] == 2


def test_preprocessing_matches_ordinary_processor(tiny_scorer):
    scorer = tiny_scorer
    # A smooth deterministic non-square image exercises actual resize/crop geometry.
    ramp = torch.linspace(0, 1, 11 * 15).reshape(1, 1, 11, 15).expand(1, 3, 11, 15)
    ordinary = scorer.processor(images=ramp[0], do_rescale=False,
                                return_tensors="pt")["pixel_values"]
    differentiable = scorer.preprocess(ramp)
    # PIL-backed older Transformers may round to uint8 while resizing; this tolerance
    # bounds that conversion and interpolation difference, not scorer accuracy.
    torch.testing.assert_close(differentiable, ordinary, rtol=0, atol=0.035)
    same_size = torch.rand(1, 3, 8, 8)
    exact = scorer.processor(images=same_size[0], do_rescale=False,
                             return_tensors="pt")["pixel_values"]
    torch.testing.assert_close(scorer.preprocess(same_size), exact, rtol=0, atol=1e-6)


def test_task_thresholds_are_evaluator_specific():
    tasks = load_tasks(Path(__file__).parents[1] / "configs/tasks.json")
    assert threshold(tasks, "appearance", "a", "margin") != threshold(tasks, "appearance", "b", "margin")
    assert protected_fields(tasks, "appearance") == ["composition", "category"]
