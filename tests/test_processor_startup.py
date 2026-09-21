"""Tokenizer startup checks, with no pretrained weights, model calls, or CUDA."""
import io
import shlex
import sys
from types import SimpleNamespace

import pytest

from utils import models


def test_missing_sentencepiece_explains_active_environment(monkeypatch):
    from transformers import AutoProcessor
    cause = ImportError('SiglipTokenizer requires the SentencePiece library')
    calls = []

    def missing(path, **kwargs):
        calls.append((path, kwargs))
        raise cause

    monkeypatch.setattr(AutoProcessor, 'from_pretrained', missing)
    with pytest.raises(ImportError, match='sentencepiece.*protobuf') as failure:
        models.load_evaluator_processor('/local/siglip')
    assert failure.value.__cause__ is cause
    assert shlex.quote(sys.executable) in str(failure.value)
    assert '-m pip install --no-deps' in str(failure.value)
    assert str(cause) in str(failure.value)
    assert calls == [('/local/siglip', {'local_files_only': True})]


def test_preflight_constructs_all_preprocessors_locally(monkeypatch):
    from transformers import AutoProcessor, AutoTokenizer
    calls = []
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained',
                        lambda path, **kwargs: calls.append(('tokenizer', path, kwargs)))
    monkeypatch.setattr(AutoProcessor, 'from_pretrained',
                        lambda path, **kwargs: calls.append(('processor', path, kwargs)))
    models.validate_preprocessors({'model': '/local/sd', 'evaluator_a': '/local/clip',
                                   'evaluator_b': '/local/siglip'})
    assert calls == [('tokenizer', '/local/sd/tokenizer', {'local_files_only': True}),
                     ('processor', '/local/clip', {'local_files_only': True}),
                     ('processor', '/local/siglip', {'local_files_only': True})]


def test_scorer_rejects_missing_tokenizer_before_loading_weights(monkeypatch):
    from transformers import AutoConfig, AutoProcessor, SiglipModel
    from utils.tasks import Scorer
    monkeypatch.setattr(models, 'prepare_evaluator', lambda args, role: '/local/siglip')
    monkeypatch.setattr(AutoConfig, 'from_pretrained',
                        lambda *args, **kwargs: SimpleNamespace(model_type='siglip'))

    def missing(*args, **kwargs):
        raise ImportError('SiglipTokenizer requires the SentencePiece library')

    monkeypatch.setattr(AutoProcessor, 'from_pretrained', missing)
    monkeypatch.setattr(SiglipModel, 'from_pretrained',
                        lambda *args, **kwargs: pytest.fail('Model weights loaded before tokenizer check'))
    args = SimpleNamespace(evaluator_b_id='local/siglip', evaluator_b_revision=None,
                           local_files_only=True)
    with pytest.raises(ImportError, match='sentencepiece.*protobuf'):
        Scorer.load(args, 'b', 'cpu')


def test_preflight_failure_stops_before_auto_launch(monkeypatch):
    from exps import collect
    from utils import cache
    monkeypatch.setattr(cache, 'skip_completed', lambda *args: False)
    monkeypatch.setattr(models, 'prepare_generator', lambda args: '/local/sd')
    monkeypatch.setattr(models, 'prepare_evaluator', lambda args, role: '/local/' + role)

    def missing(paths):
        raise ImportError('missing tokenizer dependency')

    monkeypatch.setattr(models, 'validate_preprocessors', missing)
    monkeypatch.setattr(collect, 'launch_if_needed',
                        lambda *args: pytest.fail('Workers launched before processor validation'))
    monkeypatch.setattr(collect, 'Distributed',
                        lambda *args: pytest.fail('Distributed initialized before processor validation'))
    with pytest.raises(ImportError, match='missing tokenizer dependency'):
        collect.main(['--device', 'auto'])


def test_real_siglip_processor_roundtrip_without_weights(tmp_path, monkeypatch):
    import sentencepiece
    from transformers import SiglipImageProcessor, SiglipProcessor, SiglipTokenizer, SiglipModel

    # A tiny synthetic vocabulary tests the actual optional tokenizer dependencies.
    # This is a temporary unit-test fixture, never research data or pretrained weights.
    buffer = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(['a red object', 'a blue object', 'a close up view',
                                'the whole object']),
        model_writer=buffer, vocab_size=64, hard_vocab_limit=False, minloglevel=2)
    vocab = tmp_path / 'spiece.model'
    vocab.write_bytes(buffer.getvalue())
    processor = SiglipProcessor(image_processor=SiglipImageProcessor(),
                               tokenizer=SiglipTokenizer(vocab_file=str(vocab)))
    processor.save_pretrained(tmp_path)
    monkeypatch.setattr(SiglipModel, 'from_pretrained',
                        lambda *args, **kwargs: pytest.fail('Processor check loaded model weights'))
    restored = models.load_evaluator_processor(str(tmp_path))
    assert isinstance(restored, SiglipProcessor)
    assert isinstance(restored.tokenizer, SiglipTokenizer)
    options = dict(padding='max_length', max_length=restored.tokenizer.model_max_length,
                   truncation=True, return_tensors='pt')
    actual = restored.tokenizer(['a red object', 'a blue object'], **options)['input_ids']
    expected = processor.tokenizer(['a red object', 'a blue object'], **options)['input_ids']
    assert actual.shape == (2, restored.tokenizer.model_max_length)
    assert actual.equal(expected)
    assert not list(tmp_path.glob('*.safetensors'))
