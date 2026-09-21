"""Offline cache/preparation checks with tiny file fixtures and mocked Hub calls."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import models


def _files(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        file = folder / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{}')
    return folder


def _generator(tmp_path):
    return _files(tmp_path / 'generator', [
        'model_index.json', 'scheduler/scheduler_config.json',
        'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
        'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
        'text_encoder/config.json', 'text_encoder/model.safetensors',
        'tokenizer/vocab.json', 'tokenizer/merges.txt'])


def _evaluator(tmp_path, weight='model.safetensors'):
    return _files(tmp_path / 'evaluator', [
        'config.json', 'preprocessor_config.json', 'tokenizer.json', weight])


def _args(identifier='owner/model', local_only=True):
    return SimpleNamespace(model_id=identifier, model_revision='pinned-revision',
                           evaluator_a_id=identifier, evaluator_a_revision='eval-revision',
                           evaluator_b_id=identifier, evaluator_b_revision=None,
                           local_files_only=local_only, device='cpu', dry_run=False)


def _allow_online(monkeypatch):
    # All network entry points used by these tests are still mocked.
    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    monkeypatch.delenv('TRANSFORMERS_OFFLINE', raising=False)
    import huggingface_hub.constants
    monkeypatch.setattr(huggingface_hub.constants, 'HF_HUB_OFFLINE', False)


def test_complete_local_directories_need_no_hub(tmp_path, monkeypatch):
    import huggingface_hub
    def fail(*args, **kwargs):
        pytest.fail('Local directories must not contact the Hub')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fail)
    monkeypatch.setattr(huggingface_hub, 'list_repo_files', fail)
    args = _args(str(_generator(tmp_path)))
    args.evaluator_a_id = args.evaluator_b_id = str(_evaluator(tmp_path))
    original = vars(args).copy()
    checked = []
    # This test uses filename-only fixtures. Real processor loading is tested separately.
    monkeypatch.setattr(models, 'validate_preprocessors', lambda paths: checked.append(paths))
    paths = models.prepare_models(args)
    assert paths == dict(model=args.model_id, evaluator_a=args.evaluator_a_id,
                         evaluator_b=args.evaluator_b_id)
    assert vars(args) == original
    assert checked == [paths]


def test_missing_cache_explains_explicit_offline_setting(monkeypatch):
    import huggingface_hub
    calls = []
    def missing(identifier, **kwargs):
        calls.append(kwargs)
        raise FileNotFoundError('no snapshot')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', missing)
    with pytest.raises(RuntimeError) as failure:
        models.prepare_generator(_args())
    assert len(calls) == 1
    assert calls[0]['local_files_only'] is True
    assert calls[0]['revision'] == 'pinned-revision'
    assert '--no-local-files-only' in str(failure.value)
    assert '--model-id' in str(failure.value)
    assert 'HF_HUB_OFFLINE' in str(failure.value)
    assert isinstance(failure.value.__cause__, FileNotFoundError)


def test_allowed_download_uses_pipeline_components(tmp_path, monkeypatch):
    import diffusers
    _allow_online(monkeypatch)
    import huggingface_hub
    directory = _generator(tmp_path)
    calls = []
    def cache(identifier, **kwargs):
        calls.append((identifier, kwargs))
        raise FileNotFoundError('cache empty')
    def download(identifier, **kwargs):
        calls.append((identifier, kwargs))
        return directory
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', cache)
    monkeypatch.setattr(diffusers.StableDiffusionPipeline, 'download', download)
    result = models.prepare_generator(_args(local_only=False))
    assert result == str(directory)
    assert [kwargs['local_files_only'] for _, kwargs in calls] == [True, False]
    assert all(kwargs['revision'] == 'pinned-revision' for _, kwargs in calls)
    assert calls[1][1]['safety_checker'] is None


@pytest.mark.parametrize('weight', ['model.safetensors', 'pytorch_model.bin'])
def test_evaluator_download_limits_files_and_prefers_safetensors(tmp_path, monkeypatch, weight):
    import huggingface_hub
    _allow_online(monkeypatch)
    directory = _evaluator(tmp_path, weight)
    calls = []
    def snapshot(identifier, **kwargs):
        calls.append((identifier, kwargs))
        if kwargs['local_files_only']:
            raise FileNotFoundError('cache empty')
        return str(directory)
    available = ['config.json', 'model.fp16.safetensors', 'unrelated.ckpt', weight]
    if weight == 'model.safetensors':
        available.append('pytorch_model.bin')
    monkeypatch.setattr(huggingface_hub, 'list_repo_files', lambda *a, **kw: available)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    assert models.prepare_evaluator(_args(local_only=False), 'a') == str(directory)
    kwargs = calls[-1][1]
    assert kwargs['revision'] == 'eval-revision'
    assert kwargs['local_files_only'] is False
    assert weight in kwargs['allow_patterns']
    assert 'unrelated.ckpt' not in kwargs['allow_patterns']
    assert 'model.fp16.safetensors' not in kwargs['allow_patterns']
    if weight == 'model.safetensors':
        assert 'pytorch_model.bin' not in kwargs['allow_patterns']


def test_existing_complete_snapshot_avoids_online_calls(tmp_path, monkeypatch):
    import huggingface_hub
    _allow_online(monkeypatch)
    directory = _evaluator(tmp_path)
    calls = []
    def snapshot(identifier, **kwargs):
        calls.append(kwargs)
        return str(directory)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    monkeypatch.setattr(huggingface_hub, 'list_repo_files',
                        lambda *a, **kw: pytest.fail('Complete cache should be reused'))
    models.prepare_evaluator(_args(local_only=False), 'b')
    assert len(calls) == 1 and calls[0]['local_files_only'] is True


def test_environment_offline_blocks_download_even_when_argument_allows(monkeypatch):
    calls = []
    def prepare(identifier, revision, local_only):
        calls.append(local_only)
        raise FileNotFoundError('no cache')
    monkeypatch.setattr(models, '_prepare_generator', prepare)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    with pytest.raises(RuntimeError, match='does not override offline environment'):
        models.prepare_generator(_args(local_only=False))
    assert calls == [True]


def test_incomplete_weight_shards_fail_before_model_loading(tmp_path):
    directory = _evaluator(tmp_path, 'model.safetensors.index.json')
    (directory / 'model.safetensors.index.json').write_text(
        '{"weight_map": {"layer": "model-00001-of-00001.safetensors"}}')
    args = _args(str(directory))
    with pytest.raises(RuntimeError, match='model-00001-of-00001.safetensors'):
        models.prepare_evaluator(args, 'a')
    _files(directory, ['model-00001-of-00001.safetensors'])
    assert models.prepare_evaluator(args, 'a') == str(directory)


def test_non_cache_programming_errors_are_not_relabelled():
    with pytest.raises(TypeError, match='bad API argument'):
        with models.model_access('model', 'owner/model', None, True):
            raise TypeError('bad API argument')


def test_distributed_preparation_is_before_launch_and_dry_run_skips(monkeypatch):
    calls = []
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    monkeypatch.setattr(models, 'prepare_models', lambda args: calls.append(args.device))
    args = _args()
    models.prepare_for_inference(args)
    assert calls == ['cpu']
    args.device = 'auto'
    models.prepare_for_inference(args)
    assert calls == ['cpu', 'auto']
    args.dry_run = True
    models.prepare_for_inference(args)
    assert calls == ['cpu', 'auto']
    args.device, args.dry_run = 'cpu', False
    monkeypatch.setenv('RANK', '1')
    models.prepare_for_inference(args)
    assert calls == ['cpu', 'auto', 'cpu']


def test_filesystem_permission_errors_do_not_trigger_online_retry(monkeypatch):
    _allow_online(monkeypatch)
    calls = []
    def denied(identifier, revision, local_only):
        calls.append(local_only)
        raise PermissionError('cache is unreadable')
    monkeypatch.setattr(models, '_prepare_generator', denied)
    with pytest.raises(RuntimeError, match='cache is unreadable') as failure:
        models.prepare_generator(_args(local_only=False))
    assert calls == [True]
    assert isinstance(failure.value.__cause__, PermissionError)


@pytest.mark.parametrize('role', ['model', 'evaluator_a'])
def test_actual_loaders_use_prepared_paths_offline(monkeypatch, role):
    import diffusers
    import transformers
    from utils.sampler import Sampler
    from utils.tasks import Scorer
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        raise FileNotFoundError('synthetic incomplete model')
    args = _args(local_only=False)
    args.scheduler, args.ddim_eta, args.precision = 'ddim', 0, 'float32'
    if role == 'model':
        monkeypatch.setattr(models, 'prepare_generator', lambda args: '/prepared/generator')
        monkeypatch.setattr(diffusers.StableDiffusionPipeline, 'from_pretrained', load)
        action = lambda: Sampler.load(args, 'cpu')
    else:
        monkeypatch.setattr(models, 'prepare_evaluator', lambda args, role: '/prepared/evaluator')
        monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', load)
        action = lambda: Scorer.load(args, 'a', 'cpu')
    with pytest.raises(RuntimeError, match=f"--{role.replace('_', '-')}-id"):
        action()
    assert len(calls) == 1
    assert calls[0][0].startswith('/prepared/')
    assert calls[0][1]['local_files_only'] is True
    assert args.model_id == 'owner/model'


def test_preparation_bar_precedes_model_library_imports(monkeypatch, capsys):
    from utils import progress
    calls = []

    def quiet():
        assert 'prepare: initializing model file checks' in capsys.readouterr().err
        calls.append('imports')

    monkeypatch.setattr(progress, 'quiet_library_progress', quiet)
    monkeypatch.setattr(models, 'prepare_generator', lambda args: '/fixture/generator')
    monkeypatch.setattr(models, 'prepare_evaluator', lambda args, role: '/fixture/' + role)
    monkeypatch.setattr(models, 'validate_preprocessors', lambda paths: calls.append('processors'))
    assert models.prepare_models(_args())['model'] == '/fixture/generator'
    assert calls == ['imports', 'processors']
