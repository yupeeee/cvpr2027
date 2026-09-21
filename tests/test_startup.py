"""Check main-function dispatch without starting experiments or inspecting GPUs."""
import importlib

import pytest

from utils.config import parse_args
from utils.data import scientific_fingerprint


@pytest.mark.parametrize('name', ['prepare', 'collect', 'fit', 'intervene', 'controls', 'diagnose', 'policy', 'plot'])
def test_auto_dry_run_exits_before_cuda_cache_or_models(name, monkeypatch, capsys):
    import torch
    from utils import models

    def unexpected(*args, **kwargs):
        pytest.fail('Dry-run touched model preparation or CUDA')

    monkeypatch.setattr(torch.cuda, 'is_available', unexpected)
    monkeypatch.setattr(models, 'prepare_models', unexpected)
    module = importlib.import_module('exps.' + name)
    module.main(['--config', 'configs/demo.json', '--device', 'auto', '--dry-run'])
    assert 'no models' in capsys.readouterr().out


@pytest.mark.parametrize('name', ['collect', 'fit', 'intervene', 'controls', 'diagnose', 'policy'])
def test_main_prepares_before_launch_and_does_not_initialize_parent_group(name, monkeypatch):
    from utils import models
    events = []
    module = importlib.import_module('exps.' + name)
    from utils import cache
    monkeypatch.setattr(cache, 'skip_completed', lambda *args: False)
    if name == 'controls':
        monkeypatch.setattr(module, 'skip_completed', lambda *args: False)
        monkeypatch.setattr(module, 'stage_complete', lambda *args: False)
    monkeypatch.setattr(models, 'prepare_for_inference', lambda args: events.append('prepare'))

    def launched(args, command, argv):
        assert args.device == 'auto' and command == name
        assert argv == ['--device', 'auto']
        events.append('launch')
        return True

    monkeypatch.setattr(module, 'launch_if_needed', launched)
    monkeypatch.setattr(module, 'Distributed', lambda args: pytest.fail('parent created a process group'))
    module.main(['--device', 'auto'])
    assert events == (['launch'] if name == 'fit' else ['prepare', 'launch'])


def test_device_and_download_switches_do_not_invalidate_existing_run():
    offline = parse_args('collect', ['--device', 'cuda', '--local-files-only'])
    requested = parse_args('collect', ['--device', 'auto', '--no-local-files-only'])
    assert scientific_fingerprint(offline) == scientific_fingerprint(requested)
    assert parse_args('collect', []).local_files_only is True
    assert parse_args('collect', ['--config', 'configs/demo.json']).local_files_only is True


def test_prepare_restart_defers_deep_validation_and_models(tmp_path, monkeypatch, capsys):
    from exps.prepare import main
    from utils import cache, models
    from utils import data
    import torch

    # Even an incomplete manifest is only a restart hint; collect owns validation.
    run = tmp_path / 'logs' / 'restart'
    run.mkdir(parents=True)
    manifest = run / 'manifest.json'
    manifest.write_text('{}')
    original = manifest.stat().st_mtime_ns

    def unexpected(*args, **kwargs):
        pytest.fail('Prepare scanned experiment tensors, prepared models, or inspected CUDA')

    monkeypatch.setattr(cache, 'stage_complete', unexpected)
    monkeypatch.setattr(cache, 'load_tensor', unexpected)
    monkeypatch.setattr(data, 'load_tensor', unexpected)
    monkeypatch.setattr(models, 'prepare_models', unexpected)
    monkeypatch.setattr(torch.cuda, 'is_available', unexpected)
    main(['--logs-dir', str(tmp_path / 'logs'), '--run-id', 'restart', '--device', 'auto'])
    assert 'deferred to the stages' in capsys.readouterr().err
    assert manifest.stat().st_mtime_ns == original


@pytest.mark.parametrize('existing', [False, True])
def test_prepare_fresh_or_overwrite_reports_before_preflight(tmp_path, monkeypatch, capsys, existing):
    from exps.prepare import main
    from utils import models
    argv = ['--logs-dir', str(tmp_path / 'logs'), '--run-id', 'startup']
    if existing:
        run = tmp_path / 'logs' / 'startup'
        run.mkdir(parents=True)
        (run / 'manifest.json').write_text('{}')
        argv.append('--overwrite')
    calls = []

    def prepare(args):
        assert 'checking model files and tokenizer dependencies' in capsys.readouterr().err
        calls.append(True)
        return {'model': '/fixture/model'}

    monkeypatch.setattr(models, 'prepare_models', prepare)
    main(argv)
    assert calls == [True]
    assert 'models_ready' in capsys.readouterr().out
