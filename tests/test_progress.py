"""Bounded progress checks; no pretrained models, devices, or research work."""
import pytest

from utils.config import parse_args
from utils.data import scientific_fingerprint
from utils.progress import progress, status


def test_disabled_progress_and_status_are_silent(capsys):
    args = parse_args("collect", ["--no-progress"])
    seen = []
    with progress(args, [1, 2], desc="Disabled work", total=5, initial=3) as bar:
        for item in bar:
            seen.append(item)
            bar.set_postfix(stage=item)
    status(args, "Hidden status")
    assert seen == [1, 2]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_progress_is_visible_when_stderr_is_captured_and_counts_resume(capsys):
    args = parse_args("collect", ["--progress-mininterval", "0"])
    with progress(args, [1, 2], desc="Resume cached work", total=5, initial=3,
                  unit="sample") as bar:
        assert list(bar) == [1, 2]
    output = capsys.readouterr().err
    assert "Resume cached work" in output
    assert "5/5" in output
    assert "100%" in output


def test_progress_settings_preserve_scientific_fingerprint():
    default = parse_args("collect", [])
    quiet = parse_args("collect", ["--no-progress", "--progress-mininterval", "0"])
    refreshed = parse_args("collect", ["--progress-mininterval", "3.5"])
    assert scientific_fingerprint(default) == scientific_fingerprint(quiet)
    assert scientific_fingerprint(default) == scientific_fingerprint(refreshed)


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_invalid_progress_refresh_interval_rejected(value, capsys):
    with pytest.raises(SystemExit) as error:
        parse_args("collect", ["--progress-mininterval", value])
    assert error.value.code == 2
    assert "progress_mininterval" in capsys.readouterr().err


def test_overall_progress_pins_one_total_across_nested_phases_and_ranks(tmp_path, monkeypatch, capsys):
    """Unequal rank shards and top-level child phases cannot reset overall ETA."""
    from types import SimpleNamespace
    import torch
    from utils.progress import Reporter, _ReportedProgress

    args = parse_args('controls', ['--progress-mininterval', '0'])
    monkeypatch.setattr(Reporter, '_monitor', lambda self: None)
    reporters = [Reporter(args, SimpleNamespace(rank=rank, world_size=2,
                 is_main=rank == 0, device=torch.device('cpu')), tmp_path)
                 for rank in range(2)]
    bars = [_ReportedProgress(reporter, None, 'Controls: whole command', total,
            initial, 'work', 0, True, overall=True)
            for reporter, total, initial in zip(reporters, [20, 30], [5, 7])]
    try:
        reporters[0].render()
        first_display = reporters[0].bar
        assert first_display.total == 50 and first_display.initial == 12
        for description in ('Fit wrapped probes', 'Evaluate native readouts', 'Load frozen replicas'):
            with _ReportedProgress(reporters[0], None, description, 2, 0,
                                   'item', 0, True) as child:
                child.update()
                bars[0].update(2)
                reporters[0].publish(force=True)
                reporters[0].render()
                assert reporters[0].bar is first_display
                assert len(reporters[0].phases) == 1
                assert reporters[0].bar.total == 50
                assert description in reporters[0].snapshot()['detail']
        bars[1].update(23)
        bars[1].close()  # A faster worker finishing does not hide the slower one.
        reporters[0].render()
        assert reporters[0].bar.n == 41
        assert reporters[0].bar.initial == 12
        bars[0].update(9)
        bars[0].close()
        reporters[0].render()
        assert reporters[0].bar.n == reporters[0].bar.total == 50
        assert all(len(reporter.phases) == 1 for reporter in reporters)
    finally:
        for reporter in reporters:
            reporter.close()
    output = capsys.readouterr().err
    assert 'Controls: whole command' in output
    assert 'estimated' in output and '50/50' in output
    assert 'cpu/r0' in output and 'cpu/r1' in output
