"""Exercise only the shell orchestration with a logging Python stand-in."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGES = ['prepare', 'collect', 'fit', 'intervene', 'controls', 'diagnose', 'policy', 'plot']


@pytest.fixture
def launcher(tmp_path):
    executable = tmp_path / 'python executable'
    executable.write_text(f'#!{sys.executable}\n' + '''import json
import os
from pathlib import Path
import sys

with Path(os.environ['LAUNCHER_TEST_LOG']).open('a') as handle:
    handle.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')
if sys.argv[2] == os.environ.get('LAUNCHER_TEST_FAIL'):
    sys.exit(23)
''')
    executable.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    env = dict(os.environ, PYTHON=str(executable), LAUNCHER_TEST_LOG=str(log))
    env.pop('LAUNCHER_TEST_FAIL', None)

    def invoke(args=(), fail=None):
        if fail:
            env['LAUNCHER_TEST_FAIL'] = f'exps.{fail}'
        result = subprocess.run([str(ROOT / 'run_pilot.sh'), *args], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=10)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return invoke


def test_launcher_order_and_exact_argument_forwarding(launcher):
    arguments = ['--config', '/tmp/config with spaces.json', '--device', 'cpu',
                 '--local-files-only', '--no-resume', '--overwrite', '--run-id', 'test01',
                 '--negative-prompt', 'literal "$HOME"; $(touch should-not-exist)']
    result, calls = launcher(arguments)
    assert result.returncode == 0, result.stderr
    assert [call['argv'][1] for call in calls] == [f'exps.{stage}' for stage in STAGES]
    expected = ['--config', 'configs/demo.json', '--device', 'auto', *arguments]
    assert all(call['argv'][0] == '-m' and call['argv'][2:] == expected for call in calls)
    assert all(call['cwd'] == str(ROOT) for call in calls)


@pytest.mark.parametrize('stage', ['prepare', 'fit', 'plot'])
def test_launcher_stops_on_first_failure(launcher, stage):
    result, calls = launcher(fail=stage)
    assert result.returncode == 23
    assert [call['argv'][1] for call in calls] == [
        f'exps.{name}' for name in STAGES[:STAGES.index(stage) + 1]]


@pytest.mark.parametrize('flag', ['--help', '-h'])
def test_launcher_help_never_invokes_experiments(launcher, flag):
    result, calls = launcher([flag])
    assert result.returncode == 0
    assert 'Usage:' in result.stdout
    assert calls == []


def test_launcher_forwards_dry_run_to_every_stage(launcher):
    result, calls = launcher(['--dry-run'])
    assert result.returncode == 0, result.stderr
    assert len(calls) == len(STAGES)
    assert all(call['argv'][-1] == '--dry-run' for call in calls)
