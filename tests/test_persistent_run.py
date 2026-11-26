"""Recovery behavior without contacting Slurm or loading model weights."""
import json
from pathlib import Path
import subprocess

import pytest
import persistent_run as supervisor
from src import llm_transport as transport


@pytest.mark.parametrize('terminal', ['TIMEOUT', 'FAILED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'CANCELLED', 'COMPLETED'])
def test_dead_server_replaced_without_duplicate_worker(tmp_path, monkeypatch, terminal):
    state = dict(generations=5, server='10', worker='11')
    monkeypatch.setattr(supervisor, 'job_state', lambda job: terminal if job == '10' else 'RUNNING')
    submitted = []
    monkeypatch.setattr(supervisor, 'submit', lambda script: submitted.append(script) or '12')
    assert supervisor.tick(tmp_path, tmp_path, state)
    assert submitted == [supervisor.ROOT / 'server.sh']
    assert json.loads((tmp_path / 'state.json').read_text())['server'] == '12'
    assert state['worker'] == '11'


def test_unknown_accounting_does_not_duplicate_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, 'job_state', lambda job: 'UNKNOWN')
    monkeypatch.setattr(supervisor, 'submit', lambda *args: pytest.fail('duplicate submission'))
    assert supervisor.tick(tmp_path, tmp_path, dict(generations=5, server='1', worker='2'))


@pytest.mark.parametrize('stop', [True, False])
def test_completion_and_stop_cancel_allocations(tmp_path, monkeypatch, stop):
    if stop:
        (tmp_path / 'STOP').touch()
    else:
        (tmp_path / 'checkpoint_gen_5.pkl').touch()
        (tmp_path / 'checkpoint_gen_99.pkl.tmp').touch()
    cancelled = []
    monkeypatch.setattr(supervisor, 'command', lambda *args: cancelled.append(args))
    state = dict(generations=5, server='1', worker='2', successor='3')
    assert not supervisor.tick(tmp_path, tmp_path, state)
    assert cancelled == [('scancel', '1'), ('scancel', '2'), ('scancel', '3')]
    assert state['status'] == ('stopped' if stop else 'completed')


def test_retry_rereads_hostname_after_disconnect(tmp_path, monkeypatch):
    monkeypatch.setenv('LLMGE_PERSISTENT', '1')
    host = tmp_path / 'hostname'
    host.write_text('old')
    calls = []
    success = type('Response', (), {'status_code': 200})()

    def post(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            host.write_text('replacement')
            raise transport.requests.exceptions.ConnectionError('server died')
        return success

    monkeypatch.setattr(transport.requests, 'post', post)
    monkeypatch.setattr(transport.time, 'sleep', lambda seconds: None)
    assert transport.post_generation(lambda: host.read_text(), {'prompt': 'preserve me'}, 30) is success
    assert calls == ['old', 'replacement']


def test_unsupervised_requests_keep_fail_fast_behavior(monkeypatch):
    monkeypatch.delenv('LLMGE_PERSISTENT', raising=False)
    def fail(*args, **kwargs):
        raise transport.requests.exceptions.ConnectionError('offline')
    monkeypatch.setattr(transport.requests, 'post', fail)
    with pytest.raises(transport.requests.exceptions.ConnectionError):
        transport.post_generation(lambda: 'http://offline', {}, 30)


def test_generated_jobs_use_coevolution_and_renew_supervisor(tmp_path):
    run = tmp_path / 'run with spaces'
    supervisor.write_scripts(tmp_path, run)
    worker = (tmp_path / 'worker.sbatch').read_text()
    assert 'exec uv run python run_coevolution.py --checkpoints' in worker
    assert str(run) in worker
    assert 'run_improved.py' not in worker
    for name in ('worker.sbatch', 'supervisor.sbatch'):
        subprocess.run(['bash', '-n', str(tmp_path / name)], check=True)


def test_successor_queued_before_monitoring(tmp_path, monkeypatch):
    supervisor.save(tmp_path / 'state.json', dict(status='running', port=12676, generations=5))
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    events = []
    def submit(script, *options):
        events.append(options)
        return '43'
    monkeypatch.setattr(supervisor, 'submit', submit)
    monkeypatch.setattr(supervisor, 'tick', lambda *args: events.append('tick') or False)
    supervisor.supervise(tmp_path, tmp_path)
    assert events == [('--dependency=afterany:42',), 'tick']
    assert json.loads((tmp_path / 'state.json').read_text())['successor'] == '43'


def test_squeue_purged_job_uses_accounting(monkeypatch):
    def command(*args):
        if args[0] == 'squeue':
            raise subprocess.CalledProcessError(1, args, stderr='slurm_load_jobs error: Invalid job id specified')
        return 'TIMEOUT|'
    monkeypatch.setattr(supervisor, 'command', command)
    assert supervisor.job_state('42') == 'TIMEOUT'


def test_retry_service_unavailable_then_success(monkeypatch):
    monkeypatch.setenv('LLMGE_PERSISTENT', '1')
    closed = []
    unavailable = type('Response', (), {'status_code': 503, 'close': lambda self: closed.append(True)})()
    success = type('Response', (), {'status_code': 200})()
    responses = iter([unavailable, success])
    monkeypatch.setattr(transport.requests, 'post', lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(transport.time, 'sleep', lambda seconds: None)
    assert transport.post_generation(lambda: 'http://server', {}, 30) is success
    assert closed == [True]


def test_interrupted_checkpoint_write_preserves_previous_file(tmp_path):
    import ast
    import os
    import pickle
    tree = ast.parse((supervisor.ROOT / 'run_improved.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'save_checkpoint')
    namespace = dict(os=os, pickle=pickle)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'run_improved.py', 'exec'), namespace)
    checkpoint = tmp_path / 'checkpoint_gen_1.pkl'
    checkpoint.write_bytes(pickle.dumps({'previous': True}))
    class Broken:
        def __reduce__(self):
            raise RuntimeError('interrupted serialization')
    with pytest.raises(RuntimeError):
        namespace['save_checkpoint'](1, folder_name=str(tmp_path), checkpoint_data={'broken': Broken()})
    assert pickle.loads(checkpoint.read_bytes()) == {'previous': True}


def test_requeued_supervisor_reuses_pending_successor(tmp_path, monkeypatch):
    supervisor.save(tmp_path / 'state.json', dict(status='running', port=12676, generations=5,
                                                 supervisor='42', successor='43'))
    monkeypatch.setenv('SLURM_JOB_ID', '42')
    monkeypatch.setattr(supervisor, 'job_state', lambda job: 'PENDING')
    monkeypatch.setattr(supervisor, 'submit', lambda *args: pytest.fail('duplicate successor'))
    monkeypatch.setattr(supervisor, 'tick', lambda *args: False)
    supervisor.supervise(tmp_path, tmp_path)


def test_start_repairs_stale_supervisor_preserving_children(tmp_path, monkeypatch, capsys):
    import sys
    directory = tmp_path / '.persistent'
    directory.mkdir()
    supervisor.save(directory / 'state.json', dict(status='running', generations=5,
                    port=12676, supervisor='1', successor='2', server='3', worker='4'))
    monkeypatch.setattr(supervisor, 'job_state', lambda job: 'CANCELLED')
    monkeypatch.setattr(supervisor, 'write_scripts', lambda *args: None)
    monkeypatch.setattr(supervisor, 'submit', lambda *args: '5')
    monkeypatch.setattr(sys, 'argv', ['persistent_run.py', 'start', str(tmp_path), '--generations', '10'])
    supervisor.main()
    state = json.loads((directory / 'state.json').read_text())
    assert state['supervisor'] == '5'
    assert state['server'] == '3' and state['worker'] == '4'
    assert state['generations'] == 10


def test_start_live_run_updates_target_without_duplicate_submission(tmp_path, monkeypatch, capsys):
    import sys
    directory = tmp_path / '.persistent'
    directory.mkdir()
    supervisor.save(directory / 'state.json', dict(status='running', generations=5,
                    port=12676, supervisor='1', successor='2'))
    monkeypatch.setattr(supervisor, 'job_state', lambda job: 'RUNNING')
    monkeypatch.setattr(supervisor, 'submit', lambda *args: pytest.fail('duplicate supervisor'))
    monkeypatch.setattr(sys, 'argv', ['persistent_run.py', 'start', str(tmp_path), '--generations', '10'])
    supervisor.main()
    assert json.loads((directory / 'target.json').read_text())['generations'] == 10
    assert 'target updated' in capsys.readouterr().out


def test_live_target_extension_applies_on_next_tick(tmp_path, monkeypatch):
    (tmp_path / 'checkpoint_gen_5.pkl').touch()
    supervisor.save(tmp_path / 'target.json', {'generations': 10})
    monkeypatch.setattr(supervisor, 'job_state', lambda job: 'RUNNING')
    state = dict(status='running', generations=5, server='1', worker='2')
    assert supervisor.tick(tmp_path, tmp_path, state)
    assert state['generations'] == 10


def test_server_is_held_until_exclusions_are_verified(monkeypatch):
    calls = []
    def command(*args):
        calls.append(args)
        if args[0] == 'sbatch':
            return '42'
        if args[:3] == ('scontrol', 'show', 'job'):
            return 'JobId=42 ExcNodeList=blocked'
        if args[:3] == ('scontrol', 'show', 'hostnames'):
            return 'node1\nnode2'
        return ''
    monkeypatch.setattr(supervisor, 'command', command)
    assert supervisor.submit(supervisor.ROOT / 'server.sh') == '42'
    assert '--hold' in calls[0]
    assert calls[-1] == ('scontrol', 'release', '42')


def test_unverified_exclusion_cancels_held_server(monkeypatch):
    calls = []
    def command(*args):
        calls.append(args)
        if args[0] == 'sbatch':
            return '42'
        if args[:3] == ('scontrol', 'show', 'job'):
            return 'JobId=42 ExcNodeList=(null)'
        if args[:3] == ('scontrol', 'show', 'hostnames'):
            return '' if args[-1] == '(null)' else 'node1'
        return ''
    monkeypatch.setattr(supervisor, 'command', command)
    with pytest.raises(RuntimeError, match='dropped'):
        supervisor.submit(supervisor.ROOT / 'server.sh')
    assert calls[-1] == ('scancel', '42')
    assert ('scontrol', 'release', '42') not in calls
