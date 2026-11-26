"""Keep a checkpointed coevolution run alive across Slurm allocation failures."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=60).stdout.strip()


def submit(script, *options):
    # This cluster rewrites exclusions during submission. Hold servers until
    # the scheduler confirms their exclusions, so broken GPUs cannot be assigned.
    exclusions = None
    if Path(script).resolve() == ROOT / 'server.sh':
        match = re.search(r'^#SBATCH --exclude=(\S+)', Path(script).read_text(), re.M)
        if match:
            exclusions = match.group(1)
            options = ('--hold', *options)
    result = command('sbatch', '--parsable', '--chdir', str(ROOT), '--export=ALL', *options, str(script))
    job = result.split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected sbatch response: {result}')
    if exclusions:
        try:
            command('scontrol', 'update', 'JobId=' + job, 'ExcNodeList=' + exclusions)
            record = command('scontrol', 'show', 'job', '-o', job)
            match = re.search(r'(?:^|\s)ExcNodeList=(\S+)', record)
            if not match:
                raise RuntimeError('Scheduler did not report exclusions for held server ' + job)
            actual = set(command('scontrol', 'show', 'hostnames', match.group(1)).splitlines())
            required = set(command('scontrol', 'show', 'hostnames', exclusions).splitlines())
            if not required <= actual:
                raise RuntimeError('Scheduler dropped required server exclusions: ' + record)
            command('scontrol', 'release', job)
        except Exception:
            command('scancel', job)
            raise
    return job


def job_state(job):
    if not job:
        return 'MISSING'
    try:
        queued = command('squeue', '-h', '-j', job, '-o', '%T')
    except subprocess.CalledProcessError as error:
        if 'invalid job id' not in (error.stderr or '').lower():
            raise
        queued = ''
    if queued:
        return queued.splitlines()[0]
    records = command('sacct', '-n', '-X', '-j', job, '--format=State', '--parsable2')
    # Accounting may lag behind squeue; never duplicate an unknown job.
    return records.splitlines()[0].split('|')[0].split()[0].rstrip('+') if records else 'UNKNOWN'


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def latest_generation(run):
    generations = [int(match.group(1)) for path in run.glob('checkpoint_gen_*.pkl')
                   if (match := re.fullmatch(r'checkpoint_gen_(\d+)\.pkl', path.name))]
    return max(generations, default=0)


def write_scripts(directory, run):
    import yaml
    template = yaml.safe_load((ROOT / 'slurm-config/slurm_config.yaml').read_text())['islands_bash_script']
    pattern = r'uv run python run_improved\.py\s+--checkpoints \{\}\s+--global_path \{\}\s+--llm_model \{\}\s+--prompt_group \{\}'
    worker, count = re.subn(pattern, lambda _: 'exec uv run python run_coevolution.py --checkpoints ' + shlex.quote(str(run)), template)
    if count != 1:
        raise ValueError('Expected one worker command in islands_bash_script')
    (directory / 'worker.sbatch').write_text(worker)
    python = ROOT / '.venv/bin/python'
    supervisor = ('#!/bin/bash\n#SBATCH --job-name=LLMGE_watch\n#SBATCH --time=4:00:00\n'
                  '#SBATCH --nodes=1\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=1G\n'
                  f'#SBATCH --output={directory}/supervisor-%j.out\n'
                  f'exec {shlex.quote(str(python))} {shlex.quote(str(ROOT / "persistent_run.py"))} supervise {shlex.quote(str(run))}\n')
    (directory / 'supervisor.sbatch').write_text(supervisor)


def tick(run, directory, state):
    target = directory / 'target.json'
    if target.exists():
        state['generations'] = json.loads(target.read_text())['generations']
    if (directory / 'STOP').exists() or latest_generation(run) >= state['generations']:
        state['status'] = 'stopped' if (directory / 'STOP').exists() else 'completed'
        save(directory / 'state.json', state)
        for key in ('server', 'worker', 'successor'):
            if state.get(key):
                command('scancel', state[key])
        return False
    for role, script in [('server', ROOT / 'server.sh'), ('worker', directory / 'worker.sbatch')]:
        status = job_state(state.get(role))
        if status in ACTIVE or status == 'UNKNOWN':
            continue
        print(f'{role} {state.get(role)}: {status}; submitting replacement', flush=True)
        state[role] = submit(script)
        save(directory / 'state.json', state)
    return True


def supervise(run, directory):
    with (directory / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((directory / 'state.json').read_text())
        if state['status'] != 'running' or (directory / 'STOP').exists():
            return
        os.environ.update(LLMGE_SERVER_ONLY='1', LLMGE_PERSISTENT='1',
                          LLMGE_HOSTNAME_FILE=str(directory / 'hostname.log'),
                          LLMGE_PORT=str(state['port']))
        current = os.environ['SLURM_JOB_ID']
        # Queue renewal before starting work: survives walltime and supervisor crashes.
        state['supervisor'] = current
        successor = state.get('successor')
        if not successor or successor == current or job_state(successor) not in ACTIVE | {'UNKNOWN'}:
            state['successor'] = submit(directory / 'supervisor.sbatch', '--dependency=afterany:' + current)
        save(directory / 'state.json', state)
        while True:
            try:
                if not tick(run, directory, state):
                    return
            except (subprocess.SubprocessError, OSError, RuntimeError) as error:
                print(f'Scheduler operation failed; retrying: {error}', flush=True)
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop', 'supervise'])
    parser.add_argument('run', type=Path)
    parser.add_argument('--generations', type=int, default=5, help='Stop after this completed checkpoint generation')
    parser.add_argument('--port', type=int, default=12676)
    args = parser.parse_args()
    from run_coevolution import resolve_run_directory
    run = resolve_run_directory(args.run)
    directory = run / '.persistent'
    if args.action == 'supervise':
        supervise(run, directory)
    elif args.action == 'start':
        if args.generations < 1 or not 1 <= args.port <= 65535:
            parser.error('generations must be positive and port must be 1..65535')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'state.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        # A durable label is not evidence of a live supervisor. Check both
        # allocations, since the queued successor may be taking over.
        if previous.get('status') == 'running' and not (directory / 'STOP').exists():
            states = {role: job_state(previous.get(role)) for role in ('supervisor', 'successor')}
            if any(status in ACTIVE | {'UNKNOWN'} for status in states.values()):
                save(directory / 'target.json', {'generations': args.generations})
                print(f"Run already supervised; target updated to generation {args.generations}. "
                      f"Supervisor: {previous.get('supervisor')} ({states['supervisor']}); "
                      f"successor: {previous.get('successor')} ({states['successor']}).")
                return
        with (directory / 'lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                save(directory / 'target.json', {'generations': args.generations})
                print(f'Run already supervised; target updated to generation {args.generations}.')
                return
            write_scripts(directory, run)
            (directory / 'STOP').unlink(missing_ok=True)
            save(directory / 'target.json', {'generations': args.generations})
            state = dict(status='running', generations=args.generations,
                         port=previous.get('port', args.port))
            # Adopt existing children if the supervisor chain was cancelled.
            for role in ('server', 'worker'):
                if previous.get(role):
                    state[role] = previous[role]
            save(path, state)
            try:
                state['supervisor'] = submit(directory / 'supervisor.sbatch')
            except Exception:
                state['status'] = 'submission_failed'
                save(path, state)
                raise
            save(path, state)
            print(json.dumps(state, indent=2))
    elif args.action == 'status':
        state = json.loads((directory / 'state.json').read_text())
        target = directory / 'target.json'
        if target.exists():
            state['generations'] = json.loads(target.read_text())['generations']
        print(json.dumps(state, indent=2))
        print(f'Latest checkpoint generation: {latest_generation(run)}')
    else:
        # Write durable intent first so a successor cannot restart a stopped run.
        (directory / 'STOP').touch()
        state = json.loads((directory / 'state.json').read_text())
        for role in ('successor', 'server', 'worker', 'supervisor'):
            if state.get(role):
                command('scancel', state[role])
        state['status'] = 'stopped'
        save(directory / 'state.json', state)
        print('Stopped supervisor, successor, server, and worker. Checkpoints preserved.')


if __name__ == '__main__':
    main()
