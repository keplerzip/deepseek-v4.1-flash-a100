"""Independent process deadline and rank heartbeat; imports no Torch or CUDA."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def states(directory):
    rows = []
    for rank in range(8):
        try:
            row = json.loads((directory/f'progress-rank-{rank}.json').read_text())
            rows.append(f"r{rank}={row['phase']}:{row['status']}")
        except (OSError, ValueError, KeyError):
            rows.append(f'r{rank}=not_started')
    return ' '.join(rows)


def terminate_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The launcher may have exited before its workers: kill the remaining group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--heartbeat', type=int, default=15)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.seconds > 0 and args.heartbeat > 0
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    assert command, 'Supply a Python command'
    directory = Path(os.environ['PROBE_OUTPUT'])
    directory.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    proc = subprocess.Popen([sys.executable, '-u', *command], start_new_session=True)
    report = {'status': 'RUNNING', 'deadline_seconds': args.seconds, 'child_pid': proc.pid}
    rc = 1
    try:
        while True:
            remaining = args.seconds-(time.monotonic()-began)
            if remaining <= 0:
                report.update(status='TIMEOUT', ranks=states(directory))
                print('[GPU probe TIMEOUT] '+json.dumps(report), flush=True)
                terminate_group(proc)
                rc = 124
                break
            try:
                code = proc.wait(timeout=min(args.heartbeat, remaining))
                rc = code if code >= 0 else 128-code
                report.update(status='PROCESS_EXITED' if code == 0 else 'FAILED', exit_code=rc)
                if code:
                    print('[GPU probe FAILED] '+states(directory), flush=True)
                    terminate_group(proc)
                break
            except subprocess.TimeoutExpired:
                print(f'[GPU heartbeat {int(time.monotonic()-began)}s] '+states(directory), flush=True)
    finally:
        if proc.poll() is None:
            terminate_group(proc)
        report['elapsed_seconds'] = round(time.monotonic()-began, 2)
        (directory/'supervisor.json').write_text(json.dumps(report, indent=2)+'\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
