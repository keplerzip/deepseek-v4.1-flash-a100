"""Run CPU controls and offline SM80 compilation in the exact reused image."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image', default='deepseek-v4.1-flash-a100:20260913-r1')
    p.add_argument('--checks', nargs='*', help='Optional logfile stems to resume a failed check')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.checks is None or 'frontend-mount' in a.checks:
        # Reproduce the private (0600) files installed on the target. A
        # fixed nobody UID could not read them and caused R1.7 to roll back.
        with tempfile.TemporaryDirectory(prefix='dsv41-frontend-mount-') as tmp:
            source = Path(tmp) / 'frontend.py'
            config = Path(tmp) / 'access.json'
            shutil.copyfile(ROOT / 'deploy/scripts/local_access_frontend.py', source)
            config.write_text('{}\n')
            source.chmod(0o600)
            config.chmod(0o600)
            command = ['sudo', '-n', 'docker', 'run', '--rm', '--pull', 'never',
                       '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                       '--security-opt', 'no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
                       '-e', 'PYTHONDONTWRITEBYTECODE=1',
                       '--mount', f'type=bind,src={source},dst=/frontend.py,readonly',
                       '--mount', f'type=bind,src={config},dst=/access.json,readonly',
                       '--entrypoint', '/usr/bin/python3', a.image, '-c',
                       'from pathlib import Path; compile(Path("/frontend.py").read_bytes(), "/frontend.py", "exec"); Path("/access.json").read_bytes()']
            with (a.output/'frontend-mount.log').open('w') as out:
                subprocess.run(command, check=True, stdout=out, stderr=subprocess.STDOUT)
            print('Private frontend bind mounts: PASS', flush=True)
    common = ['sudo', '-n', 'docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
              '--entrypoint', '/usr/bin/python3', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=2',
              '-v', f'{ROOT / "deploy"}:/deploy:ro', '-v', f'{ROOT / "build"}:/build:ro']
    plan = subprocess.check_output(common+[a.image, '/deploy/scripts/performance.py', '--host-deploy', str(ROOT/'deploy')])
    mounts = [s.decode() for s in plan.split(b'\0') if s]
    checks = [(['/deploy/tests/performance_runtime_cpu.py'], 'runtime-cpu.log'),
              (['/deploy/tests/performance_ep8_reference_cpu.py'], 'ep8-reference-cpu.log'),
              (['/deploy/tests/performance_r11_cpu.py'], 'r11-cpu.log'),
              (['/deploy/tests/performance_r12_cpu.py'], 'r12-cpu.log'),
              (['/deploy/tests/performance_r13_cpu.py'], 'r13-cpu.log'),
              (['/deploy/tests/performance_r14_cpu.py'], 'r14-cpu.log'),
              (['/deploy/tests/performance_r14_interpreter.py'], 'r14-interpreter.log'),
              (['/deploy/tests/vision_schema.py'], 'vision-schema.log'),
              (['/deploy/tests/guard_schema.py', '--guard-path', '/deploy/overrides/dsv41_guard.py'], 'guard-schema.log'),
              (['/build/compile_r11_kernels.py', '--output', '/results/sm80-compile.json'], 'sm80-compile.log'),
              (['/build/compile_r12_kernels.py', '--output', '/results/r12-sm80-compile.json'], 'r12-sm80-compile.log'),
              (['/build/compile_r13_kernels.py', '--output', '/results/r13-sm80-compile.json'], 'r13-sm80-compile.log'),
              (['/build/compile_r14_kernels.py', '--output', '/results/r14-sm80-compile.json'], 'r14-sm80-compile.log')]
    for command, logfile in checks:
        if a.checks is not None and Path(logfile).stem not in a.checks:
            continue
        print('Checking '+command[0], flush=True)
        with (a.output/logfile).open('w') as out:
            result = subprocess.run(common+mounts+['-v', f'{a.output.resolve()}:/results',
                                    a.image, '-u', *command], stdout=out, stderr=subprocess.STDOUT, timeout=600)
        print((a.output/logfile).read_text(), flush=True)
        result.check_returncode()


if __name__ == '__main__':
    main()
