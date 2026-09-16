"""Run CPU controls and offline SM80 compilation in the exact reused image."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image', default='deepseek-v4.1-flash-a100:20260913-r1')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    common = ['sudo', '-n', 'docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
              '--entrypoint', '/usr/bin/python3', '-e', 'PYTHONDONTWRITEBYTECODE=1',
              '-v', f'{ROOT / "deploy"}:/deploy:ro', '-v', f'{ROOT / "build"}:/build:ro']
    plan = subprocess.check_output(common+[a.image, '/deploy/scripts/performance.py', '--host-deploy', str(ROOT/'deploy')])
    mounts = [s.decode() for s in plan.split(b'\0') if s]
    checks = [(['/deploy/tests/performance_runtime_cpu.py'], 'runtime-cpu.log'),
              (['/deploy/tests/performance_r11_cpu.py'], 'r11-cpu.log'),
              (['/build/compile_r11_kernels.py', '--output', '/results/sm80-compile.json'], 'sm80-compile.log')]
    for command, logfile in checks:
        print('Checking '+command[0], flush=True)
        with (a.output/logfile).open('w') as out:
            result = subprocess.run(common+mounts+['-v', f'{a.output.resolve()}:/results',
                                    a.image, '-u', *command], stdout=out, stderr=subprocess.STDOUT, timeout=600)
        print((a.output/logfile).read_text(), flush=True)
        result.check_returncode()


if __name__ == '__main__':
    main()
