"""Exercise the actual cumulative installer against historical source trees.

Uses the installed original image with network=none. No GPU or real service
is touched. The failed-restart case supplies isolated old stop/start stubs;
the updated start fails on a deliberately absent model before GPU operations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'deepseek-v4.1-flash-a100:20260913-r1'


def files(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mode & 0o777)
            for p in root.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(root).parts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    version_expected = json.loads((ROOT / 'deploy/manifests/update.json').read_text())
    for ref, restart_failure in (('v1.0.0', False), ('d7c7d877a97bb0fc77acf90b3e6e35e0e78db1d2', False), ('v1.1.0', False), ('v1.0.0', True)):
        with tempfile.TemporaryDirectory(prefix='r11-upgrade-') as temp:
            tmp = Path(temp)
            old = subprocess.check_output(['git','archive',ref,'deploy'], cwd=ROOT)
            subprocess.run(['tar','-xf','-','-C',str(tmp)],input=old,check=True)
            target=tmp/'deploy'
            # Neither the installer nor rollback needs weights or their archive.
            missing=tmp/'deliberately-missing-model'
            (target/'deployment.env').write_text(f'MODEL_DIR={missing}\nCONTAINERD_ROOT_DIR=/kept/containerd\nAPI_PORT=8123\n')
            if restart_failure:
                (target/'start.sh').write_text('#!/bin/bash\necho OLD_START_STUB\n')
                (target/'stop.sh').write_text('#!/bin/bash\necho OLD_STOP_STUB\n')
            (target/'runtime').mkdir(exist_ok=True)
            evidence=target/'runtime/operator.log'
            evidence.write_text('preserve operator evidence\n')
            original=files(target)
            subprocess.run(['tar','-xzf',str(args.archive.resolve()),'-C',str(tmp)],check=True)
            package=tmp/('deepseek-v4.1-flash-a100-' + version_expected['delivery_version'] + '-update')
            command=['bash',str(package/'install.sh'),str(target)]+(['--restart'] if restart_failure else [])
            result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
            if restart_failure:
                assert result.returncode==1,result.stdout
                assert 'OLD_START_STUB' in result.stdout and 'Restoring the backed-up source' in result.stdout,result.stdout
            else:
                assert result.returncode==0,result.stdout
                version=json.loads((target/'manifests/update.json').read_text())
                assert version['release']==version_expected['release'] and version['cumulative']
                assert 'CONTAINERD_ROOT_DIR=/kept/containerd' in (target/'deployment.env').read_text()
                subprocess.run(['sudo','-n','docker','run','--rm','--pull','never','--network','none',
                    '--user',f'{os.getuid()}:{os.getgid()}',
                    '--entrypoint','/usr/bin/python3','-e','PYTHONDONTWRITEBYTECODE=1',
                    '-v',f'{target}:/target', IMAGE,'/target/scripts/update_manager.py',
                    '--target','/target','--rollback'],check=True,stdout=subprocess.PIPE)
            restored=files(target)
            assert original==restored, sorted(set(original)^set(restored))
            assert evidence.read_text()=='preserve operator evidence\n'
            assert json.loads((target/'runtime/latest-update.json').read_text())['status']=='ROLLED_BACK'
            row=dict(from_revision=ref,restart_failure_test=restart_failure,status='PASS',
                     exact_source_and_mode_restored=True,real_service_touched=False)
            rows.append(row)
            print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(status='PASS',archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        gpu_executed=False,cases=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
