"""Install fixed overlays only when the base bytes match the reviewed source."""
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig

root = Path(__file__).resolve().parent
site = Path(sysconfig.get_paths()['purelib'])
manifest = json.loads((root / 'runtime-patches.json').read_text())
for item in manifest['files']:
    target = site / item['path']
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != item['base_sha256']:
        raise RuntimeError('Base-image source mismatch: ' + item['path'])
    overlay = root / 'patched' / item['path']
    if hashlib.sha256(overlay.read_bytes()).hexdigest() != item['patched_sha256']:
        raise RuntimeError('Patch payload mismatch: ' + item['path'])
    shutil.copyfile(overlay, target)
shutil.copyfile(root / 'dsv41_capacity.py', site / 'dsv41_capacity.py')
shutil.copyfile(root / 'dsv41_guard.py', site / 'dsv41_guard.py')
print(json.dumps({'status': 'PASS', 'patched_files': len(manifest['files']), 'gpu_validated': False}))
