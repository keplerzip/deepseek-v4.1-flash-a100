"""Small-fixture full archive and installer tests; no GPU/real Docker touched."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import package_full
from verify_full_package import verify as verify_full

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = 'sha256:' + 'a'*64


class FullPackageTests(unittest.TestCase):
    def fixture(self, work):
        source = work/'source'
        for name in ('deploy/manifests', 'deploy/scripts', 'deploy/configs', 'docs',
                     'reports', 'examples', 'licenses', 'build'):
            (source/name).mkdir(parents=True)
        for name in ('LICENSE', 'NOTICE', 'VERSION', 'release.json', 'CHANGELOG.md', 'README.md'):
            (source/name).write_text('1.9.0\n' if name == 'VERSION' else '{}\n')
        (source/'docs/test.md').write_text('Public documentation\n')
        (source/'reports/test.json').write_text('{"gpu_executed":false}\n')
        for name in ('package_full.py', 'full-install.sh', 'update-entry.sh'):
            shutil.copyfile(ROOT/'build'/name, source/'build'/name)
        shutil.copyfile(ROOT/'deploy/scripts/update_manager.py', source/'deploy/scripts/update_manager.py')
        (source/'deploy/scripts/performance.py').write_text('# fixture only\n')
        (source/'deploy/existing.py').write_text('new-source\n')
        (source/'deploy/deployment.env').write_text(
            'MODEL_DIR=/private/build/model\nNEWAPI_BASE_URL=http://private.invalid\nCONTAINERD_ROOT_DIR=/private/store\n')
        config = dict(model='/models/DeepSeek-V4.1-Flash', served_model_name='DeepSeek-Flash',
                      max_model_len=512000, max_num_batched_tokens=1024,
                      data_parallel_size=2, tensor_parallel_size=4,
                      enable_chunked_prefill=True, disable_chunked_mm_input=True,
                      hf_overrides={'vision_max_n_token': 1020},
                      speculative_config={'method': 'dspark', 'num_speculative_tokens': 5})
        (source/'deploy/configs/runtime.json').write_text(json.dumps(config))
        constraints = dict(served_model_name='DeepSeek-Flash', max_model_len=512000,
                           max_num_batched_tokens=1024, minimum_concurrency=64,
                           maximum_concurrency=64, concurrency_formula='fixed(64)',
                           tensor_parallel_size=4, data_parallel_size=2, dspark_k=5,
                           enable_chunked_prefill=True, disable_chunked_mm_input=True,
                           vision_max_n_token=1020, requested_image_token_ceiling=1024, dspark_reserved_token_slots=4)
        (source/'deploy/configs/constraints.json').write_text(json.dumps(constraints))
        (source/'deploy/manifests/update.json').write_text(json.dumps(
            {'id': 'fixture-r19', 'release': '1.9.0', 'delivery_version': 'R1.9'}))
        (source/'deploy/manifests/allowed-image-ids.txt').write_text(IMAGE_ID+'\n')
        image = work/'runtime-image.tar'
        image.write_bytes(b'small synthetic Docker archive fixture, never GPU validated\n')
        (source/'deploy/manifests/image-integrity.json').write_text(json.dumps(
            {'status': 'PASS', 'archive_bytes': image.stat().st_size,
             'archive_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}))
        for name in ('deploy', 'start', 'stop'):
            (source/f'deploy/{name}.sh').write_text(
                '#!/bin/bash\nprintf "'+name+'\\n" >> "$MOCK_STATE/lifecycle.log"\n')
        for name in ('deploy/runtime/private-key.txt', 'deploy/weights/model.safetensors',
                     'docs/base64.sh', 'build/.env', 'build/__pycache__/private.pyc',
                     'deploy/images/stale.tar'):
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('must not be packaged\n')
        archive = work/'full.tar.gz'
        original_read_bytes = Path.read_bytes

        def guard(path):
            if path in (image, archive) or path.name == 'package.tar.gz':
                raise AssertionError('Image/archive must be hashed as a stream')
            return original_read_bytes(path)

        with patch.object(Path, 'read_bytes', guard), redirect_stdout(io.StringIO()):
            package_full.package(source, image, archive)
        extracted = work/'extracted'
        extracted.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(extracted, filter='data')
        package = extracted/'deepseek-v4.1-flash-a100-R1.9-full'
        return source, image, archive, package

    def mock_environment(self, work, *, installed=False):
        binaries, state = work/'mock-bin', work/'mock-state'
        binaries.mkdir()
        state.mkdir()
        if installed:
            (state/'image-installed').touch()
        sudo = binaries/'sudo'
        sudo.write_text(r'''#!/usr/bin/env python3
import json,os,pathlib,subprocess,sys
state=pathlib.Path(os.environ['MOCK_STATE'])
args=sys.argv[1:]
with (state/'docker.jsonl').open('a') as f:f.write(json.dumps(args)+'\n')
assert args[:2]==['-n','docker'],args
args=args[2:]
if args[:2]==['image','inspect']:
    if not (state/'image-installed').exists():sys.exit(1)
    if '--format' in args:print('sha256:'+('b' if os.environ.get('WRONG_IMAGE') else 'a')*64)
elif args[:1]==['load']:
    (state/'image-installed').touch()
elif args[:1]==['run']:
    mounts={}
    for i,a in enumerate(args):
        if a=='-v':
            host,container,*_=args[i+1].split(':');mounts[container]=host
    if '/update/deploy/scripts/update_manager.py' in args:
        translated=[]
        for a in args[args.index('/update/deploy/scripts/update_manager.py'):]:
            for container,host in mounts.items():
                if a==container or a.startswith(container+'/'):a=host+a[len(container):];break
            translated.append(a)
        sys.exit(subprocess.run([sys.executable,*translated]).returncode)
else:raise AssertionError(args)
''')
        sudo.chmod(0o755)
        # Make this test portable to root-owned CI without exercising root
        # installation: only the fake command output changes, never uid.
        uid = binaries/'id'
        uid.write_text('#!/bin/bash\n[[ "${1:-}" != -u ]] || { echo "${MOCK_UID:-1000}"; exit; }\nexec /usr/bin/id "$@"\n')
        uid.chmod(0o755)
        return {**os.environ, 'PATH': str(binaries)+os.pathsep+os.environ['PATH'],
                'MOCK_STATE': str(state)}, state

    def install(self, package, target, env, restart=False):
        return subprocess.run(['bash', str(package/'install.sh'), str(target),
                               *(['--restart'] if restart else [])],
                              text=True, capture_output=True, env=env, timeout=30)

    def test_streaming_archive_excludes_private_state_weights_and_base64(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source, image, archive, package = self.fixture(work)
            checks = subprocess.run(['sha256sum', '--quiet', '-c', 'SHA256SUMS'],
                                    cwd=package, capture_output=True)
            self.assertEqual(checks.returncode, 0, checks.stderr)
            self.assertEqual((package/'deploy/images/runtime-image.tar').read_bytes(), image.read_bytes())
            self.assertNotIn('/private/', (package/'deploy/deployment.env').read_text())
            self.assertIn('MODEL_DIR=\n', (package/'deploy/deployment.env').read_text())
            self.assertIn('CONTAINERD_ROOT_DIR=/var/lib/containerd\n', (package/'deploy/deployment.env').read_text())
            self.assertNotIn('deployment.env', (package/'deploy/manifests/deploy.sha256').read_text())
            files = [p.relative_to(package).as_posix() for p in package.rglob('*')]
            self.assertFalse(any('base64' in name or '.safetensors' in name or '/runtime/' in name for name in files))
            self.assertFalse((package/'build/.env').exists())
            expected = archive.with_name(archive.name+'.sha256').read_text().split()[0]
            self.assertEqual(expected, package_full.digest(archive))
            verified = verify_full(archive)
            self.assertEqual(verified['status'], 'PASS')
            self.assertEqual(verified['model'], 'DeepSeek-Flash')
            self.assertEqual(verified['concurrency'], 64)
            self.assertEqual(verified['per_dp_concurrency'], 32)
            self.assertNotIn('max_num_seqs', json.loads((package/'deploy/configs/runtime.json').read_text()))
            self.assertFalse(verified['extracted'])
            self.assertTrue(verified['external_checksum_checked'])
            self.assertTrue((package/'install.sh').is_file())
            self.assertFalse((package/'installer.sh').exists())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                package_full.package(source, image, archive)

    def test_full_verifier_rejects_links_weights_and_corrupt_payload(self):
        for failure in ('link', 'weight', 'checksum', 'model', 'environment'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                _, _, _, package = self.fixture(work)
                if failure == 'checksum':
                    (package/'deploy/existing.py').write_text('changed after packaging')
                if failure == 'weight':
                    (package/'deploy/model.safetensors').write_bytes(b'weight is forbidden')
                if failure == 'model':
                    config = json.loads((package/'deploy/configs/runtime.json').read_text())
                    config['served_model_name'] = 'wrong-alias'
                    (package/'deploy/configs/runtime.json').write_text(json.dumps(config))
                if failure == 'environment':
                    (package/'deploy/deployment.env').write_text('MODEL_DIR=/private/source\n')
                if failure in ('model', 'environment'):
                    (package/'deploy/manifests/deploy.sha256').write_text(
                        package_full.checksum_tree(package/'deploy', deploy=True))
                    (package/'SHA256SUMS').write_text(package_full.checksum_tree(package))
                corrupt = work/'corrupt.tar.gz'
                with tarfile.open(corrupt, 'w:gz') as tar:
                    tar.add(package, arcname=package.name)
                    if failure == 'link':
                        link = tarfile.TarInfo(package.name+'/escape')
                        link.type, link.linkname = tarfile.SYMTYPE, '/outside/private'
                        tar.addfile(link)
                with self.assertRaises(ValueError):
                    verify_full(corrupt)

    def test_source_contract_derives_c64_and_protects_vision_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _, _ = self.fixture(Path(directory))
            runtime = json.loads((source/'deploy/configs/runtime.json').read_text())
            constraints = json.loads((source/'deploy/configs/constraints.json').read_text())
            self.assertNotIn('max_num_seqs', runtime)
            self.assertEqual(package_full.configuration_metadata(runtime, constraints)['per_dp_concurrency'], 32)
            runtime['max_num_seqs'] = 32
            self.assertEqual(package_full.configuration_metadata(runtime, constraints)['global_concurrency'], 64)
            cases = [({'max_num_seqs': 64}, {}),
                     ({'hf_overrides': {'vision_max_n_token': 1024}}, {}),
                     ({'disable_chunked_mm_input': False}, {}),
                     ({'enable_chunked_prefill': False}, {}),
                     ({}, {'minimum_concurrency': 32}),
                     ({}, {'maximum_concurrency': 128}),
                     ({}, {'concurrency_formula': 'dynamic'}),
                     ({}, {'vision_max_n_token': 1024})]
            for runtime_changes, constraint_changes in cases:
                with self.assertRaises(ValueError):
                    package_full.configuration_metadata({**runtime, **runtime_changes},
                                                        {**constraints, **constraint_changes})

    def test_fresh_install_imports_only_pinned_image_and_starts_on_request(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            _, _, _, package = self.fixture(work)
            env, state = self.mock_environment(work)
            target = work/'service/deploy'
            result = self.install(package, target, env, restart=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertTrue((target/'images/runtime-image.tar').is_file())
            self.assertEqual((state/'lifecycle.log').read_text(), 'deploy\n')
            calls = [json.loads(line) for line in (state/'docker.jsonl').read_text().splitlines()]
            self.assertEqual(sum(row[2:3] == ['load'] for row in calls), 1)
            self.assertTrue(all(row[:2] == ['-n', 'docker'] for row in calls))

    def test_existing_install_uses_real_transaction_and_preserves_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            _, _, _, package = self.fixture(work)
            env, state = self.mock_environment(work, installed=True)
            target = work/'service/deploy'
            (target/'runtime/cache').mkdir(parents=True)
            (target/'deployment.env').write_text('MODEL_DIR=/kept/weights\nAPI_PORT=8123\nCONTAINERD_ROOT_DIR=/kept/containerd\n')
            (target/'runtime/cache/keep.bin').write_bytes(b'cache evidence')
            (target/'existing.py').write_text('old source\n')
            result = self.install(package, target, env)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual((target/'existing.py').read_text(), 'new-source\n')
            self.assertIn('MODEL_DIR=/kept/weights\n', (target/'deployment.env').read_text())
            self.assertIn('API_PORT=8123\n', (target/'deployment.env').read_text())
            self.assertEqual((target/'runtime/cache/keep.bin').read_bytes(), b'cache evidence')
            receipt = json.loads((target/'runtime/latest-update.json').read_text())
            self.assertEqual(receipt['status'], 'APPLIED')
            self.assertEqual(receipt['release'], '1.9.0')
            self.assertTrue((target/'images/runtime-image.tar').is_file())
            self.assertFalse((state/'lifecycle.log').exists())

    def test_existing_restart_forwards_to_transactional_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            _, _, _, package = self.fixture(work)
            env, state = self.mock_environment(work, installed=True)
            target = work/'service/deploy'
            target.mkdir(parents=True)
            (target/'deployment.env').write_text('MODEL_DIR=/kept/weights\n')
            (target/'stop.sh').write_text('#!/bin/bash\nprintf "old-stop\\n" >> "$MOCK_STATE/lifecycle.log"\n')
            result = self.install(package, target, env, restart=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual((state/'lifecycle.log').read_text(), 'old-stop\nstart\n')
            self.assertEqual(json.loads((target/'runtime/latest-update.json').read_text())['status'], 'APPLIED')
            self.assertIn('MODEL_DIR=/kept/weights\n', (target/'deployment.env').read_text())

    def test_corrupt_payload_foreign_image_and_nonempty_target_refused(self):
        for failure in ('checksum', 'image', 'nonempty', 'root'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                _, _, _, package = self.fixture(work)
                env, state = self.mock_environment(work, installed=True)
                target = work/'service/deploy'
                if failure == 'checksum':
                    (package/'deploy/existing.py').write_text('corrupt')
                if failure == 'image':
                    env['WRONG_IMAGE'] = '1'
                if failure == 'nonempty':
                    target.mkdir(parents=True)
                    (target/'keep').write_text('do not overwrite')
                if failure == 'root':
                    env['MOCK_UID'] = '0'
                result = self.install(package, target, env)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((state/'lifecycle.log').exists())
                if failure == 'nonempty':
                    self.assertEqual((target/'keep').read_text(), 'do not overwrite')
                else:
                    self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
