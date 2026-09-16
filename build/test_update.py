"""Exercise real source-update transactions and rollback in a temporary project."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('update_manager', ROOT / 'deploy/scripts/update_manager.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class UpdateChecks(unittest.TestCase):
    def setup_tree(self, root):
        payload, target = root / 'payload', root / 'target'
        (payload / 'manifests').mkdir(parents=True)
        (target / 'runtime').mkdir(parents=True)
        (target / 'deployment.env').write_text('MODEL_DIR=/kept/model\nCONTAINERD_ROOT_DIR=/kept/store\n')
        (target / 'existing.py').write_text('old source\n')
        selected = {'speculative_config': {'num_speculative_tokens': 7}, 'limit_mm_per_prompt': {'image': 4}}
        (target / 'runtime/selected-config.json').write_text(json.dumps(selected))
        (target / 'runtime/keep.log').write_text('operator evidence')
        rows = []
        for name in ('existing.py', 'new.py'):
            p = payload / name
            p.write_text('new source\n')
            rows.append(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + name + '\n')
        version = payload / 'manifests/update.json'
        version.write_text(json.dumps({'id': '20260916-perf2', 'release': '1.1.0'}))
        rows.append(hashlib.sha256(version.read_bytes()).hexdigest() + '  manifests/update.json\n')
        (payload / 'manifests/deploy.sha256').write_text(''.join(rows))
        return payload, target

    def test_apply_and_exact_rollback_preserve_settings_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, target = self.setup_tree(Path(tmp))
            original = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()}
            manager.apply(payload, target)
            env = (target / 'deployment.env').read_text()
            self.assertIn('MODEL_DIR=/kept/model\n', env)
            self.assertIn('CONTAINERD_ROOT_DIR=/kept/store\n', env)
            self.assertIn('PERF_MHC=1\n', env)
            self.assertIn('PERF_INDEXER=1\n', env)
            self.assertIn('PERF_MOE_ALIGN=1\n', env)
            self.assertEqual(json.loads((target / 'runtime/selected-config.json').read_text())['speculative_config']['num_speculative_tokens'], 5)
            receipt = json.loads((target / 'runtime/latest-update.json').read_text())
            self.assertEqual(receipt['release'], '1.1.0')
            manager.restore(target, target / receipt['backup'], receipt)
            for name, data in original.items():
                self.assertEqual((target / name).read_bytes(), data)
            self.assertFalse((target / 'new.py').exists())

    def test_cumulative_update_without_any_previous_performance_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, target = self.setup_tree(Path(tmp))
            image = target / 'images/runtime-image.tar'
            image.parent.mkdir()
            image.write_bytes(b'installed initial image archive remains untouched')
            model = target.parent / 'DeepSeek-V4.1-Flash'
            model.mkdir()
            weights = model / 'weight.safetensors'
            weights.write_bytes(b'fixed model weights remain untouched')
            before = (image.stat().st_mtime_ns, weights.stat().st_mtime_ns)
            manager.apply(payload, target)
            manager.apply(payload, target)  # Reapply with prior metadata/flags present.
            self.assertEqual((target / 'deployment.env').read_text().count('PERF_INDEXER='), 1)
            self.assertEqual(before, (image.stat().st_mtime_ns, weights.stat().st_mtime_ns))

    def test_rollback_preserves_post_update_user_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, target = self.setup_tree(Path(tmp))
            manager.apply(payload, target)
            (target / 'existing.py').write_text('new user edit')
            receipt = json.loads((target / 'runtime/latest-update.json').read_text())
            with self.assertRaisesRegex(ValueError, 'changed after update'):
                manager.restore(target, target / receipt['backup'], receipt)
            self.assertEqual((target / 'existing.py').read_text(), 'new user edit')

    def test_bad_payload_and_symlink_refuse_before_source_changes(self):
        for symlink in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload, target = self.setup_tree(root)
                if symlink:
                    external = root / 'outside'
                    external.write_text('must survive')
                    (target / 'existing.py').unlink()
                    (target / 'existing.py').symlink_to(external)
                else:
                    (payload / 'new.py').write_text('corrupt')
                with self.assertRaises(ValueError):
                    manager.apply(payload, target)
                self.assertNotIn('PERF_MHC', (target / 'deployment.env').read_text())
                if symlink:
                    self.assertEqual(external.read_text(), 'must survive')


if __name__ == '__main__':
    unittest.main(verbosity=2)
