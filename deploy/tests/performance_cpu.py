"""Source/rollback integrity contracts. No CUDA execution is claimed."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

DEPLOY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('performance', DEPLOY / 'scripts/performance.py')
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)
OVERLAY = DEPLOY / 'overrides/performance'
MANIFEST = json.loads((OVERLAY / 'manifest.json').read_text())


class OverlayChecks(unittest.TestCase):
    def fixture(self, root, patched):
        site = root / 'site'
        for row in MANIFEST['files']:
            # For vision only, a pristine fixture is not shipped. Populate the
            # updated bytes, which both old and rebuilt-image checks accept.
            variant = 'baseline' if not patched and row['group'] != 'vision' else 'patched'
            if variant == 'baseline' and row['base_sha256'] is None:
                continue
            dest = site / row['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(OVERLAY / variant / row['path'], dest)
        return site

    def test_old_and_updated_images_have_complete_reversible_layer_abi(self):
        for patched in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                site = self.fixture(Path(tmp), patched)
                enabled = perf.plan(DEPLOY, site, Path('/target/deploy'), True)
                disabled = perf.plan(DEPLOY, site, Path('/target/deploy'), False, False, False)
                self.assertEqual(len(enabled['files']), 11)
                self.assertEqual(len(disabled['files']), 8)
                restored = [r for r in disabled['files'] if r['variant'] == 'baseline']
                self.assertEqual({r['path'].rsplit('/', 1)[-1] for r in restored},
                                 {'model.py', 'dspark.py', 'sparse_attn_indexer.py', 'moe_align_block_size.py'})
                self.assertTrue(all(r['source'].startswith('/target/deploy/') for r in enabled['files']))

    def test_independent_group_switches_restore_matching_callers(self):
        from itertools import product
        with tempfile.TemporaryDirectory() as tmp:
            site = self.fixture(Path(tmp), True)
            by_path = {r['path']: r for r in MANIFEST['files']}
            for mhc, indexer, moe in product((False, True), repeat=3):
                result = perf.plan(DEPLOY, site, Path('/target/deploy'), mhc, indexer, moe)
                switches = dict(mhc=mhc, indexer=indexer, moe_align=moe, vision=True)
                for row in result['files']:
                    self.assertEqual(row['variant'], 'patched' if switches[by_path[row['path']]['group']] else 'baseline')

    def test_modified_image_is_rejected_before_mount_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = self.fixture(Path(tmp), False)
            changed = site / MANIFEST['files'][0]['path']
            changed.parent.mkdir(parents=True, exist_ok=True)
            changed.write_text('unexpected image version')
            with self.assertRaisesRegex(ValueError, 'Unsupported image source'):
                perf.plan(DEPLOY, site, Path('/target/deploy'), True)

    def test_modified_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = self.fixture(root, False)
            copy = root / 'deploy'
            shutil.copytree(OVERLAY, copy / 'overrides/performance')
            row = MANIFEST['files'][0]
            (copy / 'overrides/performance/patched' / row['path']).write_text('damaged transfer')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                perf.plan(copy, site, Path('/target/deploy'), True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
