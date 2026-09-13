"""Exercise archive integrity and both Docker identity formats without GPU/data downloads."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from verify_image import TAG, verify


class ArchiveTests(unittest.TestCase):
    def fixture(self, root, oci=False, corrupt=False):
        entries={}
        def blob(value):
            raw=value if isinstance(value,bytes) else json.dumps(value).encode()
            sha=hashlib.sha256(raw).hexdigest()
            entries['blobs/sha256/'+sha]=raw
            return 'sha256:'+sha
        layer=b'synthetic layer bytes for integrity testing only'
        diff=blob(layer)
        config={'architecture':'amd64','os':'linux','config':{'Entrypoint':['vllm','serve']},
                'rootfs':{'type':'layers','diff_ids':[diff]}}
        config_id=blob(config)
        entries['manifest.json']=json.dumps([{'RepoTags':[TAG],
            'Config':'blobs/sha256/'+config_id[7:],'Layers':['blobs/sha256/'+diff[7:]]}]).encode()
        image_id=config_id
        if oci:
            platform=blob({'schemaVersion':2,'config':{'digest':config_id},'layers':[{'digest':diff}]})
            index={'schemaVersion':2,'manifests':[{'digest':platform}]}
            image_id=blob(index)
            entries['index.json']=json.dumps({'schemaVersion':2,'manifests':[{'digest':image_id}]}).encode()
        if corrupt: entries['blobs/sha256/'+diff[7:]]=b'corrupted'
        path=root/'image.tar'
        with tarfile.open(path,'w') as tar:
            for name,raw in entries.items():
                item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
        return path,{'Id':image_id,'RootFS':{'Layers':[diff]}}

    def test_legacy_and_oci(self):
        for oci in (False,True):
            with self.subTest(oci=oci),tempfile.TemporaryDirectory() as tmp:
                path,inspect=self.fixture(Path(tmp),oci)
                result=verify(path,inspect)
                self.assertEqual(result['status'],'PASS')
                self.assertFalse(result['gpu_validated'])

    def test_reject_corrupted_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,inspect=self.fixture(Path(tmp),corrupt=True)
            with self.assertRaises(AssertionError): verify(path,inspect)

    def test_reject_unrelated_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,inspect=self.fixture(Path(tmp),True)
            inspect['Id']='sha256:'+'0'*64
            with self.assertRaises(AssertionError): verify(path,inspect)

    def test_reject_mismatched_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,inspect=self.fixture(Path(tmp))
            inspect['RootFS']['Layers']=['sha256:'+'0'*64]
            with self.assertRaises(AssertionError): verify(path,inspect)

    def test_assemble_manifest_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path,inspect=self.fixture(root,True)
            record=root/'inspect.json';record.write_text(json.dumps([inspect]))
            output=root/'bundle/deploy'
            command=[sys.executable,str(Path(__file__).with_name('assemble.py')),
                     '--archive',str(path),'--inspect',str(record),'--output',str(output)]
            completed=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            for line in (output/'manifests/deploy.sha256').read_text().splitlines():
                sha,name=line.split('  ',1)
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),sha)
            config=json.loads((output/'configs/runtime.json').read_text())
            self.assertEqual(config['speculative_config']['num_speculative_tokens'],5)
            self.assertFalse((output/'runtime').exists())
            repeated=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(repeated.returncode,0)


if __name__=='__main__': unittest.main()
