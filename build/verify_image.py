"""Validate every Docker archive blob, config, layer diff ID and image identity."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

TAG='deepseek-v4.1-flash-a100:20260913-r1'


def sha(stream):
    h=hashlib.sha256()
    while chunk:=stream.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def verify(path, inspected):
    inspected=inspected[0] if isinstance(inspected,list) else inspected
    with tarfile.open(path,'r:') as tar:
        members=tar.getmembers(); seen=set();blobs=[]
        for m in members:
            name=PurePosixPath(m.name)
            assert not name.is_absolute() and '..' not in name.parts and m.name not in seen,m.name
            assert m.isfile() or m.isdir(),m.name
            seen.add(m.name)
            if m.isfile() and m.name.startswith('blobs/sha256/'):
                actual=sha(tar.extractfile(m))
                assert actual==name.name,m.name
                blobs.append({'sha256':actual,'bytes':m.size})
        legacy=json.load(tar.extractfile('manifest.json'))
        assert len(legacy)==1 and legacy[0]['RepoTags']==[TAG]
        data=tar.extractfile(legacy[0]['Config']).read()
        config_id='sha256:'+hashlib.sha256(data).hexdigest()
        config=json.loads(data)
        assert config['architecture']=='amd64' and config['os']=='linux'
        assert config['config']['Entrypoint']==['vllm','serve']
        assert config['rootfs']['diff_ids']==inspected['RootFS']['Layers']
        assert len(legacy[0]['Layers'])==len(config['rootfs']['diff_ids'])
        layers=[]
        for name,expected in zip(legacy[0]['Layers'],config['rootfs']['diff_ids']):
            stream=tar.extractfile(name)
            magic=stream.read(2);stream.seek(0)
            compressed=magic==b'\x1f\x8b'
            actual='sha256:'+sha(gzip.GzipFile(fileobj=stream) if compressed else stream)
            assert actual==expected,(name,expected,actual)
            layers.append({'path':name,'diff_id':actual,'gzip':compressed})
            print('VERIFIED_LAYER '+actual,flush=True)
        def platform_id(index):
            for descriptor in index.get('manifests',[]):
                body=json.load(tar.extractfile('blobs/sha256/'+descriptor['digest'].split(':')[1]))
                if body.get('config',{}).get('digest')==config_id:return descriptor['digest']
                if 'manifests' in body:
                    found=platform_id(body)
                    if found:return found
        platform=platform_id(json.load(tar.extractfile('index.json'))) if 'index.json' in seen else None
        allowed={config_id}
        if platform: allowed.add(platform)
        # Docker may expose an OCI index ID instead of the config ID.
        if inspected['Id'].startswith('sha256:') and 'blobs/sha256/'+inspected['Id'][7:] in seen:
            identity=json.load(tar.extractfile('blobs/sha256/'+inspected['Id'][7:]))
            if identity.get('manifests') and platform_id(identity)==platform and platform:
                allowed.add(inspected['Id'])
        assert inspected['Id'] in allowed, 'Inspected ID is not bound to the archived image'
    with path.open('rb') as f:archive_sha=sha(f)
    report={'status':'PASS','scope':'Docker archive blobs, config, layer diff IDs and fixed identity; no GPU',
            'tag':TAG,'image_id':inspected['Id'],'image_config_id':config_id,'platform_manifest_id':platform,
            'archive':'images/runtime-image.tar','archive_bytes':path.stat().st_size,'archive_sha256':archive_sha,
            'blobs':blobs,'layers':layers,'base_amd64_manifest':'sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3',
            'gpu_validated':False}
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--inspect',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=verify(a.archive,json.loads(a.inspect.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['blobs','layers']}))


if __name__=='__main__': main()
