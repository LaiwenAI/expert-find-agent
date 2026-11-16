"""Download and verify an immutable official Qwen3 embedding checkpoint."""
import hashlib
import json
import subprocess
from pathlib import Path


def prepare():
    root=Path('artifacts/models/qwen3-embedding-0.6b');root.mkdir(exist_ok=True)
    marker=root/'download-protocol.json';repo='Qwen/Qwen3-Embedding-0.6B'
    if not marker.exists():
        body=subprocess.check_output(['curl','-fsSL','--retry','2','--max-time','60',f'https://huggingface.co/api/models/{repo}?blobs=true'])
        info=json.loads(body)
        spec=dict(repository=repo,revision=info['sha'],expected_files={f['rfilename']:dict(size=f.get('size'),sha256=f.get('lfs',{}).get('sha256')) for f in info['siblings'] if f['rfilename'] in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json')})
        marker.write_text(json.dumps(spec,indent=2))
    spec=json.loads(marker.read_text())
    if spec['repository']!=repo:raise ValueError('Wrong repository')
    for name,expected in spec['expected_files'].items():
        target=root/name
        if not target.exists():
            partial=root/(name+'.partial')
            subprocess.run(['curl','-fLsS','--retry','3','--max-time','1200','-C','-',f'https://huggingface.co/{repo}/resolve/{spec["revision"]}/{name}','-o',str(partial)],check=True)
            partial.replace(target)
        if target.stat().st_size!=expected['size']:raise ValueError('Wrong checkpoint size: '+name)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if expected['sha256'] and digest!=expected['sha256']:raise ValueError('Wrong official LFS digest: '+name)
        print(name,'verified',flush=True)
    manifest=dict(repository=repo,revision=spec['revision'],files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in spec['expected_files']},policy='Official frozen BF16 weights; no own training or quantization.')
    (root/'checkpoint-provenance.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':prepare()
