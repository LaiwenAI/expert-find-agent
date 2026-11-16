"""Download the official frozen checkpoint at a recorded immutable revision."""
import hashlib
import json
from pathlib import Path


def prepare(verify_local=False):
    from huggingface_hub import HfApi,snapshot_download
    root=Path('artifacts/models/qwen3-reranker-0.6b');root.mkdir(parents=True,exist_ok=True)
    marker=root/'checkpoint-provenance.json'
    repo='Qwen/Qwen3-Reranker-0.6B'
    if marker.exists():
        spec=json.loads(marker.read_text())
        if spec['repository']!=repo:raise ValueError('Changed checkpoint repository')
    else:
        spec=dict(repository=repo,revision=HfApi().model_info(repo).sha,
            policy='Official frozen BF16 checkpoint; no own training or quantization. Revision recorded before downloading weights.')
        marker.write_text(json.dumps(spec,indent=2))
    expected_path=root/'expected-large-files.json'
    if not expected_path.exists():
        info=HfApi().model_info(repo,revision=spec['revision'],files_metadata=True)
        expected={s.rfilename:dict(size=s.size,sha256=s.lfs.sha256 if s.lfs else None)
                  for s in info.siblings if s.rfilename in ('model.safetensors','tokenizer.json')}
        expected_path.write_text(json.dumps(expected,indent=2))
    if not verify_local:
        snapshot_download(repo_id=repo,revision=spec['revision'],local_dir=root,allow_patterns=[
            'config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
            'vocab.json','merges.txt','*.safetensors','*.safetensors.index.json'])
    expected=json.loads(expected_path.read_text())
    for name,metadata in expected.items():
        path=root/name
        if (not path.exists() or path.stat().st_size!=metadata['size'] or not metadata['sha256'] or
                hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']):
            raise ValueError('Local checkpoint differs from pinned official LFS hash: '+name)
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.iterdir()) if p.is_file() and p.name!='checkpoint-provenance.json'}
    if 'files' in spec and spec['files']!=files:raise ValueError('Checkpoint files changed')
    spec['files']=files;marker.write_text(json.dumps(spec,indent=2))
    print(json.dumps(dict(repository=repo,revision=spec['revision'],files=len(files)),indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-local',action='store_true',help='Verify manually resumed files against recorded official LFS hashes')
    prepare(parser.parse_args().verify_local)
