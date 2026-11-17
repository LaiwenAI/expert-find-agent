"""Fetch selected official MERIT benchmark assets at an explicit pinned commit.

No training, checkpoint execution or republishing of upstream data is performed.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


def run(manifest_path):
    meta=json.loads(Path(manifest_path).read_text());commit=meta['commit']
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Invalid pinned commit')
    root=Path('data/public/merit-release-20261117');root.mkdir(parents=True,exist_ok=True)
    def fetch(asset):
        name=asset['path'];relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe asset path')
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        url='https://raw.githubusercontent.com/Luli3220/MERIT/'+commit+'/'+name
        if path.exists():body=path.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'expert-search-benchmark-reproduction'}),timeout=60) as response:body=response.read()
        blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if blob!=asset['git_blob_sha'] or len(body)!=asset['bytes']:raise ValueError('Upstream snapshot mismatch '+name)
        path.write_bytes(body);return dict(**asset,url=url,sha256=hashlib.sha256(body).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:assets=list(pool.map(fetch,meta['assets']))
    record=dict(repository=meta['repository'],commit=commit,assets=assets,scope='Official pinned released benchmark assets, stored separately under data/public. Upstream terms apply; no training or checkpoint evaluation yet. Label origin/split independence still to audit, so not assumed human gold.')
    (root/'manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(path=str(root),files=len(assets),bytes=sum(a['bytes'] for a in assets),commit=commit),indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('manifest',nargs='?',default='docs/exploration/merit-release-assets-20261117.json');run(p.parse_args().manifest)
