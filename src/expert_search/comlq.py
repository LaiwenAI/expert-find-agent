"""Pinned official ComLQ download and common-interface benchmark adapter."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from .public_data import adapt

COMMIT='93d321997cad1306d64f2587a3ffba832dda87a7'
URL=f'https://raw.githubusercontent.com/xgl-git/ComLQR-main/{COMMIT}/dataset.zip'
SHA256='b893a9f4bc71dcb0675b8a90915138dbdcf14634968b953f9cab433f8ff00dce'


def prepare(archive='artifacts/public/comlq-upstream/dataset.zip',output='artifacts/public/comlq'):
    archive=Path(archive);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(URL,timeout=30) as response:
            body=response.read()
        partial=archive.with_suffix('.zip.partial');partial.write_bytes(body)
        if not zipfile.is_zipfile(partial):raise ValueError('Invalid ComLQ archive')
        partial.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest!=SHA256:raise ValueError('ComLQ archive differs from pinned official checksum')
    allowed={'dataset/corpus.jsonl','dataset/queries.jsonl','dataset/qrels/test.tsv'}
    with zipfile.ZipFile(archive) as z:
        if not allowed.issubset(z.namelist()):raise ValueError('Missing official dataset files')
        for name in allowed:
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(name))
    counts=adapt(out/'dataset',out/'adapted')
    manifest=dict(commit=COMMIT,url=URL,archive_sha256=digest,
        **counts,query_type_preserved=True,answer_field_excluded=True,
        license='No explicit dataset license found in this pinned repository; source not included in our release bundle.',
        paper='https://arxiv.org/abs/2511.12004',
        warning='Logical document retrieval, not industrial expert relevance. Do not infer negative-constraint gold from ordinary zero qrels.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
