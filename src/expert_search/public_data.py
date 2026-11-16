"""Reproducible BEIR downloader and document-task adapter (not expert gold)."""
import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from .corpus import records

URL='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'


def fetch(output='artifacts/public'):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);archive=root/'scifact.zip'
    if archive.exists() and not zipfile.is_zipfile(archive):
        raise ValueError('Incomplete or invalid existing archive; remove it only after confirming no active downloader')
    if not archive.exists():
        partial=root/'scifact.zip.partial'
        with urllib.request.urlopen(URL,timeout=30) as response, partial.open('wb') as target:
            shutil.copyfileobj(response,target)
        if not zipfile.is_zipfile(partial):raise ValueError('Invalid downloaded archive')
        partial.replace(archive)
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            dest=(root/member.filename).resolve()
            if not dest.is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
        z.extractall(root)
    (root/'scifact-source.json').write_text(json.dumps(dict(url=URL,sha256=checksum,
        license_card='https://huggingface.co/datasets/BeIR/scifact',license='CC-BY-SA-4.0 per dataset card',
        task='document retrieval, not expert relevance'),indent=2))
    return root/'scifact'


def adapt(source, output):
    src=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for name in ['experts','evidence']:
        with (out/f'{name}.jsonl').open('w') as f:
            for _,r in records(src/'corpus.jsonl'):
                row=({'id':r['_id'],'name':r.get('title','')} if name=='experts' else
                    {'id':r['_id'],'expert_id':r['_id'],'text':r.get('title','')+'\n'+r['text'],'kind':'document','source':str(src/'corpus.jsonl')})
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
    qrels={}
    with (src/'qrels/test.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            qrels.setdefault(r['query-id'],{})[r['corpus-id']]=int(r['score'])
    queries=[dict(id=r['_id'],query=r['text'],qrels=qrels[r['_id']],
                  **({'query_type':r['type']} if 'type' in r else {}))
             for _,r in records(src/'queries.jsonl') if r['_id'] in qrels]
    (out/'queries.jsonl').write_text('\n'.join(json.dumps(r) for r in queries)+'\n')
    return dict(test_queries=len(queries),task='document retrieval; IDs adapted to common interface only')
