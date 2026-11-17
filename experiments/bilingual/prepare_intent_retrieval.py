"""Freeze all construction intents for within-parent retrieval diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/bilingual-query-construction-v1');manifest=json.loads((data/'manifest.json').read_text());source=data/'synthetic-queries.jsonl'
    if sha(source)!=manifest['outputs_sha256'][source.name]:raise ValueError('Changed source dataset')
    rows=list(map(json.loads,source.read_text().splitlines()));types=set(manifest['taxonomy']);groups={}
    for row in rows:
        if row['split']!='construction' or row['id']!=row['parent_id']+':'+row['intent_type'] or row['intent_type'] not in types:raise ValueError('Changed query binding')
        groups.setdefault(row['parent_id'],[]).append(row)
    if len(rows)!=5000 or len(groups)!=1000 or any(len(x)!=5 or {r['intent_type'] for r in x}!=types for x in groups.values()):raise ValueError('Incomplete construction cohort')
    output.mkdir(parents=True,exist_ok=False)
    selected=[{k:r[k] for k in ('id','parent_id','group_id','intent_type','zh','en','status')} for r in sorted(rows,key=lambda r:r['id'])]
    (output/'queries.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    report=dict(parents=len(groups),queries=len(selected),query_sha256=sha(output/'queries.jsonl'),source_sha256=sha(source),dataset_manifest_sha256=sha(data/'manifest.json'),code_sha256=sha(__file__),
        scope='All five generated intents for each construction parent; no held-out or expert-fit labels. Text and identifiers remain local and are not authorized for public release.')
    (output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
