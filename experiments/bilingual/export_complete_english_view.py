"""Add the one local follow-up without promoting prior review status."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from expert_search.bilingual.postvalidation import check_pair
from .audit_release_identifiers import PATTERNS


def apply_followup(row, source, followup):
    """Apply only a complete, validated, Han-free replacement to a missing row."""
    if (row['id']!=source['id'] or row['id']!=followup['id'] or row['english'] is not None or row['status']!='unjudged_pair'
            or followup.get('prior_status')!=source['status']):
        raise ValueError('Follow-up identity or status mismatch')
    text=followup.get('english')
    if (followup['status']!='unjudged_followup' or not isinstance(text,str) or not text.strip() or
            PATTERNS['han_or_radical'].search(text) or followup.get('contains_han') is not False or
            followup.get('numeric_check')!=check_pair(source['en'],text)):
        raise ValueError('Invalid follow-up text')
    return dict(row,english=text,repair_status='unjudged_followup')


def main():
    old=Path('artifacts/dataset-release/english-repair-construction-v1');source=Path('artifacts/dataset-release/bilingual-query-construction-v1')
    follow=Path('artifacts/english-han-repair-followup-v1/result.json');old_manifest=json.loads((old/'manifest.json').read_text());source_manifest=json.loads((source/'manifest.json').read_text())
    old_raw=(old/'queries.jsonl').read_bytes();source_raw=(source/'synthetic-queries.jsonl').read_bytes();follow_raw=follow.read_bytes()
    if hashlib.sha256(old_raw).hexdigest()!=old_manifest['output_sha256'] or hashlib.sha256(source_raw).hexdigest()!=source_manifest['outputs_sha256']['synthetic-queries.jsonl']:
        raise ValueError('Changed source')
    rows=[json.loads(x) for x in old_raw.splitlines()];sources={r['id']:r for r in map(json.loads,source_raw.splitlines())};record=json.loads(follow_raw)
    original_repair=Path('artifacts/english-han-repair-v1/records')/(hashlib.sha256(record['id'].encode()).hexdigest()+'.json')
    if hashlib.sha256(original_repair.read_bytes()).hexdigest()!=record['prior_record_sha256']:raise ValueError('Changed earlier repair')
    replaced=0
    for i,row in enumerate(rows):
        if row['id']==record['id']:
            rows[i]=apply_followup(row,sources[row['id']],record);replaced+=1
    if len(rows)!=5000 or replaced!=1 or any(r['english'] is None for r in rows):raise ValueError('Incomplete view')
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
    if PATTERNS['han_or_radical'].search(payload.decode()):raise ValueError('Han character remains')
    out=Path('artifacts/dataset-release/english-repair-construction-v2');out.mkdir(parents=True,exist_ok=False)
    (out/'queries.jsonl').write_bytes(payload)
    manifest=dict(rows=len(rows),english_present=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
                  prior_view_sha256=old_manifest['output_sha256'],source_sha256=hashlib.sha256(source_raw).hexdigest(),
                  followup_sha256=hashlib.sha256(follow_raw).hexdigest(),output_sha256=hashlib.sha256(payload).hexdigest(),
                  scope='Local English construction view. Follow-up remains unjudged even if same-model screening passes. No independent semantic or expert-fit validation, anonymity assurance or redistribution clearance.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
