"""Create a local English-only construction view with explicit repair status."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from expert_search.bilingual.postvalidation import check_pair
from .audit_release_identifiers import PATTERNS

DATA=Path('artifacts/dataset-release/bilingual-query-construction-v1')
REPAIRS=Path('artifacts/english-han-repair-v1')
OUT=Path('artifacts/dataset-release/english-repair-construction-v1')
HAN=PATTERNS['han_or_radical']


def build_row(source, repair=None):
    """Retain all inputs without promoting unresolved views or leaking Han text."""
    old=source['en'];flagged=bool(HAN.search(old));prior=source['status']
    if flagged!=(repair is not None):raise ValueError('Missing or unexpected repair')
    english=old;state=prior;repair_status='not_needed'
    if repair is not None:
        if repair['id']!=source['id'] or repair['prior_status']!=prior:raise ValueError('Changed repair identity')
        english=repair.get('english');repair_status=repair['status']
        if english is not None:
            if repair.get('contains_han')!=bool(HAN.search(english)) or repair.get('numeric_check')!=check_pair(old,english):
                raise ValueError('Changed repair validation')
        expected=(prior=='retained_model_screened_pair' and english is not None and not HAN.search(english)
                  and repair.get('numeric_check',{}).get('status')=='match' and repair.get('audit')=='preserved')
        if (repair_status=='model_screened_repair')!=expected:raise ValueError('Invalid repair screening')
        state='retained_model_screened_pair' if expected else 'unjudged_pair'
        if english is not None and HAN.search(english):english=None
    return dict(id=source['id'],parent_id=source['parent_id'],intent_type=source['intent_type'],english=english,
                status=state,prior_status=prior,repair_status=repair_status,expert_fit_grade=None)


def main():
    manifest=json.loads((DATA/'manifest.json').read_text());source_path=DATA/'synthetic-queries.jsonl';raw=source_path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    protocol=json.loads((REPAIRS/'protocol.json').read_text())
    if digest!=manifest['outputs_sha256'][source_path.name] or digest!=protocol['dataset_sha256']:raise ValueError('Changed source')
    source=[json.loads(s) for s in raw.splitlines()];selected=set(protocol['selected_ids'])
    if len(source)!=5000 or len(selected)!=33 or len({r['id'] for r in source})!=5000:raise ValueError('Unexpected cohort')
    output=[];repair_hashes={}
    for row in source:
        repair=None
        if row['id'] in selected:
            path=REPAIRS/'records'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json');body=path.read_bytes()
            repair_hashes[row['id']]=hashlib.sha256(body).hexdigest();repair=json.loads(body)
        output.append(build_row(row,repair))
    if set(repair_hashes)!=selected:raise ValueError('Missing repair records')
    payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in output).encode()
    if HAN.search(payload.decode()):raise ValueError('Han character in English output')
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'queries.jsonl').write_bytes(payload)
    report=dict(rows=len(output),english_present=sum(r['english'] is not None for r in output),statuses=dict(Counter(r['status'] for r in output)),
                repair_statuses=dict(Counter(r['repair_status'] for r in output)),source_sha256=digest,
                repair_protocol_sha256=hashlib.sha256((REPAIRS/'protocol.json').read_bytes()).hexdigest(),repair_record_sha256=repair_hashes,
                output_sha256=hashlib.sha256(payload).hexdigest(),scope='Local English construction view. Original data and status preserved; unresolved repair text remains unjudged and Han-containing repairs are omitted. No human semantic validation, expert-fit labels, identity assurance, or redistribution decision.')
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='repair_record_sha256'},indent=2))


if __name__=='__main__':main()
