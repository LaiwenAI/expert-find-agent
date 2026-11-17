"""Build a local English candidate view while preserving unresolved flags."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from expert_search.bilingual.postvalidation import check_pair
from .audit_release_identifiers import PATTERNS

DATA=Path('artifacts/dataset-release/english-repair-construction-v2')
CANDIDATES=Path('artifacts/english-identifier-repair-v1')
OUT=Path('artifacts/dataset-release/english-deidentification-candidates-v1')


def apply_candidate(row, candidate):
    """Use a screened edit or retain its source, without promoting status."""
    if candidate['id']!=row['id'] or candidate['prior_status']!=row['status']:
        raise ValueError('Candidate source mismatch')
    text=candidate.get('english')
    if text is not None:
        if not isinstance(text,str) or not text.strip() or candidate.get('has_han')!=bool(PATTERNS['han_or_radical'].search(text)):
            raise ValueError('Invalid candidate text')
        if candidate.get('numeric_check')!=check_pair(row['english'],text):raise ValueError('Invalid numeric check')
    passed=(text is not None and not candidate.get('has_han',True) and
            candidate.get('numeric_check',{}).get('status')=='match' and candidate.get('audit')=={'identity':'no','meaning':'yes'})
    if (candidate['status']=='model_screened_candidate')!=passed:raise ValueError('Invalid screening state')
    if passed:
        expected='retained_model_screened_pair' if row['status']=='retained_model_screened_pair' else 'unjudged_pair'
        if candidate.get('output_status')!=expected:raise ValueError('Invalid output status')
        return dict(row,english=text,status=expected,deidentification='same_model_screened_candidate')
    return dict(row,deidentification='flagged_unresolved')


def main():
    manifest=json.loads((DATA/'manifest.json').read_text());raw=(DATA/'queries.jsonl').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    protocol=json.loads((CANDIDATES/'protocol.json').read_text())
    if digest!=manifest['output_sha256'] or digest!=protocol['dataset_sha256']:raise ValueError('Changed input')
    rows=[json.loads(s) for s in raw.splitlines()];selected=set(protocol['ids']);outputs=[];hashes={}
    if len(rows)!=5000 or len(selected)!=21 or len({r['id'] for r in rows})!=5000:raise ValueError('Changed cohort')
    for row in rows:
        if row['id'] in selected:
            path=CANDIDATES/'records'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json');body=path.read_bytes();hashes[row['id']]=hashlib.sha256(body).hexdigest()
            outputs.append(apply_candidate(row,json.loads(body)))
        else:outputs.append(dict(row,deidentification='not_assessed'))
    if set(hashes)!=selected:raise ValueError('Missing candidate')
    payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in outputs).encode()
    if PATTERNS['han_or_radical'].search(payload.decode()):raise ValueError('Han character in output')
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'queries.jsonl').write_bytes(payload)
    report=dict(rows=len(outputs),statuses=dict(Counter(r['status'] for r in outputs)),
                deidentification=dict(Counter(r['deidentification'] for r in outputs)),source_sha256=digest,
                candidate_protocol_sha256=hashlib.sha256((CANDIDATES/'protocol.json').read_bytes()).hexdigest(),candidate_record_sha256=hashes,
                output_sha256=hashlib.sha256(payload).hexdigest(),scope='Local candidate view, not a release. Most rows were not assessed for identity; flagged unresolved rows retain original text. Same-model decisions and numeric checks do not establish anonymity, confidentiality, semantic fidelity or redistribution rights.')
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='candidate_record_sha256'},indent=2))


if __name__=='__main__':main()
