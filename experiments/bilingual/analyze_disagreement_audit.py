"""Full selected-cohort repeated-audit accounting; never interprets it as truth."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path('artifacts/bilingual-disagreement-audit-v1')


def run():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    jobs = list(map(json.loads, (ROOT / 'jobs.jsonl').read_text().splitlines()))
    if len(jobs) != 150 or [r['id'] for r in jobs] != [r['id'] for r in protocol['selected']]:
        raise ValueError('Changed selected audit cohort')
    rows, hashes = [], {}
    for job in jobs:
        if any(key in job['payload'] for key in ['stratum','overlap','ranking','status']):
            raise ValueError('Rank or previous judgment leaked into audit input')
        canonical = hashlib.sha256(json.dumps(job['payload'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if canonical != job['input_sha256']:
            raise ValueError('Changed audit payload')
        pair = []
        for repeat in range(2):
            path = ROOT / 'predictions' / (job['key']+'-r'+str(repeat)+'.json')
            if not path.exists():
                raise ValueError('Refuse incomplete150case audit')
            record = json.loads(path.read_text())
            if record['id'] != job['id'] or record['input_sha256'] != job['input_sha256'] or record['repeat'] != repeat:
                raise ValueError('Changed audit identity')
            if record['status'] not in ('valid','unresolved'):
                raise ValueError('Invalid audit terminal status')
            if record['status']=='valid' and (set(record['labels'])!={'equivalence','grounding'} or any(v not in {'supported','changed_or_unsupported','uncertain'} for v in record['labels'].values())):
                raise ValueError('Invalid audit labels')
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            pair.append(record)
        valid = all(r['status']=='valid' for r in pair)
        row = {key: job[key] for key in ['id','intent_type','stratum','overlap']}
        row.update(both_valid=valid, labels=[r.get('labels') for r in pair], both_equivalence_supported=valid and all(r['labels']['equivalence']=='supported' for r in pair), both_grounding_supported=valid and all(r['labels']['grounding']=='supported' for r in pair), all_four_supported=valid and all(v=='supported' for r in pair for v in r['labels'].values()), any_changed=any(v=='changed_or_unsupported' for r in pair for v in r.get('labels',{}).values()), repeat_disagreement=valid and pair[0]['labels']!=pair[1]['labels'])
        rows.append(row)
    def aggregate(subset):
        return dict(cases=len(subset),mean_rank_overlap=sum(r['overlap'] for r in subset)/len(subset),**{key:sum(r[key] for r in subset) for key in ['both_valid','both_equivalence_supported','both_grounding_supported','all_four_supported','any_changed','repeat_disagreement']})
    summary = dict(cases=150,all=aggregate(rows),by_stratum={s:aggregate([r for r in rows if r['stratum']==s]) for s in ['low','middle','high']},by_intent={s:aggregate([r for r in rows if r['intent_type']==s]) for s in sorted({r['intent_type'] for r in rows})},protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),inputs_sha256=hashes,analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=protocol['scope']+' Audit approval cannot rule out translation errors or establish that source-channel/encoder effects are causal. This is not a model-error prevalence estimate or expert relevance evaluation.')
    (ROOT/'case-analysis.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':run()
