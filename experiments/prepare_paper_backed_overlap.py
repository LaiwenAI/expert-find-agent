"""Evidence-availability stratification of demand-topic candidates, no labels."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from expert_search.corpus import records


def run(candidates='artifacts/demand-topic-overlap-v1/candidates.jsonl',audits_path='artifacts/topic-overlap-audit-v1',output='artifacts/paper-backed-overlap-v1'):
    audits={}
    for p in Path(audits_path).glob('demand-*.json'):
        r=json.loads(p.read_text())
        if r.get('audit',{}).get('technical_requirement') is True:audits[r['id']]=r['audit']
    documents=defaultdict(list)
    for _,source in records('artifacts/corpus/evidence.jsonl'):
        if source['kind']=='paper' and source['text'].strip():documents[source['expert_id']].append(source)
    parent=Path(candidates);jobs=[]
    for candidate in map(json.loads,parent.read_text().splitlines()):
        did=candidate['demand_id'];eid=candidate['expert_id']
        if did not in audits or not documents[eid]:continue
        matches=[m for m in candidate['matches'] if m['topic']==audits[did]['topic']]
        if not matches:continue
        papers=sorted(documents[eid],key=lambda p:p['id'])
        jobs.append(dict(id=did+'-expert-'+eid,parent_demand=did,demand=candidate['demand'],
            technical_anchor=audits[did],topic_matches=matches,expert_id=eid,
            sources={p['id']:p for p in papers},status='unlabeled_paper_backed_candidate'))
    out=Path(output);out.mkdir(exist_ok=True)
    body=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs)
    (out/'jobs.jsonl').write_text(body)
    report=dict(audited_anchor_demands=len(audits),paper_backed_demands=len({j['parent_demand'] for j in jobs}),jobs=len(jobs),
        experts=len({j['expert_id'] for j in jobs}),source_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),jobs_sha256=hashlib.sha256(body.encode()).hexdigest(),
        warning='Evidence-availability stratum selected before capability generation. Anchors without paper-backed matched teachers remain unavailable. Missing papers do not imply lack of ability. Remaining narrow candidates are not gold or full-population coverage.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--candidates',default='artifacts/demand-topic-overlap-v1/candidates.jsonl');parser.add_argument('--audits',default='artifacts/topic-overlap-audit-v1');parser.add_argument('--output',default='artifacts/paper-backed-overlap-v1');args=parser.parse_args();run(args.candidates,args.audits,args.output)
