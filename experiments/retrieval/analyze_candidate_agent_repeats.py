"""Summarize complete repeated timing attempts, keeping failures explicit."""
import argparse
import json
import statistics
from pathlib import Path
from .enterprise_end_to_end_pilot import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    protocol=json.loads((args.root/'protocol.json').read_text());prior=Path('artifacts/candidate-agent-pilot-v1/protocol.json')
    if sha(prior)!=protocol['prior_protocol_sha256']:raise ValueError('Changed prior protocol')
    if sha(Path('experiments/retrieval/candidate_agent_repeat_pilot.py'))!=protocol['code_sha256']:raise ValueError('Changed repeat code')
    p=json.loads(prior.read_text())
    for name,digest in {**p['inputs_sha256'],**p['code_sha256']}.items():
        if sha(Path(name))!=digest:raise ValueError('Changed input or code: '+name)
    for name,digest in p['model_files_sha256'].items():
        if sha(Path('artifacts/teacher-compiled-lm')/name)!=digest:raise ValueError('Changed candidate index')
    if sha(Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl'))!=p['query_sha256']:raise ValueError('Changed queries')
    schedule=protocol['schedule'];identities=[(r['query_id'],r['arm'],r['repeat']) for r in schedule]
    expected={(q,a,rep) for q in p['query_ids'] for a in p['arms'] for rep in range(2)}
    if len(identities)!=len(expected) or set(identities)!=expected:raise ValueError('Invalid schedule')
    rows={}
    for qid,arm,repeat in identities:
        path=args.root/(qid+'-'+arm+'-'+str(repeat)+'.json')
        if not path.exists():raise ValueError('Incomplete run')
        row=json.loads(path.read_text())
        if (row['query_id'],row['arm'],row['repeat'])!=(qid,arm,repeat):raise ValueError('Invalid attempt identity')
        if row['status']=='ok':
            result=row['result'];candidates=result['candidate_ids'];rank=[x['expert']['id'] for x in result['results']]
            if len(candidates)!=len(set(candidates)) or len(rank)!=len(set(rank)) or not set(rank)<=set(candidates):raise ValueError('Invalid result identities')
            if any(not x['evidence'] or any(e['expert_id']!=x['expert']['id'] for e in x['evidence']) for x in result['results']):raise ValueError('Invalid evidence ownership')
        rows[qid,arm,repeat]=row
    output={}
    for arm in p['arms']:
        group=[r for (_,a,_),r in rows.items() if a==arm];valid=[r for r in group if r['status']=='ok']
        output[arm]={'attempts':len(group),'successes':len(valid),'mean_attempt_seconds':statistics.mean(r['attempt_seconds'] for r in group),
                     'minimum_attempt_seconds':min(r['attempt_seconds'] for r in group),'maximum_attempt_seconds':max(r['attempt_seconds'] for r in group),
                     'models':sorted({str(t.get('model')) for r in valid for t in r['result']['trace']})}
        same=0;overlap=[]
        for qid in p['query_ids']:
            first,second=rows[qid,arm,0],rows[qid,arm,1]
            if first['status']==second['status']=='ok':
                a=[r['expert']['id'] for r in first['result']['results']];b=[r['expert']['id'] for r in second['result']['results']]
                same+=a==b;overlap.append(len(set(a)&set(b)))
        output[arm]['common_success_repeated_queries']=len(overlap)
        output[arm]['identical_ordered_rankings']=same
        output[arm]['mean_returned_id_overlap']=statistics.mean(overlap) if overlap else None
    differences=[rows[q,'candidate_first',rep]['attempt_seconds']-rows[q,'single_pass',rep]['attempt_seconds'] for q in p['query_ids'] for rep in range(2)]
    print(json.dumps({'arms':output,'paired_mean_difference_seconds':statistics.mean(differences),
                      'paired_differences_seconds':differences,'scope':'All-attempt paired times include failures. Repeats reuse four development queries; observations are not independent query samples. No relevance or noninferiority claim; initialization excluded.'},indent=2))


if __name__=='__main__':main()
