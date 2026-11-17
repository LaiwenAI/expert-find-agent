"""Analyze fixed-view development comparisons with incomplete silver labels."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from expert_search.evaluation_gain import paired_actionable_gain_interval


def frozen_pool_hash(query_id, pools, snapshots, corpus):
    """Reconstruct the exact inference pool and check each frozen source span."""
    hits=[]
    candidates=pools[query_id]['candidates']
    if len(candidates)!=len(set(candidates)):
        raise ValueError('Duplicate candidate')
    for eid in candidates:
        evidence=[]
        for span in snapshots[query_id,eid]['sources']:
            source=corpus[span['source_id']]
            start,end=span['start'],span['end']
            if not (0<=start<end<=len(source['text'])):
                raise ValueError('Invalid source span')
            text=source['text'][start:end]
            if source['expert_id']!=eid or hashlib.sha256(text.encode()).hexdigest()!=span['sha256']:
                raise ValueError('Invalid source ownership or span hash')
            evidence.append({'id':source['id'],'expert_id':eid,'text':text})
        hits.append({'id':eid,'evidence':evidence})
    digest=hashlib.sha256(json.dumps(hits,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return digest,set(candidates)


def unique_index(rows, key):
    result={}
    for row in rows:
        identity=key(row)
        if identity in result:raise ValueError('Duplicate frozen input identity')
        result[identity]=row
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');protocol=json.loads((args.root/'protocol.json').read_text())
    if protocol['arms'] != ['direct', 'coverage_direct']:
        raise ValueError('Unexpected arm order; paired difference is coverage minus direct')
    for name,expected in protocol['inputs_sha256'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen input changed: '+name)
    read=lambda name:[json.loads(line) for line in (data/name).read_text().splitlines()]
    queries=unique_index((r for r in read('queries.jsonl') if r['split']=='development'),lambda r:r['id'])
    pools=unique_index((r for r in read('pools.jsonl') if r['split']=='development'),lambda r:r['query_id'])
    snapshots=unique_index((r for r in read('snapshots.jsonl') if r['split']=='development'),lambda r:(r['query_id'],r['expert_id']))
    corpus=unique_index(read('corpus.jsonl'),lambda r:r['id'])
    if not protocol['query_ids'] or len(set(protocol['query_ids']))!=len(protocol['query_ids']):raise ValueError('Invalid query cohort')
    judgments={}
    for line in (data/'judgments.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['label_status']=='model_silver':
            key=(row['query_id'],row['expert_id'])
            if key in judgments:raise ValueError('Duplicate silver judgment')
            judgments[key]=row['grade']
    output=[]
    arm_rows={arm:[] for arm in protocol['arms']}
    for qid in protocol['query_ids']:
        if qid not in queries:raise ValueError('Query is not in the development split')
        expected_hash,candidates=frozen_pool_hash(qid,pools,snapshots,corpus)
        rows=[]
        for arm in protocol['arms']:
            path=args.root/(qid+'-'+arm+'.json')
            if not path.exists():raise ValueError('Incomplete run')
            row=json.loads(path.read_text())
            if row['query_id']!=qid or row['arm']!=arm:raise ValueError('Invalid identity')
            if row['input_sha256']!=expected_hash:raise ValueError('Recorded input does not match frozen pool')
            ranking=row.get('ranking',[])
            if len(ranking)!=len(set(ranking)) or not set(ranking)<=candidates:raise ValueError('Invalid ranking candidates')
            rows.append(row)
            arm_rows[arm].append(row)
        if rows[0]['input_sha256']!=rows[1]['input_sha256']:raise ValueError('Input mismatch')
        grades={eid:grade for (query,eid),grade in judgments.items() if query==qid}
        rankings=[r.get('ranking',[]) if r['status']=='ok' else [] for r in rows]
        bounds=paired_actionable_gain_interval(rankings[1],rankings[0],grades,k=3)
        output.append({'query_id':qid,'statuses':[r['status'] for r in rows],'seconds':[r['seconds'] for r in rows],
                       'paired_gain_bounds':bounds})
    costs={}
    for arm,rows in arm_rows.items():
        complete=all(r['status']=='ok' and r.get('trace') and all(t.get('usage_present') is True for t in r['trace'])
                     and all(k in r.get('usage',{}) for k in ('prompt_tokens','completion_tokens')) for r in rows)
        costs[arm]={'mean_seconds':statistics.mean(r['seconds'] for r in rows),
                    'usage_complete':complete,
                    'mean_prompt_tokens':statistics.mean(r['usage']['prompt_tokens'] for r in rows) if complete else None,
                    'mean_completion_tokens':statistics.mean(r['usage']['completion_tokens'] for r in rows) if complete else None}
    summary={'queries':len(output),'failed_arms':sum(s!='ok' for r in output for s in r['statuses']),
             'costs':costs,
             'coverage_minus_direct_bounds':[statistics.mean(r['paired_gain_bounds'][k] for r in output) for k in ('lower','upper')],
             'judgments_sha256':hashlib.sha256((data/'judgments.jsonl').read_bytes()).hexdigest(),
             'scope':'Expert ranking against grades of the common full snapshot, not support of evidence actually shown to either model. Model silver is not independent gold; unknown grades use box bounds, not confidence intervals.'}
    print(json.dumps({'summary':summary,'queries':output},indent=2))


if __name__=='__main__':main()
