"""Analyze completed paired proxy results without dropping failed arms."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from expert_search.benchmark import metrics


def analyze(root):
    protocol=json.loads((root/'protocol.json').read_text())
    runs=Path('artifacts/benchmark/runs.jsonl')
    if hashlib.sha256(runs.read_bytes()).hexdigest()!=protocol['files_sha256'][str(runs)]:
        raise ValueError('Benchmark labels differ from the frozen input')
    qrels={}
    for line in runs.read_text().splitlines():
        row=json.loads(line);qrels.setdefault(row['query_id'],row['qrels'])
    records=[]
    for qid in protocol['query_ids']:
        pair=[]
        for arm in protocol['arms']:
            path=root/(qid+'-'+arm+'.json')
            if not path.exists():raise ValueError('Incomplete run: '+str(path))
            row=json.loads(path.read_text())
            if row['query_id']!=qid or row['arm']!=arm:raise ValueError('Invalid result identity')
            pair.append(row)
        if pair[0]['input_sha256']!=pair[1]['input_sha256']:raise ValueError('Input mismatch')
        records.append(pair)
    out={'queries':len(records),'arms':{}}
    for i,arm in enumerate(protocol['arms']):
        rows=[pair[i] for pair in records]
        scores=[metrics(r.get('ranking',[]) if r['status']=='ok' else [],qrels[r['query_id']]) for r in rows]
        out['arms'][arm]={'attempts':len(rows),'failures':sum(r['status']!='ok' for r in rows),
                          'mean_seconds':statistics.mean(r['seconds'] for r in rows),
                          'mean_mrr10':statistics.mean(s['mrr@10'] for s in scores),
                          'mean_recall10':statistics.mean(s['recall@10'] for s in scores)}
        usage=[r.get('usage') for r in rows]
        complete=all(isinstance(u,dict) and all(type(u.get(k)) is int and u[k]>=0 for k in ('prompt_tokens','completion_tokens')) and bool(r.get('trace')) and all(t.get('usage_present') is True for t in r['trace']) for r,u in zip(rows,usage))
        out['arms'][arm]['token_usage_complete']=complete
        if complete:
            out['arms'][arm]['mean_input_tokens']=statistics.mean(u['prompt_tokens'] for u in usage)
            out['arms'][arm]['mean_output_tokens']=statistics.mean(u['completion_tokens'] for u in usage)
    out['scope']='Author recovery proxy with incomplete labels; failed arms score empty; timing excludes retrieval and initialization.'
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(analyze(args.root),indent=2))
