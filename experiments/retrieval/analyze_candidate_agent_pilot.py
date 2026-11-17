"""Verify and summarize candidate-first versus single-pass development runs."""
import argparse
import json
import statistics
from pathlib import Path

from .enterprise_end_to_end_pilot import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    p=json.loads((args.root/'protocol.json').read_text())
    if p['arms']!=['candidate_first','single_pass']:raise ValueError('Unexpected arms')
    for path,digest in {**p['inputs_sha256'],**p['code_sha256']}.items():
        if sha(Path(path))!=digest:raise ValueError('Changed input or code: '+path)
    if sha(Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl'))!=p['query_sha256']:raise ValueError('Changed query snapshot')
    for name,digest in p['model_files_sha256'].items():
        if sha(Path('artifacts/teacher-compiled-lm')/name)!=digest:raise ValueError('Changed model file')
    groups={arm:[] for arm in p['arms']};overlaps=[]
    for qid in p['query_ids']:
        pair=[]
        for arm in p['arms']:
            path=args.root/(qid+'-'+arm+'.json')
            if not path.exists():raise ValueError('Incomplete run')
            row=json.loads(path.read_text())
            if (row['query_id'],row['arm'])!=(qid,arm):raise ValueError('Invalid attempt identity')
            if row['status']=='ok':
                result=row['result'];candidates=result['candidate_ids'];ranked=[v['expert']['id'] for v in result['results']]
                if len(set(candidates))!=len(candidates) or len(set(ranked))!=len(ranked) or not set(ranked)<=set(candidates):raise ValueError('Invalid candidates')
                if any(not v['evidence'] or any(e['expert_id']!=v['expert']['id'] for e in v['evidence']) for v in result['results']):raise ValueError('Invalid evidence ownership')
            groups[arm].append(row);pair.append(row)
        if all(r['status']=='ok' for r in pair):overlaps.append(len(set(pair[0]['result']['candidate_ids'])&set(pair[1]['result']['candidate_ids'])))
    output={}
    for arm,rows in groups.items():
        complete=all(r['status']=='ok' and r['result']['trace'] and all(t.get('usage_present') is True for t in r['result']['trace']) for r in rows)
        valid=[r for r in rows if r['status']=='ok']
        output[arm]={'attempts':len(rows),'successes':len(valid),'mean_attempt_seconds':statistics.mean(r['attempt_seconds'] for r in rows),
                     'usage_complete':complete,'models':sorted({str(t.get('model')) for r in valid for t in r['result']['trace']}),
                     'mean_prompt_tokens':statistics.mean(r['result']['usage']['prompt_tokens'] for r in rows) if complete else None,
                     'mean_completion_tokens':statistics.mean(r['result']['usage']['completion_tokens'] for r in rows) if complete else None,
                     'mean_returned_successes':statistics.mean(len(r['result']['results']) for r in valid) if valid else None}
        if valid:
            model_times=[sum(t['seconds'] for t in r['result']['trace']) for r in valid]
            output[arm]['mean_model_call_seconds_successes']=statistics.mean(model_times)
            output[arm]['mean_non_model_seconds_successes']=statistics.mean(r['attempt_seconds']-seconds for r,seconds in zip(valid,model_times))
            output[arm]['max_model_call_seconds_successes']=max(model_times)
            output[arm]['reasoning_tokens_complete']=all(all(t.get('reasoning_tokens') is not None for t in r['result']['trace']) for r in valid)
            output[arm]['mean_reasoning_tokens_successes']=statistics.mean(sum(t['reasoning_tokens'] for t in r['result']['trace']) for r in valid) if output[arm]['reasoning_tokens_complete'] else None
    print(json.dumps({'arms':output,'common_success_pairs':len(overlaps),'mean_candidate_overlap':statistics.mean(overlaps) if overlaps else None,
                      'scope':'Exploratory development timing, different candidates and evidence budgets; no gold, semantic citation verification or causal attribution. Initialization and index build excluded.'},indent=2))


if __name__=='__main__':main()
