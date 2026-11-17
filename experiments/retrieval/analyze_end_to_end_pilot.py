"""Summarize paired full-search attempts without fabricating relevance labels."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    protocol=json.loads((args.root/'protocol.json').read_text())
    if protocol['arms']!=['single_pass','tree_agent']:raise ValueError('Unexpected arms')
    for path,digest in protocol['input_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Changed input: '+path)
    if sha(Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl'))!=protocol['query_file_sha256']:raise ValueError('Changed queries')
    if any(sha(Path(path))!=digest for path,digest in protocol['code_sha256'].items()):raise ValueError('Changed code')
    catalog_key=hashlib.sha256(str(Path('artifacts/corpus').resolve()).encode()).hexdigest()[:16]
    if sha(Path('artifacts/expert-catalog-'+catalog_key)/'catalog.sqlite')!=protocol['catalog_sha256']:raise ValueError('Changed catalog')
    rows=[]
    for qid in protocol['query_ids']:
        for arm in protocol['arms']:
            path=args.root/(qid+'-'+arm+'.json')
            if not path.exists():raise ValueError('Incomplete run')
            row=json.loads(path.read_text())
            if row['query_id']!=qid or row['arm']!=arm:raise ValueError('Invalid attempt identity')
            if row['status']=='ok':
                result=row['result'];candidates=result['candidate_ids'];ranked=[r['expert']['id'] for r in result['results']]
                if len(candidates)!=len(set(candidates)) or len(ranked)!=len(set(ranked)) or not set(ranked)<=set(candidates):raise ValueError('Invalid candidate or result identity')
                if any(not r['evidence'] or any(e['expert_id']!=r['expert']['id'] for e in r['evidence']) for r in result['results']):raise ValueError('Invalid evidence ownership')
            rows.append(row)
    summary={}
    for arm in protocol['arms']:
        group=[r for r in rows if r['arm']==arm];valid=[r for r in group if r['status']=='ok']
        traces=[r['result']['trace'] if r['status']=='ok' else r.get('trace',[]) for r in group]
        usage_complete=all(r['status']=='ok' and trace and all(t.get('usage_present') is True for t in trace) for r,trace in zip(group,traces))
        models=sorted({str(t.get('model')) for trace in traces for t in trace})
        summary[arm]={'attempts':len(group),'successes':len(valid),'mean_attempt_seconds':statistics.mean(r['attempt_seconds'] for r in group),
                      'mean_retrieval_seconds_successes':statistics.mean(r['result']['timing']['retrieval_seconds'] for r in valid) if valid else None,
                      'mean_returned_successes':statistics.mean(len(r['result']['results']) for r in valid) if valid else None,
                      'models':models,'usage_complete':usage_complete,
                      'mean_prompt_tokens':statistics.mean(r['result']['usage']['prompt_tokens'] for r in group) if usage_complete else None,
                      'mean_completion_tokens':statistics.mean(r['result']['usage']['completion_tokens'] for r in group) if usage_complete else None}
    overlap=[]
    for i in range(0,len(rows),2):
        left,right=rows[i:i+2]
        if left['status']==right['status']=='ok':
            overlap.append(len(set(left['result']['candidate_ids'])&set(right['result']['candidate_ids'])))
    print(json.dumps({'arms':summary,'common_success_pairs':len(overlap),
                      'mean_candidate_overlap':statistics.mean(overlap) if overlap else None,
                      'scope':'One attempt per arm per development query; all-attempt timing excludes initialization and index build. No relevance or citation-support validation; no independent gold. Candidate pools and prompts differ; candidate overlap is not accuracy.'},indent=2))


if __name__=='__main__':main()
