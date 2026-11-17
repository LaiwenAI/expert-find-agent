"""Verify and summarize a paired retrieved-pool coverage pilot."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from .enterprise_end_to_end_pilot import sha
from .fixed_evidence_routing import preserve_candidate_coverage
from expert_search.fast_retrieval import FastSearch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    p=json.loads((args.root/'protocol.json').read_text())
    if p['arms']!=['direct','coverage_direct']:raise ValueError('Unexpected arms')
    for name,digest in {**p['code_sha256'],**p['input_sha256']}.items():
        if sha(Path(name))!=digest:raise ValueError('Changed input or code: '+name)
    if sha(Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl'))!=p['query_file_sha256']:raise ValueError('Changed queries')
    queries={r['id']:r for r in map(json.loads,Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl').read_text().splitlines()) if r['split']=='development'}
    searcher=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite')
    groups={arm:[] for arm in p['arms']};overlap=[];retrieval=[]
    for qid in p['query_ids']:
        raw=(args.root/(qid+'-pool.json')).read_bytes();digest=hashlib.sha256(raw).hexdigest();hits=json.loads(raw)
        if qid not in queries:raise ValueError('Query is not in development')
        rebuilt=searcher.search(queries[qid]['text'],p['candidates'])
        if json.dumps(rebuilt,sort_keys=True,ensure_ascii=False).encode()!=raw:raise ValueError('Pool differs from reconstructed retrieval')
        candidates={h['id'] for h in hits}
        if len(candidates)!=len(hits):raise ValueError('Duplicate candidate')
        evidence={e['id']:e for h in hits for e in h['evidence']}
        if len(evidence)!=sum(len(h['evidence']) for h in hits):raise ValueError('Duplicate evidence ID')
        for h in hits:
            if any(e['expert_id']!=h['id'] for e in h['evidence']):raise ValueError('Invalid evidence ownership')
        pair=[]
        for arm in p['arms']:
            path=args.root/(qid+'-'+arm+'.json')
            if not path.exists():raise ValueError('Incomplete run')
            row=json.loads(path.read_text())
            if (row['query_id'],row['arm'],row['input_sha256'])!=(qid,arm,digest):raise ValueError('Input or attempt mismatch')
            ranking=row.get('ranking',[])
            if len(ranking)!=len(set(ranking)) or not set(ranking)<=candidates:raise ValueError('Invalid ranking')
            if row['status']=='ok':
                selected=set(evidence) if arm=='direct' else set(preserve_candidate_coverage(hits,[]))
                if row['selected_count']!=len(selected):raise ValueError('Evidence count mismatch')
                if arm=='coverage_direct' and len(selected)!=sum(bool(h['evidence']) for h in hits):raise ValueError('Candidate coverage mismatch')
                supported=set()
                for result in row['trace'][-1]['output']['results']:
                    eid=result.get('expert_id')
                    if any(isinstance(c,str) and c in selected and evidence[c]['expert_id']==eid for c in result.get('citations',[])):supported.add(eid)
                if not set(ranking)<=supported:raise ValueError('Ranking lacks owned visible citation')
            groups[arm].append(row);pair.append(row)
        if pair[0]['retrieval_seconds']!=pair[1]['retrieval_seconds']:raise ValueError('Shared retrieval mismatch')
        retrieval.append(pair[0]['retrieval_seconds'])
        if all(r['status']=='ok' for r in pair):overlap.append(len(set(pair[0]['ranking'])&set(pair[1]['ranking'])))
    summary={}
    for arm,rows in groups.items():
        complete=all(r['status']=='ok' and r.get('trace') and all(t.get('usage_present') is True for t in r['trace']) for r in rows)
        summary[arm]={'attempts':len(rows),'successes':sum(r['status']=='ok' for r in rows),
                      'mean_attempt_seconds':statistics.mean(r['seconds'] for r in rows),
                      'usage_complete':complete,'models':sorted({str(t.get('model')) for r in rows for t in r.get('trace',[])}),
                      'mean_prompt_tokens':statistics.mean(r['usage']['prompt_tokens'] for r in rows) if complete else None,
                      'mean_completion_tokens':statistics.mean(r['usage']['completion_tokens'] for r in rows) if complete else None,
                      'mean_visible_evidence_successes':statistics.mean(r['selected_count'] for r in rows if r['status']=='ok') if any(r['status']=='ok' for r in rows) else None}
    print(json.dumps({'arms':summary,'mean_shared_retrieval_seconds':statistics.mean(retrieval),
                      'common_success_pairs':len(overlap),'mean_returned_id_overlap':statistics.mean(overlap) if overlap else None,
                      'scope':'Development-only exploratory one-attempt paired timing. Retrieval is shared and measured separately; initialization and index build excluded. Output overlap is not relevance or semantic evidence support; no independent gold.'},indent=2))


if __name__=='__main__':main()
