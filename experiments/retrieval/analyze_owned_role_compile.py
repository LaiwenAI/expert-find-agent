"""Check completion and structural source binding in the role-window pilot."""
import hashlib
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from expert_search.owned_role_windows import segment_source,select_owned_windows,source_key


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('artifacts/owned-role-compile-pilot-v1'));root=parser.parse_args().root
    protocol=json.loads((root/'protocol.json').read_text())
    if sha(root/'sources.json')!=protocol['source_sha256']:raise ValueError('Changed source views')
    if sha(Path('artifacts/retrieved-coverage-pilot-v1/protocol.json'))!=protocol['parent_protocol_sha256']:raise ValueError('Changed parent protocol')
    for path,digest in protocol['code_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Changed code')
    sources=json.loads((root/'sources.json').read_text())
    if len(sources)!=protocol['sources']:raise ValueError('Changed sample size')
    parent=Path('artifacts/retrieved-coverage-pilot-v1');parent_protocol=json.loads((parent/'protocol.json').read_text());inventory={}
    for qid in parent_protocol['query_ids']:
        pool=parent/(qid+'-pool.json');raw=pool.read_bytes()
        for arm in parent_protocol['arms']:
            if json.loads((parent/(qid+'-'+arm+'.json')).read_text())['input_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Changed retrieved pool')
        for hit in json.loads(raw):
            for evidence in hit['evidence']:
                inventory[evidence['id'],evidence['expert_id']]=evidence
    records=[];roles=Counter();normalization=Counter();kinds=Counter();reordered=0
    for i,source in enumerate(sources,1):
        original=inventory.get((source['id'],source['expert_id']))
        if original is None or original['text'][:6000]!=source['text']:raise ValueError('Source differs from retrieved view')
        kinds[original.get('kind','unknown')]+=1
        path=root/(f'source-{i:03d}.json')
        if not path.exists():raise ValueError('Incomplete run')
        row=json.loads(path.read_text());key=source_key(source,protocol['width'])
        if row['source_key']!=key:raise ValueError('Changed source binding')
        windows=segment_source(source,protocol['width']);ranking=list(range(len(windows)))
        selected=select_owned_windows(source,ranking,len(windows),row['role_record'],protocol['width'])
        if row['status']=='normalized_role_hints':
            if selected['status']!='role_priority':raise ValueError('Invalid normalized role record')
            roles.update({role:len(ids) for role,ids in row['role_record']['roles'].items()})
            normalization.update(row['role_record']['normalization'])
            reordered+=ranking!=[w['id'] for w in selected['windows']]
        elif row['status']!='raw_fallback' or selected['status']!='baseline_no_roles':raise ValueError('Invalid fallback')
        records.append(row)
    complete=all(r.get('usage_present') is True for r in records)
    report={'sources':len(records),'source_kinds':dict(kinds),'statuses':dict(Counter(r['status'] for r in records)),
            'role_window_counts':dict(roles),'normalization_counts':dict(normalization),'reordered_source_order':reordered,
            'mean_attempt_seconds':statistics.mean(r['seconds'] for r in records),'usage_complete':complete,
            'mean_prompt_tokens':statistics.mean(r['usage']['prompt_tokens'] for r in records) if complete else None,
            'mean_completion_tokens':statistics.mean(r['usage']['completion_tokens'] for r in records) if complete else None,
            'models':sorted({str(r.get('model')) for r in records if r['status']=='normalized_role_hints'}),
            'scope':'Structural, query-blind development pilot on fixed-width source views; role counts and reordering do not measure semantic accuracy, expert authorship, relevance, retrieval quality or runtime amortization. Missing API usage is not imputed.'}
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
