"""Build a development-only blinded dense/lexical evidence review pool."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(p):return [json.loads(line) for line in p.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');audit=Path('artifacts/fresh-lexical-pool-audit-v1')
    manifest=json.loads((root/'manifest.json').read_text());previous=json.loads((audit/'summary.json').read_text())
    inputs=('queries.jsonl','pools.jsonl','snapshots.jsonl','corpus.jsonl')
    for name in inputs:
        if sha(root/name)!=manifest['files_sha256'][name]:raise ValueError('Dataset hash mismatch')
    if sha(audit/'runs.jsonl')!=previous['runs_sha256']:raise ValueError('Lexical audit hash mismatch')
    for name,digest in previous['inputs_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Lexical audit input mismatch')
    queries={r['id']:r for r in records(root/'queries.jsonl') if r['split']=='development'}
    pools={r['query_id']:r for r in records(root/'pools.jsonl') if r['split']=='development'}
    corpus={r['id']:r for r in records(root/'corpus.jsonl') if r['split']=='development'}
    snapshots={(r['query_id'],r['expert_id']):r for r in records(root/'snapshots.jsonl') if r['split']=='development'}
    lexical={r['query_id']:r for r in records(audit/'runs.jsonl') if r['variant']=='full-demand'}
    if len(queries)!=80 or set(queries)!=set(pools) or set(queries)!=set(lexical):raise ValueError('Invalid development cohort')
    items=[];counts=Counter()
    for qid in sorted(queries):
        dense=set(pools[qid]['dense_order']);randoms=set(pools[qid]['random_candidates'])
        rank=lexical[qid]['ranking'];lex={r['expert_id'] for r in rank}
        if len(lex)!=len(rank) or dense&randoms or set(pools[qid]['candidates'])!=dense|randoms:raise ValueError('Invalid candidate pool')
        sources={eid:{} for eid in dense|randoms|lex}
        def add(eid,sid,start,end,expected=None):
            row=corpus.get(sid)
            if row is None or row['expert_id']!=eid or not 0<=start<end<=len(row['text']):raise ValueError('Invalid source view')
            text=row['text'][start:end];digest=hashlib.sha256(text.encode()).hexdigest()
            if expected is not None and expected!=digest:raise ValueError('Source view hash mismatch')
            previous_view=sources[eid].get(sid)
            if previous_view is not None and (previous_view['start'],previous_view['end'])!=(start,end):raise ValueError('Incompatible source views')
            sources[eid][sid]={'start':start,'end':end,'text':text,'sha256':digest,'kind':row['kind']}
        for eid in dense|randoms:
            snap=snapshots.get((qid,eid))
            if snap is None:raise ValueError('Missing frozen dense snapshot')
            for src in snap['sources']:add(eid,src['source_id'],src['start'],src['end'],src['sha256'])
        for hit in rank:
            for sid in hit['source_ids']:
                row=corpus.get(sid)
                if row is None:raise ValueError('Missing lexical source')
                add(hit['expert_id'],sid,0,min(1200,len(row['text'])))
        counts['queries']+=1;counts['candidates']+=len(sources);counts['lexical_outside_original']+=len(lex-(dense|randoms))
        for eid,views in sorted(sources.items()):
            if not views:raise ValueError('Candidate has no evidence')
            items.append((qid,eid,views))
    rng=random.Random(20261125);rng.shuffle(items)
    forms=[];mapping=[]
    for i,(qid,eid,views) in enumerate(items,1):
        case=f'case-{i:04d}';ordered=sorted(views.items());rng.shuffle(ordered)
        forms.append({'case_id':case,'need':queries[qid]['text'],
                      'sources':[{'id':f'source-{j}','kind':view['kind'],'text':view['text']} for j,(_,view) in enumerate(ordered,1)],
                      'method_fit':None,'constraint_support':None,'personal_contribution':None,'rationale':None,'reviewer_id':None})
        mapping.append({'case_id':case,'query_id':qid,'expert_id':eid,
                        'sources':[{'alias':f'source-{j}','id':sid,'start':v['start'],'end':v['end'],'sha256':v['sha256']} for j,(sid,v) in enumerate(ordered,1)]})
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('blind-review.jsonl',forms),('private-mapping.jsonl',mapping)]:
        (args.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report={'queries':counts['queries'],'candidate_views':counts['candidates'],'lexical_candidates_outside_original_pool':counts['lexical_outside_original'],
            'source_views':sum(len(r['sources']) for r in forms),'human_judgments':0,'seed':20261125,
            'inputs_sha256':{**{n:sha(root/n) for n in inputs},'lexical_runs':sha(audit/'runs.jsonl'),'lexical_summary':sha(audit/'summary.json'),'script':sha(Path(__file__))},
            'outputs_sha256':{n:sha(args.output/n) for n in ('blind-review.jsonl','private-mapping.jsonl')},
            'scope':'Private development-only union of frozen dense/random candidates and lexical top five, with unioned supplied source views. Names, rankings and method provenance are omitted, but source text can reveal identity. Earlier fixed-view judgments are not transferable; all new labels remain unknown. Pools exclude unreturned candidates and sources; no new independent relevance or ability assessment is made. Heldout data not read into packet.'}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
