"""Create private human-review forms for structured method hypotheses."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from expert_search.compiled_expert_search import corpus_snapshot
from expert_search.corpus import records


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run,output):
    protocol=json.loads((run/'protocol.json').read_text());summary=json.loads((run/'summary.json').read_text())
    if sha(run/'protocol.json')!=summary['protocol_sha256'] or sha(run/'view-hashes.json')!=summary['view_hashes_sha256']:raise ValueError('Changed pilot artifacts')
    corpus=Path('artifacts/corpus');queries=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(queries)!=protocol['queries_sha256']:raise ValueError('Changed query source')
    source_queries={r['id']:r for _,r in records(queries) if r['split']=='development'};attempts=[];hashes={};source_ids=set()
    for qid in protocol['query_ids']:
        path=run/(hashlib.sha256(qid.encode()).hexdigest()+'.json');row=json.loads(path.read_text());hashes[path.name]=sha(path)
        if row['query_id']!=qid or qid not in source_queries:raise ValueError('Changed query identity')
        arm=row['arms']['structured']
        if arm['status']!='valid':continue
        result=arm['result']
        if result['source_snapshot_sha256']!=corpus_snapshot(corpus):raise ValueError('Changed corpus snapshot')
        source_ids.update(result['source_view_sha256']);attempts.append((qid,result))
    sources={r['id']:r for _,r in records(corpus/'evidence.jsonl') if r['id'] in source_ids}
    if set(sources)!=source_ids:raise ValueError('Missing source views')
    items=[]
    for qid,result in attempts:
        for sid,h in result['source_view_sha256'].items():
            text=sources[sid]['text'][:protocol['source_chars']]
            if hashlib.sha256(text.encode()).hexdigest()!=h:raise ValueError('Changed source prefix')
        for rank,row in enumerate(result['results'],1):
            eid=row['expert_id'];views=[s for sid,s in sources.items() if sid in result['source_view_sha256'] and s['expert_id']==eid]
            if not views and row['evidence']:raise ValueError('Missing owned source')
            for ref in row['evidence']:
                if ref['expert_id']!=eid:raise ValueError('Foreign reference')
                for c in ref['citations']:
                    source=sources[c['evidence_id']];text=source['text'][:protocol['source_chars']]
                    if source['expert_id']!=eid or text[c['start']:c['end']]!=c['quote']:raise ValueError('Changed citation')
            items.append((qid,rank,row,views))
    random.Random(20261122).shuffle(items);forms=[];mapping=[]
    for i,(qid,rank,row,views) in enumerate(items,1):
        case=f'case-{i:03d}';aliases={s['id']:f'source-{j}' for j,s in enumerate(views,1)}
        citations=[{'source_id':aliases[c['evidence_id']],'start':c['start'],'end':c['end'],'quote':c['quote']} for ref in row['evidence'] for c in ref['citations']]
        forms.append({'case_id':case,'original_need':source_queries[qid]['text'],'level':row['level'],'demand_citations':row['demand_citations'],
                      'mechanism':row['mechanism'],'assumptions':row['assumptions'],'remaining_demand_gaps':row['remaining_demand_gaps'],
                      'source_views':[{'source_id':aliases[s['id']],'kind':s.get('kind','unknown'),'text':s['text'][:protocol['source_chars']]} for s in views],
                      'source_citations':citations,'historical_support':None,'target_alignment':None,'bridge_validity':None,'personal_contribution':None,
                      'rationale':None,'reviewer_id':None})
        mapping.append({'case_id':case,'query_id':qid,'rank':rank,'expert_id':row['expert_id'],'source_ids':[s['id'] for s in views]})
    output.mkdir(parents=True,exist_ok=False);payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode();private=(json.dumps(mapping,indent=2)+'\n').encode()
    (output/'blind-review.jsonl').write_bytes(payload);(output/'private-mapping.json').write_bytes(private)
    report={'cases':len(forms),'pilot_protocol_sha256':sha(run/'protocol.json'),'pilot_summary_sha256':sha(run/'summary.json'),
            'source_snapshot_sha256':corpus_snapshot(corpus),'attempt_sha256':hashes,'review_sha256':hashlib.sha256(payload).hexdigest(),
            'mapping_sha256':hashlib.sha256(private).hexdigest(),'levels':dict(Counter(f['level'] for f in forms)),
            'scope':'Private review of model proposals, not a blinded comparison of methods; format reveals the structured arm. IDs and ranks hidden, source text may reveal identity. Same 1,200-character source prefixes shown to the model. Unknown contribution or missing support must not be read as inability. No human judgments are supplied.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'cases':len(forms),'levels':report['levels'],'scope':report['scope']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.run,a.output)
