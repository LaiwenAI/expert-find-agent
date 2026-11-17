"""Replay agreement-gated roles on the frozen enterprise paper pilot."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from expert_search.owned_role_consensus import agree_role_records
from expert_search.owned_role_windows import segment_source,select_owned_windows
from expert_search.retrieval import tokens
from .owned_role_decision_pilot import sha


def main():
    parent=Path('artifacts/retrieved-coverage-pilot-v1');paper=Path('artifacts/owned-role-paper-pilot-v1');order=Path('artifacts/owned-role-order-pilot-v1')
    rp=json.loads((parent/'protocol.json').read_text());pp=json.loads((paper/'protocol.json').read_text());op=json.loads((order/'protocol.json').read_text())
    if sha(parent/'protocol.json')!=pp['parent_protocol_sha256'] or sha(paper/'protocol.json')!=op['parent_protocol_sha256'] or sha(paper/'sources.json')!=op['source_sha256']:raise ValueError('Changed inputs')
    for path,digest in {**pp['code_sha256'],**op['code_sha256']}.items():
        if sha(Path(path))!=digest:raise ValueError('Changed frozen code: '+path)
    qfile=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(qfile)!=rp['query_file_sha256']:raise ValueError('Changed queries')
    queries={r['id']:r for r in map(json.loads,qfile.read_text().splitlines()) if r['split']=='development'}
    sources=json.loads((paper/'sources.json').read_text());rebuilt=[];seen=set()
    for qid in rp['query_ids']:
        raw=(parent/(qid+'-pool.json')).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        for arm in rp['arms']:
            if json.loads((parent/(qid+'-'+arm+'.json')).read_text())['input_sha256']!=digest:raise ValueError('Changed retrieved pool')
        used=0
        for hit in json.loads(raw):
            for e in hit['evidence']:
                identity=(e['id'],e['expert_id'])
                if e.get('kind')!='paper' or identity in seen:continue
                if e['expert_id']!=hit['id']:raise ValueError('Invalid owner')
                seen.add(identity);rebuilt.append((qid,{'id':e['id'],'expert_id':e['expert_id'],'text':e['text'][:6000]}));used+=1
                if used==3:break
            if used==3:break
        if used!=3:raise ValueError('Incomplete source selection')
    if [s for _,s in rebuilt]!=sources:raise ValueError('Changed source selection')
    counts=Counter();costs=[];records_sha={}
    for i,(qid,source) in enumerate(rebuilt):
        windows=segment_source(source,op['width']);terms=set(tokens(queries[qid]['text']))
        ranking=sorted(range(len(windows)),key=lambda j:(-len(terms&set(tokens(windows[j]['text']))),j));records={}
        for direction in ('forward','reverse'):
            path=order/(f'source-{i+1:03d}-{direction}.json');records_sha[path.name]=sha(path);row=json.loads(path.read_text())
            if row['status']!='normalized_role_hints' or row['source_index']!=i or row['order']!=direction:raise ValueError('Incomplete role pair')
            records[direction]=row;costs.append(row['seconds'])
        combined=agree_role_records(source,records['forward']['role_record'],records['reverse']['role_record'],op['width'])
        outputs={name:select_owned_windows(source,ranking,1,rec,op['width'])['windows'][0]['id'] for name,rec in [('forward',records['forward']['role_record']),('reverse',records['reverse']['role_record']),('consensus',combined)]}
        counts['disagreed_windows']+=combined['disagreements'];counts['consensus_uncertain_windows']+=len(combined['roles']['uncertain'])
        counts['consensus_changed_from_baseline']+=outputs['consensus']!=ranking[0]
        counts['consensus_differs_from_forward']+=outputs['consensus']!=outputs['forward']
        counts['consensus_differs_from_reverse']+=outputs['consensus']!=outputs['reverse']
        counts['consensus_symmetric_under_swap']+=combined['roles']==agree_role_records(source,records['reverse']['role_record'],records['forward']['role_record'],op['width'])['roles']
    report={'sources':len(sources),'counts':dict(counts),'sum_recorded_request_seconds':sum(costs),'role_attempt_sha256':records_sha,
            'protocol_sha256':sha(order/'protocol.json'),'code_sha256':sha(Path(__file__)),
            'scope':'Post-hoc development replay, no new model calls. Two same-model judgments are not independent; agreement is not correctness or semantic confidence. Query-term window order is a control, not the production retriever. Summed sequential request times exclude all other work and do not establish benefit or break-even.'}
    out=Path('artifacts/owned-role-consensus-replay-v1');out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='role_attempt_sha256'},indent=2))


if __name__=='__main__':main()
