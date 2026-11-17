"""Replay choice-order role hints on query-ranked owned paper windows."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from expert_search.owned_role_windows import segment_source,select_owned_windows
from expert_search.retrieval import tokens
from .owned_role_decision_pilot import sha


def main():
    parent=Path('artifacts/retrieved-coverage-pilot-v1');paper=Path('artifacts/owned-role-paper-pilot-v1')
    order=Path('artifacts/owned-role-order-pilot-v1');op=json.loads((order/'protocol.json').read_text())
    pp=json.loads((paper/'protocol.json').read_text());rp=json.loads((parent/'protocol.json').read_text())
    if sha(paper/'sources.json')!=op['source_sha256'] or sha(paper/'protocol.json')!=op['parent_protocol_sha256'] or sha(parent/'protocol.json')!=pp['parent_protocol_sha256']:raise ValueError('Changed protocol or sources')
    for path,digest in {**op['code_sha256'],**pp['code_sha256']}.items():
        if sha(Path(path))!=digest:raise ValueError('Changed code: '+path)
    qfile=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(qfile)!=rp['query_file_sha256']:raise ValueError('Changed queries')
    queries={r['id']:r for r in map(json.loads,qfile.read_text().splitlines()) if r['split']=='development'}
    sources=json.loads((paper/'sources.json').read_text());rebuilt=[];seen=set()
    for qid in rp['query_ids']:
        raw=(parent/(qid+'-pool.json')).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        for arm in rp['arms']:
            if json.loads((parent/(qid+'-'+arm+'.json')).read_text())['input_sha256']!=digest:raise ValueError('Changed candidate pool')
        used=0
        for hit in json.loads(raw):
            for e in hit['evidence']:
                identity=(e['id'],e['expert_id'])
                if e.get('kind')!='paper' or identity in seen:continue
                if e['expert_id']!=hit['id']:raise ValueError('Invalid source ownership')
                seen.add(identity);rebuilt.append((qid,{'id':e['id'],'expert_id':e['expert_id'],'text':e['text'][:6000]}));used+=1
                if used==3:break
            if used==3:break
        if used!=3:raise ValueError('Incomplete paper sample')
    if [row for _,row in rebuilt]!=sources:raise ValueError('Source selection mismatch')
    counts=Counter();statuses=Counter();attempt_hashes={}
    for i,(qid,source) in enumerate(rebuilt):
        windows=segment_source(source,op['width']);query=set(tokens(queries[qid]['text']))
        ranking=sorted(range(len(windows)),key=lambda j:(-len(query&set(tokens(windows[j]['text']))),j))
        if any(query&set(tokens(w['text'])) for w in windows):counts['sources_with_lexical_overlap']+=1
        baseline=ranking[0];chosen={}
        for direction in ('forward','reverse'):
            path=order/(f'source-{i+1:03d}-{direction}.json');attempt_hashes[path.name]=sha(path);record=json.loads(path.read_text())
            if record['source_index']!=i or record['order']!=direction:raise ValueError('Invalid attempt identity')
            selected=select_owned_windows(source,ranking,1,record['role_record'],op['width'])
            statuses[direction,selected['status']]+=1;chosen[direction]=selected['windows'][0]['id']
            counts[direction+'_changed_from_lexical']+=chosen[direction]!=baseline
        counts['forward_reverse_top1_different']+=chosen['forward']!=chosen['reverse']
    report={'sources':len(sources),'counts':dict(counts),'statuses':[{'order':a,'status':b,'count':n} for (a,b),n in sorted(statuses.items())],
            'order_protocol_sha256':sha(order/'protocol.json'),'decision_attempt_sha256':attempt_hashes,
            'scope':'Post-hoc fixed-window replay on 12 reused development paper views. Baseline orders windows by count of unique overlapping query tokens with deterministic position ties, not the production retriever. Order variants differ in model output; no human support labels or evidence-quality claim.'}
    (order/'window-replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='decision_attempt_sha256'},indent=2))


if __name__=='__main__':main()
