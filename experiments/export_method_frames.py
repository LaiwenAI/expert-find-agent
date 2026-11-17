"""Export100owned source views plus26screened frames; keep raw fallback universal."""
import hashlib
import json
from pathlib import Path
from expert_search.method_frame_store import bind_frames


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run():
    root=Path('artifacts/source-method-frames-v1');sample=root/'sample.jsonl';protocol=json.loads((root/'protocol.json').read_text());sources=list(map(json.loads,sample.read_text().splitlines()))
    if sha(sample)!=protocol['sample_sha256']:raise ValueError('Changed selected source views')
    corpus=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/corpus.jsonl')
    if sha(corpus)!=protocol['corpus_sha256']:raise ValueError('Changed raw-derived source corpus')
    needed={s['id'] for s in sources};original={r['id']:r for r in map(json.loads,corpus.read_text().splitlines()) if r['id'] in needed}
    if set(original)!=needed:raise ValueError('Missing source lineage')
    bound=[];views=[];inputs={};frames=[]
    for source in sources:
        raw=original[source['id']]
        if raw['split']!='development' or source['text']!=raw['text'][:1200] or source['expert_id']!=raw['expert_id']:raise ValueError('View does not match original source/owner/split')
        path=root/'frames'/(hashlib.sha256(source['id'].encode()).hexdigest()+'.json');record=json.loads(path.read_text());inputs[path.name]=sha(path);entry=bind_frames(source,record);bound.append(entry)
        views.append(dict(source,source_sha256=entry['source_sha256'],frame_ids=[f['id'] for f in entry['frames']],retrieval_eligibility='raw_view_always_eligible; no negative capability inference'))
        frames.extend(dict(f,record_key=source['id']+'#'+f['id'],source_sha256=entry['source_sha256'],label_status='source_only_model_screened_not_expert_fit') for f in entry['frames'])
    keys=[f['record_key'] for f in frames]
    if len(set(keys))!=len(keys):raise ValueError('Repeated global frame ID')
    out=Path('artifacts/method-frame-store-v1');out.mkdir(exist_ok=True)
    for name,rows in [('sources.jsonl',views),('bound-records.jsonl',bound),('frames.jsonl',frames)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest=dict(sources=len(views),experts=len({s['expert_id'] for s in views}),frames=len(frames),sources_with_frames=sum(bool(r['frames']) for r in bound),raw_retrieval_eligible_sources=len(views),heldout_sources=0,files_sha256={p.name:sha(p) for p in sorted(out.glob('*.jsonl'))},input_records_sha256=inputs,parent_protocol_sha256=sha(root/'protocol.json'),code_sha256={str(p):sha(p) for p in [Path(__file__),Path('src/expert_search/method_frame_store.py')]},scope='Local typed frame export and source binding, not full searchable index, target-fit labels or semantic gold. No raw source is excluded for lacking frames. All field values/conditions and citation offsets preserved; not public raw-text redistribution.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k not in ['input_records_sha256','files_sha256','code_sha256']},indent=2))


if __name__=='__main__':run()
