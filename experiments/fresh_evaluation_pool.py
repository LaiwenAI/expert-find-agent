"""Raw-derived, prior-disjoint, paper-backed multi-candidate evaluation collection."""
import hashlib
import json
from pathlib import Path
import random
import re
import numpy as np

ROOT=Path('artifacts/fresh-method-data-v1')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def nh(text):return hashlib.sha256(re.sub(r'[^\w]+','',text.casefold()).encode()).hexdigest()


def run():
    lineage=json.loads((ROOT/'raw-lineage.json').read_text())
    branch_root=Path('artifacts/global-expert-branches-v1');branch_manifest=json.loads((branch_root/'manifest.json').read_text())
    if lineage['normalized_sha256']['evidence.jsonl']!=branch_manifest['source_sha256']:raise ValueError('Vectors do not trace to current original teachers')
    parent=Path('artifacts/global-expert-branch-benchmark-v1');base=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    source=branch_root/'branches.jsonl';vp=parent/'vectors.npy'
    if sha(source)!=base['roots_sha256'] or sha(vp)!=summary['vectors_sha256']:raise ValueError('Changed source/vector snapshot')
    queries=[json.loads(s) for s in (ROOT/'queries.jsonl').read_text().splitlines()];old=json.loads((ROOT/'prior-use.json').read_text())
    protocol=dict(seed=20261117,dense_candidates=5,random_candidates=2,paper_witnesses=3,detail_profile_witnesses=1,
                  inputs_sha256={str(p):sha(p) for p in [ROOT/'queries.jsonl',ROOT/'selection.json',ROOT/'prior-use.json',ROOT/'raw-lineage.json']},source_sha256=base['roots_sha256'],vectors_sha256=summary['vectors_sha256'],code_sha256=sha(Path(__file__)),
                  split='First80 shuffled fresh demands development, remaining20 sealed. Unseen eligible experts grouped by exact/normalized paper prefix and title aliases before any labels; fixed random component split targeting20percent experts for heldout. Shared exact profile prefixes across splits excluded.',
                  scope='Multi-candidate provided-evidence evaluation pool, not fully judged expert ranking gold.5dense+2random experts per new demand within its source-disjoint split; each gets up to3nearest papers+1nontrivial profile. Source views may be incomplete. No future success or actual inability labels.')
    marker=ROOT/'pool-protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed new evaluation pool protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    records=[json.loads(s) for s in source.read_text().splitlines()];full=np.load(vp,mmap_mode='r',allow_pickle=False)
    used_e=set(old['expert_ids']);used_x=set(old['source_prefix_hashes']);used_n=set(old['normalized_source_hashes']);used_t=set(old['paper_title_hashes'])
    def unseen(r):return r['expert_id'] not in used_e and hashlib.sha256(r['text'].encode()).hexdigest() not in used_x and nh(r['text']) not in used_n
    paper_positions=[]
    for i,r in enumerate(records):
        if r['kind']=='paper' and unseen(r) and nh(r['text'].splitlines()[0]) not in used_t:paper_positions.append(i)
    papers=[records[i] for i in paper_positions];eids=sorted({r['expert_id'] for r in papers});parent_id={e:e for e in eids}
    def find(e):
        while parent_id[e]!=e:parent_id[e]=parent_id[parent_id[e]];e=parent_id[e]
        return e
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent_id[max(a,b)]=min(a,b)
    seen={}
    for r in papers:
        for key in ['text:'+nh(r['text']),'title:'+nh(r['text'].splitlines()[0])]:
            if key in seen:union(r['expert_id'],seen[key])
            else:seen[key]=r['expert_id']
    groups={}
    for eid in eids:groups.setdefault(find(eid),[]).append(eid)
    components=sorted(groups.values(),key=lambda g:g[0]);random.Random(20261117).shuffle(components)
    heldout=set();target=round(.2*len(eids))
    for group in components:
        if len(heldout)<target:heldout.update(group)
    splits={e:'heldout' if e in heldout else 'development' for e in eids}
    profile_candidates=[i for i,r in enumerate(records) if r['kind']=='profile' and r['expert_id'] in splits and unseen(r)]
    profile_splits={}
    for i in profile_candidates:profile_splits.setdefault(nh(records[i]['text']),set()).add(splits[records[i]['expert_id']])
    safe_profiles=[i for i in profile_candidates if len(profile_splits[nh(records[i]['text'])])==1]
    profiles_by_owner={}
    for i in safe_profiles:
        # Detail experience supplements papers; broad field-only labels are not method witnesses.
        field=records[i]['id'].rsplit(':',1)[-1]
        if field in {'项目列表','专利列表','个人简介_CN','个人简介_EN'}:profiles_by_owner.setdefault(records[i]['expert_id'],[]).append(i)
    matrix=np.ascontiguousarray(full[paper_positions]);owners={}
    for i,r in enumerate(papers):owners.setdefault(r['expert_id'],[]).append(i)
    order=np.asarray([i for e in eids for i in owners[e]],dtype=np.int64);starts=np.asarray([0]+list(np.cumsum([len(owners[e]) for e in eids])[:-1]),dtype=np.int64)
    with np.load(ROOT/'query-vectors.npz',allow_pickle=False) as f:qvectors=f['vectors'];qids=f['ids']
    if list(qids)!=[q['id'] for q in queries] or qvectors.shape!=(100,1024):raise ValueError('Wrong new query vectors')
    jobs=[];snapshot_rows=[]
    for qi,(query,q) in enumerate(zip(queries,qvectors)):
        split='development' if qi<80 else 'heldout';pool=[e for e in eids if splits[e]==split]
        scores=np.einsum('ij,j->i',matrix,q,optimize=False);maxima=np.maximum.reduceat(scores[order],starts);by_score=dict(zip(eids,map(float,maxima)))
        dense=sorted(pool,key=lambda e:(-by_score[e],e))[:5];rng=random.Random(query['id']);randoms=rng.sample([e for e in pool if e not in dense],2)
        chosen=dense+randoms;rng.shuffle(chosen);candidates={}
        for eid in chosen:
            indices=sorted(owners[eid],key=lambda i:(-float(scores[i]),papers[i]['id']))[:3]
            sources={papers[i]['id']:papers[i] for i in indices}
            details=profiles_by_owner.get(eid,[])
            if details:
                values=np.einsum('ij,j->i',full[details],q,optimize=False);best=min(range(len(details)),key=lambda i:(-float(values[i]),records[details[i]]['id']));r=records[details[best]];sources[r['id']]=r
            candidates[eid]=sources
            snapshot_rows.append(dict(query_id=query['id'],expert_id=eid,split=split,sources=sources,source_sha256={sid:hashlib.sha256(s['text'].encode()).hexdigest() for sid,s in sources.items()},expert_ability=None,future_project_success=None,human_grade=None))
        jobs.append(dict(id=query['id'],demand=query['demand'],demand_url=query['url'],demand_source_row=query['source_row'],split=split,candidates=candidates,dense_order=dense,random_candidates=randoms,status='unjudged_new_source_pool'))
    outputs={'evaluation-jobs.jsonl':jobs,'snapshot-views.jsonl':snapshot_rows}
    for name,rows in outputs.items():
        body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);path=ROOT/name
        if path.exists() and path.read_text()!=body:raise ValueError('Changed evaluation source views')
        path.write_text(body)
    (ROOT/'expert-splits.json').write_text(json.dumps(splits,indent=2))
    (ROOT/'eligible-source-ids.json').write_text(json.dumps([r['id'] for r in papers]+[records[i]['id'] for i in safe_profiles]))
    bysplit={s:[r for r in snapshot_rows if r['split']==s] for s in ['development','heldout']}
    sets={s:dict(experts={r['expert_id'] for r in rows},sources={sid for r in rows for sid in r['sources']},texts={nh(v['text']) for r in rows for v in r['sources'].values()}) for s,rows in bysplit.items()}
    overlaps={k:len(sets['development'][k]&sets['heldout'][k]) for k in ['experts','sources','texts']}
    if any(overlaps.values()):raise ValueError('New development/heldout source leak')
    report=dict(queries=100,candidate_views=len(snapshot_rows),development_queries=80,heldout_queries=20,new_eligible_experts=len(eids),paper_source_rows=len(papers),safe_profile_rows=len(safe_profiles),paper_connected_components=len(components),largest_component=max(map(len,components)),heldout_eligible_experts=len(heldout),
                selected_experts=len({r['expert_id'] for r in snapshot_rows}),selected_sources=len({sid for r in snapshot_rows for sid in r['sources']}),development_heldout_exact_overlap=overlaps,
                output_sha256={name:sha(ROOT/name) for name in outputs},
                warning='Actually new raw-derived evaluation pool, currently unjudged. Expert/paper aliases grouped before labels; semantic duplicates and team/family dependence unverified. Heldout annotations must stay sealed during method development. Candidate-pool/evidence-view scope, not complete corpus relevance judgments.')
    (ROOT/'pool-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
