"""Query-blind source-stratified method-frame pilot on raw-derived development data."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time
from expert_search.method_frames import compile_method_frames,PROMPT,AUDIT_INSTRUCTION,CHOICES

DATA=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');ROOT=Path('artifacts/source-method-frames-v1')
BUDGET=dict(paper=40,project=15,patent=15,biography=15,research_fields=15)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    manifest=json.loads((DATA/'manifest.json').read_text());path=DATA/'corpus.jsonl'
    if sha(path)!=manifest['files_sha256'][path.name]:raise ValueError('Changed raw-derived corpus')
    strata={s:[] for s in BUDGET}
    for row in map(json.loads,path.read_text().splitlines()):
        if row['split']!='development':continue
        field=row.get('field','');s='paper' if row['kind']=='paper' else 'project' if field=='\u9879\u76ee\u5217\u8868' else 'patent' if field=='\u4e13\u5229\u5217\u8868' else 'biography' if field.startswith('\u4e2a\u4eba\u7b80\u4ecb_') else 'research_fields' if field.startswith('\u7814\u7a76\u9886\u57df_') else None
        if s in strata:strata[s].append(row)
    selected=[]
    for s,budget in BUDGET.items():
        pool=sorted(strata[s],key=lambda r:r['id']);random.Random('20261117:'+s).shuffle(pool)
        if len(pool)<budget:raise ValueError('Insufficient source stratum '+s)
        for r in pool[:budget]:
            selected.append(dict(id=r['id'],expert_id=r['expert_id'],kind=r['kind'],field=r.get('field'),text=r['text'][:1200],stratum=s,provenance=r['provenance'],split='development'))
    body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
    frozen=dict(model='gpt-6-luna',source_selection='Fixed query-blind stratified sample from development corpus, randomseed20261117:stratum, first1200characters;40papers+15projects+15patents+15biographies+15research-field lists. No demand, query labels or retrieval outputs read.',source_counts=BUDGET,corpus_sha256=sha(path),sample_sha256=hashlib.sha256(body.encode()).hexdigest(),code_sha256={str(p):sha(p) for p in [Path(__file__),Path('src/expert_search/method_frames.py'),Path('src/expert_search/span_applicability.py'),Path('src/expert_search/choice_judgments.py')]},extraction_instruction=PROMPT,audit_instruction=AUDIT_INSTRUCTION,audit_choices=CHOICES,scope='Offline representation pilot, not new target-fit labels or complete expert index.3workers,no retries. Null I/O preserved. Same-model witness-screened, not human gold/independent authorship. Heldout excluded; methodless profiles retain raw fallback.')
    ROOT.mkdir(exist_ok=True);marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=frozen:raise ValueError('Changed source-frame pilot protocol')
    sample=ROOT/'sample.jsonl'
    if sample.exists() and sample.read_text()!=body:raise ValueError('Changed source sample')
    marker.write_text(json.dumps(frozen,indent=2));sample.write_text(body);return selected


def one(source):
    path=ROOT/'frames'/(hashlib.sha256(source['id'].encode()).hexdigest()+'.json')
    if path.exists():
        r=json.loads(path.read_text())
        if r['source_id']!=source['id'] or r['source_sha256']!=hashlib.sha256(source['text'].encode()).hexdigest():raise ValueError('Stale source-frame cache')
        return r
    from laiwenai import LaiwenAI
    result=compile_method_frames(source,LaiwenAI(model='gpt-6-luna',max_retries=0));result['stratum']=source['stratum']
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2));temp.replace(path)
    print(source['stratum'],source['id'],result['status'],len(result['frames']),result.get('error_type'),flush=True);return result


def run(prepare_only=False,limit=None):
    sample=prepare()
    if prepare_only:print('Prepared',len(sample),'query-blind source views');return
    (ROOT/'frames').mkdir(exist_ok=True);start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,sample if limit is None else sample[:limit]))
    bystratum={}
    for s in BUDGET:
        rows=[r for r in results if r['stratum']==s];bystratum[s]=dict(completed_sources=len(rows),proposed_frames=sum(len(r.get('proposed_frames',[])) for r in rows),screened_frames=sum(len(r['frames']) for r in rows),sources_with_frames=sum(bool(r['frames']) for r in rows),errors=dict(Counter(r['error_type'] for r in rows if r.get('error_type'))))
    summary=dict(planned_sources=len(sample),completed_sources=len(results),strata=bystratum,api_attempts=sum(r['api_attempts'] for r in results),source_compile_seconds_sum=sum(r['offline_seconds'] for r in results),this_invocation_wall_seconds=time.perf_counter()-start,scope='API time sum includes failures and is not cohort wall time; invocation may resume cached records. Same-model screened source-only frames, not semantic gold, complete coverage, actual capability or expert-fit accuracy. No heldout sources or query-conditioned compilation.')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--limit',type=int);args=p.parse_args()
    if args.limit is not None and args.limit<1:raise ValueError('Limit must be positive')
    run(args.prepare_only,args.limit)
