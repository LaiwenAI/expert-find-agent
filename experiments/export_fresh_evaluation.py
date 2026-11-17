"""Build a portable local evaluation dataset directly from original data records."""
import hashlib
import json
from pathlib import Path
import random
from expert_search.corpus import FIELDS

ROOT=Path('artifacts/fresh-method-data-v1')
OUT=Path('artifacts/dataset-release/fresh-enterprise-eval-v1')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_jsonl(name,rows):
    path=OUT/name;partial=path.with_suffix('.partial')
    if partial.exists():raise ValueError('Inspect interrupted dataset export first')
    with partial.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    if path.exists() and sha(path)!=sha(partial):raise ValueError('Changed frozen dataset member: '+name)
    partial.replace(path)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    lineage=json.loads((ROOT/'raw-lineage.json').read_text());selection=json.loads((ROOT/'selection.json').read_text())
    for path,expected in lineage['raw_teacher_files'].items():
        if sha(Path(path))!=expected:raise ValueError('Original teacher file changed')
    workbook=next(Path('data/真实企业需求').glob('*.xlsx'))
    if sha(workbook)!=selection['workbook_sha256']:raise ValueError('Original demand workbook changed')
    queries=[json.loads(s) for s in (ROOT/'queries.jsonl').read_text().splitlines()];jobs=[json.loads(s) for s in (ROOT/'evaluation-jobs.jsonl').read_text().splitlines()]
    wanted=set(json.loads((ROOT/'eligible-source-ids.json').read_text()));splits=json.loads((ROOT/'expert-splits.json').read_text())
    selected_ids={sid for j in jobs for ss in j['candidates'].values() for sid in ss};selected_docs={};found=set();experts=[];seen=set()
    def documents():
        raw_root=Path('data/teachers/shenzhen')
        for path in sorted(raw_root.glob('*.jsonl')):
            if path.name=='papers.jsonl':continue
            with path.open('rb') as stream:
                for line,body in enumerate(stream,1):
                    if not body.strip():continue
                    row=json.loads(body);eid=row['teacher_id']
                    if eid in seen:continue
                    seen.add(eid)
                    if eid not in splits:continue
                    provenance=dict(file=str(path),line=line,record_sha256=hashlib.sha256(body).hexdigest())
                    experts.append(dict(id=eid,name=row.get('教师姓名_CN') or row.get('教师姓名_EN'),institution=row.get('机构名_CN',''),college=row.get('学院名_CN',''),url=row.get('教师详情页链接或访问方式_CN',''),split=splits[eid],provenance=provenance))
                    for field in FIELDS:
                        sid=eid+':'+field;value=row.get(field)
                        if sid not in wanted:continue
                        if not value:raise ValueError('Selected profile disappeared from raw row')
                        text=str(value)
                        doc=dict(id=sid,expert_id=eid,kind='profile',field=field,text=text,original_value=value,split=splits[eid],provenance=dict(provenance,field=field))
                        found.add(sid)
                        if sid in selected_ids:selected_docs[sid]=doc
                        yield doc
        path=raw_root/'papers.jsonl'
        with path.open('rb') as stream:
            for line,body in enumerate(stream,1):
                if not body.strip():continue
                sid='paper:'+str(line)
                if sid not in wanted:continue
                row=json.loads(body);eid=row['teacher_id']
                if eid not in splits:raise ValueError('Selected paper owner absent from raw teacher split')
                text='\n'.join(str(row.get(k) or '') for k in ['title','gs_abstract'])
                doc=dict(id=sid,expert_id=eid,kind='paper',text=text,title=row.get('title'),url=row.get('gs_link'),split=splits[eid],original_value={k:row.get(k) for k in ['title','gs_abstract']},provenance=dict(file=str(path),line=line,record_sha256=hashlib.sha256(body).hexdigest(),field='title+gs_abstract'))
                found.add(sid)
                if sid in selected_ids:selected_docs[sid]=doc
                yield doc
    write_jsonl('corpus.jsonl',documents())
    if found!=wanted or set(selected_docs)!=selected_ids:raise ValueError('Incomplete raw-derived evidence corpus')
    if {e['id'] for e in experts}!=set(splits):raise ValueError('Incomplete original teacher metadata')
    write_jsonl('experts.jsonl',experts)
    query_split={j['id']:j['split'] for j in jobs}
    write_jsonl('queries.jsonl',[dict(id=q['id'],text=q['demand'],title=q['title'],url=q['url'],split=query_split[q['id']],provenance=dict(file=str(workbook),row=q['source_row'],file_sha256=selection['workbook_sha256'])) for q in queries])
    snapshots=[];review=[]
    for job in jobs:
        for eid,sources in job['candidates'].items():
            views=[];expanded=[]
            for sid,source in sources.items():
                doc=selected_docs[sid];text=source['text']
                if doc['expert_id']!=eid or doc['split']!=job['split'] or doc['text'][:len(text)]!=text:raise ValueError('Raw source/view mismatch')
                views.append(dict(source_id=sid,start=0,end=len(text),sha256=hashlib.sha256(text.encode()).hexdigest()))
                expanded.append(dict(id=sid,kind=doc['kind'],text=text,title=doc.get('title'),url=doc.get('url'),full_text_chars=len(doc['text']),provenance=doc['provenance']))
            snapshot_id=job['id']+'::'+eid
            snapshots.append(dict(id=snapshot_id,query_id=job['id'],expert_id=eid,split=job['split'],sources=views,expert_ability=None,future_project_success=None))
            review.append(dict(snapshot_id=snapshot_id,query=job['demand'],source_view=expanded,split=job['split'],annotation=dict(grade=None,covered_operation=None,source_citations=None,mechanism=None,open_assumptions=None,remaining_scope=None,source_owner_verified=None,annotator=None),instruction='Judge only provided source spans for partial documented method fit. Do not infer actual inability from missing evidence or prior completion of future enterprise targets. No model labels or retrieval origin shown.'))
    write_jsonl('snapshots.jsonl',snapshots)
    random.Random(20261117).shuffle(review);write_jsonl('human-review.jsonl',review)
    write_jsonl('pools.jsonl',[dict(query_id=j['id'],split=j['split'],candidates=list(j['candidates']),dense_order=j['dense_order'],random_candidates=j['random_candidates']) for j in jobs])
    annotation=ROOT/'annotation-development-v1';annotation_summary=json.loads((annotation/'summary.json').read_text())
    if annotation_summary['queries']!=80 or annotation_summary['heldout_queries_annotated']!=0:raise ValueError('Unexpected annotation scope')
    judgments=[];rationales=[]
    for job in jobs:
        if job['split']=='heldout':
            if (annotation/(job['id']+'.json')).exists():raise ValueError('Heldout annotation unexpectedly exists')
            for eid in job['candidates']:judgments.append(dict(query_id=job['id'],expert_id=eid,grade=None,label_status='sealed_unjudged',expert_ability=None,future_project_success=None))
            continue
        path=annotation/(job['id']+'.json');result=json.loads(path.read_text())
        if {r['expert_id'] for r in result['judgments']}!=set(job['candidates']):raise ValueError('Annotation candidate mismatch')
        judgments.extend(result['judgments'])
        native_names={eid:'n'+str(i) for i,eid in enumerate(sorted(job['candidates']))}
        for judgment in result['judgments']:
            eid=judgment['expert_id'];proposals=[]
            for p in result['passes']:
                proposals.append(next((r for r in p.get('result',{}).get('ranking',[]) if r['expert_id']==eid),None) if p['status']=='structurally_valid' else None)
            name=native_names[eid]
            rationales.append(dict(query_id=job['id'],expert_id=eid,proposals=proposals,native_fit=result.get('native_fit',{}).get('labels',{}).get(name),witness_audits=[result.get('witness_audit',{}).get('labels',{}).get(name+':r'+str(i)) for i in range(2)],annotation_artifact_sha256=sha(path)))
    write_jsonl('judgments.jsonl',judgments);write_jsonl('silver-rationales.jsonl',rationales)
    tsv=''.join(f"{r['query_id']}\t0\t{r['expert_id']}\t{r['grade']}\n" for r in judgments if r['grade'] is not None)
    (OUT/'qrels-silver-development.tsv').write_text(tsv)
    split_data=dict(development=[j['id'] for j in jobs if j['split']=='development'],heldout=[j['id'] for j in jobs if j['split']=='heldout'])
    (OUT/'splits.json').write_text(json.dumps(split_data,indent=2))
    provenance=dict(original_teachers=lineage,selection=selection,pool_protocol=json.loads((ROOT/'pool-protocol.json').read_text()),annotation_protocol=json.loads((annotation/'protocol.json').read_text()),export_code_sha256=sha(Path(__file__)))
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    (OUT/'prior-use.json').write_bytes((ROOT/'prior-use.json').read_bytes())
    (OUT/'DATA_CARD.md').write_bytes(Path('docs/fresh-evaluation-data-card.md').read_bytes())
    for name in ['dense-baseline-runs.jsonl','dense-baseline-protocol.json']:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    for path,expected in lineage['raw_teacher_files'].items():
        if sha(Path(path))!=expected:raise ValueError('Raw teachers changed during export')
    manifest=dict(queries=len(queries),experts=len(experts),corpus_sources=len(found),candidate_snapshots=len(snapshots),selected_sources=len(selected_ids),development_queries=80,heldout_queries=20,
                  silver_judged_development=annotation_summary['judged'],silver_unjudged_development=annotation_summary['unjudged'],sealed_unjudged_candidates=140,human_labels=0,
                  label_scope='Partial documented method fit in each fixed source view; grade0=no demonstrated fit, not actual expert inability. Unjudged is not0. Source text beyond view bounds must not be silently substituted during evaluation.',
                  release_status='Local research dataset, not public redistribution or independently human-validated benchmark. Software MIT does not relicense raw source text.',
                  files_sha256={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {'manifest.json','verification.json'} and p.suffix!='.partial'})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='files_sha256'},indent=2))


if __name__=='__main__':run()
