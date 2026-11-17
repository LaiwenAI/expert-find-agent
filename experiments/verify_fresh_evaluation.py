"""Audit the new dataset against original teacher JSONL and enterprise XLSX."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from openpyxl import load_workbook


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def nh(text):return hashlib.sha256(re.sub(r'[^\w]+','',text.casefold()).encode()).hexdigest()


def run(directory):
    root=Path(directory);manifest=json.loads((root/'manifest.json').read_text());provenance=json.loads((root/'provenance.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Changed dataset member: '+name)
    raw=provenance['original_teachers']['raw_teacher_files']
    for path,expected in raw.items():
        if sha(Path(path))!=expected:raise ValueError('Changed original teacher source')
    queries={r['id']:r for r in rows(root/'queries.jsonl')};experts={r['id']:r for r in rows(root/'experts.jsonl')};documents={r['id']:r for r in rows(root/'corpus.jsonl')}
    targets=defaultdict(lambda:defaultdict(list))
    for r in experts.values():targets[r['provenance']['file']][r['provenance']['line']].append(('expert',r))
    for r in documents.values():targets[r['provenance']['file']][r['provenance']['line']].append(('source',r))
    checked=0
    for path,lines in targets.items():
        with Path(path).open('rb') as stream:
            for line,body in enumerate(stream,1):
                if line not in lines:continue
                original=json.loads(body);record_hash=hashlib.sha256(body).hexdigest()
                for kind,record in lines[line]:
                    if record['provenance']['record_sha256']!=record_hash:raise ValueError('Raw record hash mismatch')
                    if kind=='expert':
                        if record['id']!=original['teacher_id'] or record['name']!=(original.get('教师姓名_CN') or original.get('教师姓名_EN')):raise ValueError('Teacher identity mismatch')
                    else:
                        if record['expert_id']!=original['teacher_id'] or record['expert_id'] not in experts:raise ValueError('Raw teacher_id/source linkage mismatch')
                        field=record['provenance']['field']
                        expected='\n'.join(str(original.get(k) or '') for k in ['title','gs_abstract']) if field=='title+gs_abstract' else str(original[field])
                        if expected!=record['text']:raise ValueError('Source was not faithfully rebuilt from original fields')
                    checked+=1
    if checked!=len(experts)+len(documents):raise ValueError('Incomplete raw record audit')
    workbook=Path(next(iter(queries.values()))['provenance']['file'])
    if sha(workbook)!=provenance['selection']['workbook_sha256']:raise ValueError('Changed original XLSX')
    book=load_workbook(workbook,read_only=True,data_only=True);data=list(book.active.values);header=list(data[0]);book.close()
    for query in queries.values():
        original=dict(zip(header,data[query['provenance']['row']-1]));text=str(original['命题名称'])+'\n'+str(original['命题内容及答题要求'])
        if query['text']!=text:raise ValueError('Demand text does not match original workbook row')
    prior=json.loads((root/'prior-use.json').read_text())
    if set(queries)&set(prior['demand_ids']) or set(experts)&set(prior['expert_ids']):raise ValueError('Prior entity leakage')
    used_text=set(prior['source_prefix_hashes']);used_normalized=set(prior['normalized_source_hashes']);used_titles=set(prior['paper_title_hashes']);old_queries=set(prior['query_hashes'])
    for query in queries.values():
        if nh(query['text']) in old_queries or nh(query['text'].split('\n',1)[1]) in old_queries:raise ValueError('Prior demand text leakage')
    corpus_text=defaultdict(set);paper_titles=defaultdict(set)
    for doc in documents.values():
        text=doc['text'][:1200];split=doc['split']
        if doc['expert_id'] in prior['expert_ids'] or hashlib.sha256(text.encode()).hexdigest() in used_text or nh(text) in used_normalized:raise ValueError('Prior source leakage')
        if doc['split']!=experts[doc['expert_id']]['split']:raise ValueError('Wrong source split')
        corpus_text[split].add(nh(text))
        if doc['kind']=='paper':
            title=nh(text.splitlines()[0]);paper_titles[split].add(title)
            if title in used_titles:raise ValueError('Prior paper-title alias leakage')
    snapshots=list(rows(root/'snapshots.jsonl'));view_entities=defaultdict(lambda:dict(experts=set(),sources=set(),texts=set()));keys=set()
    for snapshot in snapshots:
        key=(snapshot['query_id'],snapshot['expert_id'])
        if key in keys:raise ValueError('Duplicate candidate snapshot')
        keys.add(key);split=snapshot['split'];view_entities[split]['experts'].add(snapshot['expert_id'])
        if queries[snapshot['query_id']]['split']!=split or experts[snapshot['expert_id']]['split']!=split:raise ValueError('Cross-split candidate')
        for span in snapshot['sources']:
            doc=documents[span['source_id']];text=doc['text'][span['start']:span['end']]
            if doc['expert_id']!=snapshot['expert_id'] or not 0<=span['start']<span['end']<=len(doc['text']) or hashlib.sha256(text.encode()).hexdigest()!=span['sha256']:raise ValueError('Invalid original-source view')
            view_entities[split]['sources'].add(span['source_id']);view_entities[split]['texts'].add(nh(text))
    overlap={k:len(view_entities['development'][k]&view_entities['heldout'][k]) for k in ['experts','sources','texts']}
    if any(overlap.values()):raise ValueError('Development/heldout source-view leakage')
    judgments=list(rows(root/'judgments.jsonl'));rationales={(r['query_id'],r['expert_id']):r for r in rows(root/'silver-rationales.jsonl')};mapping={'insufficient':0,'topic_only':1,'transferable_method':2,'direct_application':3}
    if {(r['query_id'],r['expert_id']) for r in judgments}!=keys or len(judgments)!=len(keys):raise ValueError('Wrong judgment pool')
    counted=0
    for r in judgments:
        if r.get('expert_ability') is not None or r.get('future_project_success') is not None:raise ValueError('Unjustified ability/success labels')
        if queries[r['query_id']]['split']=='heldout':
            if r['grade'] is not None or r['label_status']!='sealed_unjudged':raise ValueError('Heldout label leakage')
            continue
        explanation=rationales[(r['query_id'],r['expert_id'])];levels=[p['level'] if p else None for p in explanation['proposals']];native=explanation['native_fit'];expected=None
        if levels[0] is not None and levels[0]==levels[1]==native:
            if mapping[native]<2 or explanation['witness_audits']==['supported_scoped_claim']*2:expected=mapping[native]
        if r['grade']!=expected:raise ValueError('Silver admission rule mismatch')
        counted+=expected is not None
    if counted!=manifest['silver_judged_development']:raise ValueError('Silver count mismatch')
    human=list(rows(root/'human-review.jsonl'))
    if len(human)!=len(snapshots) or any(r['annotation']['grade'] is not None for r in human):raise ValueError('Unexpected human labels')
    report=dict(queries=len(queries),experts=len(experts),corpus_sources=len(documents),candidate_snapshots=len(snapshots),original_raw_records_checked=sum(len(v) for v in targets.values()),raw_entity_field_checks=checked,all_demands_match_original_workbook=True,
                original_teacher_id_links_match=True,source_authorship_independently_verified=False,prior_demand_expert_source_alias_overlap=0,development_heldout_snapshot_overlap=overlap,
                full_corpus_normalized_prefix_overlap=len(corpus_text['development']&corpus_text['heldout']),paper_title_alias_overlap=len(paper_titles['development']&paper_titles['heldout']),
                silver_admissions_recomputed=counted,heldout_labelled=0,human_labels=0,manifest_sha256=sha(root/'manifest.json'),
                scope='Raw reconstruction, metadata linkage, exact split/alias and annotation-rule audit. Does not establish source real-world authorship, semantic/team-family independence, human quality or actual future project ability.')
    (root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='artifacts/dataset-release/fresh-enterprise-eval-v1');run(p.parse_args().dataset)
