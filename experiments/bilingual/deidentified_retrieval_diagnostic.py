"""Compare local lexical rankings for Chinese, prior and edited English views."""
import hashlib
import json
import statistics
import time
from pathlib import Path
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline


def load(root,name):
    manifest=json.loads((root/'manifest.json').read_text());raw=(root/name).read_bytes()
    expected=manifest['outputs_sha256'][name] if 'outputs_sha256' in manifest else manifest['output_sha256']
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed dataset')
    return {r['id']:r for r in map(json.loads,raw.splitlines())},hashlib.sha256(raw).hexdigest()


def main():
    original,source_hash=load(Path('artifacts/dataset-release/bilingual-query-construction-v1'),'synthetic-queries.jsonl')
    previous,previous_hash=load(Path('artifacts/dataset-release/english-repair-construction-v2'),'queries.jsonl')
    revised,revised_hash=load(Path('artifacts/dataset-release/english-alignment-gated-v1'),'queries.jsonl')
    chosen=[r for r in revised.values() if r['deidentification']=='same_model_screened_candidate']
    if len(chosen)!=18 or any(r['id'] not in original or r['id'] not in previous for r in chosen):raise ValueError('Changed edited cohort')
    pipeline=CandidateEvidencePipeline();start=time.perf_counter();pipeline.evidence.prepare_adaptive_scoring();prep=time.perf_counter()-start
    rows=[]
    for i,row in enumerate(sorted(chosen,key=lambda r:r['id'])):
        qid=row['id'];inputs={'zh':original[qid]['zh'],'old':previous[qid]['english'],'new':row['english']};rankings={};seconds={}
        order=('zh','old','new');order=order[i%3:]+order[:i%3]
        for arm in order:
            start=time.perf_counter();result=pipeline.search_reranked(inputs[arm],k=10,candidates=50,evidence_per_expert=1);seconds[arm]=time.perf_counter()-start
            rankings[arm]=[r['expert']['id'] for r in result['results']]
        if any(len(r)!=10 or len(set(r))!=10 for r in rankings.values()):raise ValueError('Invalid ranking')
        overlap=lambda a,b:len(set(rankings[a])&set(rankings[b]))/10
        rows.append(dict(id=qid,changed=inputs['old']!=inputs['new'],alignment_recheck=row['alignment_recheck'],status=row['status'],
                         old_new_overlap=overlap('old','new'),zh_old_overlap=overlap('zh','old'),zh_new_overlap=overlap('zh','new'),
                         old_new_identical=rankings['old']==rankings['new'],seconds=seconds))
    pipeline.evidence.db.close();changed=[r for r in rows if r['changed']]
    def aggregate(values):
        if not values:return None
        return dict(count=len(values),old_new_overlap=statistics.mean(r['old_new_overlap'] for r in values),
                    zh_old_overlap=statistics.mean(r['zh_old_overlap'] for r in values),zh_new_overlap=statistics.mean(r['zh_new_overlap'] for r in values),
                    old_new_identical=sum(r['old_new_identical'] for r in values))
    out=Path('artifacts/deidentified-retrieval-diagnostic-v1');out.mkdir(parents=True,exist_ok=False)
    report=dict(all=aggregate(rows),changed_only=aggregate(changed),posting_count_preparation_seconds=prep,
                source_sha256=source_hash,previous_sha256=previous_hash,revised_sha256=revised_hash,
                scope='Post-hoc lexical retrieval stability for 18 previously examined construction pairs; same expert corpus and fixed 50-candidate adaptive pipeline. Changed-only subset is selected after edits. Ranking overlap is not relevance, semantic fidelity or anonymity. One timing per view excludes index construction, initialization and model reasoning.')
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
