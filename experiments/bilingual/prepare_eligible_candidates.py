"""Freeze lexical expert shortlists for bilingual selected-source scoring."""
import argparse
import json
from pathlib import Path
import statistics
import time
from expert_search.compiled_expert_search import CompiledExpertSearch
from experiments.bilingual.adaptive_effort_diagnostic import sha


def main(output):
    root=Path('artifacts/bilingual-retrieval-v1');selection=json.loads((root/'selection.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if sha(root/'queries.jsonl')!=selection['queries_sha256'] or sha(root/'evaluation.jsonl')!=summary['evaluation_sha256'] or sha(root/'protocol.json')!=summary['protocol_sha256']:raise ValueError('Changed frozen cohort')
    queries=[json.loads(x) for x in (root/'queries.jsonl').read_text().splitlines()];prior=[json.loads(x) for x in (root/'evaluation.jsonl').read_text().splitlines()]
    if len(queries)!=1000 or [r['id'] for r in queries]!=[r['id'] for r in prior]:raise ValueError('Changed query order')
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();ranker=CompiledExpertSearch();initialization=time.perf_counter()-start
    protocol={'queries_sha256':sha(root/'queries.jsonl'),'previous_evaluation_sha256':sha(root/'evaluation.jsonl'),'corpus_snapshot_sha256':ranker.snapshot,
              'index_metadata_sha256':sha('artifacts/teacher-compiled-lm/metadata.json'),'index_arrays_sha256':sha('artifacts/teacher-compiled-lm/arrays.npz'),
              'selection':'candidate-LM top 50 per language; no relevance labels or exact-ranking feedback used','pairs':1000,'views':2000,
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/compiled_expert_search.py'),Path('src/expert_search/certified_topk.py'))},
              'scope':'Reused construction queries, no expert relevance judgments; lexical shortlist against full dense expert ranking as a diagnostic, not a recall gold standard.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[]
    for q,previous in zip(queries,prior):
        for lang in ('zh','en'):
            start=time.perf_counter();result=ranker.search(q[lang],50);elapsed=time.perf_counter()-start
            ids=[r['expert']['id'] for r in result['results']];dense=previous['languages'][lang]['exact']['ranking']
            if len(ids)!=50 or len(set(ids))!=50 or len(dense)!=10:raise ValueError('Invalid shortlist')
            rows.append({'id':q['id'],'group_id':q['group_id'],'language':lang,'eligible':ids,'dense_top10':dense,
                         'dense_top10_in_shortlist':len(set(ids)&set(dense)),'seconds':elapsed})
    (output/'candidates.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'pairs':len(queries),'views':len(rows),'initialization_seconds':initialization,'mean_shortlist_seconds':statistics.mean(r['seconds'] for r in rows),
            'mean_dense_top10_shortlist_recall':statistics.mean(r['dense_top10_in_shortlist']/10 for r in rows),
            'by_language':{lang:{'views':sum(r['language']==lang for r in rows),'mean_dense_top10_shortlist_recall':statistics.mean(r['dense_top10_in_shortlist']/10 for r in rows if r['language']==lang)} for lang in ('zh','en')},
            'protocol_sha256':sha(output/'protocol.json'),'candidates_sha256':sha(output/'candidates.json'),
            'scope':protocol['scope']+' Shortlist timings exclude initialization. Candidate recall here is overlap with a previously cached dense ranking, not human relevance. Synthetic queries are not independently validated.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
