"""Test candidate-first evidence lookup after exact candidate-language-model search."""
import argparse
import json
import statistics
import time
from pathlib import Path

from expert_search.compiled_expert_search import CompiledExpertSearch
from expert_search.fast_retrieval import FastSearch
from .enterprise_end_to_end_pilot import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    query_file=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    queries=sorted((r for r in map(json.loads,query_file.read_text().splitlines()) if r['split']=='development'),key=lambda r:r['id'])[:20]
    if len(queries)!=20:raise ValueError('Expected 20 development queries')
    ranker=CompiledExpertSearch();old=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite')
    new=FastSearch('artifacts/corpus','artifacts/fast-search-selective-v1.sqlite')
    args.output.mkdir(parents=True,exist_ok=False)
    model_dir=Path('artifacts/teacher-compiled-lm')
    protocol={'query_sha256':sha(query_file),'model_file_sha256':{p.name:sha(p) for p in model_dir.iterdir() if p.is_file()},
              'baseline_db_sha256':sha(Path('artifacts/fast-search.sqlite')),'selective_db_sha256':sha(Path('artifacts/fast-search-selective-v1.sqlite')),
              'code_sha256':{p:sha(Path(p)) for p in ['src/expert_search/fast_retrieval.py','src/expert_search/compiled_expert_search.py','src/expert_search/certified_topk.py','experiments/retrieval/lm_selective_evidence_pilot.py']},
              'scope':'20 lexicographic development queries; exact candidate-LM shortlist and shared candidate evidence. Single alternating call per lookup arm, retrieval reported separately; no initialization, index build, relevance judgments, model reasoning or end-to-end claim.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[]
    for i,q in enumerate(queries):
        ranked=ranker.search(q['text'],20);ids=[r['expert']['id'] for r in ranked['results']];outputs={};elapsed={}
        for name,fn in ([('baseline',old.evidence_for),('selective',new.evidence_for_selective)] if i%2==0 else [('selective',new.evidence_for_selective),('baseline',old.evidence_for)]):
            start=time.perf_counter();outputs[name]=fn(q['text'],ids);elapsed[name]=time.perf_counter()-start
        rows.append({'query_id':q['id'],'ranking_seconds':ranked['seconds'],'identical':outputs['baseline']==outputs['selective'],'seconds':elapsed})
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'queries':len(rows),'identical':sum(r['identical'] for r in rows),'mean_ranking_seconds':statistics.mean(r['ranking_seconds'] for r in rows),
                      'mean_baseline_seconds':statistics.mean(r['seconds']['baseline'] for r in rows),
                      'mean_selective_seconds':statistics.mean(r['seconds']['selective'] for r in rows),'scope':protocol['scope']},indent=2))


if __name__=='__main__':main()
