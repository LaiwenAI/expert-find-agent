"""Replay private enterprise retrieval with traceable paper citation context."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.metadata_overlay import MetadataOverlayPipeline
from expert_search.citation_overlay import CitationOverlayPipeline


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed queries')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({r['id'] for r in queries})!=80:raise ValueError('Changed development cohort')
    output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/citation_overlay.py'),Path('src/expert_search/metadata_overlay.py'),Path('src/expert_search/candidate_reasoning_agent.py')]
    protocol={'queries':80,'queries_sha256':sha(data/'queries.jsonl'),'paper_sidecar_sha256':sha('artifacts/source-metadata-v1/papers.jsonl'),
              'raw_papers_sha256':sha('data/teachers/shenzhen/papers.jsonl'),'code_sha256':{str(p):sha(p) for p in code},
              'candidate_pool':50,'return_k':5,'scope':'Private development needs. Paper counts are unnormalized observations of unknown time, not personal contribution or relevance. No historical cutoff or protected queries.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();pipeline=CandidateEvidencePipeline(scoring='resident-adaptive');init=time.perf_counter()-start
    date=MetadataOverlayPipeline(pipeline,'artifacts/corpus','artifacts/corpus-temporal-source-v1')
    start=time.perf_counter();citations=CitationOverlayPipeline(date,'artifacts/corpus','artifacts/source-metadata-v1','data/teachers/shenzhen/papers.jsonl');citation_init=time.perf_counter()-start
    counts=Counter();timings=[]
    for i,q in enumerate(queries):
        result=date.search_reranked(q['text'],k=5,candidates=50,evidence_per_expert=1)
        start=time.perf_counter();with_citations=citations._attach(result);timings.append(time.perf_counter()-start)
        stripped=deepcopy(with_citations['results'])
        for r in stripped:
            for e in r['evidence']:e.pop('citation_context',None)
        if stripped!=result['results']:raise ValueError('Changed retrieval or date metadata')
        for r in with_citations['results']:
            for e in r['evidence']:
                counts['sources']+=1
                if e.get('kind')=='paper':
                    counts['papers']+=1;c=e['citation_context'];counts['known_citation_total']+=c['observed_total'] is not None;counts['unknown_citation_total']+=c['observed_total'] is None
                    if c['observation_date'] is not None or c['personal_credit'] is not None:raise ValueError('Unexpected temporal or personal claim')
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    pipeline.evidence.db.close()
    report={'queries':len(queries),'identical_retrieval_results':len(queries),'selected_sources':dict(counts),'pipeline_init_seconds':init,
            'citation_overlay_init_seconds':citation_init,'mean_citation_attachment_seconds':statistics.mean(timings),'protocol_sha256':sha(output/'protocol.json'),
            'scope':protocol['scope']+' One post-retrieval attachment observation per query; excludes retrieval, date overlay initialization and model inference. Source texts and identities stay private.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
