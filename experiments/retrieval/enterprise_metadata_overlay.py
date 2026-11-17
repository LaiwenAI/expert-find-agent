"""Replay development retrieval with a validated publication-metadata overlay."""
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed query input')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({q['id'] for q in queries})!=80:raise ValueError('Changed development set')
    output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/metadata_overlay.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'queries':80,'queries_sha256':sha(data/'queries.jsonl'),'base_evidence_sha256':sha('artifacts/corpus/evidence.jsonl'),
              'enriched_evidence_sha256':sha('artifacts/corpus-temporal-source-v1/evidence.jsonl'),'code_sha256':{str(p):sha(p) for p in code},
              'candidate_pool':50,'return_k':5,'evidence_per_expert':1,'scope':'Private development queries; metadata describes supplied source records, not independently verified dates, individual roles, relevance or historic availability. No as-of filtering or protected split.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();pipeline=CandidateEvidencePipeline(scoring='resident-adaptive');init=time.perf_counter()-start
    start=time.perf_counter();overlay=MetadataOverlayPipeline(pipeline,'artifacts/corpus','artifacts/corpus-temporal-source-v1');overlay_init=time.perf_counter()-start
    stats=Counter();timings=[]
    for i,q in enumerate(queries):
        plain=pipeline.search_reranked(q['text'],k=5,candidates=50,evidence_per_expert=1)
        start=time.perf_counter();enriched=overlay._attach(plain);timings.append(time.perf_counter()-start)
        stripped=deepcopy(enriched['results'])
        for row in stripped:
            for e in row['evidence']:
                for name in ('publication','contribution_status','authorship_association'):e.pop(name,None)
        if stripped!=plain['results']:raise ValueError('Overlay changed retrieval results')
        for row in enriched['results']:
            for e in row['evidence']:
                stats['evidence']+=1;stats['papers']+=e.get('kind')=='paper'
                stats['parsed_publication']+=e.get('publication',{}).get('status')=='parsed'
                stats['paper_unknown_publication']+=e.get('kind')=='paper' and e.get('publication',{}).get('status')!='parsed'
                stats['unverified_association']+=e.get('authorship_association') is not None
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    pipeline.evidence.db.close()
    report={'queries':len(queries),'identical_retrieval_results':len(queries),'selected_evidence':dict(stats),'pipeline_init_seconds':init,
            'overlay_init_seconds':overlay_init,'mean_attachment_seconds':statistics.mean(timings),'base_snapshot_sha256':overlay.base_snapshot_sha256,
            'metadata_evidence_sha256':overlay.enriched_evidence_sha256,'protocol_sha256':sha(output/'protocol.json'),
            'scope':protocol['scope']+' Attachment timing is one post-retrieval observation per query; excludes index initialization and model inference. No source text or query IDs are published.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
