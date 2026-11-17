"""Validate progressive first results on full enterprise development needs."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.progressive_candidate_agent import ProgressiveCandidateAgent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed queries')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({r['id'] for r in queries})!=80:raise ValueError('Changed cohort')
    engine=CandidateEvidencePipeline(scoring='resident');agent=ProgressiveCandidateAgent(engine,client=object());rows=[]
    for i,q in enumerate(queries):
        outputs={};seconds={}
        for arm in (('direct','progressive') if i%2==0 else ('progressive','direct')):
            start=time.perf_counter()
            if arm=='direct':outputs[arm]=engine.search_reranked(q['text'],k=20,candidates=50,evidence_per_expert=1)['results'][:5]
            else:
                events=agent.events(q['text'],k=5,candidates=20,rerank_pool=50,evidence_per_expert=1)
                event=next(events);outputs[arm]=event['results'];events.close()
                if event['stage']!='retrieval':raise ValueError('Unexpected first event')
            seconds[arm]=time.perf_counter()-start
        if outputs['direct']!=outputs['progressive']:raise ValueError('Changed preliminary results')
        rows.append({'id':q['id'],'seconds':seconds,'returned':len(outputs['progressive'])})
    engine.evidence.db.close()
    report={'queries':len(rows),'identical_first_results':len(rows),'mean_seconds':{a:statistics.mean(r['seconds'][a] for r in rows) for a in ('direct','progressive')},
            'mean_returned':statistics.mean(r['returned'] for r in rows),'pipeline_initialization_seconds':engine.initialization_seconds,
            'resident_build_seconds':engine.resident.initialization_seconds,'resident_array_bytes':engine.resident.array_bytes,
            'input_sha256':{'queries':manifest['files_sha256']['queries.jsonl'],'corpus_snapshot':engine.ranker.snapshot},
            'scope':'Post-hoc local 80 complete development needs; one alternating measurement per query after resident initialization. First event contains five unverified owned lexical candidates; no model call or explanation is performed. Identical retrieval is not semantic relevance, evidence entailment or independent quality. Timing excludes initialization, index build, downstream reasoning and queueing; no production percentile or end-to-end claim.'}
    output.mkdir(parents=True,exist_ok=False);(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report['results_sha256']=sha(output/'private-results.json');report['code_sha256']={str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/progressive_candidate_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py'))}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
