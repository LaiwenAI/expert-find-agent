"""Audit role-aware gain coverage on the fixed private development cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from expert_search.attributed_gain import attributed_gain_interval
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.metadata_overlay import MetadataOverlayPipeline


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1')
    manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed query input')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({q['id'] for q in queries})!=80:raise ValueError('Changed development cohort')
    output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/attributed_gain.py'),Path('src/expert_search/metadata_overlay.py')]
    protocol={'query_count':80,'query_sha256':sha(data/'queries.jsonl'),'code_sha256':{str(p):sha(p) for p in code},
              'candidate_pool':50,'return_k':5,'cutoff':'not specified','labels':'No independent fixed-view labels available for any dimension',
              'scope':'Private reused development needs; missing judgments remain unknown. This is coverage, not accuracy or comparison.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    engine=CandidateEvidencePipeline(scoring='resident-adaptive')
    overlay=MetadataOverlayPipeline(engine,'artifacts/corpus','artifacts/corpus-temporal-source-v1')
    rows=[]
    for q in queries:
        result=overlay.search_reranked(q['text'],k=5,candidates=50,evidence_per_expert=1)
        # Bind each key to the complete query, expert, visible evidence and cutoff.
        ranking=[hashlib.sha256(json.dumps({'query':q,'expert':r['expert'],'evidence':r['evidence'],'cutoff':None},sort_keys=True,ensure_ascii=True).encode()).hexdigest() for r in result['results']]
        bound=attributed_gain_interval(ranking,{},k=5)
        rows.append({'query_id':q['id'],'bound':bound})
    engine.evidence.db.close()
    (output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'queries':len(rows),'mean_lower':statistics.mean(r['bound']['lower'] for r in rows),
            'mean_upper':statistics.mean(r['bound']['upper'] for r in rows),
            'returned_positions':sum(r['bound']['returned'] for r in rows),
            'unjudged_joint_positions':sum(r['bound']['joint_status_counts']['unknown'] for r in rows),
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'private-results.json'),
            'scope':protocol['scope']+' No method, identity-role or cutoff judgments are manufactured from metadata. Bounds conditional on the fixed views; no population accuracy or ability is implied.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
