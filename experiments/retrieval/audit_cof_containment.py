"""Screen query/profile abstract containment using exact word shingles."""
import argparse
import hashlib
import json
from pathlib import Path

from expert_search.cof_data import unique_records
from expert_search.corpus import records
from experiments.retrieval.audit_cof_near_overlap import shingles


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    manifest=Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    provenance=json.loads(manifest.read_text())
    for path,digest in provenance['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed benchmark input: '+path)
    root=Path('artifacts/public/cof/SIGIR')
    papers=unique_records([x for _,x in records(root/'SIGIR_papers_test.json')],'paper')
    reviewers=unique_records([x for _,x in records(root/'SIGIR_reviewers_test.json')],'reviewer')
    qids=sorted(x['query_id'] for _,x in records(root/'SIGIR_queries_test_soft.json'))
    pids=sorted({p for r in reviewers.values() for p in r['papers']})
    if qids!=provenance['query_ids']:raise ValueError('Changed query selection')
    docs={p:shingles(papers[p].get('abstract','')) for p in pids}
    empty_sources=sum(not x for x in docs.values());maxima=[];matches=[]
    for q in qids:
        a=shingles(papers[q].get('abstract',''))
        if not a:
            maxima.append(dict(query_id=q,unscorable=True));continue
        best=None
        for p,b in docs.items():
            if not b:continue
            shared=len(a&b);contain=shared/min(len(a),len(b));jaccard=shared/len(a|b)
            row=dict(query_id=q,source_id=p,containment=contain,jaccard=jaccard,shared_shingles=shared,query_shingles=len(a),source_shingles=len(b))
            if best is None or (contain,jaccard)>(best['containment'],best['jaccard']):best=row
            if contain>=.8:matches.append(row)
        if best is None:raise ValueError('No scorable profile abstract')
        maxima.append(best)
    scorable=[r for r in maxima if not r.get('unscorable')]
    nonidentical=[r for r in scorable if r['containment']<1]
    report=dict(queries=len(qids),sources=len(pids),unscorable_queries=len(qids)-len(scorable),unscorable_sources=empty_sources,
        definition='Five-word shingles; casefold and regex word tokens; intersection divided by smaller set size. Exhaustive pairs, threshold 0.8 chosen post hoc.',
        queries_at_least_08=sum(r['containment']>=.8 for r in scorable),queries_at_least_05=sum(r['containment']>=.5 for r in scorable),
        next_highest_below_one=max((r['containment'] for r in nonidentical),default=None),matches=matches,per_query_maximum=maxima,
        input_manifest_sha256=sha(manifest),script_sha256=sha(__file__),shingles_script_sha256=sha('experiments/retrieval/audit_cof_near_overlap.py'),
        limitation='Unscorable short or missing abstracts are explicitly counted. Exact word shingles miss paraphrase, translated or semantically related content; the threshold is not a validated contamination classifier. Source ownership, model pretraining and dates are not assessed.')
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_query_maximum'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
