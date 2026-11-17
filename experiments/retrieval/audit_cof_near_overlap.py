"""Screen exact and near-overlap and mask matched sources for sensitivity."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from expert_search.candidate_evidence_pipeline import verify_evidence_documents
from expert_search.cof_benchmark import precision_metrics
from expert_search.cof_data import unique_records
from expert_search.corpus import records
from expert_search.fast_retrieval import FastSearch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shingles(text,n=5):
    tokens=re.findall(r'\w+',text.casefold())
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}


def main(output):
    manifest=Path('artifacts/cof-gpu-public-input-v1/input-provenance.json');provenance=json.loads(manifest.read_text())
    for p,digest in provenance['input_sha256'].items():
        if sha(p)!=digest:raise ValueError('Changed frozen input: '+p)
    previous=Path('artifacts/cof-gpu-quality-v1');summary=json.loads((previous/'summary.json').read_text())
    if sha(previous/'protocol.json')!=summary['protocol_sha256'] or sha(previous/'rows.json')!=summary['rows_sha256']:raise ValueError('Changed prior results')
    if json.loads((previous/'protocol.json').read_text())['input_manifest_sha256']!=sha(manifest):raise ValueError('Different benchmark snapshot')
    root=Path('artifacts/public/cof/SIGIR')
    papers=unique_records([x for _,x in records(root/'SIGIR_papers_test.json')],'paper')
    reviewers=unique_records([x for _,x in records(root/'SIGIR_reviewers_test.json')],'reviewer')
    labels={t:{x['query_id']:x['score'] for _,x in records(root/f'SIGIR_queries_test_{t}.json')} for t in ('soft','hard')}
    qids=sorted(labels['soft']);pids=sorted({p for r in reviewers.values() for p in r['papers']})
    if qids!=provenance['query_ids'] or set(qids)!=set(labels['hard']):raise ValueError('Changed query universe')
    prior={x['query_id']:x for x in json.loads((previous/'rows.json').read_text())}
    if set(prior)!=set(qids):raise ValueError('Changed prior result universe')
    doc_sets={p:shingles(papers[p].get('abstract','')) for p in pids}
    matches=[];maxima=[]
    for q in qids:
        a=shingles(papers[q].get('abstract',''))
        if not a:raise ValueError('Query abstract has fewer than five words')
        best=(0.,None,0.)
        for p,b in doc_sets.items():
            if not b:continue
            common=len(a & b)
            if not common:continue
            jaccard=common/len(a|b);contain=common/min(len(a),len(b))
            if jaccard>best[0]:best=(jaccard,p,contain)
            if jaccard>=.8:matches.append(dict(query_id=q,source_id=p,jaccard=jaccard,shorter_containment=contain))
        maxima.append(dict(query_id=q,source_id=best[1],jaccard=best[0],shorter_containment=best[2]))
    matches.sort(key=lambda x:(x['query_id'],x['source_id']))
    matches_by_query=defaultdict(set)
    for x in matches:matches_by_query[x['query_id']].add(x['source_id'])
    corpus=root/'unique-profile-corpus';lexical=FastSearch(corpus,corpus/'fast.sqlite');verify_evidence_documents(corpus,lexical)
    def text(pid):return (papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
    items=[dict(id='doc:'+p,text=text(p),query=False) for p in pids]+[dict(id='query:'+q,text=text(q),query=True) for q in qids]
    vecroot=root/'qwen-embedding-torch-v1';spec=json.loads((vecroot/'protocol.json').read_text())
    if hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()!=spec['input_sha256']:raise ValueError('Changed vector order')
    matrix=np.load(vecroot/'vectors.npy',mmap_mode='r');position={p:i for i,p in enumerate(pids)}
    if matrix.shape!=(len(items),1024):raise ValueError('Invalid vector matrix')
    affected=[]
    for q,excluded in sorted(matches_by_query.items()):
        query_index=qids.index(q);dense=np.einsum('ij,j->i',matrix[:len(pids)],matrix[len(pids)+query_index],optimize=False)
        lex=lexical.document_scores(text(q));scores={'cpu':{p:float(dense[position[p]]) for p in pids},'lexical':{p:lex.get(p,0.) for p in pids}}
        rankings={}
        for method,source_scores in scores.items():
            def rank(mask):
                reviewer_scores={eid:max((source_scores[p] for p in set(r['papers']) if p not in mask),default=0.) for eid,r in reviewers.items()}
                return sorted(reviewers,key=lambda eid:(-reviewer_scores[eid],eid))[:10]
            baseline=rank(set());masked=rank(excluded)
            if baseline!=prior[q]['rankings'][method]:raise ValueError('Baseline replay differs from frozen prior ranking')
            rankings[method]=dict(baseline=baseline,masked=masked,metrics={t:dict(baseline=precision_metrics(baseline,labels[t][q]),masked=precision_metrics(masked,labels[t][q])) for t in labels})
        owners=sorted(e for e,r in reviewers.items() if set(r['papers']) & excluded)
        affected.append(dict(query_id=q,excluded_source_ids=sorted(excluded),affected_reviewer_ids=owners,rankings=rankings))
    report=dict(queries=len(qids),profile_sources=len(pids),definition='Abstract word 5-gram Jaccard >= 0.8; Unicode casefold, regex word tokens; threshold selected post hoc. Pairwise exhaustive comparison.',matches=matches,
        max_jaccard_per_query=maxima,matched_queries=len(matches_by_query),affected=affected,
        provenance=dict(input_manifest_sha256=sha(manifest),prior_summary_sha256=sha(previous/'summary.json'),script_sha256=sha(__file__)),
        limitation='Lexical near-duplicate screen, not a semantic leakage audit. Mask matched source scores only for the affected query; original BM25 document frequencies are retained. Labels, ownership and individual contributions are not verified. The cached dense query is not the live-encoding run. Post-hoc diagnostic, not a corrected sealed benchmark.')
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='max_jaccard_per_query'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
