"""Fixed unsupervised profile-enrichment exploration on the complete SIGIR pool."""
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .cof_benchmark import load,precision_metrics,rank_judged
from .fast_retrieval import FastSearch
from .hybrid_benchmark import rrf
from .profile_enrichment import EnrichmentIndex
from .progressive_dense import top_indices


def run(output='artifacts/public/cof/SIGIR/enrichment-v1'):
    root,papers,reviewers,labels=load('SIGIR');out=Path(output);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'api-small/manifest.json').read_text());matrix=np.load(root/'api-small/vectors.npy');position={pid:i for i,pid in enumerate(meta['ids'])}
    text={pid:(p.get('title','')+'\n'+p.get('abstract','')).strip() for pid,p in papers.items()}
    groups=defaultdict(list)
    for pid in sorted({pid for r in reviewers.values() for pid in r['papers']}):groups[text[pid]].append(pid)
    canonical={pid:members[0] for members in groups.values() for pid in members};documents=sorted(set(canonical.values()))
    profiles={eid:{canonical[pid] for pid in r['papers']} for eid,r in reviewers.items()}
    depths=(10,50,100,200);start=time.perf_counter();index=EnrichmentIndex(documents,profiles,depths)
    build_seconds=time.perf_counter()-start
    vectors=matrix[[position[pid] for pid in documents]]
    lexical=FastSearch(root/'unique-profile-corpus',root/'unique-profile-corpus/fast.sqlite')
    spec=dict(version='enrichment-v1',depths=depths,canonicalization='Exact complete title+abstract text equality',
              embedding_input=meta['input_sha256'],profile_groups=profiles)
    spec['profile_groups']={k:sorted(v) for k,v in profiles.items()}
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest();marker=out/'input.sha256'
    if marker.exists() and marker.read_text()!=digest:raise ValueError('Changed enrichment inputs')
    marker.write_text(digest);rows=[]
    for qid in sorted(labels['soft']):
        path=out/(qid+'.json')
        if path.exists():rows.append(json.loads(path.read_text()));continue
        start=time.perf_counter();cosines=vectors@matrix[position[qid]]
        dense=[documents[i] for i in top_indices(cosines,documents,200)]
        dense_enrichment=index.score(dense);dense_seconds=time.perf_counter()-start
        start=time.perf_counter();raw=lexical.document_scores(text[qid])
        sparse=sorted(documents,key=lambda pid:(-raw.get(pid,0.),pid))[:200]
        sparse_enrichment=index.score(sparse);lexical_seconds=time.perf_counter()-start
        dense_rank=rank_judged(dense_enrichment['scores'],reviewers)
        sparse_rank=rank_judged(sparse_enrichment['scores'],reviewers)
        counts={eid:len(profile&set(dense)) for eid,profile in profiles.items()}
        candidates={
            'dense-enrichment':dense_rank,'lexical-enrichment':sparse_rank,
            'enrichment-rrf':rrf([dense_rank,sparse_rank],k=len(reviewers)),
            'raw-neighborhood-vote':rank_judged(counts,reviewers),
            'fractional-neighborhood-vote':rank_judged({eid:counts[eid]/len(profile) for eid,profile in profiles.items()},reviewers),
            'small-profile-prior':rank_judged({eid:-len(profile) for eid,profile in profiles.items()},reviewers)}
        row=dict(query_id=qid,results={task:{name:dict(ranking=ranking[:10],metrics=precision_metrics(ranking,labels[task][qid]))
                    for name,ranking in candidates.items()} for task in ('soft','hard')},
            dense_enrichment_seconds=dense_seconds,lexical_enrichment_seconds=lexical_seconds,
            top_dense_witnesses={eid:dense_enrichment['witness'].get(eid) for eid in dense_rank[:10]})
        path.write_text(json.dumps(row,indent=2));rows.append(row)
    report=dict(queries=len(rows),unique_source_texts=len(documents),source_ids=len(canonical),depths=depths,offline_table_seconds=build_seconds,
        results={task:{method:{key:statistics.mean(r['results'][task][method]['metrics'][key] for r in rows)
                    for key in rows[0]['results'][task][method]['metrics']} for method in rows[0]['results'][task]} for task in ('soft','hard')},
        median_dense_enrichment_seconds=statistics.median(r['dense_enrichment_seconds'] for r in rows),
        median_lexical_enrichment_seconds=statistics.median(r['lexical_enrichment_seconds'] for r in rows),
        warning='Exploratory standard enrichment component with fixed depths; no labels used for scores or parameter tuning. Exact text clusters are source units, not asserted world paper identities. Cached query vectors, no online embedding latency. Reference null is not expertise probability. Includes static profile-size prior to detect benchmark prior effects. Not a novel statistical theorem or a verified new-method result.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':print(json.dumps(run(),indent=2))
