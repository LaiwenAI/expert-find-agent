"""Retrieval-stage timing only; query encoding and index build excluded."""
import json
import time
from pathlib import Path
import numpy as np
from expert_search.progressive_dense import ProgressiveIndex
root=Path('artifacts/public');dm=json.loads((root/'scifact-dense/manifest.json').read_text());x=np.load(root/'scifact-dense/vectors.npy');qs=np.load(root/'scifact-query-dense/vectors.npy')
start=time.perf_counter();index=ProgressiveIndex(x,dm['ids']);build=time.perf_counter()-start
for q in qs[:10]:index.full(q);index.search(q)
times={'full':[],'progressive':[]};same=0
for repeat in range(3):
 for i,q in enumerate(qs):
  outputs={}
  order=['full','progressive'] if (i+repeat)%2==0 else ['progressive','full']
  for name in order:
   start=time.perf_counter();outputs[name]=index.full(q) if name=='full' else index.search(q);times[name].append(time.perf_counter()-start)
  same+=outputs['full']==outputs['progressive']
report={'queries':len(qs),'repeats':3,'identical_top10_trials':same,'index_build_seconds':build,'timings':{name:{'p50_seconds':float(np.median(v)),'p95_seconds':float(np.percentile(v,95)),'mean_seconds':float(np.mean(v))} for name,v in times.items()},'warning':'Local retrieval only, precomputed query embeddings; single process, configured single BLAS thread, not end-to-end latency.'}
Path('docs/exploration/progressive-latency.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
