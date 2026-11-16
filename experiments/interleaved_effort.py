import json,random
from pathlib import Path
from expert_search.cmu_gold import load_annotations
from expert_search.cmu_evidence_benchmark import run
root=Path('artifacts/public/cmu-interleaved-effort');root.mkdir(parents=True,exist_ok=True)
reviewers=sorted(random.Random(20261116).sample(sorted(load_annotations('artifacts/public/cmu-gold/data/evaluations.csv')),6))
(root/'protocol.json').write_text(json.dumps({'reviewers':reviewers,'seed':20261116,'repeats':2,'conditions':['none','medium'],'note':'Fixed random reviewer subset, no label-based selection. Model inference only, no training.'},indent=2))
for repeat in range(2):
 for i,rid in enumerate(reviewers):
  order=['none','medium'] if (i+repeat)%2==0 else ['medium','none']
  for effort in order:
   print(rid,repeat,effort,flush=True)
   run(limit=1,output=str(root/f'r{repeat}'/rid/effort),mode='rank',reasoning_effort=effort,reviewer_ids=[rid])
print('interleaved complete')
