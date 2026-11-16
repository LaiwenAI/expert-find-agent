import json,random
from itertools import combinations
from pathlib import Path
from expert_search.cmu_gold import load_annotations,pairwise_loss
root=Path('artifacts/public');labels=load_annotations(root/'cmu-gold/data/evaluations.csv')
none={p.stem:json.loads(p.read_text())['predictions'] for p in (root/'cmu-relative-none-v2').glob('*.json') if p.name!='summary.json'}
def compare(other):
 parts=[]
 for rid,grades in labels.items():
  weight=sum(abs(grades[a]-grades[b]) for a,b in combinations(grades,2))
  if not weight:continue
  a=pairwise_loss({rid:none[rid]},{rid:grades})['weighted_pairwise_loss'];b=pairwise_loss({rid:other[rid]},{rid:grades})['weighted_pairwise_loss']
  parts.append((weight,weight*(a-b)))
 rng=random.Random(42);values=[]
 for _ in range(5000):
  sampled=rng.choices(parts,k=len(parts));values.append(sum(x[1] for x in sampled)/sum(x[0] for x in sampled))
 values.sort();return {'difference':sum(x[1] for x in parts)/sum(x[0] for x in parts),'95pct_interval':[values[124],values[4874]]}
unspecified={p.stem:json.loads(p.read_text())['predictions'] for p in (root/'cmu-relative-results').glob('*.json') if p.name!='summary.json'}
mixture=json.loads((root/'cmu-dense-results/mean-max-50-50-predictions.json').read_text())
report={'none_minus_unspecified':compare(unspecified),'none_minus_dense_mixture':compare(mixture),'reviewer_bootstrap':5000,'seed':42,'warning':'Single realization and one request per condition; exploratory comparisons, no multiple-comparison correction, calls not interleaved.'}
Path('docs/exploration/effort-bootstrap.json').write_text(json.dumps(report,indent=2));print(report)
