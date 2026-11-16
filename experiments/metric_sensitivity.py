"""Controlled construct checks, not human-utility validation."""
import json
from pathlib import Path
from expert_search.constraint_metrics import evidence_gain,deadline_gain

constraints={f'c{i}':{'weight':1,'mandatory':i<2} for i in range(10)}
cases={
 'all_supported':{k:'supported' for k in constraints},
 'missing_necessary':{k:'supported' if k!='c0' else 'unknown' for k in constraints},
 'missing_optional':{k:'supported' if k!='c9' else 'unknown' for k in constraints},
 'explicit_contradiction':{k:'supported' if k!='c0' else 'contradicted' for k in constraints},
 'all_unknown':{},
}
rows={name:evidence_gain(constraints,j) for name,j in cases.items()}
assert rows['missing_necessary']['supported_coverage']==rows['missing_optional']['supported_coverage']
assert rows['missing_necessary']['gated_gain']<rows['missing_optional']['gated_gain']
assert rows['explicit_contradiction']['unknown_weight']<rows['missing_necessary']['unknown_weight']
events=[(1,rows['missing_necessary']['gated_gain']),(3,rows['all_supported']['gated_gain'])]
report=dict(cases=rows,deadline_gain={str(t):deadline_gain(events,t) for t in [.5,1,2,3,5]},
 caveat='Constructed cases establish intended behavior, not novelty, human utility correlation, or matching accuracy.')
Path('docs/exploration/metric-sensitivity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
