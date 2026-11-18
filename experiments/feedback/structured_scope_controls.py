"""Handwritten controls exposing lexical feedback-scope limitations."""
import json
from pathlib import Path
from expert_search.feedback_content import Correction, apply_corrections


def main():
    correction=Correction('b','battery','p','battery recycling','promote')
    sources={'b':{'p':'The project investigated battery recycling.'}}
    cases=[
        ('direct', 'Find an expert in battery recycling.', True),
        ('different', 'Find an expert in photovoltaic panels.', False),
        ('negated', 'Find an expert in recycling, excluding battery recycling.', False),
        ('hypothetical', 'Find an expert in solar recycling; battery recycling is only an unrelated example.', False),
        ('contrast', 'Battery recycling is out of scope; find a plastics recycling expert.', False),
        ('substring', 'Find an expert in batteryless sensors.', False),
    ]
    results=[]
    for name,query,expected in cases:
        result=apply_corrections(query,['a','b'],[correction],sources,scope_verdicts={correction:True})
        applied=result['ranking'][0]=='b'
        results.append({'case':name,'expected_applicable':expected,'applied':applied,'audit':result['audit']})
    report={'cases':results,'correct':sum(r['expected_applicable']==r['applied'] for r in results),
            'scope':'Six handcrafted English controls defined alongside the prototype; not a representative benchmark, independent annotations, or a language-understanding evaluation. A deliberately affirmative external verdict isolates the lexical scope rule and is not a valid semantic assessment. Expected applicability is a manually specified scenario assumption. The source quotation is intentionally valid in all cases; matching a quote does not establish relevance, entailment or personal contribution.'}
    output=Path('artifacts/structured-feedback-scope-controls-v3');output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
