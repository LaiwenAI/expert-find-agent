"""Blinded second-model audit; model agreement is not human ground truth."""
import json
from pathlib import Path
from laiwenai import LaiwenAI
from .agent import ExpertAgent


def run(source='artifacts/synthesis-pilot', output='artifacts/synthesis-audit', model='gpt-6-luna'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    agent=ExpertAgent(client=LaiwenAI(model=model))
    for p in sorted(Path(source).glob('*.json')):
        target=out/p.name
        if target.exists():continue
        r=json.loads(p.read_text())
        if r.get('status')=='rejected' or not r['example'].get('constraints'):
            target.write_text(json.dumps({'status':'rejected_at_generation','reason':'Source sample failed structural or evidence gates'},indent=2))
            continue
        constraints=[{'id':i,'text':c['text']} for i,c in enumerate(r['example']['constraints'])]
        agent.trace=[]
        try:
            answer=agent.ask('blind_audit',
                'Evaluate each constraint against only the supplied original expert evidence. Ignore industrial demand text as proof of expert capabilities. '
                'Return {"judgments":[{"id":int,"label":"supported|contradicted|unknown","evidence_id":str|null,"quote":str|null,"reason":str}]}. '
                'Supported requires direct evidence of the complete constraint, not merely the same broad topic. Missing evidence is unknown. Contradicted requires explicit incompatible evidence. Copy quotes exactly.',
                {'constraints':constraints,'evidence':list(r['sources'].values())})
            judgments=answer.get('judgments',[]); ids=[j.get('id') for j in judgments]
            if sorted(ids)!=list(range(len(constraints))):raise ValueError('Incomplete or duplicate constraint judgments')
            errors=[]; disagreements=[]
            for j in judgments:
                i=j['id']
                if j['label'] not in ['supported','contradicted','unknown']:errors.append('invalid label')
                if j['label']!='unknown':
                    ev=r['sources'].get(j.get('evidence_id'));quote=j.get('quote')
                    if not ev or not isinstance(quote,str) or not quote.strip() or quote not in ev['text']:errors.append('invalid quote')
                if j['label']!=r['example']['constraints'][i]['label']:disagreements.append(i)
            status='needs_review' if errors or disagreements else 'model_agreement_silver'
            row=dict(status=status,judgments=judgments,errors=errors,disagreements=disagreements,
                generator_label_provenance=r['label_provenance'],verifier_model=model,trace=agent.trace)
        except Exception as e:row=dict(status='error',error=f'{type(e).__name__}: {e}')
        target.write_text(json.dumps(row,ensure_ascii=False,indent=2))
    return str(out)
