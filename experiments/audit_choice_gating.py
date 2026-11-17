"""Diagnostic gate workload vs known silver retention; no independent accuracy claim."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ACTIONABLE={'transferable_method','direct_application'}


def run():
    source=Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl')
    jobs={j['id']:j for j in map(json.loads,source.read_text().splitlines()) if j['split']=='development'}
    truth={}
    for row in map(json.loads,Path('artifacts/dataset-release/fresh-enterprise-eval-v1/judgments.jsonl').read_text().splitlines()):
        if row['grade'] is not None and row['grade']>=2:truth.setdefault(row['query_id'],set()).add(row['expert_id'])
    reports={};inputs={}
    for repeat,root in enumerate([Path('artifacts/choice-interface-comparison-v1'),Path('artifacts/choice-interface-comparison-repeat-v1')]):
        protocol=json.loads((root/'protocol.json').read_text())
        for arm in ['native','chat']:
            for policy in ['actionable-only','not-insufficient']+(['top1','top2','top3','top5'] if arm=='native' else []):
                counters=Counter();misses=[]
                for qid in protocol['query_ids']:
                    path=root/(qid+'.json');inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();row=json.loads(path.read_text());result=row['arms'][arm];eids=list(jobs[qid]['candidates']);names=['c'+str(i) for i in range(len(eids))]
                    if result['status']!='valid':chosen=set(eids);counters['failed_gate_queries_pass_all']+=1
                    else:
                        labels=result['result']['labels']
                        if set(labels)!=set(names):raise ValueError('Foreign candidate names')
                        if policy.startswith('top'):
                            answers={a['name']:a for a in result['result']['response']['answers']}
                            scores={k:sum(p['probability'] for p in answers[k]['probabilities'] if p['value'] in ACTIONABLE) for k in names}
                            keep=sorted(names,key=lambda n:(-scores[n],n))[:int(policy[3:])]
                        else:keep=[n for n in names if (labels[n] in ACTIONABLE if policy=='actionable-only' else labels[n]!='insufficient')]
                        chosen={eids[int(n[1:])] for n in keep}
                    positives=truth.get(qid,set())&set(eids);missing=positives-chosen
                    counters.update(queries=1,candidates=len(eids),retained_candidates=len(chosen),known_silver_positives=len(positives),retained_known_positives=len(positives&chosen),queries_with_known_positive=bool(positives),queries_losing_any_known_positive=bool(missing),queries_losing_all_known_positives=bool(positives and not positives&chosen))
                    if missing:misses.append(dict(query_id=qid,expert_ids=sorted(missing)))
                reports[f'round{repeat+1}:{arm}:{policy}']=dict(counters,misses=misses)
    report=dict(policies=reports,inputs_sha256=inputs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Fresh short-choice gate decisions on first20 development queries vs prior same-model consensus silver in exact original views. This is known-positive retention, not actual relevance recall, independent accuracy, proof discarded candidates are irrelevant or measured downstream speedup. Missing labels remain unknown. Failed gates pass all; probability sums used only as uncalibrated ordering. No heldout data or policy selection based on heldout.')
    out=Path('artifacts/choice-gating-audit-v1');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2))
    for name,r in reports.items():print(name,r['retained_candidates'],'/',r['candidates'],'retained known',r['retained_known_positives'],'/',r['known_silver_positives'])


if __name__=='__main__':run()
