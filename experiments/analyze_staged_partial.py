"""Analyze actually executed bounded pipeline; conditional silver, not gold."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from expert_search.evaluation_gain import actionable_gain_interval,latest_gain_interval


def run():
    root=Path('artifacts/staged-partial-comparison-v1');protocol=json.loads((root/'protocol.json').read_text())
    rows=[json.loads((root/(qid+'.json')).read_text()) for qid in protocol['query_ids']]
    judgments={}
    for r in map(json.loads,Path('artifacts/dataset-release/fresh-enterprise-eval-v1/judgments.jsonl').read_text().splitlines()):judgments.setdefault(r['query_id'],{})[r['expert_id']]=r['grade']
    results={};evaluations=[]
    for arm in ['full','staged']:
        stats=Counter();lows=[];highs=[];deadline={s:[] for s in [2,5,10,20]}
        for row in rows:
            a=row['arms'][arm];review=a['review'];valid=review['status']=='structurally_valid';ranking=[r['expert_id'] for r in review['result']['ranking']] if valid else []
            grades=judgments[row['query_id']];gain=actionable_gain_interval(ranking,grades);positive={e for e,g in grades.items() if g is not None and g>=2}
            stats.update(selected=len(a['selected_experts']),known_positives=len(positive),selected_known_positives=len(positive&set(a['selected_experts'])),top5_known_positives=len(positive&set(ranking[:5])),valid=valid)
            lows.append(gain['lower']);highs.append(gain['upper'])
            events=[(a['seconds'],gain['lower'],gain['upper'])] if valid else []
            for s in deadline:deadline[s].append(latest_gain_interval(events,s))
            evaluations.append(dict(query_id=row['query_id'],arm=arm,status=review['status'],seconds=a['seconds'],ranking=ranking,gain=gain,omitted_known_positives=sorted(positive-set(a['selected_experts']))))
        results[arm]=dict(stats,mean_seconds=float(np.mean([r['arms'][arm]['seconds'] for r in rows])),mean_gain_lower=float(np.mean(lows)),mean_gain_upper=float(np.mean(highs)),deadline_gain={str(s):dict(lower=float(np.mean([x['lower'] for x in v])),upper=float(np.mean([x['upper'] for x in v]))) for s,v in deadline.items()})
    differences=np.asarray([r['arms']['staged']['seconds']-r['arms']['full']['seconds'] for r in rows]);rng=np.random.default_rng(20261117);boot=differences[rng.integers(0,len(rows),size=(10000,len(rows)))].mean(axis=1)
    report=dict(queries=len(rows),arms=results,mean_staged_minus_full_seconds=float(differences.mean()),query_bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),inputs_sha256={qid:hashlib.sha256((root/(qid+'.json')).read_bytes()).hexdigest() for qid in protocol['query_ids']},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Actual gate+structured comparison, excludes candidate retrieval/encoding/input initialization. All attempts timed, unresolved gives no completed output. Ranking gains conditional on incomplete prior same-model silver; not independent precision or output-claim correctness. Deadline snapshots use actual per-arm completion time. Gate policy developed on this cohort, no heldout claim.')
    (root/'evaluation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in evaluations));(root/'analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':run()
