"""Concrete bridge ablation: timing, prior silver, and fixed blind-audit outcomes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from expert_search.evaluation_gain import actionable_gain_interval


def run():
    root=Path('artifacts/concrete-partial-comparison-v1');audit=Path('artifacts/concrete-witness-audit-v1');protocol=json.loads((root/'protocol.json').read_text());rows=[json.loads((root/(q+'.json')).read_text()) for q in protocol['query_ids']]
    grades={}
    for r in map(json.loads,Path('artifacts/dataset-release/fresh-enterprise-eval-v1/judgments.jsonl').read_text().splitlines()):grades.setdefault(r['query_id'],{})[r['expert_id']]=r['grade']
    summaries={};audit_by_query={q:json.loads((audit/(q+'.json')).read_text()) for q in protocol['query_ids']}
    for arm in ['focused','concrete']:
        stats=Counter();levels=Counter();gains=[];tokens=[];screen=Counter()
        for row in rows:
            result=row['arms'][arm]['review'];valid=result['status']=='structurally_valid';ranking=result['result']['ranking'] if valid else [];names=[p['expert_id'] for p in ranking];positive={e for e,g in grades[row['query_id']].items() if g is not None and g>=2}
            stats.update(valid=valid,returned=len(names),known_silver_positives=len(positive),returned_known_silver_positives=len(positive&set(names)))
            gains.append(actionable_gain_interval(names,grades[row['query_id']],k=3));levels.update(p['level'] for p in ranking)
            for p in ranking:
                if p['level'] in ('transferable_method','direct_application'):
                    stats['actionable_mechanisms']+=1
                    stats['literal_ellipsis_or_trailing_hyphen']+=bool(re.search(r'…|\.\.\.|[-–—]\s*$',p['mechanism']))
            for round in result.get('rounds',[]):
                usage=round['response'].get('usage') or {}
                if type(usage.get('completion_tokens')) is int:tokens.append(usage['completion_tokens'])
            a=audit_by_query[row['query_id']]
            for name,meta in a['origins'].items():
                if meta['arm']==arm:screen[a.get('audit',{}).get('labels',{}).get(name,'unresolved')]+=1
        summaries[arm]=dict(stats,levels=dict(levels),mean_seconds=float(np.mean([r['arms'][arm]['seconds'] for r in rows])),mean_completion_tokens=float(np.mean(tokens)) if tokens else None,token_observed_requests=len(tokens),gain_lower=float(np.mean([g['lower'] for g in gains])),gain_upper=float(np.mean([g['upper'] for g in gains])),blind_witness_counts=dict(screen),screen_supported_per_60_slots=screen['supported_scoped_claim']/60,screen_supported_per_actionable_claim=screen['supported_scoped_claim']/sum(screen.values()) if screen else None)
    differences=np.asarray([r['arms']['concrete']['seconds']-r['arms']['focused']['seconds'] for r in rows]);rng=np.random.default_rng(20261117);boot=differences[rng.integers(0,len(rows),size=(10000,len(rows)))].mean(axis=1)
    report=dict(queries=len(rows),arms=summaries,mean_concrete_minus_focused_seconds=float(differences.mean()),latency_query_bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for q in protocol['query_ids'] for p in [root/(q+'.json'),audit/(q+'.json')]},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Both same3display slots and same7raw candidate inputs. All attempts timed. Prior silver is same-model and incomplete; exact output views retained. Fixed arm-blind same-model witness screen is not independent accuracy or semantic ground truth; screen counts cannot prove transfer validity, precision or recall. Non-actionable output contributes no positive screen credit; unresolved audits explicitly counted. Ellipsis statistic is a literal string diagnostic, not a completeness validator. No heldout usage, own training or search E2E claim.')
    (root/'analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':run()
