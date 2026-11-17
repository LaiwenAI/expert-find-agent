"""Verify calibration accounting and identify same-policy timing comparisons."""
import argparse
import json
from pathlib import Path
import statistics
from experiments.bilingual.adaptive_effort_diagnostic import sha


def audit(folder):
    summary=json.loads((folder/'summary.json').read_text());protocol=json.loads((folder/'protocol.json').read_text());cal=json.loads((folder/'calibration.json').read_text())
    for name,key in [('protocol.json','protocol_sha256'),('calibration.json','calibration_sha256'),('results.json','results_sha256')]:
        if sha(folder/name)!=summary[key]:raise ValueError('Changed frozen results')
    rows=json.loads((folder/'results.json').read_text());split=protocol['calibration_pairs']*2
    if len(rows)!=(protocol['calibration_pairs']+protocol['evaluation_pairs'])*2 or len({r['group_id'] for r in rows[:split]})!=protocol['calibration_pairs'] or len({r['group_id'] for r in rows[split:]})!=protocol['evaluation_pairs'] or {r['group_id'] for r in rows[:split]}&{r['group_id'] for r in rows[split:]}:raise ValueError('Invalid group split')
    def summarize(subset,threshold):
        retain=[];times=[];ordered=0;escalated=0
        for r in subset:
            low=r['arms']['low'];hi=r['arms']['high'];ref=r['arms']['exact']
            up=threshold is not None and (low['gap'] is None or low['gap']<=threshold);arm=hi if threshold is None or up else low
            retain.append(len(set(arm['ids'])&set(ref['ids']))/10);ordered+=arm['ids']==ref['ids'] and arm['sources']==ref['sources']
            times.append(arm['seconds']+(low['seconds'] if up else 0.));escalated+=up
        return {'views':len(subset),'mean_retention':statistics.mean(retain),'ordered_equal':ordered,'simulated_mean_seconds':statistics.mean(times),'escalated':escalated}
    thresholds=[None]+protocol['thresholds']
    if [r['threshold'] for r in cal['calibration']]!=thresholds:raise ValueError('Changed policy inventory')
    for row in cal['calibration']:
        recomputed=summarize(rows[:split],row['threshold'])
        if any(recomputed[k]!=row[k] for k in recomputed):raise ValueError('Changed calibration accounting')
    eligible=[r for r in cal['calibration'] if r['mean_retention']>=protocol['retention_target']]
    selected=min(eligible,key=lambda r:(r['simulated_mean_seconds'],r['threshold'] is not None)) if eligible else None
    threshold=selected['threshold'] if selected else None
    if cal['threshold']!=threshold or cal['target_met']!=bool(eligible) or summary['selected_threshold']!=threshold:raise ValueError('Changed policy selection')
    if summarize(rows[split:],threshold)!=summary['evaluation_selected_simulated'] or summarize(rows[split:],None)!=summary['evaluation_high']:raise ValueError('Changed evaluation accounting')
    if any('actual_gate' not in row for row in rows[split:]) or summary['actual_results_match_frozen']!=len(rows)-split:raise ValueError('Incomplete actual replay')
    return {'calibration_views':split,'evaluation_views':len(rows)-split,'eligible_policies':sum(r['mean_retention']>=protocol['retention_target'] for r in cal['calibration']),
            'selected_policy':'direct_high' if threshold is None else 'gap_gate','selected_threshold':threshold,
            'same_algorithm_as_high_baseline':threshold is None,'evaluation_retention':summary['evaluation_selected_simulated']['mean_retention'],
            'evaluation_ordered_equal':summary['evaluation_selected_simulated']['ordered_equal'],'input_summary_sha256':sha(folder/'summary.json'),
            'scope':'Recomputes selection and observed fidelity. When direct high is selected, its two timings are the same algorithm in different passes; their difference is not an algorithmic speedup. Exact-score fidelity is not relevance, and this is not a sealed test.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.folder);a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('Audit exists')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
