"""Matched top3 output-budget analysis using incomplete development silver."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from expert_search.evaluation_gain import actionable_gain_interval,latest_gain_interval


def run():
    root=Path('artifacts/focused-partial-comparison-v1');protocol=json.loads((root/'protocol.json').read_text());rows=[json.loads((root/(q+'.json')).read_text()) for q in protocol['query_ids']]
    grades={}
    for r in map(json.loads,Path('artifacts/dataset-release/fresh-enterprise-eval-v1/judgments.jsonl').read_text().splitlines()):grades.setdefault(r['query_id'],{})[r['expert_id']]=r['grade']
    arms={};evaluations=[]
    for arm in ['full','focused']:
        stats=Counter();gains=[];completion=[];input_chars=[];deadlines={s:[] for s in [2,3,5,10,20]}
        for row in rows:
            a=row['arms'][arm];review=a['review'];valid=review['status']=='structurally_valid';ranking=[r['expert_id'] for r in review['result']['ranking']][:3] if valid else [];judgments=grades[row['query_id']]
            gain=actionable_gain_interval(ranking,judgments,k=3);positive={e for e,g in judgments.items() if g is not None and g>=2}
            stats.update(valid=valid,known_positive_candidates=len(positive),returned_known_positive_candidates=len(positive&set(ranking)),returned_candidates=len(ranking))
            for round in review.get('rounds',[]):
                input_chars.append(round['input_chars']);usage=round['response'].get('usage') or {}
                if type(usage.get('completion_tokens')) is int:completion.append(usage['completion_tokens'])
            events=[(a['seconds'],gain['lower'],gain['upper'])] if valid else []
            for s in deadlines:deadlines[s].append(latest_gain_interval(events,s))
            gains.append(gain);evaluations.append(dict(query_id=row['query_id'],arm=arm,status=review['status'],seconds=a['seconds'],ranking=ranking,gain=gain))
        times=[r['arms'][arm]['seconds'] for r in rows]
        arms[arm]=dict(stats,mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),empirical_p95_seconds=float(np.quantile(times,.95)),mean_gain_lower=float(np.mean([g['lower'] for g in gains])),mean_gain_upper=float(np.mean([g['upper'] for g in gains])),completion_tokens_observed_requests=len(completion),mean_completion_tokens=float(np.mean(completion)) if completion else None,input_chars_observed_requests=len(input_chars),mean_input_chars=float(np.mean(input_chars)) if input_chars else None,deadline_gain={str(s):dict(lower=float(np.mean([x['lower'] for x in v])),upper=float(np.mean([x['upper'] for x in v]))) for s,v in deadlines.items()})
    common=[r for r in rows if all(a['review']['status']=='structurally_valid' for a in r['arms'].values())];equal=0;overlap=[]
    for r in common:
        lists=[[x['expert_id'] for x in r['arms'][a]['review']['result']['ranking']][:3] for a in ['full','focused']];equal+=lists[0]==lists[1];overlap.append(len(set(lists[0])&set(lists[1]))/3)
    observed=[r for r in rows if all(a['review'].get('rounds') for a in r['arms'].values())]
    if any(r['arms']['full']['review']['rounds'][0]['input_chars']!=r['arms']['focused']['review']['rounds'][0]['input_chars'] for r in observed):raise ValueError('Different raw input character counts')
    paired_valid_latency=float(np.mean([r['arms']['focused']['seconds']-r['arms']['full']['seconds'] for r in common])) if common else None
    differences=np.asarray([r['arms']['focused']['seconds']-r['arms']['full']['seconds'] for r in rows]);rng=np.random.default_rng(20261117);boot=differences[rng.integers(0,len(rows),size=(10000,len(rows)))].mean(axis=1)
    report=dict(queries=len(rows),cutoff=3,arms=arms,paired_valid_queries=len(common),paired_valid_mean_focused_minus_full_seconds=paired_valid_latency,paired_observed_equal_input_chars_queries=len(observed),ordered_top3_agreements=equal,mean_top3_set_overlap=float(np.mean(overlap)) if overlap else None,mean_focused_minus_full_seconds=float(differences.mean()),query_bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),conditional_gain_difference_bounds=[arms['focused']['mean_gain_lower']-arms['full']['mean_gain_upper'],arms['focused']['mean_gain_upper']-arms['full']['mean_gain_lower']],inputs_sha256={q:hashlib.sha256((root/(q+'.json')).read_bytes()).hexdigest() for q in protocol['query_ids']},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Same top3 budget and raw inputs, actual API attempts including failure latency. Gains conditional on incomplete same-model development silver, not independent semantic accuracy, confidence or actual ability. Difference bounds are conservative silver-label ranges, not statistical confidence intervals. Latency bootstrap clusters by20 fixed queries. No retrieval/encoding/init or independent citation correctness audit. Token averages only requests with observed usage, not imputed failures. Tail outputs not compared or negated.')
    (root/'evaluation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in evaluations));(root/'analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':run()
