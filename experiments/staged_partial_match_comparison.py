"""Paired actual API execution: full-view comparison vs short-gate then comparison."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from expert_search.choice_judgments import decide
from expert_search.partial_method_match import build_view
from expert_search.partial_match_schema import StructuredPartialMethodMatcher
from annotate_fresh_evaluation import FIT_INSTRUCTION,FIT_CHOICES

ROOT=Path('artifacts/staged-partial-comparison-v1')


def one(item):
    index,job=item;path=ROOT/(job['id']+'.json')
    if path.exists():return json.loads(path.read_text())
    from laiwenai import LaiwenAI
    client=LaiwenAI(model='gpt-6-luna',max_retries=0);matcher=StructuredPartialMethodMatcher(client);arms={}
    order=['full','staged'] if index%2==0 else ['staged','full']
    for arm in order:
        start=time.perf_counter();selected=job['candidates'];gate=None
        if arm=='staged':
            records,mapping=build_view(job['demand'],selected,{},'raw')
            questions=[dict(type='choice',name=c['id'],instructions=FIT_INSTRUCTION+' Candidate: '+c['id'],choices=FIT_CHOICES) for c in records['candidates']]
            try:
                gate=decide(client,records,questions)
                keep={mapping['experts'][name] for name,level in gate['labels'].items() if level!='insufficient'}
                selected={eid:s for eid,s in selected.items() if eid in keep}
            except Exception as error:gate=dict(status='unresolved_pass_all',error_type=type(error).__name__)
        if selected:
            try:review=matcher.match(job['demand'],selected,{},'raw')
            except Exception as error:review=dict(status='unresolved',error_type=type(error).__name__)
        else:review=dict(status='no_candidate_selected',result=dict(ranking=[]))
        arms[arm]=dict(selected_experts=list(selected),gate=gate,review=review,seconds=time.perf_counter()-start,scope='Partial-method model output; excluded candidates are not judged incapable or irrelevant.')
    row=dict(query_id=job['id'],split='development',order=order,arms=arms)
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2));print(job['id'],[(a,round(r['seconds'],2),len(r['selected_experts']),r['review']['status']) for a,r in arms.items()],flush=True);return row


def run():
    source=Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl');jobs=[j for j in map(json.loads,source.read_text().splitlines()) if j['split']=='development'][:20]
    ROOT.mkdir(exist_ok=True)
    protocol=dict(model='gpt-6-luna',query_ids=[j['id'] for j in jobs],jobs_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),policy='Native gate retain all except insufficient; error passes all. Then original strict structured comparator on retained exact raw source views. Full arm compares all7. Alternate arm order,3workers,no retries. Gate+comparison time charged to staged arm. Policy selected after development gate audit, not independent test. Empty selection returns explicit no-candidate status, not negative ability.',code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/annotate_fresh_evaluation.py'),Path('src/expert_search/partial_match_schema.py'),Path('src/expert_search/partial_method_match.py'),Path('src/expert_search/choice_judgments.py')]},scope='Bounded candidate pipeline, not full retrieval E2E. Quality diagnostics against same-model silver only; no independent expert accuracy. Heldout unused.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed frozen staged comparison')
    marker.write_text(json.dumps(protocol,indent=2))
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,enumerate(jobs)))
    summary=dict(queries=len(rows),arms={a:dict(mean_seconds=sum(r['arms'][a]['seconds'] for r in rows)/len(rows),selected_candidates=sum(len(r['arms'][a]['selected_experts']) for r in rows),valid_reviews=sum(r['arms'][a]['review']['status']=='structurally_valid' for r in rows)) for a in ['full','staged']},scope=protocol['scope'])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
