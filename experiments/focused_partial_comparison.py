"""Paired whole-pool input, explain-all vs explain-top3 on next20 dev queries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from expert_search.partial_match_schema import StructuredPartialMethodMatcher
from expert_search.focused_partial_match import FocusedPartialMethodMatcher

ROOT=Path('artifacts/focused-partial-comparison-v1')


def one(item):
    index,job=item;path=ROOT/(job['id']+'.json')
    if path.exists():return json.loads(path.read_text())
    from laiwenai import LaiwenAI
    client=LaiwenAI(model='gpt-6-luna',max_retries=0);full=StructuredPartialMethodMatcher(client);focused=FocusedPartialMethodMatcher(client);arms={}
    order=['full','focused'] if index%2==0 else ['focused','full']
    for arm in order:
        start=time.perf_counter()
        try:review=full.match(job['demand'],job['candidates'],{},'raw') if arm=='full' else focused.match(job['demand'],job['candidates'],k=3)
        except Exception as error:review=dict(status='unresolved',error_type=type(error).__name__)
        arms[arm]=dict(review=review,seconds=time.perf_counter()-start)
    row=dict(query_id=job['id'],split='development',order=order,candidates=list(job['candidates']),arms=arms)
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2));print(job['id'],[(a,round(r['seconds'],2),r['review']['status']) for a,r in arms.items()],flush=True);return row


def run():
    source=Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl');jobs=[j for j in map(json.loads,source.read_text().splitlines()) if j['split']=='development'][20:40]
    ROOT.mkdir(exist_ok=True)
    protocol=dict(model='gpt-6-luna',query_ids=[j['id'] for j in jobs],jobs_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selection='Next20 fixed development queries (positions20:40), distinct from prior20 latency/gate cohort; still development, not sealed or independently labeled test.',
                  comparison='Same seven raw candidate views and reference-specific schema. One request each; all7 explanations vs only top3. Both max_completion_tokens2400,reasoningnone,no retries. Alternate arm order,3workers. Both evaluated at top3; missing outputs remain unresolved and tail candidates have no negative labels.',
                  code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/expert_search/partial_match_schema.py'),Path('src/expert_search/partial_method_match.py'),Path('src/expert_search/focused_partial_match.py')]},scope='Candidate-pool comparison, excludes full retrieval/encoding/init. Prior same-model silver cannot establish independent accuracy or evidence-claim semantic correctness. Heldout unused.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed frozen focused experiment')
    marker.write_text(json.dumps(protocol,indent=2))
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,enumerate(jobs)))
    print('Completed',len(rows),'paired queries',flush=True)


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
