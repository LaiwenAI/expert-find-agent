"""Fresh paired calls for a fixed, development-selected selective policy."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from expert_search.merit_suitability import prompts, build_case, gold, parse_final_label
from experiments.merit_suitability_comparison import DATA, CHOICES

ROOT = Path('artifacts/merit-selective-live-pilot-v1')


def main():
    source = DATA / 'MERIT-Assessor/test/test.json'; prompt_path = DATA / 'MERIT-Assessor/test/prompts.py'
    prior = json.loads(Path('artifacts/merit-suitability-comparison-v1/protocol.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(source) != prior['dataset_sha256'] or sha(prompt_path) != prior['official_prompts_sha256']:
        raise ValueError('Frozen source changed')
    official = prompts(prompt_path); raw = json.loads(source.read_text())
    all_cases = [(build_case(row, i, official), gold(row)) for i, row in enumerate(raw)]
    # Select one case per exact-title group from the replay test split without labels.
    eligible = [(case, label) for case, label in all_cases if int(hashlib.sha256(case['paper_cluster'].encode()).hexdigest()[:8], 16) % 2 == 1]
    eligible.sort(key=lambda item: hashlib.sha256(('pilot-v1:' + item[0]['id']).encode()).hexdigest())
    selected = []; groups = set()
    for case, label in eligible:
        if case['paper_cluster'] not in groups:
            groups.add(case['paper_cluster']); selected.append((case, label))
            if len(selected) == 40: break
    if len(selected) != 40: raise ValueError('Insufficient distinct groups')
    ROOT.mkdir(parents=True, exist_ok=False); (ROOT / 'predictions').mkdir()
    protocol = dict(model='gpt-6-luna', threshold=0.75, cases=[dict(id=c['id'], input_sha256=c['input_sha256'], paper_cluster=c['paper_cluster']) for c, _ in selected],
                    dataset_sha256=sha(source), official_prompts_sha256=sha(prompt_path),
                    selection='First 40 distinct test-title groups by fixed pair-ID hash; labels unused; prior full cohort was already examined.',
                    order='Alternate cascade/full arm first by selected index; 3 workers; no retries.',
                    scope='Fresh calls on previously studied public self-ratings, not untouched data or industrial suitability. Cascade request time includes both serial calls when escalated; client initialization excluded.')
    (ROOT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    load_dotenv('.env')
    def execute(item):
        index, (case, _) = item; client = LaiwenAI(model='gpt-6-luna', max_retries=0); result = {}
        def full():
            answer = client.chat.completions.create(messages=[dict(role='system',content=official['action_system_prompt']), dict(role='user',content=case['input'])],reasoning_effort='none',max_completion_tokens=4096)
            return parse_final_label(answer.choices[0].message.content)
        def cascade():
            q = [dict(type='choice',name='suitability',instructions=prior['criterion'],choices=CHOICES)]
            answer = decide(client, dict(target_and_profile=case['input']), q)
            probs = answer['response']['answers'][0]['probabilities']; confidence = max(p['probability'] for p in probs)
            if not .5 <= confidence <= 1: raise ValueError('Invalid decision score')
            escalate = confidence < 0.75
            return (full() if escalate else int(answer['labels']['suitability']=='qualified')), escalate, confidence
        for arm in (('cascade','full') if index % 2 == 0 else ('full','cascade')):
            started = time.perf_counter(); record = {'status':'unresolved', 'label':None}
            try:
                if arm == 'full': record.update(status='valid', label=full())
                else:
                    label, escalate, confidence = cascade()
                    record.update(status='valid', label=label, escalated=escalate, maximum_choice_probability=confidence)
            except Exception as exc: record['error_type'] = type(exc).__name__
            record['seconds'] = time.perf_counter() - started; result[arm] = record
        record = dict(id=case['id'], input_sha256=case['input_sha256'], arms=result)
        path = ROOT / 'predictions' / (case['id']+'.json'); path.write_text(json.dumps(record, indent=2)+'\n')
        print(f"{index+1}/40 {case['id']} " + str({k:v['status'] for k,v in result.items()}), flush=True)
        return record
    with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(execute, enumerate(selected)))
    labels={c['id']: label for c,label in selected}
    summary={'cases':len(results), 'arms':{}}
    for arm in ('cascade','full'):
        values=[r['arms'][arm] for r in results]; valid=[(r, r['arms'][arm]) for r in results if r['arms'][arm]['status']=='valid']
        correct=sum(v['label']==labels[r['id']] for r,v in valid)
        summary['arms'][arm]=dict(valid=len(valid), unresolved=len(values)-len(valid), correct=correct,
                                   full_cohort_accuracy_bounds=[correct/len(values),(correct+len(values)-len(valid))/len(values)],
                                   mean_seconds=sum(v['seconds'] for v in values)/len(values))
    summary['cascade_escalations_observed']=sum(r['arms']['cascade'].get('escalated',False) for r in results)
    summary['scope']=protocol['scope']+' Pilot is not powered for equivalence; failed requests remain unresolved.'
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n'); print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
