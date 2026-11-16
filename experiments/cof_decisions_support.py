"""Training-free native Decisions support-probability ranking, fixed NIPS pool."""
import hashlib
import inspect
import json
from pathlib import Path
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.decisions_backend import DecisionsBackend
from expert_search.shared_evidence import expand_candidates


def run():
    root,papers,reviewers,labels=load('NIPS')
    parent=root/'packing-shared-v2';out=root/'decisions-support-v1';out.mkdir(exist_ok=True)
    backend=DecisionsBackend();rows=[]
    for qid in sorted(labels['soft']):
        source=parent/(qid+'-similarity-input.json')
        payload=json.loads(source.read_text())
        original=json.loads((parent/(qid+'-similarity.json')).read_text())
        aliases=original['aliases'];candidates=expand_candidates(payload)
        if len(candidates)!=len(aliases):raise ValueError('Alias mismatch')
        pairs=[]
        for candidate in candidates:
            alias=candidate['candidate_id'];eid=aliases[alias]
            for e in candidate['evidence']:
                paper=papers[e['paper_id']];text=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
                if e['paper_id'] not in reviewers[eid]['papers'] or text[e['start']:e['end']]!=e['text']:
                    raise ValueError('Changed source ownership or span')
            pairs.append(dict(id=str(alias),constraint='The attributed publications establish direct research expertise in the specific problem and methods of this target paper: '+payload['target_paper'],
                              text=json.dumps(candidate['evidence'],ensure_ascii=False)))
        fingerprint=hashlib.sha256(json.dumps([pairs,inspect.getsource(DecisionsBackend)],sort_keys=True).encode()).hexdigest()
        target=out/(qid+'.json')
        if target.exists():
            row=json.loads(target.read_text())
            if row['fingerprint']!=fingerprint:raise ValueError('Changed request')
        else:
            row=dict(query_id=qid,fingerprint=fingerprint,status='error')
            try:
                response=backend.judge(pairs)
                scores={aliases[int(j['id'])]:j['probabilities']['supported'] for j in response['judgments']}
                ranking=sorted(scores,key=lambda eid:(-scores[eid],eid))
                row.update(status='valid',response=response,ranking=ranking,
                    metrics={task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()})
            except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,indent=2));temporary.replace(target)
        rows.append(row);print(qid,row['status'],flush=True)
    valid=[r for r in rows if r['status']=='valid']
    report=dict(dataset='NIPS',planned=len(rows),valid=len(valid),
        protocol='Same frozen similarity evidence and candidate pool as packing benchmark; rank by direct-expertise support probability, lexical expert ID tie break. Missing support is not inability. No tuning.',
        warning='Model probabilities are uncalibrated; released topic relevance labels do not validate entailment.')
    if len(valid)==len(rows):
        report['metrics']={task:{metric:sum(r['metrics'][task][metric] for r in rows)/len(rows) for metric in rows[0]['metrics'][task]} for task in labels}
        report['mean_request_seconds']=sum(r['response']['seconds'] for r in rows)/len(rows)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
