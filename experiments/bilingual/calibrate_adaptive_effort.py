"""Calibrate a fixed gap gate on construction pairs and evaluate later pairs."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from expert_search.owned_vector_index import ExactOwnedVectorIndex,OwnedVectorIndex
from expert_search.statistics import clustered_paired_interval
from experiments.bilingual.adaptive_effort_diagnostic import Effort,sha


THRESHOLDS=(0.,0.0005,0.001,0.002,0.004,0.008,0.016,0.032,0.064)
TARGET=0.995


def main(output):
    import faiss
    root=Path('artifacts/bilingual-retrieval-v1');summary=json.loads((root/'summary.json').read_text());encoding=json.loads((root/'encoding.json').read_text());selection=json.loads((root/'selection.json').read_text());proto=json.loads((root/'protocol.json').read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(root/'queries.jsonl')!=selection['queries_sha256'] or sha(root/'query-vectors.npz')!=encoding['query_vectors_sha256'] or sha(root/'protocol.json')!=summary['protocol_sha256'] or sha(root/'protocol.json')!=encoding['protocol_sha256'] or sha(source)!=proto['source_sha256'] or sha(vp)!=proto['source_vectors_sha256']:raise ValueError('Changed frozen inputs')
    queries=[json.loads(s) for s in (root/'queries.jsonl').read_text().splitlines()]
    with np.load(root/'query-vectors.npz',allow_pickle=False) as arrays:
        q=arrays['vectors'].copy();qids=arrays['ids'].tolist()
    if len(queries)!=1000 or qids!=[r['id'] for r in queries] or q.shape!=(2000,1024) or not np.isfinite(q).all():raise ValueError('Changed query inventory')
    if len({r['group_id'] for r in queries})!=1000:raise ValueError('Company groups overlap')
    records=[json.loads(s) for s in source.read_text().splitlines()];ids=[r['id'] for r in records];owners=[[r['expert_id']] for r in records];matrix=np.load(vp,mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(len(records),1024):raise ValueError('Changed source inventory')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'queries_sha256':sha(root/'queries.jsonl'),'query_vectors_sha256':sha(root/'query-vectors.npz'),'source_sha256':sha(source),'source_vectors_sha256':sha(vp),
              'previous_protocol_sha256':sha(root/'protocol.json'),'previous_summary_sha256':sha(root/'summary.json'),
              'calibration_pairs':400,'evaluation_pairs':600,'thresholds':list(THRESHOLDS),'retention_target':TARGET,
              'selection':'Among preset gap thresholds and direct ef512, choose the fastest simulated calibration policy reaching mean exact top-ten retention >= 0.995. Ties favor direct ef512.',
              'gate':'Search ef64 for 11 experts; escalate to ef512 if fewer than 11 results or tenth-minus-eleventh score <= threshold. Direct ef512 avoids low search.',
              'methods':'one rotating-order exact, ef64 and ef512 timing per view; then a separate actual selected-policy replay on evaluation views',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('experiments/bilingual/adaptive_effort_diagnostic.py'),Path('src/expert_search/owned_vector_index.py'),Path('src/expert_search/source_routing.py'),Path('src/expert_search/statistics.py'))},
              'scope':'Previously explored construction queries and frozen vectors; no expert-fit labels. Exact-score fidelity is not relevance or semantic correctness. Fixed score gap is heuristic, not an exactness guarantee.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    faiss.omp_set_num_threads(1);start=time.perf_counter();exact=ExactOwnedVectorIndex(matrix,ids,owners);graph=OwnedVectorIndex(matrix,ids,owners);low=OwnedVectorIndex(matrix,ids,owners,index=Effort(graph.index,64));high=OwnedVectorIndex(matrix,ids,owners,index=Effort(graph.index,512));build=time.perf_counter()-start
    def execute(engine,vector,k):
        started=time.perf_counter();hits=engine.search_vector(vector,k=k)['hits'];return hits,time.perf_counter()-started
    for engine,k in ((exact,10),(low,11),(high,11)):execute(engine,q[0],k)
    rows=[];engines={'exact':(exact,10),'low':(low,11),'high':(high,11)};names=('exact','low','high')
    for i,row in enumerate(queries):
        for j,language in enumerate(('zh','en')):
            offset=(2*i+j)%3;order=names[offset:]+names[:offset];observed={}
            for name in order:
                hits,t=execute(*((engines[name][0],q[2*i+j],engines[name][1])));observed[name]={'ids':[h['id'] for h in hits[:10]],'sources':[h['evidence'][0]['id'] for h in hits[:10]],'seconds':t}
                if name=='low':observed[name]['gap']=None if len(hits)<11 else float(hits[9]['score']-hits[10]['score'])
            if len(observed['exact']['ids'])!=10 or len(observed['high']['ids'])!=10:raise ValueError('Incomplete baseline')
            rows.append({'id':row['id'],'group_id':row['group_id'],'language':language,'arms':observed})
        if (i+1)%100==0:print('Completed baseline pairs',i+1,flush=True)
    def pick(row,threshold):
        if threshold is None:return 'high'
        gap=row['arms']['low']['gap'];return 'high' if gap is None or gap<=threshold else 'low'
    def outcome(subset,threshold):
        quality=[];times=[];ordered=[];escalated=0
        for row in subset:
            chosen=pick(row,threshold);escalated+=chosen=='high' and threshold is not None
            result=row['arms'][chosen];ref=row['arms']['exact'];quality.append(len(set(result['ids'])&set(ref['ids']))/10)
            ordered.append(result['ids']==ref['ids'] and result['sources']==ref['sources'])
            times.append(result['seconds']+(row['arms']['low']['seconds'] if threshold is not None and chosen=='high' else 0.))
        return {'views':len(subset),'mean_retention':statistics.mean(quality),'ordered_equal':sum(ordered),'simulated_mean_seconds':statistics.mean(times),'escalated':escalated}
    calibration=rows[:800];evaluation=rows[800:];candidates=[]
    for threshold in (None,)+THRESHOLDS:
        report=outcome(calibration,threshold);candidates.append({'threshold':threshold,**report})
    eligible=[x for x in candidates if x['mean_retention']>=TARGET]
    chosen=min(eligible,key=lambda r:(r['simulated_mean_seconds'],r['threshold'] is not None)) if eligible else None
    selected_policy=chosen['threshold'] if chosen else None
    # Failure to reach target is distinct from selecting the direct-high policy.
    decision={'target_met':bool(eligible),'threshold':selected_policy,'fallback':'direct_high' if selected_policy is None else None,'calibration':candidates}
    (output/'calibration.json').write_text(json.dumps(decision,indent=2)+'\n')
    actual=[]
    for index,row in enumerate(evaluation):
        vector=q[800+index];started=time.perf_counter();escalated=False
        if selected_policy is None:hits,_=execute(high,vector,11)
        else:
            hits,_=execute(low,vector,11);escalated=len(hits)<11 or hits[9]['score']-hits[10]['score']<=selected_policy
            if escalated:hits,_=execute(high,vector,11)
        seconds=time.perf_counter()-started;hits=hits[:10];expected=row['arms'][pick(row,selected_policy)]
        if [h['id'] for h in hits]!=expected['ids'] or [h['evidence'][0]['id'] for h in hits]!=expected['sources']:raise ValueError('Actual gate differs from frozen baseline')
        actual.append({'seconds':seconds,'escalated':escalated})
        if (index+1)%200==0:print('Completed evaluation views',index+1,flush=True)
    for row,actual_row in zip(evaluation,actual):row['actual_gate']=actual_row
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    comparison=clustered_paired_interval([r['actual_gate']['seconds']-r['arms']['high']['seconds'] for r in evaluation],[r['group_id'] for r in evaluation]);comparison['sampling_unit']='whole company group; language-view-weighted mean'
    report={'calibration_pairs':400,'evaluation_pairs':600,'target_retention':TARGET,'calibration_target_met':bool(eligible),'selected_threshold':selected_policy,
            'index_build_and_copies_seconds':build,'faiss':faiss.__version__,'node':platform.node(),
            'calibration_selected':outcome(calibration,selected_policy),'evaluation_selected_simulated':outcome(evaluation,selected_policy),
            'evaluation_high':outcome(evaluation,None),'evaluation_low':{'views':len(evaluation),'mean_retention':statistics.mean(len(set(r['arms']['low']['ids'])&set(r['arms']['exact']['ids']))/10 for r in evaluation),'ordered_equal':sum(r['arms']['low']['ids']==r['arms']['exact']['ids'] and r['arms']['low']['sources']==r['arms']['exact']['sources'] for r in evaluation),'mean_seconds':statistics.mean(r['arms']['low']['seconds'] for r in evaluation)},
            'evaluation_actual_mean_seconds':statistics.mean(x['seconds'] for x in actual),'evaluation_actual_escalated':sum(x['escalated'] for x in actual),
            'evaluation_actual_gate_minus_high_seconds':comparison,'actual_results_match_frozen':len(actual),
            'protocol_sha256':sha(output/'protocol.json'),'calibration_sha256':sha(output/'calibration.json'),'results_sha256':sha(output/'results.json'),
            'scope':protocol['scope']+' Calibration is post-hoc on previously inspected development data; evaluation pairs were also previously explored and are not a sealed test. Gating is heuristic and has no worst-case guarantee. Actual gate timing is a separate pass from high baseline, excludes initialization, encoding, index construction and model calls. Group bootstrap does not address timing drift.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
