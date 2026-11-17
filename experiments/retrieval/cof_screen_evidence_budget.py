"""Paired exploratory screen with one versus three evidence views per candidate."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.progressive_decision_agent import ProgressiveDecisionAgent
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_progressive_screen_pilot import FixedPipeline, sha, verify_prior


def main(output):
    root,papers,reviewers,labels=load('SIGIR');prior=Path('artifacts/cof-owned-vote-quality-v2')
    hashes=verify_prior(root,papers,reviewers,prior)
    if set(labels['soft'])!=set(labels['hard']) or any(set(labels[t][q])!=set(reviewers) for t in labels for q in labels[t]):raise ValueError('Incomplete judgments')
    ids=sorted(labels['soft'])[:20];output.mkdir(parents=True,exist_ok=False)
    protocol={'query_ids':ids,'selection':'First 20 lexically sorted previously studied SIGIR IDs','model':'gpt-6-luna','workers':4,'retries':0,
              'candidate_pool':50,'screen_candidates':20,'evidence_per_candidate':[1,3], 'order':'Alternating arm order by query position; serial per query',
              'input_sha256':hashes,'prior_artifact_sha256':{str(p.relative_to(prior)):sha(p) for p in (prior/'corpus/experts.jsonl',prior/'corpus/evidence.jsonl',prior/'lm/metadata.json',prior/'lm/arrays.npz',prior/'fast.sqlite')},
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/progressive_decision_agent.py'),Path('src/expert_search/choice_judgments.py'),Path('src/expert_search/candidate_evidence_pipeline.py'))},
              'scope':'Exploratory reused public reviewer matching; reviewer-level topic labels do not establish source entailment or enterprise fit. Source copies change document frequencies. A paired view-budget contrast is not an end-to-end explanation evaluation.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env')
    start=time.perf_counter();engine=CandidateEvidencePipeline(prior/'corpus',prior/'lm',prior/'fast.sqlite',scoring='resident');init=time.perf_counter()-start
    frozen=[]
    for qid in ids:
        p=papers[qid];query=(p.get('title','')+'\n'+p.get('abstract','')).strip();start=time.perf_counter()
        three=engine.search_reranked(query,k=20,candidates=50,evidence_per_expert=3);retrieval=time.perf_counter()-start
        one=deepcopy(three)
        for r in one['results']:r['evidence']=r['evidence'][:1]
        frozen.append((qid,query,one,three,retrieval))
    engine.evidence.db.close()
    snapshots={qid:{arm:hashlib.sha256(json.dumps(res['results'],sort_keys=True,ensure_ascii=False).encode()).hexdigest() for arm,res in [('one',one),('three',three)]} for qid,_,one,three,_ in frozen}
    (output/'candidate-snapshots.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    def run(item,position):
        qid,query,one,three,retrieval=item;client=LaiwenAI(model='gpt-6-luna',max_retries=0)
        row={'query_id':qid,'retrieval_three_seconds':retrieval,'arms':{}}
        for arm in (('one','three') if position%2==0 else ('three','one')):
            result=one if arm=='one' else three; n=1 if arm=='one' else 3
            agent=ProgressiveDecisionAgent(FixedPipeline(result),client=client);stream=agent.events(query,k=5,candidates=20,rerank_pool=50,explanation_candidates=20,evidence_per_expert=n)
            first=next(stream);screen=next(stream);entry={'stage':screen['stage'],'seconds':screen['seconds']-first['seconds']}
            if screen['stage']=='screen':
                ranking=[x['expert']['id'] for x in screen['results'][:5]]
                entry.update(ranking=ranking,judgments=screen['judgments'],selected=screen['selected'],metrics={t:precision_metrics(ranking,labels[t][qid]) for t in labels})
            else:entry['error_type']=screen.get('error_type',screen.get('reason'))
            row['arms'][arm]=entry
        print(qid,*(row['arms'][a]['stage'] for a in ('one','three')),flush=True);return row
    (output/'checkpoints').mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,item,i) for i,item in enumerate(frozen)];results={}
        for future in as_completed(futures):
            row=future.result();qid=row['query_id'];results[qid]=row;dest=output/'checkpoints'/(hashlib.sha256(qid.encode()).hexdigest()+'.json')
            tmp=dest.with_suffix('.partial');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(dest)
    rows=[results[qid] for qid in ids];(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    valid=[r for r in rows if all(r['arms'][a]['stage']=='screen' for a in ('one','three'))]
    report={'planned':len(rows),'joint_valid':len(valid),'arm_valid':{a:sum(r['arms'][a]['stage']=='screen' for r in rows) for a in ('one','three')},
            'initialization_seconds':init,'mean_three_view_retrieval_seconds':statistics.mean(r['retrieval_three_seconds'] for r in rows),
            'mean_attempt_seconds':{a:statistics.mean(r['arms'][a]['seconds'] for r in rows) for a in ('one','three')},
            'paired':{t:paired_interval([r['arms']['three']['metrics'][t]['p@5']-r['arms']['one']['metrics'][t]['p@5'] for r in valid]) if valid else None for t in labels},
            'valid_mean_p@5':{t:{a:statistics.mean(r['arms'][a]['metrics'][t]['p@5'] for r in valid) if valid else None for a in ('one','three')} for t in labels},
            'mean_returned':{a:statistics.mean(len(r['arms'][a]['ranking']) for r in valid) if valid else None for a in ('one','three')},
            'protocol_sha256':sha(output/'protocol.json'),'candidate_snapshots_sha256':sha(output/'candidate-snapshots.json'),'results_sha256':sha(output/'results.json'),
            'scope':protocol['scope']+' Bootstrap assumes independent queries, unadjusted for multiple comparisons. Valid-only relevance estimates exclude failures. Timings are concurrent and exclude initialization and explanation.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
