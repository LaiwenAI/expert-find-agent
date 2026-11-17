"""Evaluate retrieved abstracts, rationale selection and source verdicts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import verify_evidence_documents
from expert_search.choice_judgments import decide
from expert_search.fast_retrieval import FastSearch
from experiments.scifact.decisions_oracle_pilot import QUESTION,load,sha
from experiments.scifact.rationale_selection_pilot import lexical_order,metrics


def main(archive,parent,corpus,database,output):
    protocol=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    if sha(archive)!=protocol['archive_sha256'] or sha(parent/'protocol.json')!=summary['protocol_sha256'] or sha(parent/'results.json')!=summary['results_sha256']:raise ValueError('Changed source or cohort')
    source,claims=load(archive);papers={str(p['doc_id']):p for p in source};by_claim={str(c['id']):c for c in claims}
    inputs=[json.loads(x) for x in (corpus/'queries.jsonl').read_text().splitlines()];queries={q['id']:q for q in inputs}
    if len(queries)!=len(inputs):raise ValueError('Duplicate adapted queries')
    output.mkdir(parents=True,exist_ok=False);(output/'attempts').mkdir();load_dotenv('.env')
    frozen={'source_sha256':sha(archive),'parent_protocol_sha256':sha(parent/'protocol.json'),'adapted_corpus_sha256':sha(corpus/'evidence.jsonl'),
            'adapted_experts_sha256':sha(corpus/'experts.jsonl'),'adapted_queries_sha256':sha(corpus/'queries.jsonl'),'index_sha256':sha(database),
            'case_ids':[c['claim_id'] for c in protocol['cases']],'retrieval':'BM25 top 1','rationale':'sentence BM25 top 3','question':QUESTION,'model':'gpt-6-luna','workers':3,'retries':0,
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('experiments/scifact/rationale_selection_pilot.py'),Path('src/expert_search/choice_judgments.py'),Path('src/expert_search/fast_retrieval.py'))},
            'scope':'Post-hoc same 40 SciFact development claims; top-one abstract retrieval, fixed lexical rationale, then model verdict. Gold may be incomplete and is withheld from model. Not enterprise expert fit.'}
    (output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    search=FastSearch(corpus,database);verify_evidence_documents(corpus,search)
    prepared=[]
    for case in protocol['cases']:
        qid=str(case['claim_id']);claim=by_claim[qid]
        if queries[qid]['query']!=claim['claim']:raise ValueError('Adapted query mismatch')
        start=time.perf_counter();hits=search.search(claim['claim'],10,'bm25-max');elapsed=time.perf_counter()-start
        if not hits:raise ValueError('No retrieval result')
        ids=[str(hit['id']) for hit in hits];doc=ids[0]
        if doc not in papers:raise ValueError('Unknown retrieved paper')
        p=papers[doc];expected=(p.get('title','')+'\n'+' '.join(p['abstract'])).strip()
        if hits[0]['evidence'][0]['text'].split()!=expected.split():raise ValueError('Adapted document text mismatch beyond whitespace')
        start=time.perf_counter();selected=lexical_order(claim['claim'],p['abstract'])[:3];selection_seconds=time.perf_counter()-start
        groups=claim['evidence'].get(doc,[]);known=bool(groups)
        if known and len({g['label'] for g in groups})!=1:raise ValueError('Mixed document labels')
        prepared.append({'claim_id':case['claim_id'],'doc_id':doc,'top10':ids,'retrieval_seconds':elapsed,'selection_seconds':selection_seconds,
                         'selected':selected,'known_evidence':known,'label':groups[0]['label'].lower() if known else None,
                         'rationale':metrics(selected,groups) if known else None})
    search.db.close()
    def run(row):
        claim=by_claim[str(row['claim_id'])];p=papers[row['doc_id']];start=time.perf_counter();row=dict(row,status='unresolved')
        try:
            result=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),{'claim':claim['claim'],'title':p['title'],'abstract':p['abstract']},[QUESTION])
            row.update(status='valid',predicted=result['labels']['verdict'],decision_seconds=result['seconds'])
        except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        row['attempt_seconds']=time.perf_counter()-start;(output/'attempts'/(str(row['claim_id'])+'.json')).write_text(json.dumps(row,indent=2)+'\n');return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,prepared))
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    known=[r for r in rows if r['known_evidence']];joint=[r for r in known if r['status']=='valid' and r['predicted']==r['label'] and r['rationale']['complete_rationale']]
    report={'planned':len(rows),'top1_annotated':len(known),'top10_annotated':sum(any(doc in by_claim[str(r['claim_id'])]['evidence'] for doc in r['top10']) for r in rows),
            'valid':sum(r['status']=='valid' for r in rows),'known_verdict_correct':sum(r.get('predicted')==r['label'] for r in known),
            'known_complete_rationale':sum(r['rationale']['complete_rationale'] for r in known),'known_joint':len(joint),
            'unannotated_top1':len(rows)-len(known),'prediction_counts':dict(Counter(r.get('predicted','unresolved') for r in rows)),
            'mean_retrieval_seconds':statistics.mean(r['retrieval_seconds'] for r in rows),'mean_selection_seconds':statistics.mean(r['selection_seconds'] for r in rows),'mean_decision_attempt_seconds':statistics.mean(r['attempt_seconds'] for r in rows),
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':frozen['scope']+' Observed joint hits require annotated top-one document, correct verdict and a complete annotated rationale. Unannotated documents are not verified negatives; observed hits are not full precision or official scorer. Retrieval and selection run sequentially locally; model calls use three workers. Initialization, index build, queueing and concurrency are excluded from retrieval timings.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--database',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.archive,a.parent,a.corpus,a.database,a.output)
