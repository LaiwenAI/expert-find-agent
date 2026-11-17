"""Audit exact query/profile text overlap and its observed ranking sensitivity."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from expert_search.cof_data import unique_records
from expert_search.corpus import records
from expert_search.statistics import paired_interval


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    return ' '.join(value.casefold().split())


def audit(output):
    provenance_path=Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    provenance=json.loads(provenance_path.read_text())
    for path,digest in provenance['input_sha256'].items():
        if sha(path)!=digest: raise ValueError('Changed benchmark input: '+path)
    run=Path('artifacts/cof-single-query-v1')
    summary=json.loads((run/'summary.json').read_text())
    if sha(run/'protocol.json')!=summary['protocol_sha256'] or sha(run/'rows.json')!=summary['rows_sha256']:
        raise ValueError('Changed live retrieval results')
    run_protocol=json.loads((run/'protocol.json').read_text())
    if run_protocol['input_manifest_sha256']!=sha(provenance_path): raise ValueError('Different dataset snapshot')
    root=Path('artifacts/public/cof/SIGIR')
    papers=unique_records([x for _,x in records(root/'SIGIR_papers_test.json')],'paper')
    reviewers=unique_records([x for _,x in records(root/'SIGIR_reviewers_test.json')],'reviewer')
    labels={t:{x['query_id']:x['score'] for _,x in records(root/f'SIGIR_queries_test_{t}.json')} for t in ('soft','hard')}
    qids=sorted(labels['soft']);pids=sorted({p for r in reviewers.values() for p in r['papers']})
    if qids!=provenance['query_ids'] or set(qids)!=set(labels['hard']): raise ValueError('Changed query universe')
    rows={x['query_id']:x for x in json.loads((run/'rows.json').read_text())}
    if set(rows)!=set(qids): raise ValueError('Changed result query universe')
    owners=defaultdict(set)
    for eid,r in reviewers.items():
        for pid in set(r['papers']):owners[pid].add(eid)
    fields={};abstract_matches=[]
    for field in ('title','abstract'):
        lookup=defaultdict(list)
        for pid in pids:
            text=normalize(papers[pid].get(field,''))
            if text:lookup[text].append(pid)
        found={qid:sorted(lookup[normalize(papers[qid].get(field,''))]) for qid in qids if normalize(papers[qid].get(field,'')) in lookup and normalize(papers[qid].get(field,''))}
        fields[field]=dict(queries_with_exact_match=len(found),queries_without_text=sum(not normalize(papers[qid].get(field,'')) for qid in qids),matched_query_ids=sorted(found))
        if field=='abstract':
            for qid,matched in found.items():
                eids=sorted({e for pid in matched for e in owners[pid]})
                abstract_matches.append(dict(query_id=qid,source_ids=matched,owners=[dict(id=eid,labels={t:labels[t][qid][eid] for t in labels},rank={m:(rows[qid]['rankings'][m].index(eid)+1 if eid in rows[qid]['rankings'][m] else None) for m in ('dense','lexical')}) for eid in eids]))
    excluded=set(fields['abstract']['matched_query_ids']);filtered=[rows[q] for q in qids if q not in excluded]
    if not filtered: raise ValueError('No remaining queries')
    metrics={t:{m:{k:sum(r['metrics'][t][m][k] for r in filtered)/len(filtered) for k in ('p@5','ndcg@10')} for m in ('dense','lexical')} for t in labels}
    intervals={t:{k:paired_interval([r['metrics'][t]['dense'][k]-r['metrics'][t]['lexical'][k] for r in filtered]) for k in ('p@5','ndcg@10')} for t in labels}
    report=dict(queries=len(qids),unique_profile_papers=len(pids),query_profile_same_id=len(set(qids)&set(pids)),normalization='Unicode casefold and whitespace collapse; exact field equality, nonempty strings only',fields=fields,abstract_matches=abstract_matches,
        sensitivity=dict(excluded_query_ids=sorted(excluded),remaining_queries=len(filtered),metrics=metrics,dense_minus_lexical=intervals),
        provenance=dict(input_manifest_sha256=sha(provenance_path),retrieval_summary_sha256=sha(run/'summary.json'),script_sha256=sha(__file__)),
        limitation='Post-hoc exclusion is sensitivity analysis, not a sealed evaluation or leakage correction. Only exact normalized titles and abstracts are checked; paraphrases, source reuse, model pretraining contamination and substantive relatedness are not assessed. IDs and original reviewer labels do not establish actual authorship or contribution.')
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);audit(parser.parse_args().output)
