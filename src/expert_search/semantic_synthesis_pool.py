"""Prepare demand-aligned synthesis jobs using frozen research-topic retrieval.

Offline preparation makes no model API calls. Generation/auditing reuses scope-v5;
candidate similarity never supplies a relevance or capability label.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .corpus import records
from .fast_retrieval import FastSearch


def prepare(output='artifacts/synthesis-semantic-v1'):
    from .chinese_concept_index import ConceptSearch
    concept = ConceptSearch()
    lexical = FastSearch('artifacts/corpus','artifacts/fast-search.sqlite')
    fields = {}
    for _,row in records('artifacts/corpus/evidence.jsonl'):
        if row.get('field') in ('研究领域_CN','研究领域_EN'):
            fields.setdefault(row['expert_id'],[]).append(row)
    demands = [r for _,r in records('artifacts/demands/queries.jsonl')]
    jobs,overlap = [],0
    for demand in demands:
        selected = concept.search(demand['title'],k=2,depth=100)['results']
        ids = [r['id'] for r in selected]
        evidence = lexical.evidence_for(demand['query'],ids,n=3)
        old_ids = {r['id'] for r in lexical.search(demand['query'],2,'bm25-max')}
        for rank,candidate in enumerate(selected):
            eid = candidate['id']
            overlap += eid in old_ids
            sources = {e['id']:dict(e,text=e['text'][:4000]) for e in evidence[eid]+fields.get(eid,[])}
            if any(e['expert_id'] != eid for e in sources.values()):
                raise ValueError('Cross-expert source in synthesis job')
            jobs.append(dict(id=demand['id']+'-candidate-semantic-'+str(rank),parent_demand=demand['id'],
                demand=demand['query'],expert_id=eid,sources=sources,protocol='scope-v5',
                selection=dict(method='Frozen BGE Chinese research-area retrieval',query_field='demand title',
                    score=candidate['score'],concepts=candidate['evidence'],
                    also_in_old_bm25_top2=eid in old_ids)))
    snapshots = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        map(Path,['artifacts/corpus/evidence.jsonl','artifacts/demands/queries.jsonl','artifacts/chinese-concepts/vectors.npy'])}
    serialized = ''.join(json.dumps(job,ensure_ascii=False,sort_keys=True)+'\n' for job in jobs)
    protocol = dict(version='semantic-synthesis-pool-v1',generator_protocol='scope-v5',jobs=len(jobs),
        demands=len(demands),old_bm25_candidate_overlap=overlap,snapshots=snapshots,
        concept_input_sha256=concept.meta['input_sha256'],jobs_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        source_policy='Common BM25 top3 from each selected expert plus available CN/EN research-area fields; each source prefix <=4000 characters.',
        limitations='Exploratory change to both candidate retrieval and source policy, not a causal retrieval-only ablation. Model probabilities and retrieval similarity are not labels. Same demand cohort as v5; no held-out generalization claim. All generated/rejected/unresolved outputs retained; only audited narrow subtasks may be silver. No complete industrial-fit negatives.')
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    marker = out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Changed synthesis cohort or sources')
    marker.write_text(json.dumps(protocol,indent=2))
    (out/'jobs.jsonl').write_text(serialized)
    return protocol


def execute(output='artifacts/synthesis-semantic-v1'):
    from .synthesis_v2 import worker
    out = Path(output)
    protocol = json.loads((out/'protocol.json').read_text())
    if hashlib.sha256((out/'jobs.jsonl').read_bytes()).hexdigest() != protocol['jobs_sha256']:
        raise ValueError('Changed prepared jobs')
    for path,digest in protocol['snapshots'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Changed corpus snapshot')
    results = out/'results'
    results.mkdir(exist_ok=True)
    statuses = []
    for _,job in records(out/'jobs.jsonl'):
        status = worker(job,results)
        statuses.append(status)
        print(job['id'],status,flush=True)
    report = dict(jobs=len(statuses),statuses={s:statuses.count(s) for s in sorted(set(statuses))},
                  warning=protocol['limitations'])
    (results/'summary.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    print(json.dumps(execute() if args.execute else prepare(),indent=2))
