"""Retrieve expert evidence from demand-first research facets, without labels."""
import hashlib
import json
from pathlib import Path
from expert_search.chinese_concept_index import ConceptSearch
from expert_search.fast_retrieval import FastSearch
from expert_search.corpus import records


def run(chinese=False):
    parent=Path('artifacts/demand-first-facets-v2')
    demands={r['id']:r for _,r in records('artifacts/demands/queries.jsonl')}
    if not (parent/'summary.json').exists():raise ValueError('Facet extraction has not completed')
    if {p.stem for p in parent.glob('demand-*.json')}!=set(demands):raise ValueError('Incomplete demand cohort')
    concept=ConceptSearch();lexical=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite')
    out=Path('artifacts/demand-first-pool-v2' if chinese else 'artifacts/demand-first-pool-v1');out.mkdir(exist_ok=True);jobs=[]
    for path in sorted(parent.glob('demand-*.json')):
        row=json.loads(path.read_text())
        if row['status']!='structurally_valid':continue
        for index,facet in enumerate(row['facets']):
            if facet['role']!='research_topic':continue
            candidates=concept.search(facet['text'] if chinese else facet['search_query'],k=2,depth=100)['results']
            ids=[r['id'] for r in candidates]
            evidence=lexical.evidence_for(facet['search_query'],ids,n=3)
            for candidate in candidates:
                eid=candidate['id'];sources=evidence[eid]
                if any(s['expert_id']!=eid for s in sources):raise ValueError('Cross-owner evidence')
                jobs.append(dict(id=f"{row['id']}-facet-{index}-expert-{eid}",parent_demand=row['id'],
                    demand=demands[row['id']]['query'],facet=facet,expert_id=eid,sources={s['id']:s for s in sources},
                    retrieval_score=candidate['score'],facet_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    status='retrieved_unlabeled'))
    protocol=dict(version=2 if chinese else 1,area_query='original Chinese facet text' if chinese else 'generated search query',facet_protocol=json.loads((parent/'protocol.json').read_text()),
        corpus_sha256=hashlib.sha256(Path('artifacts/corpus/evidence.jsonl').read_bytes()).hexdigest(),
        concept_input_sha256=concept.meta['input_sha256'],
        policy='Research-topic-only; top2 frozen area retrieval, top3 owned lexical evidence; no relevance labels.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed pool protocol')
    marker.write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (out/'jobs.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
    report=dict(jobs=len(jobs),demands=len({j['parent_demand'] for j in jobs}),
        source_facets=len(list(parent.glob('demand-*.json'))),
        warning='Research-topic retrieval only; facet roles and expert relevance not independently audited. Future targets and prior qualifications are not used as existing capability filters.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--chinese',action='store_true');run(parser.parse_args().chinese)
