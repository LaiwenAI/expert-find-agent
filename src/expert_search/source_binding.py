"""Exact source-record binding diagnostics, not expert ability or semantic entailment."""
import hashlib
import itertools
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from .corpus import records


def canonical_title(title):
    return ' '.join(title.split()).casefold()


def record_key(title):
    return hashlib.sha256(canonical_title(title).encode()).hexdigest()


def binding_witness(facet_records, mode):
    """Return source keys, or None, under explicitly lexical record semantics.

    distinct_records is ordinary bipartite matching (not a novel algorithm).
    Duplicate titles must already have been mapped to a common source key.
    """
    if mode not in ('same_record', 'distinct_records'):
        raise ValueError('Unknown binding mode')
    if not facet_records or any(not isinstance(s, (set, frozenset)) for s in facet_records):
        raise ValueError('Require nonempty list of source-key sets')
    if mode == 'same_record':
        shared = set.intersection(*map(set, facet_records))
        return [min(shared)] * len(facet_records) if shared else None
    assignment = {}

    def augment(facet, visited):
        for source in sorted(facet_records[facet]):
            if source in visited:
                continue
            visited.add(source)
            if source not in assignment or augment(assignment[source], visited):
                assignment[source] = facet
                return True
        return False

    for facet in sorted(range(len(facet_records)), key=lambda i: (len(facet_records[i]), i)):
        if not augment(facet, set()):
            return None
    result = [None] * len(facet_records)
    for source, facet in assignment.items():
        result[facet] = source
    return result


def phrase_index(papers):
    postings = defaultdict(lambda: defaultdict(set))
    titles = {}
    edges = defaultdict(list)
    for paper in papers:
        title = paper.get('title') or ''
        if not title.strip():
            continue
        key = record_key(title)
        titles.setdefault(key, title)
        edges[paper['expert_id'], key].append(paper['id'])
        # Literal Chinese phrases only: no translation or inferred synonym.
        phrases = set()
        for run in re.findall(r'[\u4e00-\u9fff]+', title):
            for length in range(3, min(8, len(run)) + 1):
                phrases.update(run[i:i+length] for i in range(len(run)-length+1))
        for phrase in phrases:
            postings[phrase][paper['expert_id']].add(key)
    return postings, titles, edges


def candidate_families(demand, postings, limit=20, screen=True):
    eligible = {phrase: by_expert for phrase, by_expert in postings.items()
                if 2 <= len(by_expert) <= 150 and phrase in demand['query']}
    # Longest matched substrings reduce redundant nested phrases; this is NOT
    # a technical-term or requirement-span validator.
    maximal = sorted(p for p in eligible if not any(p != other and p in other for other in eligible))
    # Mining without this gate produces fragments such as “技术的”/“与应用”.
    # Passing is only a syntactic screen, never proof of technical relevance.
    functional = set('的与于及和为在对是其该本了中以按从由将所等')
    generic = {'影响因素', '系统设计', '研究方法', '研究结果', '发展趋势'}
    if screen:
        maximal = [p for p in maximal if len(p) >= 4 and not any(c in functional for c in p) and p not in generic]
    pairs = list(itertools.combinations(maximal, 2))
    random.Random(42).shuffle(pairs)
    accepted = []
    for first, second in pairs:
        if first in second or second in first:
            continue
        common = set(eligible[first]) & set(eligible[second])
        same, distinct, joint_only, split_only = {}, {}, [], []
        for expert in sorted(common):
            sources = [eligible[first][expert], eligible[second][expert]]
            joint = binding_witness(sources, 'same_record')
            split = binding_witness(sources, 'distinct_records')
            if joint is not None:
                same[expert] = joint
            if split is not None:
                distinct[expert] = split
            if joint is not None and split is None:
                joint_only.append(expert)
            if split is not None and joint is None:
                split_only.append(expert)
        # Both directions of the scope contrast must have an actual witness.
        if joint_only and split_only:
            accepted.append(dict(facets=[first, second], same_record=same, distinct_records=distinct,
                                 joint_only=joint_only, split_only=split_only))
            if len(accepted) == limit:
                break
    return accepted


def build(output='artifacts/source-binding', per_demand=20):
    if type(per_demand) != int or per_demand < 1:
        raise ValueError('Positive family limit required')
    corpus = Path('artifacts/corpus/evidence.jsonl')
    demand_file = Path('artifacts/demands/queries.jsonl')
    papers = [r for _, r in records(corpus) if r['kind'] == 'paper']
    experts = [r['id'] for _, r in records('artifacts/corpus/experts.jsonl')]
    postings, titles, edges = phrase_index(papers)
    rows,raw_candidates = [],[]
    for _, demand in records(demand_file):
        for family in candidate_families(demand,postings,per_demand,screen=False):
            raw_candidates.append(dict(parent_demand=demand['id'],facets=family['facets'],
                                       status='unvalidated raw substring family; not accepted data'))
        for family in candidate_families(demand, postings, per_demand):
            facets = family['facets']
            fid = hashlib.sha256(json.dumps([demand['id'], facets], ensure_ascii=False).encode()).hexdigest()[:20]
            demand_spans = [dict(start=demand['query'].index(f), end=demand['query'].index(f)+len(f), quote=f) for f in facets]
            queries = []
            for mode in ('same_record', 'distinct_records'):
                rule = ('至少一条题名记录同时逐字包含' if mode == 'same_record' else
                        '至少两条不同题名记录，分别逐字包含以下各项（同一条记录不能重复使用）')
                query = '检索当前资料库中的专家：'+rule+'“'+facets[0]+'”和“'+facets[1]+'”。仅判断题名记录，不推断解决企业问题的能力；规范化后相同题名算同一记录。'
                certificates = {}
                for expert, witness in family[mode].items():
                    certificates[expert] = [dict(record_key=key, evidence_ids=sorted(edges[expert, key]),
                        title_sha256=hashlib.sha256(titles[key].encode()).hexdigest(),
                        start=titles[key].index(facet), end=titles[key].index(facet)+len(facet), quote=facet)
                        for key, facet in zip(witness, facets)]
                queries.append(dict(id=fid+'-'+mode, mode=mode, query=query,
                                    relevant_experts=sorted(certificates), certificates=certificates))
            rows.append(dict(id=fid, parent_demand=demand['id'], demand_source=demand['source'],
                             facets=facets, demand_spans=demand_spans, queries=queries,
                             joint_only=family['joint_only'], split_only=family['split_only']))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'families.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (out/'raw-candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw_candidates))
    manifest = dict(protocol='literal-title-binding-v2', raw_candidate_families=len(raw_candidates),families=len(rows), queries=len(rows)*2,
        parent_demands=len({r['parent_demand'] for r in rows}), experts=len(experts),
        per_demand_limit=per_demand, seed=42,
        snapshots={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (corpus, demand_file)},
        source_units=len(titles),
        label_semantics='Exhaustive lexical property of current normalized title records; absent matches are false ONLY for this database predicate, never proof of lack of expertise.',
        limitations=['Diagnostic only, not complete industrial relevance or semantic gold.',
            'Character substrings have only a functional-word screen, not validated technical terms or requirement clauses.',
            'Selection guarantees scope contrasts and creates a biased distribution.',
            'Exact-title normalization defines records, not actual publication identity.',
            'Families share demand, phrases and experts; unsplit pilot, not independent test data.',
            'Symbolic matching is a required oracle baseline and a standard database operation.',
            'Source references are local; no data redistribution license or publication asserted.'])
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
