"""Search all Re-DocRED documents for a work-scoped role and named person."""

import argparse
from collections import Counter, defaultdict
from heapq import nsmallest
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json


HASHES = {
    "dev": "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f",
    "test": "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c",
}
ROLES = {
    "P50": "author wrote written writing novel",
    "P57": "director directed direction",
    "P58": "screenwriter screenplay script",
    "P86": "composer composed music score",
    "P112": "founder founded established",
    "P161": "cast actor stars starring played portrays",
    "P162": "producer produced production",
    "P170": "creator created designed",
    "P175": "performer performed recorded sung singer",
    "P178": "developer developed development",
    "P676": "lyricist lyrics words",
}
ARMS = ("flat_four", "alias_boost_four", "alias_route_four")
LIMIT = 4
TOP = 5


def words(text):
    return tuple(tokens(text))


def canonical(mentions):
    return max((m["name"].strip() for m in mentions), key=lambda name: (len(words(name)),len(name),name))


def bm25_index(documents):
    terms = [Counter(token for sentence in doc["sents"] for token in words(" ".join(sentence))) for doc in documents]
    lengths = [sum(row.values()) for row in terms]
    average = sum(lengths) / len(lengths)
    frequency = Counter(token for row in terms for token in row)
    index = defaultdict(dict)
    for i, row in enumerate(terms):
        denominator = 1.2 * (.25 + .75 * lengths[i]/average)
        for token, count in row.items():
            freq = frequency[token]
            idf = math.log1p((len(terms)-freq+.5)/(freq+.5))
            index[token][i] = idf * count * 2.2 / (count + denominator)
    return index


def mentions_index(documents):
    aliases = defaultdict(set)
    profiles = []
    for doc_id, doc in enumerate(documents):
        people, work = [], defaultdict(set)
        for entity in doc["vertexSet"]:
            if entity[0]["type"] == "PER":
                person = words(canonical(entity))
                if person:
                    people.append((person, tuple((m["sent_id"],m["pos"][0],m["pos"][1]) for m in entity)))
                continue
            if entity[0]["type"] in {"TIME","NUM"}:
                continue
            for mention in entity:
                alias = words(mention["name"])
                if alias:
                    aliases[alias].add(doc_id)
                    work[alias].add(mention["sent_id"])
        cue_positions = defaultdict(list)
        for i,sentence in enumerate(doc["sents"]):
            for j,token in enumerate(sentence):
                for term in tokens(token):
                    cue_positions[term].append((i,j))
        profiles.append((people,work,cue_positions))
    return aliases, profiles


def study_queries(documents, first_test):
    # A reference atlas of explicit role instances, built separately from the
    # retrieval structures. A retrieved candidate is keyed by document and name.
    support, competing = defaultdict(set), defaultdict(set)
    requested = {}
    all_roles = defaultdict(lambda: defaultdict(set))
    for doc_id, doc in enumerate(documents):
        names = [words(canonical(entity)) for entity in doc["vertexSet"]]
        kinds = [entity[0]["type"] for entity in doc["vertexSet"]]
        for label in doc["labels"]:
            h, t, role = (label[field] for field in ("h","t","r"))
            if role in ROLES and kinds[t] == "PER" and kinds[h] != "PER" and names[h] and names[t]:
                all_roles[names[h]][role].add((doc_id,names[t]))
                support[names[h],role].add((doc_id,names[t]))
    for work, role_sets in all_roles.items():
        for role, positive in role_sets.items():
            others = set().union(*(values for k,values in role_sets.items() if k != role)) - positive
            by_doc = {d for d,_ in positive} & {d for d,_ in others}
            # A contrast must exist in at least one test document. Other test
            # and distractor documents remain eligible for a positive match.
            test_refs = sorted(d for d in by_doc if d>=first_test)
            if test_refs:
                requested[(work,role)] = test_refs[0]
                competing[(work,role)] = {(d,p) for d,p in others if d in by_doc}
    return [{"work":work,"role":role,"group":group,"positive":support[work,role],
             "rivals":competing[work,role]} for (work,role),group in sorted(requested.items())]


def shortlist(query, postings, aliases, arm):
    work = query["work"]
    cues = words(ROLES[query["role"]])
    exact = aliases.get(work,set())
    scoped = exact if arm == "alias_route_four" and exact else None
    scores = defaultdict(float)
    visited = 0
    for token in sorted(set(work+cues)):
        posting = postings.get(token, {})
        if scoped is not None and len(scoped) < len(posting):
            for d in scoped:
                visited += 1
                scores[d] += posting.get(d, 0.)
        else:
            for d,value in posting.items():
                visited += 1
                if scoped is not None and d not in scoped:
                    continue
                scores[d] += value
    if scoped is not None:
        for d in scoped:
            scores.setdefault(d,0.)
    selected = nsmallest(LIMIT,scores,key=lambda d:(-int(arm!="flat_four" and d in exact),-scores[d],d))
    return selected, visited, len(scores)


def rank_people(query, docs, profiles):
    result = []
    cues = words(ROLES[query["role"]])
    for rank, doc in enumerate(docs):
        people, work, positions = profiles[doc]
        anchored = work.get(query["work"], set())
        found_cues = [where for cue in cues for where in positions.get(cue,())]
        for name, mentions in people:
            affinities=[]
            for sent,start,end in mentions:
                same = [min(abs(start-pos),abs(end-1-pos)) for i,pos in found_cues if i==sent]
                local = 1/(1+min(same)) if same else 0.
                work_distance = min((abs(sent-i) for i in anchored),default=4)
                affinities.append((local+.10)/(1+.30*work_distance))
            score = max(affinities,default=0.) + .20/(1+rank)
            result.append((-score,rank,doc,name))
    top=nsmallest(TOP,result)
    return [(d,name) for _,_,d,name in top], len(result)


def evaluate(queries, postings, aliases, profiles):
    result = {arm:[] for arm in ARMS}
    for i,query in enumerate(queries):
        offset=i%len(ARMS)
        for arm in ARMS[offset:]+ARMS[:offset]:
            started=perf_counter()
            docs,visited,scored=shortlist(query,postings,aliases,arm)
            predictions,people=rank_people(query,docs,profiles)
            ms=(perf_counter()-started)*1000
            positive=query["positive"]
            rivals=query["rivals"]
            result[arm].append(dict(
                documents=len(docs), postings_visited=visited, documents_scored=scored,
                person_occurrences_scored=people, positive_source_recalled=any(doc in docs for doc,_ in positive),
                positives=sum(x in positive for x in predictions), rivals=sum(x in rivals for x in predictions),
                unjudged=sum(x not in positive and x not in rivals for x in predictions),
                any_positive=any(x in positive for x in predictions), ms=ms, predictions=predictions,
            ))
    return result


def run(snapshot, output):
    if output.exists():
        raise FileExistsError(output)
    documents=[]
    for split,expected in HASHES.items():
        path=snapshot/f"{split}_revised.json"
        if digest(path)!=expected:
            raise ValueError("Changed corpus")
        if split=="test":
            first_test=len(documents)
        documents.extend(json.loads(path.read_bytes()))
    if len({doc["title"] for doc in documents})!=len(documents):
        raise ValueError("Unexpected repeated document identity")
    queries=study_queries(documents,first_test)
    postings=bm25_index(documents)
    aliases,profiles=mentions_index(documents)
    evaluate(queries[:10],postings,aliases,profiles)
    trials=[evaluate(queries,postings,aliases,profiles) for _ in range(3)]
    groups=[q["group"] for q in queries]
    result={}
    anchors=[q["work"] in aliases for q in queries]
    for arm in ARMS:
        first=trials[0][arm]
        if any(row["predictions"]!=rep[arm][i]["predictions"] for i,row in enumerate(first) for rep in trials[1:]):
            raise ValueError("Nonrepeatable routing")
        baseline=trials[0]["alias_boost_four"]
        fields=("documents","postings_visited","documents_scored","person_occurrences_scored","positive_source_recalled",
                "positives","rivals","unjudged","any_positive")
        result[arm]={field:sum(row[field] for row in first) for field in fields}
        elapsed=np.median([ [r["ms"] for r in rep[arm]] for rep in trials],axis=0)
        result[arm].update(
            p50_p95_ms=np.quantile(elapsed,[.5,.95]).tolist(),
            positive_p5_delta_vs_boost=clustered_paired_interval(
                [(a["positives"]-b["positives"])/TOP for a,b in zip(first,baseline,strict=True)],groups),
            differing_ranked_results_vs_boost=sum(a["predictions"]!=b["predictions"] for a,b in zip(first,baseline,strict=True)),
            positive_source_recalled_no_exact_alias=sum(not anchor and row["positive_source_recalled"]
                                                        for anchor,row in zip(anchors,first,strict=True)),
        )
    write_json(output,dict(
        inputs=HASHES, code_sha256=digest(Path(__file__)), documents=len(documents), queries=len(queries),
        query_reference_documents=len(set(groups)), query_work_seen_in_alias_dictionary=sum(anchors),
        queries_with_multiple_positive_documents=sum(len({d for d,_ in q["positive"]})>1 for q in queries),
        distinct_positive_pairs=sum(len(q["positive"]) for q in queries), max_documents=LIMIT, top=TOP,
        postings=sum(map(len,postings.values())), arms=result,
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument("output",type=Path)
    run(**vars(parser.parse_args()))
