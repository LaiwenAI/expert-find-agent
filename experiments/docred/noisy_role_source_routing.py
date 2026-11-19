"""Stress test indexed work routing with missing and ambiguous query aliases."""

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from hashlib import sha256
from heapq import nsmallest
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import global_role_search as search


ARMS = ("flat_four", "exact_then_flat", "fuzzy_only", "guarded_fuzzy", "swap_then_guard")
STYLES = ("observed_alias", "token_drop", "transposition")
# Frozen from the prior exact-anchor comparison's alias-boost arm.
SCORE_CUT = .25
GRAM_CANDIDATES = 48


def grams(alias):
    padded = "^" + " ".join(alias) + "$"
    return {padded[i:i+3] for i in range(len(padded)-2)}


class AliasLookup:
    def __init__(self, aliases):
        self.aliases = aliases
        self.postings = defaultdict(set)
        for alias in aliases:
            for gram in grams(alias):
                self.postings[gram].add(alias)

    def possible(self, query):
        exact = self.aliases.get(query)
        if exact:
            return [(1.,query)], 0
        candidates = Counter()
        for gram in sorted(grams(query)):
            candidates.update(self.postings.get(gram, ()))
        rough = nsmallest(GRAM_CANDIDATES, candidates, key=lambda a:(-candidates[a],a))
        q = " ".join(query)
        scores = [(SequenceMatcher(None,q," ".join(alias),autojunk=False).ratio(),alias)
                  for alias in rough]
        matches = sorted(((-score,alias) for score,alias in scores if score>=.65))
        return [(-score,alias) for score,alias in matches[:2]], len(rough)


def bm25_scores(query, postings):
    words = search.words(search.ROLES[query["role"]])
    scores = defaultdict(float)
    for token in sorted(set(query["work"]+words)):
        for doc,value in postings.get(token,{}).items():
            scores[doc] += value
    return nsmallest(search.LIMIT,scores,key=lambda doc:(-scores[doc],doc))


def ranked_alias_docs(postings, lookup, key, query, length):
    docs = lookup.aliases[key]
    terms=set(query["work"]+search.words(search.ROLES[query["role"]]))
    return nsmallest(length, docs, key=lambda doc:(
        -sum(postings.get(term,{}).get(doc,0.) for term in sorted(terms)),doc,
    ))


def swapped_aliases(query, aliases):
    candidates = set()
    comparisons = 0
    for i, token in enumerate(query):
        if len(token) < 4 or not token.isalpha():
            continue
        for j in range(len(token) - 1):
            if token[j] == token[j+1]:
                continue
            transposed = token[:j] + token[j+1] + token[j] + token[j+2:]
            candidate = query[:i] + (transposed,) + query[i+1:]
            comparisons += 1
            if candidate in aliases:
                candidates.add(candidate)
    return candidates, comparisons


def route(query, postings, lookup, arm):
    work = query["work"]
    if arm in ("flat_four","exact_then_flat"):
        docs, _, _ = search.shortlist(
            query,postings,lookup.aliases,"flat_four" if arm=="flat_four" else "alias_route_four")
        return docs, 0, arm=="flat_four" or work not in lookup.aliases
    alternate_lookups = 0
    if arm=="swap_then_guard" and work not in lookup.aliases:
        matches, alternate_lookups = swapped_aliases(work,lookup.aliases)
        if len(matches)==1:
            key=next(iter(matches))
            if len(lookup.aliases[key])==1:
                return ranked_alias_docs(postings,lookup,key,query,search.LIMIT), alternate_lookups, False
    options, comparisons = lookup.possible(work)
    comparisons += alternate_lookups
    if not options:
        return bm25_scores(query,postings), comparisons, True
    best, alias = options[0]
    if arm=="fuzzy_only":
        return ranked_alias_docs(postings,lookup,alias,query,search.LIMIT), comparisons, False

    exact_rare = best==1. and len(lookup.aliases[alias])==1 and len(work)>1
    if exact_rare:
        return ranked_alias_docs(postings,lookup,alias,query,search.LIMIT), comparisons, False
    if best < .78:
        return bm25_scores(query,postings), comparisons, True

    baseline = bm25_scores(query,postings)
    reserve = baseline[:1]
    priority = ranked_alias_docs(postings,lookup,alias,query,search.LIMIT-1)
    if len(options)>1 and best-options[1][0] <= .07:
        contender = ranked_alias_docs(postings,lookup,options[1][1],query,1)
        priority = priority[:2]+contender
    selected = list(dict.fromkeys(reserve+priority+baseline))[:search.LIMIT]
    return selected, comparisons, True


def stable(*fields):
    return sha256(repr(fields).encode()).digest()


def query_styles(queries, documents):
    views = []
    for source in queries:
        original = source["work"]
        document = documents[source["group"]]
        observed = set()
        for mentions in document["vertexSet"]:
            if search.words(search.canonical(mentions)) != original:
                continue
            observed.update(search.words(m["name"]) for m in mentions)
        alternative = {a for a in observed if a!=original and a and len("".join(a))>=4
                       and a not in {("this","film"),("the","film"),("it",),("they",)}}
        variants = {}
        if alternative:
            variants["observed_alias"] = min(alternative,key=lambda a:stable(original,a))
        if len(original)>1:
            drop = min(range(len(original)),key=lambda i:stable(original,i))
            variants["token_drop"] = original[:drop]+original[drop+1:]
        eligible = [(i,j) for i,word in enumerate(original) if len(word)>=5 and word.isalpha()
                    for j in range(1,len(word)-2) if word[j]!=word[j+1]]
        if eligible:
            i,j = min(eligible,key=lambda p:stable(original,p))
            word = original[i]
            typo = word[:j]+word[j+1]+word[j]+word[j+2:]
            variants["transposition"] = original[:i]+(typo,)+original[i+1:]
        for style, variant in variants.items():
            views.append(dict(source,work=variant,style=style,original=original))
    return views


def evaluate(views, index, reader=None):
    postings,lookup,profiles,context=index
    results=defaultdict(lambda:defaultdict(list))
    for i, query in enumerate(views):
        offset=i%len(ARMS)
        for arm in ARMS[offset:]+ARMS[:offset]:
            start=perf_counter()
            docs, comparisons, backfill = route(query,postings,lookup,arm)
            pairs,_=search.rank_people(query,docs,profiles)
            margins=reader(query,pairs,context) if reader else None
            elapsed=(perf_counter()-start)*1000
            scores=[((d,p) in query["positive"],(d,p) in query["rivals"]) for d,p in pairs]
            item=dict(
                source=any(d in docs for d,_ in query["positive"]), opened=len(docs), compared=comparisons,
                used_bm25=backfill, exact_alias=query["work"] in lookup.aliases,
                proposed_positive=sum(pos for pos,_ in scores), proposed_rival=sum(rival for _,rival in scores),
                proposed_pairs=len(pairs), ms=elapsed,
            )
            if margins is not None:
                accepted=np.asarray(margins)>SCORE_CUT
                item.update(
                    accepted_positive=sum(bool(a) and p for a,(p,_) in zip(accepted,scores,strict=True)),
                    accepted_rival=sum(bool(a) and r for a,(_,r) in zip(accepted,scores,strict=True)),
                    accepted_unjudged=sum(bool(a) and not (p or r) for a,(p,r) in zip(accepted,scores,strict=True)),
                )
            results[query["style"]][arm].append(item)
    return results


def summarize(views, data, gpu):
    report={}
    styles={style:[q["group"] for q in views if q["style"]==style] for style in STYLES}
    fields=("source","opened","compared","used_bm25","exact_alias","proposed_positive","proposed_rival","proposed_pairs")
    if gpu:
        fields+=("accepted_positive","accepted_rival","accepted_unjudged")
    for style,methods in data.items():
        baseline=methods["flat_four"]
        groups=styles[style]
        metrics={}
        for arm,rows in methods.items():
            totals={key:sum(row[key] for row in rows) for key in fields}
            outcome="accepted_positive" if gpu else "proposed_positive"
            delta=[(a[outcome]-b[outcome])/search.TOP for a,b in zip(rows,baseline,strict=True)]
            totals.update(p50_p95_ms=np.quantile([r["ms"] for r in rows],[.5,.95]).tolist(),
                          known_positive_p5_delta_vs_flat=clustered_paired_interval(delta,groups))
            metrics[arm]=totals
        report[style]=dict(queries=len(baseline), reference_documents=len(set(groups)), arms=metrics)
    return report


def run(snapshot,output,gpu_model=None):
    if output.exists():
        raise FileExistsError(output)
    splits={}
    for split,expected in search.HASHES.items():
        path=snapshot/f"{split}_revised.json"
        if search.digest(path)!=expected:
            raise ValueError("Changed Re-DocRED corpus")
        splits[split]=json.loads(path.read_bytes())
    docs=splits["dev"]+splits["test"]
    queries=search.study_queries(docs,len(splits["dev"]))
    views=query_styles(queries,docs)
    aliases,profiles=search.mentions_index(docs)
    lookup=AliasLookup(aliases)
    bm25=search.bm25_index(docs)
    info={}
    context=None
    reader=None
    if gpu_model:
        import torch
        from experiments.docred import oracle_role_evidence as oracle
        from experiments.docred import person_bound_evidence as base
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise ValueError("Require a Slurm CUDA evaluation")
        tokenizer, model, provenance=oracle.load_reader(gpu_model)
        score=oracle.reader_scorer(tokenizer,model)
        info=dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                  checkpoint=provenance["revision"], frozen_numeric_score_cut=SCORE_CUT)
        context=[]
        for doc in docs:
            sentences=[" ".join(s) for s in doc["sents"]]
            context.append((sentences,base.profile(sentences)))

        def reader(query, pairs, context):
            prepared=[]
            work=" ".join(query["work"])
            role=query["role"]
            for doc,person in pairs:
                name=" ".join(person)
                sentences, parsed=context[doc]
                case=dict(work=work,role=role)
                found=base.choose(case,name,parsed,"lexical_two") if len(sentences)>=2 else range(len(sentences))
                question=f"Was {name} credited in the role of {base.ROLES[role]} for {work}?"
                prepared.append((question,"\n".join(sentences[i] for i in found)))
            if not prepared:
                return []
            scores, clipped=score(prepared)
            if clipped or not np.isfinite(scores).all():
                raise ValueError("Invalid paired result")
            return scores

    index=bm25,lookup,profiles,context
    evaluate(views[:8],index,reader)
    output_data=dict(
        code_sha256=search.digest(Path(__file__)),global_search_code_sha256=search.digest(Path(search.__file__)),
        inputs=search.HASHES, documents=len(docs), original_queries=len(queries),
        panels=summarize(views,evaluate(views,index,reader),bool(gpu_model)), **info,
    )
    search.write_json(output,output_data)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--gpu-model",type=Path)
    run(**vars(parser.parse_args()))
