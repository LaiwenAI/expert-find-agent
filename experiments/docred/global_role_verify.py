"""Check globally retrieved person-source candidates against frozen role evidence."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.docred import global_role_search as search
from experiments.docred import oracle_role_evidence as reader
from experiments.docred import person_bound_evidence as evidence


def search_and_verify(query, arm, index, context, model):
    postings, aliases, profiles = index
    start = perf_counter()
    docs, _, _ = search.shortlist(query, postings, aliases, arm)
    predictions, _ = search.rank_people(query, docs, profiles)
    prepared = []
    work = " ".join(query["work"])
    role = query["role"]
    for doc, person in predictions:
        sentences, parsed = context[doc]
        name = " ".join(person)
        case = dict(work=work, role=role)
        if len(sentences) < 2:
            selected = range(len(sentences))
        else:
            selected = evidence.choose(case, name, parsed, "lexical_two")
        target = f"Was {name} credited in the role of {evidence.ROLES[role]} for {work}?"
        prepared.append((target, "\n".join(sentences[i] for i in selected)))
    if prepared:
        scores, clipped = model(prepared)
        if clipped or not np.isfinite(scores).all():
            raise ValueError("Clipped or invalid proof scores")
    else:
        scores = []
    elapsed = (perf_counter()-start)*1000
    return dict(
        margins=[(score, "positive" if pair in query["positive"] else
                  "competing" if pair in query["rivals"] else "unjudged")
                 for score,pair in zip(scores,predictions,strict=True)],
        source_recalled=any(d in docs for d,_ in query["positive"]),
        evaluated_pairs=len(prepared), ms=elapsed,
    )


def replay(queries, index, context, model):
    rows={name:[] for name in search.ARMS}
    for i, query in enumerate(queries):
        offset=i%len(search.ARMS)
        for name in search.ARMS[offset:]+search.ARMS[:offset]:
            rows[name].append(search_and_verify(query,name,index,context,model))
    return rows


def cutoff(rows):
    competitors=sorted((score for row in rows for score,kind in row["margins"] if kind=="competing"),reverse=True)
    if not competitors:
        raise ValueError("Need development rivals for calibration")
    cap=len(competitors)//20
    return competitors[cap], len(competitors), cap


def summary(rows, threshold):
    kinds=("positive","competing","unjudged")
    return dict(
        proposed={kind:sum(label==kind for row in rows for _,label in row["margins"]) for kind in kinds},
        accepted={kind:sum(label==kind and s>threshold for row in rows for s,label in row["margins"]) for kind in kinds},
        requests_with_confirmed_pair=sum(any(s>threshold and k=="positive" for s,k in row["margins"]) for row in rows),
        proposed_pair_count=sum(row["evaluated_pairs"] for row in rows),
        p50_p95_ms=np.quantile([row["ms"] for row in rows],[.50,.95]).tolist(),
    )


def run(snapshot, model, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new GPU evaluation")
    splits={}
    for split,expected in search.HASHES.items():
        path=snapshot/f"{split}_revised.json"
        if search.digest(path)!=expected:
            raise ValueError("Changed Re-DocRED corpus")
        splits[split]=json.loads(path.read_bytes())
    documents=splits["dev"]+splits["test"]
    if len({row["title"] for row in documents})!=len(documents):
        raise ValueError("Overlapping source identities")
    test=search.study_queries(documents,len(splits["dev"]))
    test_works={row["work"] for row in test}
    dev=[q for q in search.study_queries(splits["dev"],0) if q["work"] not in test_works]
    aliases,profiles=search.mentions_index(documents)
    index=(search.bm25_index(documents),aliases,profiles)
    context=[]
    for doc in documents:
        sentences=[" ".join(s) for s in doc["sents"]]
        context.append((sentences,evidence.profile(sentences)))
    tokenizer,checkpoint,provenance=reader.load_reader(model)
    score=reader.reader_scorer(tokenizer,checkpoint)
    replay(dev[:2],index,context,score)
    fitted=replay(dev,index,context,score)
    cuts={method:cutoff(rows) for method,rows in fitted.items()}
    held=replay(test,index,context,score)
    metrics={}
    baseline=held["alias_boost_four"]
    base_cut=cuts["alias_boost_four"][0]
    common_cap=min(values[2] for values in cuts.values())
    groups=[row["group"] for row in test]
    for method in search.ARMS:
        threshold,n,cap=cuts[method]
        reference=summary(fitted[method],threshold)
        if reference["accepted"]["competing"]>cap:
            raise ValueError("Calibration budget exceeded")
        observed=summary(held[method],threshold)
        candidates=sorted((s for row in fitted[method] for s,k in row["margins"] if k=="competing"),reverse=True)
        absolute_cutoff=candidates[common_cap]
        delta=[]
        for a,b in zip(held[method],baseline,strict=True):
            count=lambda row,t:sum(k=="positive" and s>t for s,k in row["margins"])
            delta.append((count(a,threshold)-count(b,base_cut))/search.TOP)
        metrics[method]=dict(
            threshold=threshold, development_competing=n, development_cap=cap, development=reference,test=observed,
            shared_numeric_cutoff_test=summary(held[method],base_cut),
            matched_absolute_acceptance_cap=dict(cap=common_cap,threshold=absolute_cutoff,
                                                  development=summary(fitted[method],absolute_cutoff),
                                                  test=summary(held[method],absolute_cutoff)),
            accepted_positive_p5_delta_vs_boost=clustered_paired_interval(delta,groups),
        )
    search.write_json(output,dict(
        gpu=torch.cuda.get_device_name(),job=os.environ["SLURM_JOB_ID"],checkpoint=provenance["revision"],
        code_sha256=search.digest(Path(__file__)), retrieval_code_sha256=search.digest(Path(search.__file__)),
        input_hashes=search.HASHES, test=len(test), development=len(dev), arms=metrics,
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot","model","output"):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
