"""Cross-fit text-only evidence gates after author-linked source retrieval."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.retrieval import tokens
from experiments.credit.scoped_credit_search import (
    FIXTURE, STOPS, VIEWS, Search, digest, save, source_rows, stable, whitespace,
)


GATES = ("dense_rank_gate", "absolute_gate", "relative_gate")
ROUTES = ("top4", "expanded", *GATES, *(f"selective_{gate}" for gate in GATES))
VANILLA = ("top4", "expanded")
TASK = "Judge whether the document is the specific paper described by the clue. A related paper is insufficient."
MISSING = ("present", "missing_left", "missing_right")


def graph(documents):
    owners = defaultdict(list)
    for doi, row in documents.items():
        for owner in row["identities"]:
            owners[owner].append(doi)
    edges = sorted({pair for dois in owners.values() for pair in combinations(sorted(dois), 2)})
    components = {doi:doi for pair in edges for doi in pair}

    def find(doi):
        while doi != components[doi]:
            doi = components[doi]
        return doi

    for a,b in edges:
        components[find(a)] = find(b)
    return [(a,b,find(a)) for a,b in edges]


def prepare(fixture_path, frozen, output, sources):
    if digest(fixture_path)!=FIXTURE:
        raise ValueError("Changed scoped fixture")
    fixture=json.loads(fixture_path.read_text())
    document_rows=source_rows(fixture,sources)
    pinned=json.loads(frozen.read_text())
    actual = {doi: {k: v for k, v in row.items() if k != "result"} for doi, row in document_rows.items()}
    # Role tuples and JSON arrays have the same serialized representation.
    if json.loads(json.dumps(actual)) != pinned["documents"]:
        raise ValueError("Unexpected publication corpus")
    corpus_terms=Counter(word for row in document_rows.values() for word in set(tokens(row["index"])))
    pairs=graph(pinned["documents"])
    probes={}
    for doi in sorted({source for a,b,_ in pairs for source in (a,b)}):
        row=document_rows[doi]
        result=row["result"]
        options={word for word in tokens(result) if len(word)>3 and not word.isdigit() and word not in STOPS}
        useful=sorted(options,key=lambda word:(corpus_terms[word]==0,corpus_terms[word],stable(doi,word)))[:8]
        if len(useful)<3 or whitespace(result) in whitespace(row["index"]):
            raise ValueError("Invalid independently compared clue")
        probes[doi]=dict(result=result,keywords=" ".join(useful))
    if any(probes.get(doi)!=views for doi,views in pinned["clues"].items() if doi in probes):
        raise ValueError("Clues differ from the earlier replay")
    save(output,dict(frozen_sha256=digest(frozen),pairs=pairs,clues=probes))
    print(dict(pairs=len(pairs),groups=len({p[2] for p in pairs}),sources=len(probes)),flush=True)


def observation(index, corpus, verifier, view, removed):
    started=perf_counter()
    dense, fused=index.views(view,removed)
    narrowed=index.rank(dense,fused,"linked_top4")
    primary=index.rank(dense,fused,"linked_expand")
    base_ms=(perf_counter()-started)*1000
    if primary is None:
        return dict(top4=narrowed,expanded=primary,base_ms=base_ms,cross_ms=0.,checked_pairs=0)
    ids=[index.ids[x] for x in primary]
    worst_rank=max(order.index(doc)+1 for order,doc in zip(dense,ids,strict=True))
    request=[]
    alternate=[]
    for side, doi in enumerate(primary):
        request.append((view[side],corpus[doi]["index"]))
    for side,doi in enumerate(primary):
        other=index.dois[fused[side][0]]
        if other!=doi:
            alternate.append((side,len(request)))
            request.append((view[side],corpus[other]["index"]))
    started=perf_counter()
    scores,clipped=verifier.score_pairs(request,task=TASK)
    extra_ms=(perf_counter()-started)*1000
    if clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid source-confirmation scores")
    gaps={j:scores[j]-scores[k] for j,k in alternate}
    relative=min([*gaps.values(),*scores[:2]]) if gaps else min(scores[:2])
    return dict(top4=narrowed,expanded=primary,dense_rank_gate=-float(worst_rank),
                absolute_gate=min(scores[:2]),relative_gate=relative,base_ms=base_ms,
                cross_ms=extra_ms,checked_pairs=len(request))


def calibrate(rows, policy):
    selective = policy.startswith("selective_")
    score = policy.removeprefix("selective_")
    eligible = [r for r in rows if score in r and (not selective or r["top4"] is None)]
    false_scores=[r[score] for r in eligible if r["truth"]!=r["expanded"]]
    cutoff=max(false_scores,default=-float("inf"))
    # Use a strict boundary, so ties with any observed false candidate are rejected.
    return cutoff


def accepted(rows, policy, thresholds):
    answers=[]
    for r in rows:
        if policy in VANILLA:
            answers.append(r[policy])
        elif policy.startswith("selective_") and r["top4"] is not None:
            answers.append(r["top4"])
        else:
            value=r.get(policy.removeprefix("selective_"),-float("inf"))
            answers.append(r["expanded"] if value>thresholds[r["component"],policy] else None)
    return answers


def report(rows):
    groups=sorted({r["component"] for r in rows})
    thresholds={(component,policy):calibrate([r for r in rows if r["component"]!=component],policy)
                for component in groups for policy in ROUTES if policy not in VANILLA}
    result={}
    for name, sample in (("all",rows),*((view,[r for r in rows if r["view"]==view]) for view in VIEWS)):
        arms={}
        for route in ROUTES:
            metrics=Counter()
            times=[]
            for row,answer in zip(sample,accepted(sample,route,thresholds),strict=True):
                text_model = route.removeprefix("selective_") in {"absolute_gate","relative_gate"}
                selective = route.startswith("selective_")
                check = text_model and row["expanded"] is not None and (not selective or row["top4"] is None)
                metrics["cross_encoder_calls"] += check
                times.append(row["base_ms"] + (row["cross_ms"] if check else 0.))
                truth=row["truth"]
                if truth is None:
                    metrics["missing"]+=1
                    metrics["missing_false_accepts"]+=answer is not None
                else:
                    metrics["present"]+=1
                    metrics["present_exact"]+=answer==truth
                    metrics["present_wrong_accepts"]+=answer is not None and answer!=truth
                    metrics["present_unresolved"]+=answer is None
            arms[route]=dict(metrics, estimated_factored_p50_p95_ms=np.quantile(times,[.5,.95]).tolist())
        result[name]=arms
    return result


def run(fixture_path, frozen, cases, embed_path, rerank_path, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or digest(fixture_path)!=FIXTURE):
        raise ValueError("Require a new Slurm CUDA run")
    pinned=json.loads(frozen.read_text())
    panel=json.loads(cases.read_text())
    if panel["frozen_sha256"]!=digest(frozen) or panel["pairs"]!=[list(row) for row in graph(pinned["documents"])]:
        raise ValueError("Unexpected source pairs")
    corpus=pinned["documents"]
    encoder_meta=verify_checkpoint(embed_path)
    encoder=QwenTorchEmbedding(embed_path,batch_size=24)
    encoder.model.config.use_cache=False
    verifier=Reranker(rerank_path)
    records=[]
    with TemporaryDirectory() as directory:
        idx=Search(pinned,encoder,Path(directory))
        for offset,(a,b,component) in enumerate(panel["pairs"]):
            for view in VIEWS:
                clues=[panel["clues"][doi][view] for doi in (a,b)]
                for condition in MISSING:
                    removed=() if condition=="present" else ((a,) if condition=="missing_left" else (b,))
                    outcome=observation(idx,corpus,verifier,clues,removed)
                    if offset==0 and view==VIEWS[0] and condition=="present":
                        outcome=observation(idx,corpus,verifier,clues,removed)
                    records.append(outcome | dict(component=component,view=view,
                                                  truth=(a,b) if condition=="present" else None))
            if (offset+1)%5==0:
                print(offset+1,"anchor pairs evaluated",flush=True)
    seen=[r for r in records if r["checked_pairs"]]
    save(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),script_sha256=digest(Path(__file__)),
        frozen_sha256=digest(frozen),panel_sha256=digest(cases),anchor_pairs=len(panel["pairs"]),
        components=len({p[2] for p in panel["pairs"]}),evaluated_queries=len(records),
        expanded_proposals_checked=len(seen),verification_pairs=sum(r["checked_pairs"] for r in records),
        retrieval_p50_p95_ms=np.quantile([r["base_ms"] for r in records],[.5,.95]).tolist(),
        confirmation_p50_p95_ms=np.quantile([r["cross_ms"] for r in seen],[.5,.95]).tolist() if seen else None,
        embedding_revision=encoder_meta["revision"],reranker_revision=verifier.provenance["revision"],results=report(records),
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(required=True)
    prepare_cmd=sub.add_parser("prepare")
    for name in ("fixture_path","frozen","output"):
        prepare_cmd.add_argument(name,type=Path)
    prepare_cmd.add_argument("sources",type=Path,nargs="+")
    prepare_cmd.set_defaults(action=prepare)
    run_cmd=sub.add_parser("run")
    for name in ("fixture_path","frozen","cases","embed_path","rerank_path","output"):
        run_cmd.add_argument(name,type=Path)
    run_cmd.set_defaults(action=run)
    values=vars(parser.parse_args())
    values.pop("action")(**values)
