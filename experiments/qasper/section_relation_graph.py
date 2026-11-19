"""Evaluate hierarchical shortlist expansion and learned paragraph relations on QASPER."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.qasper import adaptive_reranking_depth as depth
from experiments.qasper import certified_reranking as base
from experiments.qasper import evidence_sets as earlier
from experiments.qasper.evidence_shortlist import online_scores, paper_indexes, terms

WIDTH = 24
SALT = "qasper-section-relation-graph-v1"
WEIGHTS = (0., .03, .06, .12, .24)
ROUTES = ("flat16", "hierarchy16", "flat32")
PICK = ("point", "node", "graph", "forced_graph")


def section_shortlist(dense, sections, question):
    order = np.argsort(-dense, kind="stable")
    if len(order) <= 16:
        return order.tolist()
    selected = order[:8].tolist()
    chosen = set(selected)
    section_scores = {}
    query = set(terms(question))
    for i in order[:32]:
        section = sections[i]
        if section not in section_scores:
            heading = set(terms(section))
            section_scores[section] = float(dense[i]) + 0.02 * len(query & heading) / max(1, len(heading))
    counts = Counter(sections[i] for i in selected)
    for _ in range(8):
        candidate = max((int(i) for i in order[:32] if i not in chosen), key=lambda i: (
            .65 * float(dense[i]) + .35 * section_scores[sections[i]] - .015 * counts[sections[i]], -i))
        selected.append(candidate)
        chosen.add(candidate)
        counts[sections[candidate]] += 1
    return [int(i) for i in order[:32] if i in chosen]


def observe(case, encoder, reader, route):
    question, paragraphs, sections, vectors, statistics = case
    start = perf_counter()
    dense = online_scores(encoder, question, vectors)
    order = np.argsort(-dense, kind="stable")
    if route == "hierarchy16":
        chosen = section_shortlist(dense, sections, question)
    else:
        chosen = order[:32 if route == "flat32" else 16].tolist()
    margins = []
    context = [f"{sections[i]}\n{paragraphs[i]}" for i in range(len(paragraphs))]
    for j in range(0, len(chosen), base.BLOCK):
        values, _ = reader.score(question, context, chosen[j:j+base.BLOCK], task=base.TASK)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite source judgment")
        margins.extend(values)
    probability = 1/(1+np.exp(-np.clip(margins,-40,40)))
    return dict(selected=chosen, dense=dense[chosen], probability=probability,
                point=(1-base.WEIGHT)*dense[chosen]+base.WEIGHT*probability,
                ms=(perf_counter()-start)*1000)


def features(case, observed):
    question, paragraphs, sections, vectors, statistics = case
    chosen = observed["selected"]
    terms_q = set(terms(question))
    source_counts = statistics[0]
    words = [set(source_counts[i]) for i in chosen]
    coverage = [word & terms_q for word in words]
    lengths = np.array([len(paragraphs[i]) for i in chosen])
    heading = [len(terms_q & set(terms(sections[i]))) for i in chosen]
    local = np.array([len(item)/max(1,len(terms_q)) for item in coverage])
    node = np.column_stack((observed["dense"], observed["probability"], observed["point"],
                           np.log1p(lengths), heading, local))
    ij = np.triu_indices(len(chosen), 1)
    left, right = ij
    embed = vectors[chosen]
    section = [sections[i] for i in chosen]
    extra = np.array([(len(a | b) - max(len(a), len(b))) / max(1, len(terms_q))
                      for a,b in ( (coverage[i],coverage[j]) for i,j in zip(left,right))])
    lexical_link = np.array([len(words[i] & words[j]) / max(1,len(words[i] | words[j]))
                             for i,j in zip(left,right)])
    pair = np.column_stack((np.minimum(node[left],node[right]), np.maximum(node[left],node[right]),
                            (embed[left]*embed[right]).sum(1),
                            np.log1p(np.abs(np.array(chosen)[left]-np.array(chosen)[right])),
                            np.array([section[i]==section[j] for i,j in zip(left,right)]), extra, lexical_link))
    return node, pair, ij


def ridge(x, y, weights=None):
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    data=(x-center)/scale
    if weights is None:
        weights=np.ones(len(y))
    intercept=float(np.average(y,weights=weights))
    coef=np.linalg.solve(data.T @ (weights[:,None]*data) + 32*np.eye(data.shape[1]),
                        data.T @ (weights*(y-intercept)))
    return center,scale,coef,intercept


def infer(matrix, model):
    center,scale,coef,intercept=model
    return (matrix-center)/scale @ coef + intercept


def training(rows):
    nodes, pairs, node_y, pair_y = [], [], [], []
    for case, observed, references in rows:
        if not references:
            continue
        node,pair,(left,right)=features(case,observed)
        text=[case[1][i] for i in observed["selected"]]
        membership=np.array([[line in reference for line in text] for reference in references], dtype=bool)
        nodes.append(node)
        pairs.append(pair)
        node_y.extend(membership.any(0))
        pair_y.extend((membership[:,left] & membership[:,right]).any(0))
    node_x, pair_x = np.concatenate(nodes),np.concatenate(pairs)
    node_y, pair_y = np.array(node_y),np.array(pair_y)
    return (ridge(node_x,node_y),ridge(pair_x,pair_y,np.where(pair_y,6.,1.))), len(node_y), len(pair_y), int(pair_y.sum())


def build_state(case, observed, model):
    node,pair,indices=features(case,observed)
    relevance=infer(node, model[0])
    relation=np.zeros((len(node),len(node)))
    relation[indices]=infer(pair,model[1])
    relation+=relation.T.copy()
    return relevance,relation


def optimize(point, relevance, relation, node_weight, edge_weight):
    priority=point + node_weight*relevance
    n=len(priority)
    count=min(base.TOP,n)
    if edge_weight==0 or n==count:
        return np.argsort(-priority,kind="stable")[:count].tolist()
    beam=[(0.,())]
    for depth in range(count):
        previous = [p for _,p in beam]
        generated = set(tuple(sorted(p+(i,))) for p in previous for i in range(n) if i not in p)
        states = np.asarray(sorted(generated), dtype=np.int64)
        values = priority[states].sum(1)
        if depth:
            a,b=np.triu_indices(depth+1,1)
            values += edge_weight*relation[states[:,a],states[:,b]].sum(1)
        ranking=np.argsort(-values,kind="stable")[:WIDTH]
        beam=[(float(values[i]),tuple(states[i])) for i in ranking]
    return sorted(beam[0][1],key=lambda i:(-priority[i],i))


def complete(case, observed, positions, gold):
    text=[case[1][observed["selected"][i]] for i in positions]
    return base.quality(text,gold)


def calibrate(rows, model):
    settings={}
    for route, batch in rows.items():
        points=[(case,obs,gold,build_state(case,obs,model)) for case,obs,gold in batch if gold]
        def choose(grid, node_weight=None):
            result=[]
            for w in grid:
                n,e=(w,0.) if node_weight is None else (node_weight,w)
                value=[complete(case,obs,optimize(obs["point"],*state,n,e),gold)["complete_evidence"]
                       for case,obs,gold,state in points]
                result.append((sum(value),w))
            return max(result,key=lambda row:(row[0],-row[1]))[1]
        node=choose(WEIGHTS)
        settings[route]=dict(node=node,edge=choose(WEIGHTS,node),forced_edge=choose(WEIGHTS[1:],node))
    return settings


def dataset_cases(dataset, selection, encoder):
    indexes, offline=paper_indexes(dataset,selection,encoder)
    output=[]
    for pid,qa in selection:
        paragraphs,sections,vectors,stats=indexes[pid]
        output.append((pid,(qa["question"],paragraphs,sections,vectors,stats),base.evidence_sets(qa,paragraphs)))
    return output,offline


def test_rows(cases, encoder, reader, model, settings):
    output=[]
    for i,(pid,case,gold) in enumerate(cases):
        row={}
        offset=i%len(ROUTES)
        for route in ROUTES[offset:]+ROUTES[:offset]:
            obs=observe(case,encoder,reader,route)
            begun=perf_counter()
            state=build_state(case,obs,model)
            state_ms=(perf_counter()-begun)*1000
            setting=settings[route]
            arms={}
            for arm,n,e in (("point",0.,0.),("node",setting["node"],0.),
                            ("graph",setting["node"],setting["edge"]),
                            ("forced_graph",setting["node"],setting["forced_edge"])):
                started=perf_counter()
                positions=optimize(obs["point"],*state,n,e)
                ms=obs["ms"] + (perf_counter()-started)*1000+(state_ms if n or e else 0)
                arms[arm]=dict(quality=complete(case,obs,positions,gold),ms=ms)
            row[route]=dict(arms=arms,reads=len(obs["selected"]),
                shortlist=base.quality([case[1][k] for k in obs["selected"]],gold))
        output.append((pid,row))
        if (i+1)%32==0:
            print(f"Evaluated {i+1} requests",flush=True)
    return output


def summarize(rows):
    report={}
    for route in ROUTES:
        scored=[(p,r[route]) for p,r in rows if r[route]["shortlist"] is not None]
        report[route]=dict(
            queries=len(rows),evaluable=len(scored),
            shortlist_complete=sum(x["shortlist"]["complete_evidence"] for _,x in scored),
            mean_reads=mean(r[route]["reads"] for _,r in rows),
            arms={arm:dict(
                complete=sum(x["arms"][arm]["quality"]["complete_evidence"] for _,x in scored),
                f1=mean(x["arms"][arm]["quality"]["f1"] for _,x in scored),
                p95_ms=float(np.quantile([r[route]["arms"][arm]["ms"] for _,r in rows],.95)),
                delta_vs_point=clustered_paired_interval([
                    x["arms"][arm]["quality"]["complete_evidence"]-x["arms"]["point"]["quality"]["complete_evidence"]
                    for _,x in scored],[p for p,x in scored]),
            ) for arm in PICK})
    return report


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh sink and Slurm CUDA")
    dev,test=(base.read_data(args.data,split) for split in ("dev","test"))
    if set(dev)&set(test):
        raise ValueError("Official paper splits overlap")
    provenance=base.verify_checkpoint(args.embedding)
    encoder=base.QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache=False
    reader=base.Reranker(args.reranker)
    cases,offline=dataset_cases(dev,depth.selected_dev(dev),encoder)
    fit_rows=[]; calibration={route:[] for route in ROUTES}
    for i,(pid,case,gold) in enumerate(cases):
        if depth.partition(pid)=="fit":
            fit_rows.append((case,observe(case,encoder,reader,"flat32"),gold))
        else:
            for route in ROUTES:
                calibration[route].append((case,observe(case,encoder,reader,route),gold))
        if (i+1)%64==0:
            print(f"Prepared {i+1} development requests",flush=True)
    model,nodes,pairs,positives=training(fit_rows)
    settings=calibrate(calibration,model)
    print("Frozen settings:",settings,flush=True)
    panels=earlier.test_panels(test)
    known={pid for entries in panels.values() for pid,_ in entries} | {pid for pid,_ in base.panel(test)}
    def order(k):
        return sha256(f"{SALT}:{k}".encode()).digest()
    fresh=[(pid,min(test[pid]["qas"],key=lambda qa:order(qa["question_id"]))) for pid in sorted(set(test)-known,key=order)]
    report={}
    for name,panel in (("additional_reused",panels["additional"]),("remaining_papers",fresh)):
        selected,times=dataset_cases(test,panel,encoder)
        report[name]=dict(offline=times, report=summarize(test_rows(selected,encoder,reader,model,settings)))
    base.write_json(args.output,dict(job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
        script_sha256=base.digest(Path(__file__)), data_sha256=base.ASSETS,
        embedding_revision=provenance["revision"],reranker_revision=reader.provenance["revision"],
        settings=settings,fit_nodes=nodes,fit_edges=pairs,positive_edges=positives,development_offline=offline,panels=report))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data","embedding","reranker","output"):
        parser.add_argument(name,type=Path)
    run(parser.parse_args())
