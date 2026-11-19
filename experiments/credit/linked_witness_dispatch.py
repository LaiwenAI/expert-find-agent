"""Budgeted selection of source-scoped publisher witnesses for one identity."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


PROJECTION = "1b9d9b1fae5397b790be93df66b42d8295105f4a57ade1ca3618e6ee40d2c322"
EXAMPLES = "c107b4e780218e58aec4e2a13f2d7072533962a6a73b82d23e4273b43aafd528"
ARMS = ("independent", "scoped", "owner_mean", "owner_bottleneck", "adaptive")
BUDGETS = (2, 4)


def stable(*parts):
    return sha256(json.dumps(parts, sort_keys=True).encode()).digest()


def gold(side_records, requested):
    common = side_records[0].keys() & side_records[1].keys()
    return {owner for owner in common
            if all(role in side_records[side][owner] for side,role in requested)}


def panel(documents):
    by_owner = defaultdict(list)
    for doi, record in documents.items():
        for row in record["owners"]:
            by_owner[row["owner"]].append(doi)
    links = sorted({pair for dois in by_owner.values() for pair in combinations(sorted(set(dois)), 2)})
    requests = []
    for dois in links:
        records = [{row["owner"]:set(row["roles"]) for row in documents[doi]["owners"]} for doi in dois]
        declared = [set().union(*row.values()) for row in records]
        families = {"scope_swap": {}, "role_edit": {}}
        for owner in records[0].keys() & records[1].keys():
            for duplicated in (0,1):
                other = 1-duplicated
                for first, second in combinations(sorted(records[duplicated][owner]),2):
                    for third in records[other][owner] - {first,second}:
                        before = tuple(sorted(((duplicated,first),(duplicated,second),(other,third))))
                        for moved, stay in ((first,second),(second,first)):
                            after = tuple(sorted(((duplicated,stay),(duplicated,third),(other,moved))))
                            if (all(role in declared[side] for side,role in after)
                                    and not gold(records,after)):
                                key = stable(dois,before,after).hex()
                                families["scope_swap"][key] = (before,after)
                        for replacement in declared[other] - {first,second,third}:
                            after = tuple(sorted(((duplicated,first),(duplicated,second),(other,replacement))))
                            if not gold(records,after):
                                key = stable(dois,before,after).hex()
                                families["role_edit"][key] = (before,after)
        for kind, limit in (("scope_swap",6),("role_edit",4)):
            for key,(before,after) in sorted(families[kind].items())[:limit]:
                for variant, requested in (("present",before),(kind,after)):
                    requests.append((dois,key,variant,requested,gold(records,requested)))
    return requests, len(links)


def choose_read(scores, rows, scopes, required, arm, opened):
    available = [i for i in range(len(rows)) if i not in opened]
    if not available:
        return None
    scoped = np.array([[row[0]==side for side in scopes] for row in rows])
    relevance = np.where(scoped, scores, -np.inf)
    if arm == "independent":
        return max(available, key=lambda i:(float(scores[i].max()),-i))
    if arm == "scoped":
        return max(available, key=lambda i:(float(relevance[i].max()),-i))
    owners = defaultdict(dict)
    for i, (side,owner,_) in enumerate(rows):
        owners[owner][side] = i
    plans = []
    for owner, indices in owners.items():
        if set(indices) != {0,1}:
            continue
        visible = [indices[side] for side in scopes]
        values = [float(relevance[i,j]) for j,i in enumerate(visible)]
        if arm == "adaptive":
            # Decoding a read reveals whether the owned witness can complete its side.
            if any(i in opened and required[j] not in rows[i][2] for j,i in enumerate(visible)):
                continue
            values = [1. if i in opened else value for i,value in zip(visible,values,strict=True)]
        unopened = [i for i in set(visible) if i not in opened]
        if not unopened:
            continue
        value = float(np.mean(values)) if arm == "owner_mean" else min(values)
        next_row = max(unopened,key=lambda i:(float(relevance[i].max()),-i))
        plans.append((value,sum(values),-next_row,next_row))
    return max(plans)[3] if plans else None


def output(rows, opened, requested, ignore_owner=False):
    observed = set(opened)
    if ignore_owner:
        union = defaultdict(set)
        linked = defaultdict(set)
        for side,owner,tags in rows:
            linked[owner].add(side)
        for i in opened:
            side,_,roles = rows[i]
            union[side].update(roles)
        return {owner for owner,sides in linked.items()
                if len(sides)==2 and all(role in union[side] for side,role in requested)}
    values = [{owner:set(roles) for i,(scope,owner,roles) in enumerate(rows)
               if i in observed and scope==side} for side in (0,1)]
    return gold(values,requested)


def request(rows, scores, requested, budget, arm):
    opened = []
    scopes, roles = list(zip(*requested,strict=True))
    for _ in range(budget):
        next_row = choose_read(scores,rows,scopes,roles,arm,opened)
        if next_row is None:
            break
        opened.append(next_row)
    return output(rows,opened,requested), opened


def run(projection, glossary, checkpoint, result):
    if result.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    if digest(projection)!=PROJECTION or digest(glossary)!=EXAMPLES:
        raise ValueError("Unexpected inputs")
    documents = json.loads(projection.read_bytes())["documents"]
    examples = json.loads(glossary.read_bytes())
    phrases = defaultdict(set)
    for row in examples["examples"]:
        phrases[row["role"]].add(row["text"])
    items, linked_pairs = panel(documents)
    encoder_info = verify_checkpoint(checkpoint)
    encoder = QwenTorchEmbedding(checkpoint,batch_size=24)
    encoder.model.config.use_cache = False
    inventory = {}
    for dois in sorted({row[0] for row in items}):
        structure = [(side,row["owner"],set(row["roles"])) for side,doi in enumerate(dois)
                     for row in documents[doi]["owners"]]
        descriptions = [" ".join(examples["definitions"][role] for role in sorted(roles))
                        for _,_,roles in structure]
        inventory[dois] = (structure,descriptions)
    descriptions = sorted({text for _,views in inventory.values() for text in views})
    cache = dict(zip(descriptions,encoder.encode(descriptions),strict=True))
    indexed = {key:(rows,np.stack([cache[text] for text in views])) for key,(rows,views) in inventory.items()}

    totals = defaultdict(Counter)
    times = defaultdict(list)
    pair_correct = defaultdict(lambda:defaultdict(list))
    for i,(dois,family,variant,requested,truth) in enumerate(items):
        rows, embeddings = indexed[dois]
        text = [min(phrases[role],key=lambda phrase:stable(family,role,phrase)) for _,role in requested]
        begin = perf_counter()
        vectors = encoder.encode(text,query=True)
        scores = embeddings @ vectors.T
        prefix = (perf_counter()-begin)*1000
        offset = i % len(ARMS)
        for budget in BUDGETS:
            for arm in ARMS[offset:]+ARMS[:offset]:
                begin = perf_counter()
                found, opened = request(rows,scores,requested,budget,arm)
                elapsed = prefix + (perf_counter()-begin)*1000
                unsafe = output(rows,opened,requested,ignore_owner=True)
                bucket = totals[budget,arm]
                bucket["queries"] += 1
                bucket["exact"] += int(found==truth)
                bucket["unsupported_ids"] += len(found-truth)
                bucket["flattened_unsupported_ids"] += len(unsafe-truth)
                bucket["opened"] += len(opened)
                if variant=="present":
                    bucket["positive_cases"] += 1
                    bucket["positive_any"] += bool(found & truth)
                    bucket["positive_exact"] += found==truth
                else:
                    bucket[f"{variant}_cases"] += 1
                    bucket[f"{variant}_unsafe_cases"] += bool(unsafe)
                times[budget,arm].append(elapsed)
                pair_correct[budget,arm][family].append(found==truth)
    report = {budget:{arm:dict(totals[budget,arm],
                               exact_pairs=sum(all(row) and len(row)==2 for row in pair_correct[budget,arm].values()),
                               p95_ms=float(np.quantile(times[budget,arm],.95)))
                      for arm in ARMS} for budget in BUDGETS}
    write_json(result,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),script_sha256=digest(Path(__file__)),
        projection_sha256=PROJECTION,examples_sha256=EXAMPLES,embedding_revision=encoder_info["revision"],
        article_links=linked_pairs,tested_article_links=len(indexed),requests=len(items),reports=report,
    ))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection","glossary","checkpoint","result"):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
