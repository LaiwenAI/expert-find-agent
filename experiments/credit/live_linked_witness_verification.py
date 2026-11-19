"""Check owner-aware witness dispatch with a live verifier and no role oracle."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.credit import linked_witness_dispatch as replay
from experiments.followir.core17_verifier import Reranker

TASK = ("Judge whether this declaration explicitly supports the specified research contribution. "
        "Adjacent activities and writing about the activity alone are insufficient.")
ARMS = (
    ("scoped", "scoped", 4, 0.),
    ("owner_mean", "owner_mean", 2, 0.),
    ("adaptive", "adaptive", 4, 0.),
    ("adaptive_strict", "adaptive", 4, 1.),
)


def verify(rows, scores, descriptions, phrases, requested, reader, route, budget, threshold):
    scopes, roles = tuple(zip(*requested,strict=True))
    observed = [(side,owner,set()) for side,owner,_ in rows]
    opened = []
    requested_checks = []
    for _ in range(budget):
        index = replay.choose_read(scores,observed,scopes,roles,route,opened)
        if index is None:
            break
        side,owner,_ = observed[index]
        offsets = [i for i, target in enumerate(scopes) if target==side]
        margins, clipped = reader.score_pairs([(phrases[i],descriptions[index]) for i in offsets],task=TASK)
        if clipped or not np.isfinite(margins).all():
            raise ValueError("Unexpected truncated or nonfinite declaration score")
        observed[index] = (side,owner,{roles[i] for i,margin in zip(offsets,margins,strict=True) if margin>=threshold})
        requested_checks.extend((index,roles[i],margin>=threshold) for i,margin in zip(offsets,margins,strict=True))
        opened.append(index)
    return replay.output(observed,opened,requested), opened, requested_checks


def run(projection, glossary, checkpoint, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    if replay.digest(projection)!=replay.PROJECTION or replay.digest(glossary)!=replay.EXAMPLES:
        raise ValueError("Unexpected inputs")
    documents = json.loads(projection.read_bytes())["documents"]
    glossary_rows = json.loads(glossary.read_bytes())
    phrase_inventory = defaultdict(set)
    for row in glossary_rows["examples"]:
        phrase_inventory[row["role"]].add(row["text"])
    items,_ = replay.panel(documents)
    encoder_info = replay.verify_checkpoint(checkpoint)
    encoder = replay.QwenTorchEmbedding(checkpoint,batch_size=24)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    prepared = {}
    for dois in sorted({row[0] for row in items}):
        rows = [(side,row["owner"],set(row["roles"])) for side,doi in enumerate(dois)
                for row in documents[doi]["owners"]]
        texts = [" ".join(glossary_rows["definitions"][role] for role in sorted(roles))
                 for _,_,roles in rows]
        prepared[dois] = rows,texts
    ordered_texts = sorted({text for _,texts in prepared.values() for text in texts})
    stored = dict(zip(ordered_texts,encoder.encode(ordered_texts),strict=True))
    embeddings = {key:np.stack([stored[x] for x in texts]) for key,(_,texts) in prepared.items()}
    counts = defaultdict(Counter)
    timings = defaultdict(list)
    groups = defaultdict(lambda:defaultdict(list))
    for i,(dois,family,variant,requested,truth) in enumerate([items[0],*items]):
        rows,texts = prepared[dois]
        phrases = [min(phrase_inventory[role],key=lambda phrase:replay.stable(family,role,phrase))
                   for _,role in requested]
        start = perf_counter()
        scores = embeddings[dois] @ encoder.encode(phrases,query=True).T
        prefix = (perf_counter()-start)*1000
        shift = i % len(ARMS)
        for arm,route,budget,threshold in ARMS[shift:]+ARMS[:shift]:
            start = perf_counter()
            found,opened,checked = verify(rows,scores,texts,phrases,requested,verifier,route,budget,threshold)
            elapsed = prefix + (perf_counter()-start)*1000
            if not i:
                continue
            bucket = counts[arm]
            bucket["queries"] += 1
            bucket["opened"] += len(opened)
            bucket["checked_atoms"] += len(checked)
            bucket["exact"] += found==truth
            bucket["unsupported_people"] += len(found-truth)
            bucket["queries_with_unsupported"] += bool(found-truth)
            bucket["positive_any"] += bool(found&truth)
            bucket[f"{variant}_queries"] += 1
            bucket[f"{variant}_unsupported"] += bool(found-truth)
            bucket["false_atom_accepts"] += sum(accept and role not in rows[j][2] for j,role,accept in checked)
            bucket["false_atom_rejects"] += sum(not accept and role in rows[j][2] for j,role,accept in checked)
            groups[arm][family].append(found==truth)
            timings[arm].append(elapsed)
    reports = {arm:dict(counts[arm],exact_pairs=sum(all(x) and len(x)==2 for x in groups[arm].values()),
                        p95_ms=float(np.quantile(timings[arm],.95))) for arm,*_ in ARMS}
    replay.write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),script_sha256=replay.digest(Path(__file__)),
        dispatch_sha256=replay.digest(Path(replay.__file__)),projection_sha256=replay.PROJECTION,
        embedding_revision=encoder_info["revision"],reranker_revision=verifier.provenance["revision"],
        queries=len(items),reports=reports,
    ))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection","glossary","checkpoint","reranker","output"):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
