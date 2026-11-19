"""Diagnose shortlist and final-selection errors with labeled QASPER paragraphs."""

import argparse
from collections import Counter
from hashlib import sha256
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.qasper import certified_reranking as base
from experiments.qasper.adaptive_evidence import paper_inputs
from experiments.qasper.evidence_shortlist import online_scores, paper_indexes, terms
from experiments.qasper.section_relation_graph import section_shortlist


ARMS = ("flat16", "section16", "dense_anchor16", "verified_anchor16", "flat32")
SALT = "qasper-evidence-gap-audit-v1"
HEAD = 8
ANCHORS = 2
GAP_SLOTS = 4


def expand(dense, order, visible, sections, statistics, question, anchors):
    """Add evidence near anchors and rare question terms those anchors omit."""
    read = set(visible)
    counts, frequency = statistics
    uncovered = set(terms(question)).difference(*(counts[i].keys() for i in anchors))
    idf = {word: math.log1p((len(dense)+.5)/(frequency[word]+.5)) for word in uncovered}
    denominator = max(sum(idf.values()), 1e-12)
    neighborhood = {i+offset for i in anchors for offset in (-1, 1)
                    if 0 <= i+offset < len(dense) and sections[i] == sections[i+offset]}
    selected = []
    for _ in range(GAP_SLOTS):
        if not uncovered:
            break
        candidates = [i for i in order[:32] if i not in read and uncovered.intersection(counts[i])]
        if not candidates:
            break
        chosen = max(candidates, key=lambda i: (
            float(dense[i]) + .20*sum(idf[w] for w in sorted(uncovered & counts[i].keys()))/denominator, -i))
        selected.append(chosen)
        read.add(chosen)
        uncovered.difference_update(counts[chosen])
    for i in order:
        if i in neighborhood and i not in read and len(selected) < HEAD:
            selected.append(i)
            read.add(i)
    for i in order:
        if i not in read and len(selected) < HEAD:
            selected.append(i)
            read.add(i)
    return selected


def score(reader, question, context, dense, chosen):
    values = []
    clipped = 0
    for offset in range(0, len(chosen), base.BLOCK):
        ids = chosen[offset:offset+base.BLOCK]
        margins, truncation = reader.score(question, context, ids, task=base.TASK)
        if not all(math.isfinite(margin) for margin in margins):
            raise ValueError("Invalid reranker margin")
        for i, margin in zip(ids, margins, strict=True):
            probability = 1/(1+math.exp(-max(-40., min(40., margin))))
            values.append(((1-base.WEIGHT)*dense[i]+base.WEIGHT*probability, i))
        clipped += truncation
    return values, clipped


def request(encoder, reader, case, arm):
    question, paragraphs, sections, vectors, stats = case
    start = perf_counter()
    dense = online_scores(encoder, question, vectors)
    order = np.argsort(-dense, kind="stable").tolist()
    context = [f"{section}\n{paragraph}" for section, paragraph in zip(sections, paragraphs, strict=True)]
    if arm in {"dense_anchor16", "verified_anchor16"}:
        prefix = order[:HEAD]
        first, clipped = score(reader, question, context, dense, prefix)
        anchors = (prefix[:ANCHORS] if arm == "dense_anchor16" else
                   [i for value, i in sorted(first, key=lambda row: (-row[0], row[1]))[:ANCHORS]])
        new = expand(dense, order, prefix, sections, stats, question, anchors)
        extra, clipped_extra = score(reader, question, context, dense, new)
        values = first + extra
        clipped += clipped_extra
    else:
        chosen = section_shortlist(dense, sections, question) if arm == "section16" else order[:32 if arm == "flat32" else 16]
        values, clipped = score(reader, question, context, dense, chosen)
    selected = [i for _, i in sorted(values, key=lambda row: (-row[0], row[1]))[:base.TOP]]
    total_ms = 1000*(perf_counter()-start)
    return [i for _, i in values], selected, total_ms, clipped


def diagnosis(paragraphs, candidates, output, references):
    feasible = [ref for ref in references if len(ref) <= base.TOP]
    if not references:
        return "unscored"
    if not feasible:
        return "output_capacity"
    shortlist = {paragraphs[i] for i in candidates}
    result = {paragraphs[i] for i in output}
    if any(ref <= result for ref in feasible):
        return "complete"
    if any(ref <= shortlist for ref in feasible):
        return "selection_miss"
    return "shortlist_miss"


def choose(dataset, multi=False):
    def order(key):
        return sha256(f"{SALT}:{key}".encode()).digest()

    selected = []
    for pid, paper in sorted(dataset.items()):
        questions = paper["qas"]
        if multi:
            paragraphs, _ = paper_inputs(paper)
            questions = [qa for qa in questions if (refs := base.evidence_sets(qa, paragraphs))
                         and 2 <= min(map(len, refs)) <= base.TOP]
        if questions:
            selected.append((pid, min(questions, key=lambda qa: order(qa["question_id"]))))
    return selected


def summarize(rows):
    report = {}
    for arm in ARMS:
        current = [row[arm] for row in rows]
        scored = [r for r in current if r["label"] not in {"unscored", "output_capacity"}]
        paired = [(row[arm],row["flat16"]) for row in rows
                  if row[arm]["label"] not in {"unscored", "output_capacity"}]
        report[arm] = dict(
            audit=dict(Counter(x["label"] for x in current)),
            checks=mean(r["checks"] for r in current), clipped=sum(r["clipped"] for r in current),
            latency_ms=np.quantile([r["ms"] for r in current],[.5,.95]).tolist(),
            complete_delta_vs_flat16=paired_interval([
                int(a["label"]=="complete")-int(b["label"]=="complete") for a,b in paired]),
            shortlist_delta_vs_flat16=paired_interval([
                int(a["label"]!="shortlist_miss")-int(b["label"]!="shortlist_miss") for a,b in paired]),
            feasible=len(scored),
        )
    return report


def run(data, embedding, reranker, output, multi=False):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    dataset = base.read_data(data,"test")
    chosen = choose(dataset, multi)
    provenance = base.verify_checkpoint(embedding)
    encoder = base.QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = base.Reranker(reranker)
    indexes, offline = paper_indexes(dataset,chosen,encoder)
    first_pid, first_question = chosen[0]
    for arm in ARMS:
        request(encoder,reader,(first_question["question"],*indexes[first_pid]),arm)
    rows = []
    for offset,(pid,qa) in enumerate(chosen):
        case = (qa["question"],*indexes[pid])
        paragraphs = case[1]
        references = base.evidence_sets(qa,paragraphs)
        row = {}
        order = offset % len(ARMS)
        for arm in ARMS[order:]+ARMS[:order]:
            shortlist, final, elapsed, clipped = request(encoder,reader,case,arm)
            row[arm] = dict(label=diagnosis(paragraphs,shortlist,final,references),
                            ms=elapsed,checks=len(shortlist),clipped=clipped)
        rows.append(row)
        if (offset+1)%40==0:
            print("Scored papers:",offset+1,flush=True)
    base.write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),papers=len(rows),offline=offline,
        panel="multi_evidence" if multi else "all_papers",
        data_sha256=base.ASSETS,script_sha256=base.digest(Path(__file__)),
        split_sha256=sha256("\n".join(qa["question_id"] for _,qa in chosen).encode()).hexdigest(),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        arms=summarize(rows),
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data","embedding","reranker","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--multi", action="store_true")
    run(**vars(parser.parse_args()))
