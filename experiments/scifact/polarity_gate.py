"""Verify opposite evidence requirements on human-judged scientific abstracts."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tarfile

import numpy as np

from experiments.followir.core17_reranking import digest, write_json

ARCHIVE = "8f4dc238b2eff422bab5c7918fcbddb6996b29d37cb887942ef6ee1a16fb3415"
IFIR = "0d4b89b098c2361fe208d719c8b869baf78e18de5da93d2a50defc7c88f60494"
WINDOW = 1800
MAX_FALSE = 0.10
LABELS = ("SUPPORT", "CONTRADICT")
TASK = ("Decide whether the document supplies evidence for the requested relation to the claim. "
        "The direction of the evidence is a mandatory requirement.")


def prepare(args):
    if args.output.exists() or digest(args.archive) != ARCHIVE or digest(args.ifir) != IFIR:
        raise ValueError("Require pinned sources and a fresh projection path")
    ifir = {row["_id"]: row for row in json.loads(args.ifir.read_bytes())}
    with tarfile.open(args.archive) as archive:
        claims = [json.loads(line) for line in archive.extractfile("data/claims.jsonl")]
        if not set(ifir) <= {str(row["id"]) for row in claims}:
            raise ValueError("Unexpected IFIR claim inventory")
        for row in claims:
            shared = ifir.get(str(row["id"]))
            if shared is not None and shared["text"].strip() != row["claim"].strip():
                raise ValueError("Claim text did not match")
        wanted = {int(doc) for row in claims for doc in row["evidence"]}
        docs = {}
        for member in ("data/corpus.jsonl", "data/corpus_candidates.jsonl"):
            for line in archive.extractfile(member):
                row = json.loads(line)
                if row["doc_id"] in wanted:
                    docs.setdefault(row["doc_id"], "\n".join([row["title"], *row["abstract"]]))
            if docs.keys() == wanted:
                break
    if docs.keys() != wanted:
        raise ValueError("Missing judged abstract")
    dev_docs = {doc for row in claims if str(row["id"]) in ifir for doc in row["evidence"]}
    rows = {"calibration": [], "test": []}
    excluded = 0
    for row in claims:
        part = "calibration" if str(row["id"]) in ifir else "test"
        for doc, evidence in row["evidence"].items():
            if part == "test" and doc in dev_docs:
                excluded += 1
                continue
            if evidence["label"] not in LABELS:
                raise ValueError("Unexpected annotation")
            rows[part].append(dict(claim_id=row["id"], document=doc, criterion=row["claim"],
                                   source=docs[int(doc)][:WINDOW], label=evidence["label"]))
    write_json(args.output, dict(rows=rows, excluded_shared_doc_pairs=excluded,
                                archive_sha256=ARCHIVE, ifir_sha256=IFIR,
                                window_characters=WINDOW))


def small_scores(reader, rows):
    inputs = [(f"Claim: {row['criterion']}\nRequired relation: {label.lower()} the claim.", row["source"])
              for row in rows for label in LABELS]
    scores, clipped = [], 0
    for i in range(0, len(inputs), 8):
        batch, shortened = reader.score_pairs(inputs[i:i + 8], task=TASK)
        scores.extend(batch)
        clipped += shortened
    return np.asarray(scores).reshape(-1, 2), clipped


def cutoff(scores, gold, limit=MAX_FALSE):
    limits = []
    for j in range(2):
        negatives = sorted(scores[gold != j, j], reverse=True)
        limits.append(float(negatives[int(len(negatives) * limit)]))
    return limits


def tally(scores, gold, limits):
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite stance scores")
    accepts = scores > limits
    i = np.arange(len(gold))
    right, wrong = accepts[i, gold], accepts[i, 1 - gold]
    def side(j):
        which = gold == j
        return dict(total=int(which.sum()), correct=int(right[which].sum()), opposite=int(wrong[which].sum()))
    return dict(support=side(0), contradict=side(1), both=int((right & wrong).sum()),
                neither=int((~right & ~wrong).sum()), only_correct=int((right & ~wrong).sum()),
                only_opposite=int((wrong & ~right).sum()))


def run(args):
    import torch

    from experiments.followir.core17_verifier import Reranker
    from experiments.qasper.paired_reader import load_reader
    from experiments.scifact.criterion_reader import measure

    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and a Slurm GPU")
    data = json.loads(args.projection.read_bytes())
    if data["archive_sha256"] != ARCHIVE or data["ifir_sha256"] != IFIR or data["window_characters"] != WINDOW:
        raise ValueError("Changed frozen sources")
    samples = data["rows"]
    for key in ("claim_id", "document"):
        if {row[key] for row in samples["calibration"]} & {row[key] for row in samples["test"]}:
            raise ValueError(f"Overlapping calibration and test {key}")
    gold = {part: np.array([LABELS.index(row["label"]) for row in rows]) for part, rows in samples.items()}
    verifier = Reranker(args.reranker)
    tokenizer, reader, provenance = load_reader(args.chat)
    order = tuple(samples)
    if args.order_offset:
        order = order[::-1]
    scores = {name: {} for name in ("separate_binary", "contrastive_binary", "ternary_reader")}
    clipped = {}
    for part in order:
        raw, clipped[part] = small_scores(verifier, samples[part])
        direct, _ = measure(tokenizer, reader, samples[part])
        scores["separate_binary"][part] = raw
        scores["contrastive_binary"][part] = raw - raw[:, ::-1]
        scores["ternary_reader"][part] = direct[:, :2] - np.maximum(direct[:, 1::-1], direct[:, 2:])
        print(f"{part}: completed {len(samples[part])} evidence pairs", flush=True)
    reports = {}
    for name, bank in scores.items():
        limits = cutoff(bank["calibration"], gold["calibration"])
        if name != "separate_binary":
            limits = [max(0, value) for value in limits]
        reports[name] = {
            "cutoff": limits,
            "calibrated": {part: tally(bank[part], gold[part], limits) for part in samples},
            "zero": {part: tally(bank[part], gold[part], [0, 0]) for part in samples},
        }
    write_json(args.output, dict(
        reports=reports, binary_clipped=clipped, order_offset=args.order_offset,
        claim_counts={part: len({row["claim_id"] for row in rows}) for part, rows in samples.items()},
        excluded_shared_doc_pairs=data["excluded_shared_doc_pairs"],
        projection_sha256=digest(args.projection), gpu=torch.cuda.get_device_name(),
        reranker_revision=verifier.provenance["revision"], chat_revision=provenance["revision"],
        script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    build, gpu = commands.add_parser("prepare"), commands.add_parser("run")
    for name in ("archive", "ifir", "output"):
        build.add_argument(name, type=Path)
    for name in ("projection", "reranker", "chat", "output"):
        gpu.add_argument(name, type=Path)
    gpu.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    args = cli.parse_args()
    (prepare if args.command == "prepare" else run)(args)
