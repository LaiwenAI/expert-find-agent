"""Locate the losses after fixed community answer selection."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from experiments.community import joint_answer_witness as base

THRESHOLDS = (-4, -2, 0, 2, 4)
PANELS = ("stats2024", "stats2025", "mathoverflow_original")


def known_pair(pair, proof, queries):
    return all(d in queries[q]["known"] for q, d in zip(pair, proof, strict=True))


def audit(queries, documents, owners, requests, plans, prefix, *, reader, canonical):
    full = {}
    # Use the same pairs and model context limit. The only change is
    # removing the application-level 1,800-character answer cutoff.
    clipped = base.score_pairs(reader, queries, documents, prefix, full, window=None)
    alternate = {}
    known_items = {(q, d) for q, d in prefix if d in queries[q]["known"]}
    if not canonical:
        # Use a synthetic query key per answer to handle canonical
        # parents that differ across the accepted answers of one query.
        original = {d: {"text": row["question"] + "\n" + row["question_text"]}
                    for d, row in documents.items()}
        alternate_clipped = base.score_pairs(
            reader, original, documents, {(d, d) for _, d in known_items}, alternate
        )
    else:
        alternate_clipped = None

    eligible = {arm: [] for arm in base.ARMS}
    for (pair, ranks, *_), plan in zip(requests, plans, strict=True):
        for arm, docs in plan.items():
            proposals = base.paired_sources(ranks, docs, owners)
            eligible[arm].append((pair, ranks, proposals))

    outcomes = {}
    for name, margins in (("prefix", prefix), ("no_character_cutoff", full)):
        arm_outcomes = {}
        for arm, rows in eligible.items():
            thresholds = {str(t): Counter(known_available=0, no_known_pass=0,
                                          known_pass_other_wins=0, known=0, issued=0,
                                          unresolved=0) for t in THRESHOLDS}
            for pair, ranks, proposals in rows:
                if not proposals:
                    continue

                def weak(proof):
                    return min(margins[q, d] for q, d in zip(pair, proof, strict=True))

                def order(proof):
                    total = sum(margins[q, d] for q, d in zip(pair, proof, strict=True))
                    return -weak(proof), -total, base.byline_join.pair_key(proof, ranks)

                best = min(proposals, key=order)
                best_score = weak(best)
                gold_scores = [weak(proof) for proof in proposals if known_pair(pair, proof, queries)]
                selected_known = known_pair(pair, best, queries)
                for t in THRESHOLDS:
                    counts = thresholds[str(t)]
                    passed = best_score > t
                    counts["issued"] += passed
                    counts["unresolved"] += passed and not selected_known
                    counts["known"] += passed and selected_known
                    if gold_scores:
                        counts["known_available"] += 1
                        known_passes = max(gold_scores) > t
                        counts["no_known_pass"] += not known_passes
                        counts["known_pass_other_wins"] += known_passes and not selected_known
            arm_outcomes[arm] = thresholds
        outcomes[name] = arm_outcomes

    question_contrast = None
    if not canonical:
        question_contrast = Counter()
        for q, d in sorted(known_items):
            key = f"later_{'pass' if prefix[q, d] > 0 else 'fail'}"
            key += f"_parent_{'pass' if alternate[d, d] > 0 else 'fail'}"
            question_contrast[key] += 1
        for key in ("later_pass_parent_pass", "later_pass_parent_fail",
                    "later_fail_parent_pass", "later_fail_parent_fail"):
            question_contrast.setdefault(key, 0)

    return dict(outcomes=outcomes, full_context_clipped=clipped,
                parent_context_clipped=alternate_clipped, known_question_answer_pairs=len(known_items),
                known_question_wording_contrast=question_contrast)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*PANELS, "embedding", "reranker", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on a Slurm GPU")
    reference = json.loads(args.reference.read_bytes())["panels"]
    checkpoint = base.live.verify_embedding(args.embedding)
    embed, reader = base.QwenTorchEmbedding(args.embedding), base.Reranker(args.reranker)
    reports = {}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        for name in PANELS:
            canonical = name == "mathoverflow_original"
            record = {}

            def observe(queries, documents, owners, requests, plans, margins):
                record.update(audit(queries, documents, owners, requests, plans, margins,
                                    reader=reader, canonical=canonical))

            original = base.evaluate(getattr(args, name), canonical, embed, reader,
                                     Path(scratch) / name, args.order_offset, observe=observe)
            if original != reference[name]:
                raise ValueError(f"Previous benchmark did not reproduce for {name}")
            for arm in base.ARMS:
                check = record["outcomes"]["prefix"][arm]["0"]
                for key in ("known_available", "known", "issued", "unresolved"):
                    if check[key] != original["natural"][arm][key]:
                        raise ValueError(f"Prefix outcome does not reproduce for {name}/{arm}")
            record["baseline_context_clipped"] = original["clipped_pairs"]
            reports[name] = record
            print(f"{name}: baseline reproduced and reader analysis completed", flush=True)
    base.write_json(args.output, dict(
        panels=reports, thresholds=THRESHOLDS, order_offset=args.order_offset,
        gpu=torch.cuda.get_device_name(), embedding_checkpoint=checkpoint,
        reranker_checkpoint=reader.provenance["revision"],
        reference_sha256=base.digest(args.reference),
        script_sha256=base.digest(Path(__file__)),
        baseline_script_sha256=base.digest(Path(base.__file__)),
    ))


if __name__ == "__main__":
    main()
