"""Identify excerpt-level disagreements without altering blinded reviewer forms."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from experiments.retrieval.enterprise_routed_review import digest, write
from experiments.scifact import requirement_verifier_transfer as transfer


def run(aggregates, projection, models, review_order, private_flags, output):
    import torch

    from experiments.followir.core17_verifier import Reranker
    from experiments.qasper.paired_reader import load_reader
    from experiments.scifact.criterion_verifier_calibration import ENTERPRISE_TASK

    if (any(x.exists() for x in (review_order, private_flags, output)) or not os.environ.get("SLURM_JOB_ID")
            or not torch.cuda.is_available()):
        raise ValueError("Require fresh outputs and Slurm CUDA")
    summary = json.loads(aggregates.read_bytes())
    if summary["script_sha256"] != digest(Path(transfer.__file__)):
        raise ValueError("Changed requirement scoring script")
    views = json.loads(projection.read_bytes())["views"]
    flat = [dict(id=row["id"], criterion=criterion, source=s["text"], criterion_index=i + 1,
                 excerpt_index=j + 1)
            for row in views for i, criterion in enumerate(row["criteria"])
            for j, s in enumerate(row["views"])]
    if len(views) != summary["private_views"] or len(flat) != summary["private_criterion_excerpt_pairs"]:
        raise ValueError("Changed private view inventory")
    ranker = Reranker(models/"qwen3-reranker-0.6b")
    tok, reader, provenance = load_reader(models/"qwen3-4b-instruct-2507")
    if ranker.provenance["revision"] != summary["ranker"] or provenance["revision"] != summary["reader"]:
        raise ValueError("Changed reviewer models")
    scores, _, _ = transfer.measure(flat, ranker, tok, reader, ENTERPRISE_TASK, private=True)
    thresholds = {name: summary["public"][name]["frozen_cutoff"] for name in transfer.PRIVATE_ARMS}
    supported = {arm: [value > thresholds[arm] for value in scores[arm]] for arm in transfer.PRIVATE_ARMS}
    supported["ranker_native_zero"] = [value > 0. for value in scores["ranker_requirement"]]
    for arm, positive in supported.items():
        if sum(positive) != summary["private_summary"][arm]["supported_pairs"]:
            raise ValueError("Replayed aggregate support count changed")
    records = []
    counts, priority = Counter(), Counter()
    for i, unit in enumerate(flat):
        decision = {arm: values[i] for arm, values in supported.items()}
        native = decision["ranker_native_zero"]
        ensemble = decision["reader_agreement"]
        counts.update(ranker_only=int(native and not ensemble), reader_only=int(ensemble and not native),
                      both_support=int(native and ensemble), both_defer=int(not native and not ensemble),
                      direct_mapping_disagreement=int(decision["reader_requirement"] != decision["reader_permuted"]))
        priority[unit["id"]] += int(native != ensemble)
        records.append(dict(case_id=unit["id"], criterion_index=unit["criterion_index"],
                            excerpt_index=unit["excerpt_index"], flags=decision))
    write(private_flags, records)
    # Case order stays separate from the immutable forms, with no verdicts.
    write(review_order, [case for case in sorted(priority, key=lambda key: (-priority[key], key))])
    write(output, dict(job=os.environ["SLURM_JOB_ID"], script_sha256=digest(Path(__file__)),
                       public_summary_sha256=digest(aggregates), pairs=len(flat), review_views=len(views),
                       cases_with_model_disagreement=sum(value > 0 for value in priority.values()),
                       comparisons=dict(counts), human_judgments=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("aggregates","projection","models","review_order","private_flags","output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
