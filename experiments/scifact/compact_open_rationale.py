"""Audit short-window errors with annotated SciFact-Open sentence locations."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
import tarfile

from experiments.scifact.compact_open_holdout import CUTS, PINS, digest


PROJECTION = "86fe38b4638787a1696c0d987c89b6ac8c4d618eacd2a0917b9b6cbddef70461"
LOCATIONS = "9e3841d0caacace29d41d2e560fd5c16b9b9ae6640f5f535a97e1604656fc2da"
ARMS = ("full", "compact", "rationale_swap", "rationale_only")


def annotated_locations(archive, rows):
    required = {(row["case"], row["doc"]): row for row in rows}
    locations = {}
    with tarfile.open(archive, "r:gz") as source:
        for line in source.extractfile("data/claims.jsonl"):
            claim = json.loads(line)
            for paper, annotation in claim["evidence"].items():
                key = (claim["id"], paper)
                if key not in required:
                    continue
                row = required[key]
                positions = tuple(sorted(set(annotation["sentences"])))
                if (not positions or min(positions) < 0 or max(positions) >= len(row["sentences"])
                        or annotation["label"].lower() != row["grade"]
                        or claim["claim"] != row["criterion"] or key in locations):
                    raise ValueError("Annotation does not match the frozen projection")
                locations[key] = positions
    if locations.keys() != required.keys():
        raise ValueError("A retained pair has no annotated sentence locations")
    return locations


def swap_rationale(compact, gold):
    """Keep the amount of text fixed in sentences and preserve source order."""
    missing = set(gold) - set(compact)
    removable = tuple(i for i in compact if i not in gold)
    if len(missing) > len(removable):
        return None
    if not missing:
        return compact
    # Among equally small edits, keep the original sentences closest to a rationale sentence.
    drop = max(combinations(removable, len(missing)), key=lambda group: (
        sum(min(abs(i - j) for j in gold) for i in group), group,
    ))
    return tuple(sorted((set(compact) - set(drop)) | missing))


def cells(records):
    return dict(
        pairs=len(records),
        full_support=sum(row["full"] for row in records),
        compact_support=sum(row["compact"] for row in records),
        compact_gained=sum(row["compact"] and not row["full"] for row in records),
        compact_lost=sum(row["full"] and not row["compact"] for row in records),
        rationale_only_support=sum(row["rationale_only"] for row in records),
        same_length_swap_pairs=sum(row["rationale_swap"] is not None for row in records),
        swap_gained=sum(row["rationale_swap"] is True and not row["compact"] for row in records),
        swap_lost=sum(row["rationale_swap"] is False and row["compact"] for row in records),
        recovered_full_only=sum(row["rationale_swap"] is True and row["full"] and not row["compact"]
                                for row in records),
    )


def prepare(archive, projection, output):
    if digest(archive) != PINS["archive"] or digest(projection) != PROJECTION:
        raise ValueError("Changed archives or frozen pair projection")
    rows = [json.loads(line) for line in projection.read_text().splitlines()]
    locations = annotated_locations(archive, rows)
    with output.open("x", encoding="utf-8") as stream:
        json.dump([locations[(r["case"], r["doc"])] for r in rows], stream, separators=(",", ":"))
    print(digest(output), flush=True)


def run(locations, projection, checkpoint, output):
    import torch

    from experiments.followir.core17_reranking import write_json
    from experiments.qasper.paired_reader import load_reader
    from experiments.scifact.criterion_reader import measure
    from experiments.scifact.nli_evidence_windows import choose_windows
    from experiments.scifact.selective_evidence_verifier import support_margin

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh Slurm GPU run")
    if digest(locations) != LOCATIONS or digest(projection) != PROJECTION:
        raise ValueError("Changed sentence locations or frozen pair projection")
    rows = [json.loads(line) for line in projection.read_text().splitlines()]
    labels = json.loads(locations.read_text())
    tokenizer, reader, model = load_reader(checkpoint)
    audits = []
    for i, (row, positions) in enumerate(zip(rows, labels, strict=True)):
        sentences = row["sentences"]
        gold = tuple(positions)
        _, compact = choose_windows(row["criterion"], sentences)
        swapped = swap_rationale(compact, gold)
        available = dict(full=tuple(range(len(sentences))), compact=compact,
                         rationale_swap=swapped, rationale_only=gold)
        same_inputs = {}
        record = dict(grade=row["grade"], provenance=row["provenance"],
                      contained=set(gold) <= set(compact))
        for arm in ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]:
            chosen = available[arm]
            if chosen is None:
                record[arm] = None
                continue
            if chosen not in same_inputs:
                example = dict(criterion=row["criterion"], source=" ".join(sentences[j] for j in chosen))
                scores, _ = measure(tokenizer, reader, [example])
                same_inputs[chosen] = float(support_margin(scores)[0])
            cutoff = CUTS["full"] if arm == "full" else CUTS["two_lexical_windows"]
            record[arm] = same_inputs[chosen] > cutoff
        audits.append(record)
    by_label = {}
    for label in ("support", "contradict"):
        entries = [record for record in audits if record["grade"] == label]
        by_label[label] = {
            name: cells(selected)
            for name, selected in (("overall", entries),
                                   ("rationale_present", [record for record in entries if record["contained"]]),
                                   ("rationale_missing", [record for record in entries if not record["contained"]]))
        }
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), revision=model["revision"],
        cases=len(rows), projection_sha256=PROJECTION, location_sha256=LOCATIONS,
        script_sha256=digest(Path(__file__)),
        thresholds=CUTS, inputs=dict(Counter(record["provenance"] for record in audits)),
        results=by_label,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    for action, method, names in (
        ("prepare", prepare, ("archive", "projection", "output")),
        ("run", run, ("locations", "projection", "checkpoint", "output")),
    ):
        command = subparsers.add_parser(action)
        for name in names:
            command.add_argument(name, type=Path)
        command.set_defaults(method=method)
    parsed = vars(parser.parse_args())
    parsed.pop("method")(**parsed)
