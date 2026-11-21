"""Prepare and merge a blinded exact-anchor qualification review batch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from experiments.retrieval.enterprise_routed_review import digest, write
from experiments.retrieval.score_enterprise_requirement_role_review import load_forms, read_submission

REVIEWERS = ("a", "b")
MUTABLE = ("role", "rationale", "reviewer_id")


def frozen(packet):
    form = packet / "review.jsonl"
    manifest = json.loads((packet / "manifest.json").read_bytes())
    originals = load_forms(form)
    if digest(form) != manifest["original_form_sha256"] or len(originals) != manifest["cases"]:
        raise ValueError("Changed frozen review")
    return form, manifest, originals


def prepare(packet, output):
    form, master, originals = frozen(packet)
    link_path = packet / "private-method-map.jsonl"
    if digest(link_path) != master["private_mapping_sha256"]:
        raise ValueError("Changed source linkage")
    linkage = load_forms(link_path)
    priorities = load_forms(packet / "private-priorities.jsonl")
    if originals.keys() != linkage.keys() or originals.keys() != priorities.keys():
        raise ValueError("Priority flags, source linkage and review inventory disagree")
    primary = sorted(key for key, row in priorities.items() if len(row["boundary_families"]) >= 2)
    controls = {key for key, row in priorities.items() if not row["boundary_families"]}
    if not primary or len(controls) < len(primary):
        raise ValueError("Need a matched comparison excerpt for every focal excerpt")

    def distance(anchor, control):
        a, b = originals[anchor], originals[control]
        return (linkage[anchor]["request"] != linkage[control]["request"],
                abs(len(a["target"]) - len(b["target"])),
                abs(len(a["full_request"]) - len(b["full_request"])), control)

    pairs = []
    for anchor in primary:
        control = min(controls, key=lambda key: distance(anchor, key))
        controls.remove(control)
        pairs.append((anchor, control))
    selected = {key for pair in pairs for key in pair}
    batch = [originals[key] for key in sorted(
        selected, key=lambda key: sha256(f"scope-first-batch-v1:{key}".encode()).digest())]
    if any(row[field] is not None for row in batch for field in MUTABLE):
        raise ValueError("The source template must be blank")

    output.mkdir(mode=0o700)
    for name in ("review.jsonl", *(f"reviewer-{key}.jsonl" for key in REVIEWERS)):
        write(output / name, batch, lines=True)
    write(output / "manifest.json", dict(cases=len(batch), parent_form_sha256=digest(form),
          original_form_sha256=digest(output / "review.jsonl")))
    write(output / "private-matching.json", {"pairs": pairs})
    same_request = sum(linkage[a]["request"] == linkage[b]["request"] for a, b in pairs)
    write(output / "summary.json", dict(blind_cases=len(batch), focal_cases=len(primary),
          comparison_cases=len(pairs), same_request_pairs=same_request,
          represented_requests=len({linkage[key]["request"] for key in selected}), judgments=0))


def transfer(packet, batch, batch_a, batch_b, output, master_a=None, master_b=None):
    main_form, _, originals = frozen(packet)
    batch_form, batch_manifest, subset = frozen(batch)
    if batch_manifest["parent_form_sha256"] != digest(main_form) or any(
            subset[key] != originals.get(key) for key in subset):
        raise ValueError("Batch is not an exact subset of the original review")

    submissions = {}
    reviewer_ids = {}
    counts = {}
    for suffix, returned, existing in zip(REVIEWERS, (batch_a, batch_b), (master_a, master_b), strict=True):
        reviewer, _ = read_submission(returned, subset)
        present, _ = read_submission(existing, originals)
        if reviewer and present and reviewer != present:
            raise ValueError("The batch and larger packet have different reviewer IDs")
        reviewer_ids[suffix] = reviewer or present
        merged = load_forms(existing) if existing else {key: dict(row) for key, row in originals.items()}
        count = 0
        for key, row in load_forms(returned).items():
            if row["role"] is None:
                continue
            previous = merged[key]
            if previous["role"] is not None and any(row[field] != previous[field] for field in MUTABLE):
                raise ValueError("Conflicting judgments at the same anchor")
            merged[key] = previous | {field: row[field] for field in MUTABLE}
            count += 1
        submissions[suffix] = list(merged.values())
        counts[f"batch_labels_{suffix}"] = count
    if reviewer_ids["a"] is not None and reviewer_ids["a"] == reviewer_ids["b"]:
        raise ValueError("Two distinct reviewer IDs are required")

    output.mkdir(mode=0o700)
    for suffix, merged in submissions.items():
        write(output / f"reviewer-{suffix}.jsonl", merged, lines=True)
    write(output / "summary.json", dict(master_form_sha256=digest(main_form),
          batch_form_sha256=digest(batch_form), **counts))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(required=True)
    prep = commands.add_parser("prepare")
    for key in ("packet", "output"):
        prep.add_argument(key, type=Path)
    prep.set_defaults(action=prepare)
    merge = commands.add_parser("transfer")
    for key in ("packet", "batch", "batch_a", "batch_b", "output"):
        merge.add_argument(key, type=Path)
    for key in ("master-a", "master-b"):
        merge.add_argument(f"--{key}", type=Path)
    merge.set_defaults(action=transfer)
    args = vars(cli.parse_args())
    args.pop("action")(**args)
