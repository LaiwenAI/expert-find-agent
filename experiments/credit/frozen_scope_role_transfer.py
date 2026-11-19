"""Replay legacy role-verification thresholds on modern publisher metadata."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import anchored_clause_rescue as anchored
from experiments.credit import factored_article_role_verification as factored
from experiments.credit import legacy_cascade_transfer as legacy
from experiments.credit import legacy_contribution_text as old_credit
from experiments.credit.plos_role_inventory import declared_roles
from experiments.followir.core17_reranking import digest, write_json

ARMS = ("joint", "scoped_joint", "scoped_role", "scoped_head")
NEIGHBORS = (
    ("Conceptualization", "Methodology", "Investigation", "Validation"),
    ("Data curation", "Formal analysis", "Software", "Visualization"),
    ("Funding acquisition", "Resources", "Project administration", "Supervision"),
    ("Writing – original draft", "Writing – review & editing"),
)
ADJACENT = {role: set(group) - {role} for group in NEIGHBORS for role in group}
LEGACY_JOB = "43101"


def local_authors(xml, doi):
    roots = ET.fromstring(xml).findall("./front/article-meta/contrib-group/contrib[@contrib-type='author']")
    names = []
    for node in roots:
        parts = [" ".join(node.findtext(f"./name/{part}", "").split())
                 for part in ("given-names", "surname")]
        names.append(" ".join(part for part in parts if part))
    return names, declared_roles(xml, doi)


def legacy_names(old, transfer):
    earlier = json.loads((old / "capture.json").read_bytes())
    external = json.loads((transfer / "capture.json").read_bytes())
    groups = [(old / split, data["checksums"]) for split, data in earlier.items()]
    groups.append((transfer, external["checksums"]))
    seen, names = set(), set()
    for directory, mapping in groups:
        for doi, sha in mapping.items():
            seen.add(doi)
            names.update(legacy.byline_names(legacy.checked_xml(directory, doi, sha)))
    return seen, names


def make_cases(old, transfer, modern, manifest):
    records = json.loads(manifest.read_bytes())["sources"]
    seen, previous_authors = legacy_names(old, transfer)
    excluded = Counter()
    cases = []
    clusters = []
    for record in records:
        doi = record["doi"]
        raw = legacy.checked_xml(modern, doi, record["source_sha256"])
        author_names = legacy.byline_names(raw)
        if doi in seen or not previous_authors.isdisjoint(author_names):
            excluded["overlap_legacy_doi_or_byline"] += 1
            continue
        names, roles = local_authors(raw, doi)
        if roles != [tuple(row) for row in record["roles_by_author"]]:
            raise ValueError("Publisher source no longer matches the receipt")
        if len(set(name.casefold() for name in names)) != len(names) or not all(names):
            excluded["ambiguous_byline"] += 1
            continue
        article = []
        for role in sorted(ADJACENT):
            holders = [i for i, row in enumerate(roles) if role in row]
            if len(holders) < 2:
                continue
            choices = [(i, other, foil) for i in holders for other in holders if i != other
                       for foil in (set(roles[i]) & ADJACENT[role])]
            if not choices:
                continue
            i, other, foil = min(choices, key=lambda item: old_credit.stable(doi, role, *item))
            article.append(dict(doi=doi, role=role, author=i, names=names,
                                payloads=((i, role), (i, foil), (other, role)), paper=len(clusters)))
        if article:
            cases.extend(article)
            clusters.append(author_names)
        else:
            excluded["missing_same_role_and_neighbor_role_controls"] += 1
    linked = legacy.clusters(clusters)
    return cases, linked, dict(excluded)


def observe(cases, phrases, reader):
    result = []
    for j, case in enumerate(cases):
        doi, role = case["doi"], case["role"]
        options = phrases[role]
        phrase = min(options, key=lambda text: old_credit.stable(doi, role, text))
        rows = {}
        order = j % len(ARMS)
        for arm in ARMS[order:] + ARMS[:order]:
            start = perf_counter()
            inputs, indices = [], []
            for k, (author, credit) in enumerate(case["payloads"]):
                # Match tree positions, not displayed name similarity.
                if arm != "joint" and author != case["author"]:
                    continue
                name = case["names"][author]
                original = f"Author: {name}. Contribution: {credit}."
                if arm in ("joint", "scoped_joint"):
                    query = f"Author: {case['names'][case['author']]}. Requested activity: {phrase}"
                else:
                    query = phrase
                indices.append(k)
                inputs.append((query, credit if arm == "scoped_head" else original))
            instruction = anchored.PERSONAL if arm in ("joint", "scoped_joint") else factored.ROLE_TASK
            scores, clipped = reader.score_pairs(inputs, task=instruction)
            if clipped or not np.isfinite(scores).all():
                raise ValueError("Invalid reader outputs")
            positions = dict(zip(indices, scores, strict=True))
            rows[arm] = dict(
                scores=[positions.get(k, factored.NO_SCORE) for k in range(3)], pairs=len(inputs),
                ms=(perf_counter() - start) * 1000,
            )
        result.append(rows)
    return result



def role_diagnostic(measured):
    """Describe score separation without claiming an operating point."""
    positives = np.asarray([row["scores"][0] for row in measured])
    foils = np.asarray([row["scores"][1] for row in measured])
    cuts = np.unique(np.concatenate((positives, foils)))
    points = []
    for cut in np.concatenate(([np.nextafter(cuts[0], -np.inf)], cuts)):
        points.append((int((foils > cut).sum()), int((positives > cut).sum())))
    return dict(
        ordered_pairs=int((positives > foils).sum()),
        tied_pairs=int((positives == foils).sum()),
        retrospective_max_positives={
            str(int(fraction * 100)): max(positive for wrong, positive in points
                                          if wrong <= int(fraction * len(foils)))
            for fraction in (0., .05, .10)
        },
    )


def summarize(cases, groups, rows, thresholds):
    clean = {}
    outcomes = {}
    for arm in ARMS:
        measured = [row[arm] for row in rows]
        accepted = [[score != factored.NO_SCORE and score > thresholds[arm] for score in row["scores"]]
                    for row in measured]
        clean[arm] = [int(good and not wrong_role and not wrong_owner)
                      for good, wrong_role, wrong_owner in accepted]
        per_role = defaultdict(Counter)
        for case, (good, wrong_role, wrong_owner) in zip(cases, accepted, strict=True):
            counts = per_role[case["role"]]
            counts.update(cases=1, positive=good, wrong_role=wrong_role, wrong_owner=wrong_owner)
        outcomes[arm] = dict(
            cases=len(cases), accepted_positive=sum(flags[0] for flags in accepted),
            accepted_wrong_role=sum(flags[1] for flags in accepted),
            accepted_wrong_owner=sum(flags[2] for flags in accepted), clean=sum(clean[arm]),
            reader_pairs=sum(row["pairs"] for row in measured),
            verifier_p95_ms=float(np.quantile([row["ms"] for row in measured], .95)),
            same_owner_role_diagnostic=role_diagnostic(measured),
            per_role={role: dict(counts) for role, counts in sorted(per_role.items())},
        )
    for arm in ARMS:
        outcomes[arm]["clean_difference_vs_joint"] = clustered_paired_interval(
            [a - b for a, b in zip(clean[arm], clean["joint"], strict=True)],
            [groups[case["paper"]] for case in cases],
        )
    return outcomes


def run(old, transfer, modern, external_fixture, straightforward, frozen, checkpoint, output):
    import torch
    from experiments.followir.core17_verifier import Reranker

    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID") or output.exists():
        raise ValueError("Require Slurm CUDA and a new output file")
    saved = json.loads(frozen.read_bytes())
    receipts = {"original_capture_sha256": old / "capture.json", "transfer_capture_sha256": transfer / "capture.json"}
    if saved["job"] != LEGACY_JOB or any(saved[k] != digest(p) for k, p in receipts.items()):
        raise ValueError("Legacy comparison inputs and thresholds do not match")
    manifest = external_fixture / "manifest.json"
    cases, groups, excluded = make_cases(old, transfer, modern, manifest)
    banks = {"straightforward": straightforward, "boundary": external_fixture / "boundary-phrases.json"}
    reader = Reranker(checkpoint)
    if reader.provenance["revision"] != saved["checkpoint"] or not cases:
        raise ValueError("Changed reader revision or missing transfer cases")
    thresholds = saved["thresholds"]["section_lexical"]
    panels = {}
    for wording, bank in banks.items():
        phrases = json.loads(bank.read_bytes())
        observe(cases[:3], phrases, reader)
        rows = observe(cases, phrases, reader)
        panels[wording] = summarize(cases, groups, rows, thresholds)

    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), checkpoint=reader.provenance["revision"],
        code_sha256=digest(Path(__file__)), frozen_sha256=digest(frozen), modern_manifest_sha256=digest(manifest),
        wording_hashes={name: digest(path) for name, path in banks.items()}, thresholds=thresholds,
        papers=len(groups), author_clusters=len(set(groups)), roles=sorted({case["role"] for case in cases}),
        excluded=excluded, panels=panels,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("old", "transfer", "modern", "external_fixture", "straightforward", "frozen", "checkpoint", "output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
