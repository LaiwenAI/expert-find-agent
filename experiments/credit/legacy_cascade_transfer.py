"""Evaluate the frozen note cascade on a newly captured publisher cohort."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
from xml.etree import ElementTree as ET

from experiments.credit import legacy_contribution_text as legacy


REFERENCE = "35b5fdc101f2cc6207366ed343e610f7898d1252f5d38db2d1a9972ce315f249"
TOPICS = ("traffic flow", "traffic congestion", "road network", "public transport", "urban mobility",
          "vehicle traffic", "traffic prediction", "groundwater", "river flow", "watershed",
          "stormwater", "urban drainage", "rainfall runoff", "water consumption")
MAX_SOURCES = 64
REPEATS = 5
ARMS = ("reader", "character", "semantic", "eager_agreement", "lazy_agreement")


def capture(old, output):
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads((old / "capture.json").read_text())
    visited = {doi for part in previous.values() for doi in part["selected"]}
    inventory, queries = legacy.inventory(TOPICS)
    eligible = inventory.keys() - visited
    chosen = sorted(eligible, key=lambda doi: legacy.stable("independent-cascade-v1", doi))[:MAX_SOURCES]
    output.mkdir()
    checks, failed = legacy.take(output, chosen)
    (output / "capture.json").write_text(json.dumps(dict(
        prior_sha256=sha256((old / "capture.json").read_bytes()).hexdigest(),
        queries=queries, candidates=len(inventory), eligible=len(eligible), selected=chosen, checksums=checks,
        failed=failed), indent=2) + "\n")
    print(f"Captured {len(checks)} of {len(chosen)} selected publications from {len(eligible)} eligible titles.", flush=True)


def restore(receipt, output):
    if output.exists():
        raise FileExistsError(output)
    raw = receipt.read_bytes()
    expected = json.loads(raw)["checksums"]
    output.mkdir()
    checked, failed = legacy.take(output, expected)
    if failed or checked != expected:
        raise ValueError("Cannot restore the previously captured publisher files")
    (output / "capture.json").write_bytes(raw)


def checked_xml(folder, doi, expected):
    path = folder / (doi.rsplit(".", 1)[-1] + ".xml")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Unexpected publisher file for {doi}")
    return raw


def byline_names(raw):
    root = ET.fromstring(raw)
    authors = root.findall("./front/article-meta/contrib-group/contrib[@contrib-type='author']/name")
    result = set()
    for author in authors:
        parts = legacy.NAME_PARTS.findall(f"{author.findtext('given-names') or ''} {author.findtext('surname') or ''}")
        if len(parts) >= 2:
            result.add(" ".join(part.casefold() for part in parts))
    return result


def cohort(old, snapshot):
    prior_bytes = (old / "capture.json").read_bytes()
    new = json.loads((snapshot / "capture.json").read_text())
    previous = json.loads(prior_bytes)
    if new["prior_sha256"] != sha256(prior_bytes).hexdigest():
        raise ValueError("Mismatched prior capture")
    seen_dois = {doi for split in previous.values() for doi in split["selected"]}
    if seen_dois & set(new["selected"]):
        raise ValueError("Shared publication across development and transfer")
    prior_names = set()
    for split, desc in previous.items():
        for doi, digest in desc["checksums"].items():
            prior_names.update(byline_names(checked_xml(old / split, doi, digest)))

    cases, omissions, name_sets = [], Counter(), []
    for doi, digest in new["checksums"].items():
        raw = checked_xml(snapshot, doi, digest)
        parsed = legacy.note(raw, doi)
        if parsed is None:
            omissions["unsupported_note_format_or_license"] += 1
            continue
        _, text, claims, names = parsed
        if prior_names & names:
            omissions["previous_author_name_overlap"] += 1
            continue
        selected = legacy.choose_case(doi, claims)
        if selected is None:
            omissions["no_two_duty_pair"] += 1
            continue
        tasks, correct, wrong = selected
        cases.append((doi, text, tasks, correct, wrong))
        name_sets.append(names)
    return cases, name_sets, dict(omissions)


def clusters(name_sets):
    leaders, named = list(range(len(name_sets))), {}
    def root(index):
        while leaders[index] != index:
            index = leaders[index]
        return index
    for i, names in enumerate(name_sets):
        for person in sorted(names):
            if person in named:
                a, b = root(i), root(named[person])
                leaders[max(a, b)] = min(a, b)
            else:
                named[person] = i
    return [root(i) for i in range(len(name_sets))]


def load_gate(gate, constructor):
    if gate is None:
        return None
    return constructor(tuple(gate["features"]),
                       tuple(-float("inf") if value is None else value for value in gate["reject_below"]),
                       tuple(float("inf") if value is None else value for value in gate["accept_above"]))


def evaluate(rows, tags, catalog, encoder, engines, gates, threshold, cascade):
    import numpy as np
    from expert_search.statistics import clustered_paired_interval

    metrics = defaultdict(list)
    for i, row in enumerate(rows):
        reports = defaultdict(list)
        for repetition in range(REPEATS):
            shift = (i + repetition) % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                gate = gates[arm]
                if arm == "lazy_agreement" and gate is not None:
                    out, ms, calls, n_vec = cascade.lazy_agreement([row], catalog, encoder, engines, gate, threshold)
                else:
                    out, ms, calls = cascade.live([row], catalog, encoder, engines, gate, threshold)
                    n_vec = int(gate is not None and 1 in gate.features)
                reports[arm].append((tuple(bool(v) for v in out[0]), bool(calls[0]), n_vec, ms[0]))
        for arm, runs in reports.items():
            if len({trial[:3] for trial in runs}) != 1:
                raise ValueError(f"Repeated outcomes changed for {arm}")
            outcome, read, vector_calls, _ = runs[0]
            metrics[arm].append((outcome, read, vector_calls, median(trial[3] for trial in runs)))
        if metrics["eager_agreement"][-1][:2] != metrics["lazy_agreement"][-1][:2]:
            raise ValueError("Lazy scheduling changed the eager output or mandatory reader path")
    result = {}
    for arm, values in metrics.items():
        outcomes = np.asarray([row[0] for row in values], dtype=bool)
        base = np.asarray([row[0] for row in metrics["reader"]], dtype=bool)
        runtime = np.asarray([row[3] for row in values])
        reader_ms = np.asarray([row[3] for row in metrics["reader"]])
        result[arm] = dict(
            accepted_supported=int(outcomes[:, 0].sum()), accepted_wrong=int(outcomes[:, 1].sum()),
            exact_pairs=int((outcomes[:, 0] & ~outcomes[:, 1]).sum()),
            reader_calls=sum(line[1] for line in values), vector_calls=sum(line[2] for line in values),
            mean_ms=float(runtime.mean()), p95_ms=float(np.quantile(runtime, .95)),
            support_delta_vs_reader=clustered_paired_interval(outcomes[:, 0].astype(int) - base[:, 0], tags),
            wrong_delta_vs_reader=clustered_paired_interval(outcomes[:, 1].astype(int) - base[:, 1], tags),
            mean_ms_delta_vs_reader=clustered_paired_interval(runtime - reader_ms, tags),
        )
    return result


def run(old, snapshot, embedding, reader, reference, output):
    import torch
    from experiments.credit import compile_legacy_notes as compiled
    from experiments.credit import legacy_note_cascade as cascade

    if (output.exists() or compiled.digest(reference) != REFERENCE
            or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise ValueError("Require the frozen gate, new output and Slurm CUDA")
    freeze = json.loads(reference.read_text())
    if freeze["capture_sha256"] != compiled.digest(old / "capture.json"):
        raise ValueError("New evaluation requires the same prior capture")
    rows, names, omissions = cohort(old, snapshot)
    if len(rows) < 16:
        raise ValueError("Insufficient disjoint transferable corpus")
    tags = clusters(names)
    provenance = compiled.verify_checkpoint(embedding)
    encoder = compiled.QwenTorchEmbedding(embedding, batch_size=8)
    encoder.model.config.use_cache = False
    catalog, build = compiled.catalog({"transfer": (rows, omissions)}, encoder)
    tokenizer, scorer, info = compiled.load_reader(reader)
    if provenance["revision"] != freeze["embedding_revision"] or info["revision"] != freeze["reader_revision"]:
        raise ValueError("Changed frozen model revisions")
    enc = [tokenizer.encode(x, add_special_tokens=False) for x in ("1", "0")]
    if any(len(token) != 1 for token in enc) or enc[0] == enc[1]:
        raise ValueError("Expected two distinct label tokens")
    engines = (scorer, tokenizer, [token[0] for token in enc])
    gates = {"reader": None, **{arm: load_gate(freeze["routes"][arm]["selected"]["gate"], cascade.Gate)
                               for arm in ("character", "semantic")}}
    gates["eager_agreement"] = gates["lazy_agreement"] = load_gate(
        freeze["routes"]["agreement"]["selected"]["gate"], cascade.Gate)
    for gate in gates.values():
        cascade.live(rows[:1], catalog, encoder, engines, gate, freeze["reader_cutoff"])
    result = evaluate(rows, tags, catalog, encoder, engines, gates, freeze["reader_cutoff"], cascade)
    compiled.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=compiled.digest(Path(__file__)),
        frozen_runs_sha256=REFERENCE, transfer_capture_sha256=compiled.digest(snapshot / "capture.json"),
        test_papers=len(rows), group_count=len(set(tags)), omissions=omissions,
        doi_inventory=[row[0] for row in rows], repetitions=REPEATS, index=build, test=result,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action, fields in (("capture", ("old", "output")), ("restore", ("receipt", "output")),
                           ("run", ("old", "snapshot", "embedding", "reader", "reference", "output"))):
        sub = commands.add_parser(action)
        for field in fields:
            sub.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    {"capture": capture, "restore": restore, "run": run}[args.pop("command")](**args)
