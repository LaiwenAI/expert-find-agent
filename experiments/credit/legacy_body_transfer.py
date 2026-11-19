"""Transfer frozen source gates to disjoint PLOS publications and authors."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from experiments.credit import legacy_contribution_text as legacy


TOPICS = ("building energy", "urban heat island", "land surface temperature",
          "climate change adaptation", "urban growth", "soil erosion", "land subsidence",
          "ground deformation", "satellite imagery", "remote sensing",
          "land cover classification", "recycling", "municipal solid waste", "urban expansion")
LIMIT = 112
BODY_RESULTS = "d3fcc064ca446c48bdf4ce81e4864f8cc9172a8f2d9363aeeb5fd13d546a19d2"
ARMS = ("dense", "lexical_body")


def captures(old, first):
    previous = json.loads((old / "capture.json").read_text())
    tested = json.loads((first / "capture.json").read_text())
    return [(old / split, part) for split, part in previous.items()] + [(first, tested)]


def capture(old, first, output):
    prior = {doi for _, part in captures(old, first) for doi in part["selected"]}
    available, receipts = legacy.inventory(TOPICS)
    selected = sorted(available.keys() - prior,
                      key=lambda doi: legacy.stable("frozen-body-transfer-v1", doi))[:LIMIT]
    output.mkdir()
    checks, failed = legacy.take(output, selected)
    (output / "capture.json").write_text(json.dumps(dict(
        queries=receipts, candidates=len(available), selected=selected, checksums=checks, failed=failed,
        earlier_captures=[sha256((folder / "capture.json").read_bytes()).hexdigest() for folder in (old, first)],
    ), indent=2) + "\n")
    print(f"Captured {len(checks)} of {len(selected)} selected publications.", flush=True)


def prepare(old, first, fresh):
    from experiments.credit import legacy_body_confirmation as experiment
    from experiments.credit import legacy_open_attribution as base

    originals = captures(old, first)
    new = json.loads((fresh / "capture.json").read_text())
    if new["earlier_captures"] != [base.indexed.digest(folder / "capture.json") for folder in (old, first)]:
        raise ValueError("Changed earlier captures")
    if set(new["selected"]) & {doi for _, part in originals for doi in part["selected"]}:
        raise ValueError("Fresh DOI intersects the earlier captures")
    earlier_names = set()
    for folder, part in originals:
        for doi, checksum in part["checksums"].items():
            earlier_names.update(base.transfer.byline_names(base.transfer.checked_xml(folder, doi, checksum)))

    notes, omissions = base.inventory(old, first)
    bodies = experiment.compile_body(old, first, notes)
    rows, author_groups = [], []
    for doi, checksum in new["checksums"].items():
        raw = base.transfer.checked_xml(fresh, doi, checksum)
        note = base.publication(raw, doi)
        if note is None:
            omissions["ineligible_new_source"] = omissions.get("ineligible_new_source", 0) + 1
            continue
        notes[doi] = note
        bodies[doi] = experiment.body_windows(raw, note.result)
        names = base.transfer.byline_names(raw)
        case = base.legacy.choose_case(doi, note.memberships)
        if names and not names & earlier_names and case is not None:
            rows.append((doi, case[0]))
            author_groups.append(names)
        else:
            omissions["ineligible_new_query"] = omissions.get("ineligible_new_query", 0) + 1
    return notes, bodies, rows, base.transfer.clusters(author_groups), omissions


def replay(index, bodies, rows, groups, form, withheld, cutoff, lexical_cutoff, engine, gate, author_cutoff):
    import numpy as np
    from expert_search.statistics import clustered_paired_interval
    from experiments.credit import legacy_body_confirmation as experiment
    from experiments.credit import legacy_open_attribution as base

    outputs = {arm: [] for arm in ARMS}
    for doi, tasks in rows:
        clue = index.clue(doi, form)
        begin = perf_counter()
        source, score = index.find(clue, doi if withheld else None)
        retrieval = 1e3 * (perf_counter() - begin)
        dense_accept = score > cutoff
        lexical_accept = dense_accept
        body_ms = 0.
        if not dense_accept:
            begin = perf_counter()
            coverage, _ = experiment.select_windows(clue, bodies[source], index.idf)
            body_ms = 1e3 * (perf_counter() - begin)
            lexical_accept = coverage is not None and coverage > lexical_cutoff
        correct = not withheld and source == doi
        truth = (set.intersection(*(index.notes[doi].memberships[t] for t in tasks)) if correct else set())
        authors, calls, author_ms = set(), 0, 0.
        if lexical_accept:
            begin = perf_counter()
            accepted, calls = base.verify(index.notes[source], source, tasks, True, engine, gate, author_cutoff)
            authors = set(accepted)
            author_ms = 1e3 * (perf_counter() - begin)
        for arm, passed in zip(ARMS, (dense_accept, lexical_accept), strict=True):
            recommended = authors if passed else set()
            outputs[arm].append(dict(
                correct_nomination=int(correct), correct_source=int(passed and correct),
                wrong_source=int(passed and not correct), supported=int(bool(recommended & truth)),
                clean_support=int(bool(recommended) and recommended <= truth),
                wrong_author=int(bool(recommended - truth)),
                wrong_author_on_correct_source=int(correct and bool(recommended - truth)),
                wrong_source_recommendation=int(bool(recommended) and not correct),
                author_calls=calls if passed else 0,
                ms=retrieval + (author_ms if passed else 0.) + (body_ms if arm == "lexical_body" else 0.),
            ))
    report = {}
    for arm, items in outputs.items():
        totals = Counter()
        for item in items:
            totals.update({k: v for k, v in item.items() if k != "ms"})
        report[arm] = dict(totals, factored_p95_ms=float(np.quantile([v["ms"] for v in items], .95)))
    delta = {}
    for field in ("supported", "clean_support", "wrong_author_on_correct_source", "wrong_source_recommendation"):
        interval = clustered_paired_interval(
            [a[field] - b[field] for a, b in zip(outputs["lexical_body"], outputs["dense"], strict=True)], groups)
        interval["sampling_unit"] = "connected author-name component; query-weighted mean"
        delta[field] = interval
    return dict(results=report, lexical_minus_dense=delta)


def run(old, first, fresh, embedding, reader, open_results, body_results, frozen, output):
    import torch
    from experiments.credit import legacy_body_confirmation as experiment
    from experiments.credit import legacy_open_attribution as base

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output on Slurm CUDA")
    if (base.indexed.digest(open_results) != experiment.REFERENCE
            or base.indexed.digest(frozen) != base.transfer.REFERENCE
            or base.indexed.digest(body_results) != BODY_RESULTS):
        raise ValueError("Changed frozen source or author policy")
    notes, bodies, rows, groups, omissions = prepare(old, first, fresh)
    if len(rows) < 40 or len(set(groups)) < 35:
        raise ValueError("Too few independent publisher queries")
    body = json.loads(body_results.read_bytes())
    if body["script_sha256"] != base.indexed.digest(Path(experiment.__file__)) or body["reference_sha256"] != experiment.REFERENCE:
        raise ValueError("Unexpected body policy provenance")
    open_data = json.loads(open_results.read_bytes())
    freeze = json.loads(frozen.read_bytes())
    encoder = base.indexed.QwenTorchEmbedding(embedding, batch_size=12)
    encoder.model.config.use_cache = False
    index = base.Search(notes, encoder)
    tokenizer, model, reader_info = base.indexed.load_reader(reader)
    if (base.indexed.verify_checkpoint(embedding)["revision"] != freeze["embedding_revision"]
            or reader_info["revision"] != freeze["reader_revision"]):
        raise ValueError("Changed model checkpoint")
    ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("1", "0")]
    if any(len(token) != 1 for token in ids) or ids[0] == ids[1]:
        raise ValueError("Expected distinct single-token output scores")
    engine = model, tokenizer, [token[0] for token in ids]
    gate = base.transfer.load_gate(freeze["routes"]["character"]["selected"]["gate"], base.cascade.Gate)
    if gate is None or gate.features != (0,):
        raise ValueError("Changed character-gate format")
    first_doi, first_tasks = rows[0]
    base.cascade.reader_pair((first_doi, notes[first_doi].note, first_tasks,
                              notes[first_doi].authors[0], notes[first_doi].authors[1]), 1, engine)
    report = {}
    for form in base.FORMS:
        for state in ("present", "withheld"):
            report[f"{form}/{state}"] = replay(
                index, bodies, rows, groups, form, state == "withheld", open_data["thresholds"][form],
                body["thresholds"][form]["lexical_body"]["threshold"], engine, gate, freeze["reader_cutoff"])
    base.indexed.write_json(output, dict(
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        body_results_sha256=base.indexed.digest(body_results),
        capture_sha256=base.indexed.digest(fresh / "capture.json"), job=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), queries=len(rows), author_components=len(set(groups)),
        candidates=len(notes), body_windows=sum(map(len, bodies.values())), omissions=omissions, report=report,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, fields in (("capture", ("old", "first", "output")),
                         ("run", ("old", "first", "fresh", "embedding", "reader",
                                  "open_results", "body_results", "frozen", "output"))):
        cmd = commands.add_parser(name)
        for field in fields:
            cmd.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    {"capture": capture, "run": run}[args.pop("command")](**args)


if __name__ == "__main__":
    main()
