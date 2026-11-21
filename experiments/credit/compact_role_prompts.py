"""Measure compact role prompts with the same guarded proof resolver."""

import argparse
from collections import Counter, defaultdict
from copy import copy
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.distilled_role_recheck import route
from experiments.credit.elife_one_role_transfer import build
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, TOPIC_TASK, Catalog, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_joint_calibration import judge
from experiments.credit.engineering_one_role_recheck import RoleReader, combined, resolve
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


COMPACT = {
    "Conceptualization": "Formulate research ideas, goals and aims.",
    "Data curation": "Clean, annotate and maintain reusable data.",
    "Formal analysis": "Analyze study data with formal methods.",
    "Funding acquisition": "Obtain financial support for the project.",
    "Investigation": "Perform experiments or collect research evidence.",
    "Methodology": "Design research methods or develop models.",
    "Project administration": "Coordinate research planning and execution.",
    "Resources": "Provide materials, samples, equipment or computing resources.",
    "Software": "Develop, implement or test research code.",
    "Supervision": "Oversee, lead or mentor the research.",
    "Validation": "Verify research results, replication or reproducibility.",
    "Visualization": "Create figures or visual data presentations.",
    "Writing – original draft": "Write the initial manuscript draft.",
    "Writing – review & editing": "Critique, comment on or revise the manuscript.",
}


def predict(readers, text, roles, order):
    result = {}
    for arm in order:
        start = perf_counter()
        values = readers[arm].read(text, roles)
        result[arm] = values, (perf_counter() - start) * 1000
    return result


def evaluate(articles, rows, encoder, banks, wording, readers, bands, cutoff, reverse):
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    counts, times = defaultdict(Counter), defaultdict(list)
    if reverse:
        rows = rows[::-1]
    for i, row in enumerate([rows[0], *rows]):
        intended = tuple(row["papers"])
        excerpts = [articles[paper]["clue"] for paper in intended]
        text = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        order = tuple(readers)
        order = order[i % len(order):] + order[:i % len(order)]
        for view, clues in (("excerpt", excerpts), ("keywords", [catalog.keywords(clue) for clue in excerpts])):
            start = perf_counter()
            prompts = [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
            prompts += [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in text]
            vectors = encoder.encode(prompts)
            scores = {name: vectors[2:] @ banks[name].T for name in bands}
            embed_ms = (perf_counter() - start) * 1000
            cache = {}
            for availability, masked in zip(AVAILABILITY, (None, *intended), strict=True):
                start = perf_counter()
                topic = catalog.rank(vectors[:2], clues, masked)
                pair, _ = select(catalog, topic, row, [set(), set()], "anchored", masked)
                case = row | {"papers": pair, "gold": row["gold"] if pair == intended else []}
                retrieve_ms = embed_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                baseline = resolve(articles, case, required_roles(case, scores, "prototype_band", bands)) if pair else {}
                results = {"prototype_band": (baseline, 0, 0, (perf_counter() - start) * 1000)}
                start = perf_counter()
                sets, initial, slot, approved = route(articles, case, scores, bands) if pair else ([], {}, None, set())
                guard_ms = (perf_counter() - start) * 1000
                if approved and slot not in cache:
                    cache[slot] = predict(readers, text[slot], sets[slot], order)
                for arm in readers:
                    start = perf_counter()
                    output, calls, tokens, read_ms = initial, 0, 0, 0.
                    if approved:
                        (role, margin, tokens), read_ms = cache[slot][arm]
                        calls = 1
                        if role in approved and margin > cutoff:
                            output = combined(articles, case, sets, slot, role)
                    results[arm] = output, calls, tokens, guard_ms + read_ms + (perf_counter() - start) * 1000
                for arm, (output, calls, tokens, ms) in results.items():
                    start = perf_counter()
                    observed = judge(case, output, 0.)
                    json.dumps(observed)
                    elapsed = retrieve_ms + ms + (perf_counter() - start) * 1000
                    if i:
                        key = "|".join((view, availability, arm))
                        counts[key].update(observed, cases=1, calls=calls, tokens=tokens,
                                           substitutions=observed["issued"] and pair != intended,
                                           swap_issued=row["variant"] == "swap" and observed["issued"])
                        times[key].append(elapsed)
    return {key: dict(totals=val, mean_ms=float(np.mean(times[key])), p95_ms=float(np.quantile(times[key], .95)))
            for key, val in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    full = RoleReader(args.reader, definitions)
    compact, names = copy(full), copy(full)
    compact.definitions = [f"{role}: {COMPACT[role]}" for role in ROLES]
    names.definitions = ROLES
    readers = {"full": full, "compact": compact, "name_only": names}
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    articles, panel, census = build(args.elife)
    engineering = json.loads(args.engineering.read_bytes())
    cohorts = {"elife": (articles, panel), "engineering": (engineering["articles"], engineering["queries"])}
    results = {name: evaluate(*data, encoder, banks, wording, readers, bands, cutoff, args.reverse)
               for name, data in cohorts.items()}
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), elife_sha256=digest(args.elife), engineering_sha256=digest(args.engineering),
        elife_panel=census, embedding_revision=checkpoint["revision"], reader_revision=full.provenance["revision"],
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, bands=bands, cutoff=cutoff, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("elife", "engineering", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
