"""Factor source and role confirmation for a bounded proof repair."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.elife_one_role_transfer import build
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, TOPIC_TASK, Catalog, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_joint_calibration import features, fit_source, judge, open_calibration, source_score
from experiments.credit.engineering_one_role_recheck import RoleReader, combined, pick, resolve, state
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = {
    "recheck": ("no_source_guard", False),
    "role_agreement": ("no_source_guard", True),
    "calibrated_source": ("calibrated", False),
    "calibrated_both": ("calibrated", True),
    "two_leaders": ("two_leaders", False),
    "two_leaders_and_role": ("two_leaders", True),
}


def guards(catalog, topic, pair, clues, model):
    start = perf_counter()
    leaders = all(doc == side["fused"][0] for doc, side in zip(pair, topic, strict=True))
    leaders_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    calibrated = all(source_score(model, features(catalog, side, clue, doc)) > model["threshold"]
                     for doc, side, clue in zip(pair, topic, clues, strict=True))
    return {"no_source_guard": (True, 0.), "two_leaders": (leaders, leaders_ms),
            "calibrated": (calibrated, (perf_counter() - start) * 1000)}


def evaluate(articles, rows, encoder, banks, wording, reader, bands, cutoff, source_models, reverse):
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    counts, latency = defaultdict(Counter), defaultdict(list)
    if reverse:
        rows = rows[::-1]
    for i, row in enumerate([rows[0], *rows]):
        intended = tuple(row["papers"])
        excerpts = [articles[paper]["clue"] for paper in intended]
        request = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for view, clues in (("excerpt", excerpts), ("keywords", [catalog.keywords(clue) for clue in excerpts])):
            read_cache = {}
            start = perf_counter()
            queries = [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
            queries += [f"Instruct: {TASK}\nQuery:{clause}" for clause in request]
            vectors = encoder.encode(queries)
            scores = {bank: vectors[2:] @ banks[bank].T for bank in bands}
            encoded_ms = (perf_counter() - start) * 1000
            for availability, withheld in zip(AVAILABILITY, (None, *intended), strict=True):
                start = perf_counter()
                topic = catalog.rank(vectors[:2], clues, withheld)
                pair, _ = select(catalog, topic, row, [set(), set()], "anchored", withheld)
                case = row | {"papers": pair, "gold": row["gold"] if pair == intended else []}
                fetch_ms = (perf_counter() - start) * 1000 + encoded_ms
                start = perf_counter()
                prototype = required_roles(case, scores, "prototype_band", bands)
                reference = resolve(articles, case, prototype) if pair is not None else {}
                result = {"prototype_band": (reference, 0, 0, fetch_ms + (perf_counter() - start) * 1000)}
                start = perf_counter()
                slot, initial = None, {}
                if pair is not None:
                    sets, initial, branches = state(articles, case, scores["ridge"], bands)
                    slot = pick(branches, scores["ridge"])["proof_aware_gated"]
                common_ms = fetch_ms + (perf_counter() - start) * 1000
                if slot is None:
                    result.update({arm: (initial, 0, 0, common_ms) for arm in ARMS})
                else:
                    source_match = guards(catalog, topic, pair, clues, source_models[view])
                    start = perf_counter()
                    approved = {role for role in sets[slot] & prototype[slot]
                                if combined(articles, case, sets, slot, role)}
                    agreement_ms = (perf_counter() - start) * 1000
                    if slot not in read_cache:
                        start = perf_counter()
                        reading = reader.read(request[slot], sets[slot])
                        read_cache[slot] = reading, (perf_counter() - start) * 1000
                    (role, margin, tokens), reader_ms = read_cache[slot]
                    for arm, (source_guard, role_guard) in ARMS.items():
                        start = perf_counter()
                        allowed, guard_ms = source_match[source_guard]
                        call = allowed and (not role_guard or bool(approved))
                        output = initial
                        if call and role is not None and margin > cutoff and (not role_guard or role in approved):
                            output = combined(articles, case, sets, slot, role)
                        ms = common_ms + guard_ms
                        ms += (agreement_ms if role_guard and allowed else 0) + (reader_ms if call else 0)
                        result[arm] = output, int(call), tokens if call else 0, ms + (perf_counter() - start) * 1000
                for arm, (output, calls, tokens, ms) in result.items():
                    start = perf_counter()
                    outcome = judge(case, output, 0.)
                    if i:
                        key = "|".join((view, availability, arm))
                        counts[key].update(outcome, requests=1, calls=calls, tokens=tokens,
                                           substitutions=outcome["issued"] and pair != intended,
                                           swap_issued=row["variant"] == "swap" and outcome["issued"])
                        latency[key].append(ms + (perf_counter() - start) * 1000)
    return {key: dict(totals=total, mean_ms=float(np.mean(latency[key])),
                      p95_ms=float(np.quantile(latency[key], .95))) for key, total in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    revision = verify_checkpoint(args.embedding)["revision"]
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    reader = RoleReader(args.reader, definitions)
    plos, clues, _ = open_calibration(args.plos)
    catalog = Catalog(plos, encoder)
    source_fit = {view: fit_source(catalog, clues, encoder, view) for view in ("excerpt", "keywords")}
    models = {view: pair[0] for view, pair in source_fit.items()}
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    articles, cases, census = build(args.elife)
    engineering = json.loads(args.engineering.read_bytes())
    cohorts = {"elife": (articles, cases), "engineering": (engineering["articles"], engineering["queries"])}
    results = {key: evaluate(*pool, encoder, banks, wording, reader, bands, cutoff, models, args.reverse)
               for key, pool in cohorts.items()}
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), elife_sha256=digest(args.elife), engineering_sha256=digest(args.engineering),
        calibration_sha256=digest(args.plos), elife_panel=census, bands=bands, cutoff=cutoff,
        source_fit={view: pair[1] for view, pair in source_fit.items()}, gpu=torch.cuda.get_device_name(),
        embedding_revision=revision, reader_revision=reader.provenance["revision"], reverse=args.reverse, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("elife", "engineering", "plos", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
