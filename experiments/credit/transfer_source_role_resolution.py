"""Transfer unquoted source–activity resolution to different papers and phrasing."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.budgeted_source_role_service import Service
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.engineering_cross_source_search import phrase_bank
from experiments.credit.engineering_one_role_recheck import RoleReader
from experiments.credit.engineering_role_banks import projections, queries
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.offset_budgeted_source_role import (
    TRAIN_FORMS, fit_context, joint_parse, pools, score_search, synthetic,
)
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.unquoted_budgeted_source_role import sample_families
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


FORMS = {
    "activities_first": ("Which researcher handled {x} for a study of {a} "
                         "and carried out {y} for separate work on {b}?"),
    "numbered_reverse": ("The two topics are {a} and {b}. For the second topic confirm {y}; "
                         "for the first topic check {x}. Return one person with both contributions."),
    "indirect_reverse": ("Consider {a} alongside {b}. In the study named last, look for {y}. "
                         "In the study named first, verify {x}. Both credits must belong to one person."),
}
CUE = re.compile(r"\bthe\s+(first|second|former|latter)\b"
                 r"(?:\s+(?:topic|study|paper|article|project)\b)?", flags=re.IGNORECASE)
CLAUSE = re.compile(r"[.!?;]\s+")
SIDE = {"first": 0, "former": 0, "second": 1, "latter": 1}


def ordinal_repair(text, pairs):
    """Rebind two copied activities if each has one separate antecedent cue."""
    if not pairs:
        return ()
    found = {}
    for _, activity in pairs:
        starts = list(re.finditer(rf"(?<!\w){re.escape(activity)}(?!\w)", text))
        if len(starts) != 1:
            return ()
        start, end = starts[0].span()
        boundaries = [match.span() for match in CLAUSE.finditer(text)]
        left = max((b for _, b in boundaries if b <= start), default=0)
        right = min((a for a, _ in boundaries if a >= end), default=len(text))
        cues = list(CUE.finditer(text, left, right))
        if len(cues) != 1 or cues[0].end() > start:
            return ()
        index = SIDE[cues[0][1].casefold()]
        if index in found:
            return ()
        found[index] = activity
    return tuple((topic, found[i]) for i, (topic, _) in enumerate(pairs))


def compare(service, rows, wording, model, partner, reverse):
    bank = phrase_bank(wording)
    rows = rows[::-1] if reverse else rows
    counts, timings = defaultdict(Counter), defaultdict(list)
    for n, row in enumerate([rows[0], *rows]):
        topics = tuple(service.index.keywords(service.index.articles[key]["clue"]) for key in row["papers"])
        activities = tuple(bank[r][int(row["family"][:16], 16) % len(bank[r])].strip().rstrip(".?!")
                           for r in row["roles"])
        if not all(topics) or len(set(topics)) != 2 or len(set(activities)) != 2:
            raise ValueError("Two distinct nonempty source and activity phrases required")
        expected = tuple(zip(topics, activities, strict=True))
        swapped = tuple(zip(topics, activities[::-1], strict=True))
        for style, template in FORMS.items():
            text = template.format(a=topics[0], b=topics[1], x=activities[0], y=activities[1])
            _, _, joint, _, joint_ms, agreement, agreement_ms = joint_parse(
                model, service.encoder, text, partner)
            begin = perf_counter()
            directed = ordinal_repair(text, joint)
            repair_ms = (perf_counter() - begin) * 1000
            variants = (
                ("supplied", expected, 0., False),
                ("joint", joint, joint_ms, False),
                ("agreement", agreement, agreement_ms, False),
                ("ordinal", directed, joint_ms + repair_ms, bool(directed)),
                ("ordinal_then_agreement", directed or agreement,
                 (joint_ms if directed else agreement_ms) + repair_ms, bool(directed)),
            )
            for arm, binding, parse_ms, explicit in variants:
                begin = perf_counter()
                scored = score_search(service, binding, row)
                elapsed = parse_ms + (perf_counter() - begin) * 1000
                if n:
                    key = f"{style}|{arm}"
                    counts[key].update(scored, complete=bool(binding), exact_binding=binding == expected,
                                       swap_only=binding == swapped, ordinal_applicable=explicit,
                                       ordinal_changed=explicit and binding != joint)
                    timings[key].append((parse_ms, elapsed))
    return {key: dict(totals=dict(counts[key]),
                      parser_p95_ms=float(np.quantile([p for p, _ in samples], .95)),
                      total_p95_ms=float(np.quantile([t for _, t in samples], .95)))
            for key, samples in sorted(timings.items())}


def load_engineering(panel_path, manifest_path):
    manifest = json.loads(manifest_path.read_bytes())
    if digest(panel_path) != manifest["panel_sha256"]:
        raise ValueError("Expected the pinned engineering panel")
    panel = json.loads(panel_path.read_bytes())
    return panel["articles"], panel["queries"]


def checked_original(path, manifest_path, old):
    articles, rows, _ = build(path, json.loads(manifest_path.read_bytes()))
    old_authors = {person for article in old.values() for person in article["roles"]}
    blocked = {key for key, article in articles.items() if key in old or old_authors.intersection(article["roles"])}
    fresh = {key: article for key, article in articles.items() if key not in blocked}
    return fresh, [row for row in rows if not blocked.intersection(row["papers"])], len(blocked)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    torch.set_num_threads(1)
    development, cases, _ = build(args.development, json.loads(args.dev_manifest.read_bytes()))
    elife_test, elife_rows, excluded = checked_original(args.elife, args.elife_manifest, development)
    engineering, engineering_rows = load_engineering(args.engineering, args.engineering_manifest)
    definitions, phrases, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    fit_topics, _ = pools(development, PostingCatalog(development, encoder), sample_families(cases))
    activities = sorted({phrase.strip().rstrip(".?!") for phrase in phrases})
    random.Random(792).shuffle(activities)
    data = synthetic(fit_topics, activities[:64], TRAIN_FORMS, 2048, 805, outside=True)
    begin = perf_counter()
    primary, first_loss = fit_context(encoder, data, args.seed)
    partner_seed = 29 if args.seed == 17 else 17
    partner, second_loss = fit_context(encoder, data, partner_seed)
    elapsed = perf_counter() - begin
    banks = projections(encoder, definitions, phrases)
    reader = RoleReader(args.role_reader, definitions)
    reader.definitions = ROLES
    source = Reranker(args.source_reader)
    frozen = json.loads(args.source_frozen.read_bytes())["runs"][0]
    if source.provenance["revision"] != frozen["verifier"]["revision"]:
        raise ValueError("Unexpected source verifier")
    cutoffs = dict(source=frozen["thresholds"]["keywords"]["weakest"]["cutoff"],
                   role=json.loads(args.role_frozen.read_bytes())["runs"][0]["fit"]["cutoff"])
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    panels = {}
    for name, catalog, rows in (("elife_earlier", elife_test, elife_rows),
                                ("engineering", engineering, engineering_rows)):
        selected = sample_families(rows)
        if not selected:
            raise ValueError(f"No eligible families in {name}")
        index = PostingCatalog(catalog, encoder)
        service = Service(index, encoder, banks, bands, source, reader, cutoffs)
        hashes = {sha256(repr(tuple(row["papers"])).encode()).hexdigest() for row in selected}
        panels[name] = dict(articles=len(catalog), pairs=len(hashes), pair_hashes=sorted(hashes),
                            query_families=sorted({row["family"] for row in selected}),
                            results=compare(service, selected, wording, primary, partner, args.reverse))
        print(f"{name}: {len(catalog)} articles, {len(hashes)} article pairs", flush=True)
    files = ("development", "dev_manifest", "elife", "elife_manifest", "engineering",
             "engineering_manifest", "niso", "training", "bands", "source_frozen", "role_frozen")
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)),
        inputs={k: digest(getattr(args, k)) for k in files},
        upstream_modules={name: digest(Path(__file__).with_name(f"{name}.py"))
                          for name in ("offset_budgeted_source_role", "source_role_offset_lattice")},
        embedding_revision=checkpoint["revision"], source_reader_revision=source.provenance["revision"],
        role_reader_revision=reader.provenance["revision"], seed=args.seed, partner_seed=partner_seed,
        reverse=args.reverse, gpu=torch.cuda.get_device_name(), training_seconds=elapsed,
        training_requests=len(data), fit_topics=len(fit_topics), losses=(first_loss, second_loss),
        elife_overlap_removed=excluded, panels=panels,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("development", "dev_manifest", "elife", "elife_manifest", "engineering",
                     "engineering_manifest", "niso", "training", "bands", "source_frozen", "role_frozen",
                     "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--seed", type=int, choices=(17, 29), default=17)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
