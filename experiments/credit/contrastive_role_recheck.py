"""Contrast ambiguous roles under the existing two-source proof guard."""

import argparse
from collections import Counter, defaultdict
from copy import copy
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.distilled_role_recheck import route, student_values
from experiments.credit.elife_one_role_transfer import build as development
from experiments.credit.elife_prompt_holdout import build as holdout
from experiments.credit.engineering_cross_source_search import AVAILABILITY, TOPIC_TASK, Catalog, phrase_bank, select
from experiments.credit.engineering_joint_calibration import SUPPORT
from experiments.credit.engineering_one_role_recheck import RoleReader, replace
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_issuance_audit import outcome
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("names", "names_calibrated", "pair_names_raw", "pair_names", "pair_contrast", "pair_stable")
BOUNDARIES = {
    "Conceptualization": "Define the core research question or aims. Method design alone does not meet this role.",
    "Data curation": "Clean, document or maintain reusable data. Collecting evidence and analyzing it are separate.",
    "Formal analysis": "Apply analytical methods to study data. Designing the method or just collecting data is separate.",
    "Funding acquisition": "Secure financial support. Providing equipment or materials alone is separate.",
    "Investigation": "Run the study or collect evidence. Designing or independently verifying the study is separate.",
    "Methodology": "Design the procedure or model. Applying it to data or coding it is separate.",
    "Project administration": "Coordinate schedules, logistics or execution. Scientific oversight and mentorship are separate.",
    "Resources": "Supply material, equipment, samples or computing. Securing funding is separate.",
    "Software": "Implement, maintain or test code. Designing the research method without code is separate.",
    "Supervision": "Lead or oversee the research, including mentorship. Routine coordination alone is separate.",
    "Validation": "Check correctness, replication or reproducibility. Performing the initial study alone is separate.",
    "Visualization": "Create or prepare data displays. General analysis and prose writing are separate.",
    "Writing – original draft": "Produce the first manuscript text or substantive translation. Revising a draft is separate.",
    "Writing – review & editing": "Review or revise manuscript text. Creating the first draft is separate.",
}


def shortlist(scores, slot, roles):
    values = student_values("dense_agreement", None, scores, {}, slot)
    return sorted(roles, key=lambda role: (-values[SUPPORT[role]], role))[:2]


class ContrastReader:
    def __init__(self, reader):
        self.reader = reader
        self.reader.tokenizer.padding_side = "left"

    @torch.inference_mode()
    def read(self, phrase, options):
        tokenizer = self.reader.tokenizer
        ordered = sorted(options, key=lambda role: sha256(f"{phrase}\n{role}".encode()).digest())
        def measure(orderings):
            start = perf_counter()
            prompts = []
            for a, b in orderings:
                messages = [
                    dict(role="system", content="Choose the role entailed by the activity. Return one letter."),
                    dict(role="user", content=f"Activity: {phrase}\nA. {a}: {BOUNDARIES[a]}\nB. {b}: {BOUNDARIES[b]}\n"
                         "C. Neither role fits or the activity does not distinguish them."),
                ]
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
            positions = (inputs["attention_mask"].cumsum(1) - 1).clamp(min=0)
            logits = self.reader.model(**inputs, position_ids=positions, use_cache=False,
                                       logits_to_keep=1).logits[:, -1, self.reader.digits[:3]].float().cpu().numpy()
            choices = []
            for values, labels in zip(logits, orderings, strict=True):
                top, second = np.argsort(-values, kind="stable")[:2]
                choices.append((labels[top] if top < 2 else None, float(values[top] - values[second])))
            tokens = int(inputs["attention_mask"].sum().item())
            return choices, tokens, (perf_counter() - start) * 1000

        [single], single_tokens, single_ms = measure((ordered,))
        [first, second], paired_tokens, paired_ms = measure((ordered, ordered[::-1]))
        return (*single, single_tokens, single_ms), (
            first[0] if first[0] == second[0] else None, min(first[1], second[1]), paired_tokens, paired_ms)


def timed(reader, text, options):
    start = perf_counter()
    role, margin, tokens = reader.read(text, options)
    return role, margin, tokens, (perf_counter() - start) * 1000


def compare(articles, rows, encoder, banks, wording, names, contrast, bands, reverse, fit=None):
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    observations = defaultdict(list)
    for i, row in enumerate([rows[0], *(rows[::-1] if reverse else rows)]):
        intended = tuple(row["papers"])
        excerpts = [articles[paper]["clue"] for paper in intended]
        text = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for view, clues in (("excerpt", excerpts), ("keywords", [catalog.keywords(clue) for clue in excerpts])):
            start = perf_counter()
            vectors = encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{c}" for c in clues] +
                                     [f"Instruct: {TASK}\nQuery:{t}" for t in text])
            scores = {name: vectors[2:] @ banks[name].T for name in bands}
            encode_ms = (perf_counter() - start) * 1000
            cache = {}
            masks = (None, *intended) if fit is not None else (None,)
            for availability, masked in zip(AVAILABILITY, masks):
                start = perf_counter()
                topic = catalog.rank(vectors[:2], clues, masked)
                pair, _ = select(catalog, topic, {}, [set(), set()], "anchored", masked)
                scope = {"papers": pair, "roles": (None, None)}
                sets, slot, approved = [set(), set()], None, set()
                if pair is not None:
                    sets, _, slot, approved = route(articles, scope, scores, bands)
                base_ms = encode_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                baseline, _ = outcome(catalog, row, pair, sets, "initial", None)
                base_ms += (perf_counter() - start) * 1000
                if approved and slot not in cache:
                    start = perf_counter()
                    pool = shortlist(scores, slot, sets[slot])
                    shortlist_ms = (perf_counter() - start) * 1000
                    normal = timed(names, text[slot], sets[slot])
                    pair_names = timed(names, text[slot], pool)
                    single, stable = contrast.read(text[slot], pool)
                    pair_names = (*pair_names[:3], pair_names[3] + shortlist_ms)
                    cache[slot] = dict(names=normal, names_calibrated=normal,
                                       pair_names_raw=pair_names, pair_names=pair_names,
                                       pair_contrast=(*single[:3], single[3] + shortlist_ms),
                                       pair_stable=(*stable[:3], stable[3] + shortlist_ms))
                for arm in ARMS:
                    start = perf_counter()
                    alternative, available, margin, tokens, read_ms = baseline, False, 0., 0, 0.
                    if approved:
                        role, margin, tokens, read_ms = cache[slot][arm]
                        if role in approved:
                            alternative, _ = outcome(catalog, row, pair, replace(sets, slot, role), "reader", slot)
                            available = True
                    more_ms = (perf_counter() - start) * 1000
                    if i:
                        observations[view, availability, arm].append(
                            dict(before=baseline, after=alternative, margin=margin, eligible=available,
                                 calls=int(bool(approved)), tokens=tokens, ms=base_ms + read_ms + more_ms))
    return observations


def decision(row, cutoff):
    return row["after"] if row["eligible"] and row["margin"] > cutoff else row["before"]


def calibrate(observations, teacher_cutoff):
    chosen = {"names": {"cutoff": teacher_cutoff}, "pair_names_raw": {"cutoff": teacher_cutoff}}
    for arm in (name for name in ARMS if name not in chosen):
        rows = [row for (_, _, name), group in observations.items() if name == arm for row in group]
        best = None
        for cutoff in sorted({0., *(r["margin"] for r in rows if r["eligible"])}):
            totals = Counter()
            for row in rows:
                totals.update(decision(row, cutoff))
            if totals["adverse"] > .05 * totals["issued"]:
                continue
            key = totals["proof"], -totals["adverse"], cutoff
            if best is None or key > best[0]:
                best = key, {"cutoff": cutoff, "totals": dict(totals)}
        if best is None:
            raise ValueError(f"No feasible development threshold for {arm}")
        chosen[arm] = best[1]
    return chosen


def aggregate(observations, fit):
    results = {}
    for key, rows in sorted(observations.items()):
        cutoff = fit[key[-1]]["cutoff"]
        totals = Counter()
        for row in rows:
            totals.update(decision(row, cutoff), calls=row["calls"], tokens=row["tokens"], requests=1)
        results["|".join(key)] = dict(totals=dict(totals), p95_ms=float(np.quantile([r["ms"] for r in rows], .95)))
    return results


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    model = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    full = RoleReader(args.reader, definitions)
    names = copy(full)
    names.definitions = ROLES
    contrast = ContrastReader(full)
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    sources, cases, dev = development(args.development)
    obs = compare(sources, cases, encoder, banks, wording, names, contrast, bands, args.reverse)
    fit = calibrate(obs, cutoff)
    articles, panel, census = holdout(args.holdout, json.loads(args.manifest.read_bytes()))
    measured = compare(articles, panel, encoder, banks, wording, names, contrast, bands, args.reverse, fit=fit)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), dev_sha256=digest(args.development), holdout_sha256=digest(args.holdout),
        embedding_revision=model["revision"], reader_revision=full.provenance["revision"],
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, fit=fit, dev=dev, holdout=census,
        development=aggregate(obs, fit), results=aggregate(measured, fit),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development", "holdout", "manifest", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
