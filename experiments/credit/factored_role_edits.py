"""Compare shared intent parsing and evidence-driven routing on matched role edits."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.attribution_controls import LABELS, Reader
from experiments.credit.counterfactual_role_edits import POSITIVE, TASKS, fingerprint, label
from experiments.credit.evaluate_role_edits import MEANINGS, prompt
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_language_router import NISO_SHA256, SYMBOLS, TASK, Router
from experiments.qasper.adaptive_evidence import digest
from experiments.qasper.paired_reader import load_reader
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


METHODS = ("direct", "dense", "full", "shortlist", "invariance")
REQUEST = "Which authors contributed to both of these activities?"
FIELD = "\nRequired activity: "
FORMAT = re.compile(re.escape(REQUEST) + re.escape(FIELD) + r"([^\n]+)" + re.escape(FIELD) + r"([^\n]+)")


def wording(group, style, phrases):
    if style == "authored":
        variants = TASKS
    else:
        variants = {}
        for role in set(group["before"] + group["after"]):
            options = phrases[role]
            variants[role] = options[int(fingerprint(group["id"], role), 16) % len(options)]
    return [REQUEST + "".join(FIELD + " ".join(variants[role].split()) for role in group[view])
            for view in ("before", "after")]


def candidate_lists(router, atoms, dense, k):
    return [sorted(set(router.lexical_rank(atom)[0][:k] + ranks[:k]))
            for atom, ranks in zip(atoms, dense, strict=True)]


def choose_batched(router, atoms, options, rotation):
    orders, prompts = [], []
    for i, (atom, choices) in enumerate(zip(atoms, options, strict=True)):
        shift = (rotation + i) % len(choices)
        ordered = choices[shift:] + choices[:shift]
        glossary = "\n".join(f"{symbol}: {router.docs[j]}"
                             for symbol, j in zip(SYMBOLS, ordered))
        prompts.append(f"{TASK} Choose one role from the glossary.\n{glossary}\nTask: {atom}\nAnswer:")
        orders.append(ordered)
    batch = router.tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
    positions = (batch.attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    output = router.reader(**batch, position_ids=positions, use_cache=False, logits_to_keep=1)
    predictions = [ordered[int(output.logits[i, -1, router.ids[:len(ordered)]].argmax().item())]
                   for i, ordered in enumerate(orders)]
    return predictions


def unanimous(options, evidence):
    results = {tuple(label(source, (ROLES[a], ROLES[b])) for source in evidence)
               for a in options[0] for b in options[1] if a != b}
    return list(next(iter(results))) if len(results) == 1 else None


@torch.inference_mode()
def run_request(router, reader, request, authors, evidence, method, rotation, option_order):
    start = perf_counter()
    roles, candidates, calls = None, None, 0
    if method == "direct":
        prompts = [prompt(request, a, [(a, visible)], option_order) for a, visible in zip(authors, evidence)]
        choices, _ = reader.classify(prompts)
        answers = [option_order[LABELS.index(symbol)] for symbol, _ in choices]
        calls = 1
    else:
        match = FORMAT.fullmatch(request)
        if match is None:
            raise ValueError("Request outside the controlled two-field protocol")
        atoms = match.groups()
        if method == "full":
            choices = choose_batched(router, atoms, [list(range(len(ROLES)))] * 2, rotation)
            calls = 1
        else:
            vectors = router.embedder.encode([f"Instruct: {TASK}\nQuery: {atom}" for atom in atoms])
            dense = np.argsort(-(vectors @ router.vectors.T), axis=1, kind="stable").tolist()
            if method == "dense":
                choices = [rank[0] for rank in dense]
            else:
                candidates = candidate_lists(router, atoms, dense, 3)
                answers = unanimous(candidates, evidence) if method == "invariance" else None
                if answers is not None:
                    return dict(answers=answers, roles=None, calls=0, candidates=candidates,
                                ms=(perf_counter() - start) * 1000)
                choices = choose_batched(router, atoms, candidates, rotation)
                calls = 1
        roles = [ROLES[role] for role in choices]
        # Duplicate predictions supply one distinct activity for a two-activity request.
        answers = [label(source, roles) if len(set(roles)) == 2 else "unavailable" for source in evidence]
    return dict(answers=answers, roles=roles, calls=calls, candidates=candidates,
                ms=(perf_counter() - start) * 1000)


def summarize(rows):
    output = {}
    for style in ("authored", "niso"):
        group = [row for row in rows if row["style"] == style]
        report = {}
        for method in METHODS:
            estimates = [row["arms"][method] for row in group]
            flat = [(view, expected, role) for row, views in zip(group, estimates, strict=True)
                    for view, expected, role in zip(views, row["gold"], row["required"], strict=True)]
            exact = [all(v["answers"] == y for v, y in zip(views, row["gold"], strict=True))
                     for row, views in zip(group, estimates, strict=True)]
            direct = [all(v["answers"] == y for v, y in zip(row["arms"]["direct"], row["gold"], strict=True))
                      for row in group]
            report[method] = dict(
                exact_groups=sum(exact), groups=len(exact),
                exact_role_requests=sum(v["roles"] is not None and set(v["roles"]) == set(r) for v, _, r in flat),
                role_requests_with_explicit_prediction=sum(v["roles"] is not None for v, _, _ in flat),
                false_promotions=sum(a == POSITIVE and b != POSITIVE for v, y, _ in flat
                                     for a, b in zip(v["answers"], y, strict=True)),
                skipped=sum(v["calls"] == 0 for v, _, _ in flat),
                shortlist_coverage=sum(v["candidates"] is not None and all(ROLES.index(role) in slots
                                       for role, slots in zip(r, v["candidates"], strict=True)) for v, _, r in flat),
                incorrect_skips=sum(v["calls"] == 0 and list(v["answers"]) != y for v, y, _ in flat)
                if method == "invariance" else None,
                p50_p95_ms=np.quantile([v["ms"] for v, _, _ in flat], [.5, .95]).tolist(),
                group_delta=clustered_paired_interval([int(a) - int(b) for a, b in zip(exact, direct, strict=True)],
                                                      [row["doi"] for row in group]),
            )
        output[style] = report
    return output


def run(fixture, glossary, embedding, checkpoint, output):
    digest_glossary = digest(glossary)
    if digest_glossary != NISO_SHA256:
        raise ValueError("Changed NISO snapshot")
    upstream = json.loads(glossary.read_bytes())
    docs = [f"{r}: {upstream['definitions'][r]}" for r in ROLES]
    phrases = defaultdict(list)
    for example in upstream["examples"]:
        phrases[example["role"]].append(example["text"])
    provenance = json.loads((embedding / "checkpoint-provenance.json").read_bytes())
    for name, expected in provenance["files"].items():
        if digest(embedding / name) != expected:
            raise ValueError(f"Changed model file: {name}")
    embedder = QwenTorchEmbedding(embedding)
    embedder.model.config.use_cache = False
    tokenizer, model, reader_provenance = load_reader(checkpoint)
    reader = Reader(tokenizer, model)
    with TemporaryDirectory() as temporary:
        lexical = build_lexical({text: i for i, text in enumerate(docs)}, Path(temporary))
    router = Router(docs, lexical, embedder.encode(docs), embedder, model, tokenizer, 3)
    m = json.loads((fixture / "manifest.json").read_bytes())
    path = fixture / "pairs.jsonl"
    if digest(path) != m["pairs_sha256"]:
        raise ValueError("Changed benchmark")
    papers = {p["doi"]: p["roles_by_author"] for p in m["sources"] if p["split"] == "test"}
    groups = [g for line in path.read_text().splitlines() if (g := json.loads(line))["split"] == "test"]
    rows = []
    for i, group in enumerate(groups):
        roles = papers[group["doi"]]
        authors = [group["changed_author"], group["stable_author"]]
        visible = [roles[a - 1] for a in authors]
        required = [group["before"], group["after"]]
        gold = [list(pair) for pair in zip(group["changed_labels"], group["stable_labels"], strict=True)]
        rotation = int(fingerprint(group["id"], "options")[:8], 16)
        option_order = sorted(MEANINGS, key=lambda m: fingerprint(group["id"], m))
        for style in ("authored", "niso"):
            formatted = wording(group, style, phrases)
            shift = i % len(METHODS)
            if i == 0:
                for method in METHODS:
                    run_request(router, reader, formatted[0], authors, visible, method, rotation, option_order)
            choices = {}
            for method in METHODS[shift:] + METHODS[:shift]:
                views = [None, None]
                for j in (i % 2, 1 - i % 2):
                    views[j] = run_request(router, reader, formatted[j], authors, visible, method, rotation, option_order)
                choices[method] = views
            rows.append(dict(doi=group["doi"], id=group["id"], style=style, required=required,
                             gold=gold, arms=choices))
    with output.open("x") as stream:
        json.dump(dict(summary=summarize(rows), rows=rows, pair_sha256=digest(path), niso_sha256=digest_glossary,
                       script_sha256=digest(Path(__file__)), embedding=provenance, reader=reader_provenance,
                       gpu=torch.cuda.get_device_name()), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "glossary", "embedding", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.fixture, args.glossary, args.embedding, args.checkpoint, args.output)
