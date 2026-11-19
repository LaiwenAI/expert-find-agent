"""Measure the local text-to-expert route on human-authored task descriptions."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import minimal_evidence_bundles as bundles
from experiments.credit.role_language_router import NISO_SHA256, SYMBOLS, TASK
from experiments.credit.role_prototypes import ROLES, encode_tasks, train_banks
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.paired_reader import load_reader


ARMS = ("definition", "ridge", "reader", "guarded", "known_roles")
CACHE = 50
HEADER = "Find researchers with documented experience in all of these tasks:"


def key(value):
    return sha256(f"credit-live-bundle-v1:{value}".encode()).digest()


def phrase_split(glossary):
    groups = defaultdict(set)
    for row in glossary["examples"]:
        groups[row["role"]].add(" ".join(row["text"].split()))
    appearances = Counter(p for phrases in groups.values() for p in phrases)
    calibration, held = [], {}
    for role in ROLES:
        phrases = sorted((p for p in groups[role] if appearances[p] == 1), key=key)
        if len(phrases) < 2:
            raise ValueError("Each role needs disjoint calibration and test phrases")
        split = max(1, len(phrases) // 3)
        calibration.extend((phrase, role) for phrase in phrases[:split])
        held[role] = phrases[split:]
    return calibration, held


def panel(owners, labels, bits, held):
    groups = defaultdict(list)
    for roles in combinations(ROLES, 3):
        mask = bundles.role_mask(roles, bits)
        expected = bundles.reference(owners, labels, mask)
        cohort = ("empty" if not expected else "complementary"
                  if any(len(citations) > 1 for _, citations in expected) else "single")
        order = tuple(sorted(roles, key=lambda role: key((roles, role))))
        phrases = [min(held[role], key=lambda p: key((roles, p))) for role in order]
        request = "\n".join([HEADER, *(f"- {p}" for p in phrases)])
        groups[cohort].append(dict(id=key(roles).hex()[:20], roles=order, expected=expected,
                                  mask=mask, cohort=cohort, request=request))
    return sorted([row for rows in groups.values() for row in sorted(rows, key=lambda r: r["id"])[:40]],
                  key=lambda row: row["id"])


class Router:
    def __init__(self, glossary, encoder, tokenizer, model, banks, chat_reader):
        self.encoder, self.tokenizer, self.model, self.banks = encoder, tokenizer, model, banks
        self.definitions = [f"{role}: {glossary['definitions'][role]}" for role in ROLES]
        self.chat_reader = chat_reader
        labels = [tokenizer.encode(x, add_special_tokens=False) for x in SYMBOLS]
        if any(len(ids) != 1 for ids in labels):
            raise ValueError("Expected single-token role options")
        self.token_ids = [ids[0] for ids in labels]

    def dense(self, phrases, mode):
        scores = encode_tasks(self.encoder, phrases) @ self.banks[mode].T
        order = np.argsort(-scores, axis=1, kind="stable")
        ordered = np.take_along_axis(scores, order, axis=1)
        return [ROLES[i] for i in order[:, 0]], ordered[:, 0] - ordered[:, 1]

    @torch.inference_mode()
    def read_roles(self, phrases, shift):
        order = np.roll(np.arange(len(ROLES)), shift % len(ROLES))
        options = "\n".join(f"{tag}: {self.definitions[i]}" for tag, i in zip(SYMBOLS, order, strict=True))
        prompts = [f"{TASK} Choose the single best glossary option.\n{options}\n\nResearch task: {p}\nAnswer:"
                   for p in phrases]
        if self.chat_reader:
            prompts = [self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "Respond with only the letter of the best glossary option."},
                 {"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True) for text in prompts]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, self.token_ids]
        return [ROLES[order[i]] for i in logits.argmax(dim=1).cpu().tolist()]


def calibrate(router, pairs):
    phrases = [phrase for phrase, _ in pairs]
    ridge, gaps = router.dense(phrases, "ridge")
    reader = []
    for start in range(0, len(phrases), 6):
        reader.extend(router.read_roles(phrases[start:start+6], start // 6))
    trials = []
    for threshold in [0., *map(float, np.quantile(gaps, [.05, .10, .15, .20, .25]))]:
        fallback = gaps < threshold
        predictions = np.where(fallback, reader, ridge)
        trials.append(dict(threshold=threshold, fallback_tasks=int(fallback.sum()),
                           correct=sum(str(guess) == role for guess, (_, role) in zip(predictions, pairs, strict=True))))
    return dict(phrases=len(pairs), winner=max(trials, key=lambda r: (r["correct"], -r["fallback_tasks"])),
                grid=trials)


def evaluate(arm, case, index, router, threshold, owners, cache, paths, labels, bits):
    started = perf_counter()
    title, *lines = case["request"].splitlines()
    if title != HEADER or len(lines) != 3 or any(not line.startswith("- ") for line in lines):
        raise ValueError("Expected a three-task request")
    phrases = [line[2:] for line in lines]
    tasks = 0
    if arm == "known_roles":
        roles = case["roles"]
    elif arm == "reader":
        roles, tasks = router.read_roles(phrases, index), len(phrases)
    else:
        roles, gaps = router.dense(phrases, "definitions" if arm == "definition" else "ridge")
        if arm == "guarded":
            low = [i for i, gap in enumerate(gaps) if gap < threshold]
            if low:
                for i, role in zip(low, router.read_roles([phrases[i] for i in low], index), strict=True):
                    roles[i] = role
                tasks = len(low)
    routing_ms = (perf_counter() - started) * 1000
    unresolved = len(set(roles)) != 3
    decoded = {}

    def load(owner, source):
        if source not in decoded:
            decoded[source] = json.loads(paths[source].read_bytes())
        return decoded[source][owner]

    mask = bundles.role_mask(set(roles), bits)
    matches = [] if unresolved else bundles.rank(owners, cache, mask, load, "frontier")[0]
    response = json.dumps({"status": "unresolved" if unresolved else "completed", "matches": matches},
                          ensure_ascii=True, separators=(",", ":"))
    elapsed_ms = (perf_counter() - started) * 1000
    unsupported = 0
    for owner, sources in matches:
        declared = 0
        for source in sources:
            declared |= labels[owner, source]
        unsupported += declared & case["mask"] != case["mask"]
    return dict(response=response, ms=elapsed_ms, routing_ms=routing_ms, exact_parse=mask == case["mask"],
                exact=not unresolved and matches == case["expected"], unsupported=unsupported,
                unresolved=unresolved, returned=len(matches), reader_tasks=tasks, files=len(decoded))


def summarize(rows):
    scores = {}
    for arm in ARMS:
        first = [row["arms"][arm][0] for row in rows]
        scores[arm] = dict(
            cases=len(rows), exact_roles=sum(row["exact_parse"] for row in first),
            exact_ranked_bundles=sum(row["exact"] for row in first),
            unsupported_recommendations=sum(row["unsupported"] for row in first),
            queries_with_unsupported=sum(row["unsupported"] > 0 for row in first),
            unresolved=sum(row["unresolved"] for row in first), returned=sum(row["returned"] for row in first),
            reader_requests=sum(row["reader_tasks"] > 0 for row in first),
            file_loads=sum(row["files"] for row in first),
            p95_ms=float(np.quantile([median(v["ms"] for v in row["arms"][arm]) for row in rows], .95)),
            routing_p95_ms=float(np.quantile([median(v["routing_ms"] for v in row["arms"][arm]) for row in rows], .95)),
        )
    return scores


def run(fixture, glossary_file, training, embedding, reader, scratch, output, chat_reader=False):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Require a new output and Slurm CUDA")
    if bundles.digest(fixture) != bundles.FIXTURE_SHA256 or bundles.digest(glossary_file) != NISO_SHA256:
        raise ValueError("Changed benchmark input")
    sources = json.loads(fixture.read_bytes())["sources"]
    glossary = json.loads(glossary_file.read_bytes())
    owners, role_records = bundles.owner_data(sources)
    owners = [(owner, papers) for owner, papers in owners if len(papers) > 1]
    bits = {role: 1 << i for i, role in enumerate(ROLES)}
    labels = {record: bundles.role_mask(values, bits) for record, values in role_records.items()}
    caps = {row["doi"]: bundles.role_mask({r for a in row["authors"] for r in a["roles"]}, bits) for row in sources}
    cache = bundles.cached(owners, labels, caps, CACHE)
    calibration, held = phrase_split(glossary)
    cases = panel(owners, labels, bits, held)
    embedding_info = verify_checkpoint(embedding)
    if embedding_info["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected embedding checkpoint")
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    definitions = encoder.encode([glossary["definitions"][role] for role in ROLES])
    banks, trained = train_banks(training, SimpleNamespace(embedder=encoder, vectors=definitions), glossary["examples"])
    tokenizer, model, reader_info = load_reader(reader)
    tokenizer.padding_side = "left"
    router = Router(glossary, encoder, tokenizer, model, banks, chat_reader)
    tuned = calibrate(router, calibration)
    threshold = tuned["winner"]["threshold"]
    rows = []
    with TemporaryDirectory(prefix="credit-live-bundles-", dir=scratch) as temp:
        paths = {doi: Path(temp) / f"{i}.json" for i, doi in enumerate(sorted({d for _, p in owners for d in p}))}
        for doi, path in paths.items():
            path.write_text(json.dumps({owner: value for (owner, source), value in labels.items() if source == doi}))
        for arm in ARMS:
            evaluate(arm, cases[0], 0, router, threshold, owners, cache, paths, labels, bits)
        for i, case in enumerate(cases):
            result = {arm: [] for arm in ARMS}
            for repeat in range(2):
                shift = (i + repeat) % len(ARMS)
                for arm in ARMS[shift:] + ARMS[:shift]:
                    item = evaluate(arm, case, i, router, threshold, owners, cache, paths, labels, bits)
                    if arm == "known_roles" and (not item["exact"] or item["unsupported"]):
                        raise AssertionError("Known roles changed the exhaustive result")
                    result[arm].append(item)
            if any(values[0]["response"] != values[1]["response"] for values in result.values()):
                raise AssertionError("Warm repeats returned different results")
            rows.append(dict(id=case["id"], cohort=case["cohort"], arms=result))
            if len(rows) % 24 == 0:
                print("Completed", len(rows), flush=True)
    bundles.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=bundles.digest(Path(__file__)),
        fixture_sha256=bundles.digest(fixture), glossary_sha256=bundles.digest(glossary_file),
        training_sha256=bundles.digest(training), bundle_script_sha256=bundles.digest(Path(bundles.__file__)),
        embedding=embedding_info, reader=reader_info, tuning=tuned, trained=trained, author_cache_target=CACHE,
        chat_reader=chat_reader, held_phrases=sum(map(len, held.values())),
        report=summarize(rows), by_stratum={name: summarize([row for row in rows if row["cohort"] == name])
                                          for name in sorted({row["cohort"] for row in rows})}, rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("fixture", "glossary_file", "training", "embedding", "reader", "scratch", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--chat-reader", action="store_true")
    run(**vars(parser.parse_args()))
