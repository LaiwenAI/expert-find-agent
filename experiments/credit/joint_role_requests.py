"""Measure complete-request parsing and same-author attribution against oracle splits."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_language_router import NISO_SHA256, Router, TASK, load_fixture
from experiments.qasper.adaptive_evidence import digest
from experiments.qasper.paired_reader import load_reader
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


LABELS = "ABCDEFGHIJKLMNO"
METHODS = ("oracle_boundary_readers", "oracle_boundary_batch", "joint_full_glossary", "joint_pair_shortlist")
CONDITIONS = ("conjunction", "excluded_distractor")
REPEATS = 2


def stable(key, field):
    return sha256(f"joint-role-v1:{key}:{field}".encode()).digest()


def select_cases(papers, examples):
    grouped = defaultdict(list)
    for doi, available in papers.items():
        for pair in combinations(sorted(available), 2):
            grouped[pair].append(doi)
    cases = []
    for pair in sorted(grouped):
        doi = min(grouped[pair], key=lambda candidate: stable(pair, candidate))
        ordered = sorted(pair, key=lambda role: stable(doi + str(pair), role))
        requested = []
        key = doi + str(pair)
        for role in ordered:
            options = examples[role]
            selected = int.from_bytes(stable(key, role), "big") % len(options)
            requested.append(options[selected])
        foil = min(set(ROLES) - set(pair), key=lambda role: stable(key, "foil " + role))
        foil_options = examples[foil]
        excluded = foil_options[int.from_bytes(stable(key, foil), "big") % len(foil_options)]
        cases.append((doi, pair, requested, excluded))
    return cases


def make_request(parts, foil, condition):
    first, second = [part.rstrip(". ") for part in parts]
    if condition == "conjunction":
        return f"Which single author contributed to both of these activities: {first}; and {second}?"
    omitted = foil.rstrip(". ")
    return (
        f"I am looking for a single author whose contribution included {first}. "
        f"I do not need {omitted}. The same author must also have contributed to {second}."
    )


class JointReader:
    def __init__(self, router):
        self.router = router
        self.tokenizer = router.tokenizer
        self.model = router.reader
        encoded = [self.tokenizer.encode(symbol, add_special_tokens=False) for symbol in LABELS]
        if any(len(token) != 1 for token in encoded):
            raise ValueError("Expected one token per role symbol")
        self.ids = [token[0] for token in encoded]

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def inputs(self, ids):
        return torch.tensor([ids], dtype=torch.long, device="cuda")

    def best(self, output, options, excluded=None):
        logits = output.logits[0, -1, self.ids[:options]].float()
        if not torch.isfinite(logits).all().item():
            raise ValueError("Unusable decision scores")
        if excluded is not None:
            logits[excluded] = -torch.inf
        return int(logits.argmax().item())

    def full(self, query, rotation):
        options = list(range(len(ROLES)))
        shift = rotation % len(options)
        options = options[shift:] + options[:shift]
        glossary = "\n".join(f"{symbol}: {self.router.docs[i]}" for symbol, i in zip(LABELS, options))
        prompt = ("Map the two required tasks to two distinct contributor roles. "
                  "Ignore any activity the request excludes. Output the first required role, then the second.\n"
                  f"{glossary}\n\nRequest: {query}\nFirst role:")
        start = self.encode(prompt)
        first_output = self.model(input_ids=self.inputs(start), logits_to_keep=1, use_cache=True)
        first = self.best(first_output, len(options))
        tail = [self.ids[first], *self.encode("\nSecond role:")]
        second_output = self.model(input_ids=self.inputs(tail), past_key_values=first_output.past_key_values,
                                   logits_to_keep=1, use_cache=True)
        second = self.best(second_output, len(options), excluded=first)
        return [ROLES[options[first]], ROLES[options[second]]], len(start) + len(tail), list(ROLES)

    def shortlist(self, query, rotation):
        lexical, _ = self.router.lexical_rank(query)
        semantic = self.router.dense_rank(query)
        roles = sorted(set(lexical[:3] + semantic[:3]))
        pairs = list(combinations(roles, 2))
        shift = rotation % len(pairs)
        pairs = pairs[shift:] + pairs[:shift]
        glossary = "\n".join(self.router.docs[i] for i in roles)
        answer_pairs = "\n".join(f"{symbol}: {ROLES[a]} + {ROLES[b]}" for symbol, (a, b) in zip(LABELS, pairs))
        prompt = ("Select the pair of roles matching the two required tasks in the request. "
                  "Ignore any task the request excludes.\n"
                  f"Glossary:\n{glossary}\n\nOptions:\n{answer_pairs}\n\nRequest: {query}\nAnswer:")
        ids = self.encode(prompt)
        output = self.model(input_ids=self.inputs(ids), logits_to_keep=1, use_cache=False)
        pair = pairs[self.best(output, len(pairs))]
        return [ROLES[i] for i in pair], len(ids), [ROLES[i] for i in roles]

    def oracle_batch(self, original_atoms, rotation):
        vectors = self.router.embedder.encode([f"Instruct: {TASK}\nQuery: {atom}" for atom in original_atoms])
        prompts, choices = [], []
        for i, (atom, vector) in enumerate(zip(original_atoms, vectors, strict=True)):
            lexical, _ = self.router.lexical_rank(atom)
            dense = np.argsort(-(self.router.vectors @ vector), kind="stable").tolist()
            candidates = sorted(set(lexical[:3] + dense[:3]))
            shift = (rotation + i) % len(candidates)
            candidates = candidates[shift:] + candidates[:shift]
            glossary = "\n".join(f"{label}: {self.router.docs[k]}" for label, k in zip(LABELS, candidates))
            prompts.append(f"{TASK} Choose the single best option from the glossary.\n"
                           f"{glossary}\n\nResearch task: {atom}\nAnswer:")
            choices.append(candidates)
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        positions = (inputs.attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        ans = self.model(**inputs, position_ids=positions, logits_to_keep=1, use_cache=False)
        roles = [ROLES[candidates[int(ans.logits[i, -1, self.ids[:len(candidates)]].argmax().item())]]
                 for i, candidates in enumerate(choices)]
        return roles, int(inputs.attention_mask.sum().item()), [[ROLES[j] for j in option] for option in choices]

    def evaluate(self, text, original_atoms, method, index, declarations):
        start = perf_counter()
        if method == "oracle_boundary_readers":
            atoms = [self.router.run(atom, "shortlist_reader", index + i) for i, atom in enumerate(original_atoms)]
            roles = [atom["predicted"] for atom in atoms]
            tokens = sum(atom["reader_tokens"] for atom in atoms)
            candidates = [sorted(ROLES[i] for i in atom["candidates"]) for atom in atoms]
        elif method == "oracle_boundary_batch":
            roles, tokens, candidates = self.oracle_batch(original_atoms, index)
        else:
            operation = self.full if method == "joint_full_glossary" else self.shortlist
            roles, tokens, candidates = operation(text, index)
        predicted_authors = None if roles[0] == roles[1] else sorted(declarations[roles[0]] & declarations[roles[1]])
        return dict(roles=roles, authors=predicted_authors, reader_tokens=tokens, candidates=candidates,
                    ms=(perf_counter() - start) * 1000)


def score(rows):
    summary = {}
    for condition in CONDITIONS:
        cohort = [row for row in rows if row["condition"] == condition]
        details = {}
        for method in METHODS:
            outs = [row["arms"][method][0] for row in cohort]
            medians = [np.median([trial["ms"] for trial in row["arms"][method]]) for row in cohort]
            fields = {
                "exact_role_pair": sum(set(ans["roles"]) == set(row["roles"]) for ans, row in zip(outs, cohort)),
                "exact_author_set": sum(ans["authors"] == row["authors"] for ans, row in zip(outs, cohort)),
                "unsupported_author": sum(bool(set(ans["authors"] or ()) - set(row["authors"]))
                                          for ans, row in zip(outs, cohort)),
                "duplicate_roles_unresolved": sum(ans["authors"] is None for ans in outs),
                "p50_p95_ms": list(map(float, np.quantile(medians, [.5, .95]))),
                "mean_reader_tokens": mean(ans["reader_tokens"] for ans in outs),
            }
            if method == "joint_pair_shortlist":
                fields["both_roles_in_shortlist"] = sum(
                    set(row["roles"]) <= set(ans["candidates"]) for ans, row in zip(outs, cohort))
            details[method] = fields
        summary[condition] = {"requests": len(cohort), "arms": details}
    return summary


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.examples) != NISO_SHA256:
        raise ValueError("Changed NISO release")
    data = json.loads(args.examples.read_bytes())
    if tuple(data["definitions"]) != ROLES:
        raise ValueError("Unexpected glossary")
    docs = [f"{role}: {data['definitions'][role]}" for role in ROLES]
    example_groups = defaultdict(list)
    for row in data["examples"]:
        example_groups[row["role"]].append(row["text"])
    papers, declarations, fixture_hash = load_fixture(args.fixture, args.sources)
    trials = select_cases(papers, example_groups)
    embed_provenance = json.loads((args.embedding / "checkpoint-provenance.json").read_bytes())
    for name, expect in embed_provenance["files"].items():
        if digest(args.embedding / name) != expect:
            raise ValueError(f"Changed embedding file: {name}")
    embedder = QwenTorchEmbedding(args.embedding)
    embedder.model.config.use_cache = False
    vectors = embedder.encode(docs)
    tokenizer, reader, checkpoint = load_reader(args.reader)
    with TemporaryDirectory() as folder:
        lexical = build_lexical({text: i for i, text in enumerate(docs)}, Path(folder))
    model = JointReader(Router(docs, lexical, vectors, embedder, reader, tokenizer, 3))

    rows = []
    with torch.inference_mode():
        for i, (doi, roles, atoms, foil) in enumerate(trials):
            actual = sorted(papers[doi][roles[0]] & papers[doi][roles[1]])
            for variation, condition in enumerate(CONDITIONS):
                request = make_request(atoms, foil, condition)
                results = defaultdict(list)
                if i == 0 and variation == 0:
                    for method in METHODS:
                        model.evaluate(request, atoms, method, i, declarations[doi])
                for repeat in range(REPEATS):
                    offset = (i + variation + repeat) % len(METHODS)
                    for method in METHODS[offset:] + METHODS[:offset]:
                        result = model.evaluate(request, atoms, method, i + args.option_offset, declarations[doi])
                        results[method].append(result)
                if any(len({tuple(trial["roles"]) for trial in runs}) != 1 for runs in results.values()):
                    raise ValueError("Unstable repeated role mapping")
                rows.append(dict(doi=doi, roles=roles, authors=actual, condition=condition, arms=results))
            if (i + 1) % 20 == 0:
                print("Completed role pairs", i + 1, "/", len(trials), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    result = args.output / "requests.json"
    result.write_text(json.dumps(rows, indent=2) + "\n")
    summary = dict(cases=len(trials), repeat_per_arm=REPEATS, gpu=torch.cuda.get_device_name(),
                   checkpoint=checkpoint, embedding=embed_provenance, example_sha256=NISO_SHA256,
                   fixture_sha256=fixture_hash, rows_sha256=digest(result), metrics=score(rows),
                   option_offset=args.option_offset)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("benchmarks/plos-credit-pilot-v1"))
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    parser.add_argument("--reader", type=Path, default=Path("artifacts/models/qwen3-4b-instruct-2507"))
    parser.add_argument("--option-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
