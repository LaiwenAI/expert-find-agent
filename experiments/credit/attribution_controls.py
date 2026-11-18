"""Test how author-scoped evidence and missing declarations affect attribution."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit.plos_role_inventory import declared_roles
from experiments.qasper.adaptive_evidence import digest
from experiments.qasper.paired_reader import load_reader


LABELS = ("A", "B", "C")
MODES = ("all_authors", "author_scoped", "candidate_redacted", "byline_only")
AVAILABLE = {"all_authors", "author_scoped"}
MEANINGS = (
    "the target author's supplied declaration lists the requested role.",
    "the target author's supplied complete declaration lists other roles but omits the requested role.",
    "the target author's declaration is unavailable.",
)
EVIDENCE_INSTRUCTION = (
    "Use only declarations tied to the target author. Never infer a role from the abstract, "
    "the byline, author order, or a different author's declaration.\n\n"
)
BATCH_SIZE = 16


@dataclass(frozen=True)
class Paper:
    abstract: str
    roles: tuple[tuple[str, ...], ...]


def read_inputs(fixture, sources):
    manifest = json.loads((fixture / "manifest.json").read_bytes())
    input_path = fixture / "queries.json"
    if digest(input_path) != manifest["rows_sha256"]:
        raise ValueError("Changed contributor benchmark")
    papers = {}
    for record in manifest["sources"]:
        doi = record["doi"]
        path = sources / f"pone.{doi.split('.')[-1]}.xml"
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != record["source_sha256"]:
            raise ValueError(f"Changed source: {doi}")
        root = ET.fromstring(raw)
        meta = root.find("./front/article-meta")
        abstracts = meta.findall("./abstract")
        abstract = "\n".join(" ".join("".join(p.itertext()).split()) for a in abstracts for p in a.findall(".//p"))
        roles = tuple(declared_roles(raw, doi))
        papers[doi] = Paper(abstract, roles)
    return json.loads(input_path.read_bytes()), papers, manifest["rows_sha256"]


def case_inventory(queries):
    cases = []
    seen = set()
    for row in queries:
        relevant = set(row["matching_author_ordinals"])
        others = set(row["other_author_ordinals"])
        if not relevant or not others or relevant & others:
            raise ValueError("Invalid contrastive contributor query")
        for author in sorted(relevant | others):
            key = (row["doi"], row["requested_role"], author)
            if key in seen:
                raise ValueError("Duplicate contributor case")
            seen.add(key)
            cases.append((row["doi"], row["publication_title"], row["requested_role"], author,
                          "A" if author in relevant else "B"))
    return cases


def prompt(title, role, author, abstract, roles, mode, meanings):
    visible = []
    for ordinal, listed in enumerate(roles, 1):
        if mode == "byline_only" or (mode == "candidate_redacted" and ordinal == author):
            continue
        if mode == "author_scoped" and ordinal != author:
            continue
        visible.append(f"Complete declaration for Author {ordinal}: {', '.join(listed)}")
    return (
        "Determine what the supplied publication declares about the requested contributor role.\n"
        + "".join(f"{letter}: {MEANINGS[LABELS.index(meaning)]}\n"
                  for letter, meaning in zip(LABELS, meanings, strict=True))
        + f"{EVIDENCE_INSTRUCTION}Title: {title}\nAbstract: {abstract}\n"
        f"Byline: {', '.join(f'Author {i}' for i in range(1, len(roles) + 1))}\n"
        f"Target: Author {author}\nRequested role: {role}\n"
        + "\n".join(visible) + "\n\nAnswer:"
    )


class Reader:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.tokenizer.padding_side = "left"
        symbol_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LABELS]
        if any(len(ids) != 1 for ids in symbol_ids):
            raise ValueError("Attribution labels are not one token")
        self.label_ids = [ids[0] for ids in symbol_ids]

    def classify(self, texts):
        start = perf_counter()
        encoded = self.tokenizer(texts, padding=True, truncation=False, add_special_tokens=False,
                                 return_tensors="pt").to("cuda")
        if encoded.input_ids.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("Attribution context overflow")
        positions = (encoded.attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.inference_mode():
            output = self.model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1)
            scores = output.logits[:, -1, self.label_ids].float()
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite attribution logits")
            decisions = scores.argmax(dim=1).cpu().tolist()
            lengths = encoded.attention_mask.sum(dim=1).cpu().tolist()
        return [(LABELS[x], n) for x, n in zip(decisions, lengths, strict=True)], perf_counter() - start


def structured(roles, author, role, mode):
    if mode not in AVAILABLE:
        return "C"
    return "A" if role in roles[author - 1] else "B"


def summarize(rows):
    summary = {}
    query_count = len({(row["doi"], row["role"]) for row in rows})
    for mode in MODES:
        expected = [row["declared"] if mode in AVAILABLE else "C" for row in rows]
        for arm in ("byline", "structure", "reader"):
            choices = [row["arms"][mode][arm] for row in rows]
            correct = [a == b for a, b in zip(choices, expected, strict=True)]
            incorrect_sets = {
                (row["doi"], row["role"])
                for row, predicted, gold in zip(rows, choices, expected, strict=True)
                if (predicted == "A") != (gold == "A")
            }
            summary[f"{mode}/{arm}"] = dict(
                accuracy=mean(correct),
                exact_positive_sets=query_count - len(incorrect_sets),
                role_queries=query_count,
                output_counts=dict(Counter(choices)),
                false_promotions=sum(g != "A" and pred == "A" for pred, g in zip(choices, expected, strict=True)),
                promotion_denominator=sum(label != "A" for label in expected),
                false_role_exclusions=sum(g == "A" and pred == "B" for pred, g in zip(choices, expected, strict=True)),
                missed_declared_roles=sum(g == "A" and pred != "A" for pred, g in zip(choices, expected, strict=True)),
                claimed_absence_on_missing_record=sum(g == "C" and pred == "B" for pred, g in zip(choices, expected, strict=True)),
            )
    summary["scoped_minus_all_reader_accuracy"] = clustered_paired_interval([
        int(row["arms"]["author_scoped"]["reader"] == row["declared"])
        - int(row["arms"]["all_authors"]["reader"] == row["declared"]) for row in rows
    ], [row["doi"] for row in rows])
    return summary


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    queries, papers, fixture_hash = read_inputs(args.fixture, args.sources)
    cases = case_inventory(queries)
    tokenizer, model, checkpoint = load_reader(args.model)
    reader = Reader(tokenizer, model)
    reader.classify(["A single warm-up input. Answer:"])

    rows = [{"doi": doi, "role": role, "author_ordinal": author, "declared": answer, "arms": {}}
            for doi, _, role, author, answer in cases]
    timing = {mode: [] for mode in MODES}
    for start in range(0, len(cases), BATCH_SIZE):
        group = cases[start:start + BATCH_SIZE]
        shifts = [i % len(LABELS) if args.rotate_labels else 0 for i in range(start, start + len(group))]
        mappings = [LABELS[shift:] + LABELS[:shift] for shift in shifts]
        rotate = (start // BATCH_SIZE) % len(MODES)
        for mode in MODES[rotate:] + MODES[:rotate]:
            prompt_rows = [prompt(title, role, author, papers[doi].abstract, papers[doi].roles, mode, mapping)
                           for (doi, title, role, author, _), mapping in zip(group, mappings, strict=True)]
            estimates, seconds = reader.classify(prompt_rows)
            timing[mode].append(seconds)
            for i, (item, (est, tokens), mapping) in enumerate(zip(group, estimates, mappings, strict=True), start):
                doi, _, role, author, answer = item
                direct = structured(papers[doi].roles, author, role, mode)
                if mode in AVAILABLE and direct != answer:
                    raise ValueError("Fixture and publisher declaration disagree")
                rows[i]["arms"][mode] = dict(byline="A", structure=direct, reader=mapping[LABELS.index(est)],
                                              input_tokens=tokens)
        if start % (10 * BATCH_SIZE) == 0:
            print("Attribution cases", min(len(cases), start + BATCH_SIZE), "/", len(cases), flush=True)
    summary = dict(
        questions=len(queries), checks=len(rows), papers=len({row["doi"] for row in rows}),
        queries_sha256=fixture_hash, checkpoint=checkpoint, gpu=torch.cuda.get_device_name(),
        batch_size=BATCH_SIZE, rotate_labels=args.rotate_labels, metrics=summarize(rows),
        typed_cost={mode: {
            "input_tokens_mean": mean(row["arms"][mode]["input_tokens"] for row in rows),
            "classify_seconds_total": sum(timing[mode]),
            "batch_p95_seconds": float(np.quantile(timing[mode][:-1], .95)),
        } for mode in MODES},
    )
    args.output.mkdir(parents=True, exist_ok=False)
    result = args.output / "rows.json"
    result.write_text(json.dumps(rows, indent=2) + "\n")
    summary["rows_sha256"] = digest(result)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("benchmarks/plos-credit-pilot-v1"))
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-4b-instruct-2507"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotate-labels", action="store_true")
    run(parser.parse_args())
