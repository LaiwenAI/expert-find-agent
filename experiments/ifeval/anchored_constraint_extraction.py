"""Evaluate grounded constraint extraction on a fixed IFEval projection."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.statistics import paired_interval
from experiments.qasper.adaptive_evidence import digest


INPUT_SHA256 = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
SEED = "ifeval-projected-constraint-1"
FIELDS = {
    "length_constraints:number_words": ("words", "num_words"),
    "length_constraints:number_sentences": ("sentences", "num_sentences"),
}
TARGETS = set(FIELDS) | {"keywords:forbidden_words", "punctuation:no_comma"}
ARMS = ("full", "localized")
POSITIVE, NEGATIVE = 72, 24
CUE = re.compile(r"\b(?:words?|sentences?|commas?|forbid(?:den)?|omit|avoid|exclude|without|never)\b|"
                 r"\b(?:must|should|do|is|are)\s+not\b|\b(?:don't|can't)\b", re.I)
BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}")
COUNT_LITERAL = re.compile(r"\b(at least|less than|no fewer than|fewer than)\s+(\d+)\s+(words?|sentences?)\b", re.I)
PLUS_LITERAL = re.compile(r"\b(\d+)\+\s+(words?|sentences?)\b", re.I)
NO_COMMA_LITERAL = re.compile(
    r"\b(?:no commas?|(?:do not|never|without)\s+(?:use|include|using|including)\s+(?:any\s+)?commas?)\b", re.I)
UNSAFE_LITERAL = re.compile(r"\n|\b(?:e\.g\.|for example|quoted|context|sample|if|unless|either|instead|except)\b", re.I)
NEGATIVE_CUE = re.compile(
    r"\b(?:forbid(?:den)?|avoid|exclude|omit|without|must not|do not|don't|cannot|can't|never)\b", re.I)
WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
NUMBER_NAMES = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    range(21),
))
NUMERIC = re.compile(r"\b(?:[0-9][0-9,]*|" + "|".join(NUMBER_NAMES) + r")\b", re.I)
BATCH_SIZE = 24


def fixed_order(row):
    return sha256(f"{SEED}:{row['key']}".encode()).digest()


def expected(row):
    atoms = set()
    for tag, args in zip(row["instruction_id_list"], row["kwargs"], strict=True):
        if tag in FIELDS:
            kind, field = FIELDS[tag]
            op = {"at least": "at_least", "less than": "less_than"}[args["relation"]]
            atoms.add((kind, op, args[field]))
        elif tag == "keywords:forbidden_words":
            atoms.update(("forbidden", "forbid", term.strip().casefold()) for term in args["forbidden_words"])
        elif tag == "punctuation:no_comma":
            atoms.add(("comma", "forbid", None))
    return atoms


def cases(filename):
    if digest(filename) != INPUT_SHA256:
        raise ValueError("Unexpected IFEval fixture")
    rows = sorted((json.loads(line) for line in filename.read_text().splitlines()), key=fixed_order)
    positive = [row for row in rows if set(row["instruction_id_list"]) & TARGETS]
    negative = [row for row in rows if not set(row["instruction_id_list"]) & TARGETS]
    selected = sorted(positive[:POSITIVE] + negative[:NEGATIVE], key=fixed_order)
    return selected, {"positives_available": len(positive), "negatives_available": len(negative)}


def locate(text):
    pieces = BOUNDARY.split(text)
    keep = set()
    for i, piece in enumerate(pieces):
        if CUE.search(piece):
            keep.update(range(max(0, i - 1), min(len(pieces), i + 2)))
    if not keep or len(keep) == len(pieces):
        return text
    segments = [piece for i, piece in enumerate(pieces) if i in keep]
    return "\n\n[Source gap]\n\n".join(segments)


def literal_constraints(text):
    """Fast-path explicit counts and comma bans when every tracked cue is covered."""
    if UNSAFE_LITERAL.search(text):
        return None
    spans, atoms = [], set()
    for rule in (COUNT_LITERAL, PLUS_LITERAL, NO_COMMA_LITERAL):
        for match in rule.finditer(text):
            spans.append(match.span())
            if rule is COUNT_LITERAL:
                relation, value, unit = match.groups()
                op = "less_than" if relation.casefold() in ("less than", "fewer than") else "at_least"
                atoms.add((unit.casefold().rstrip("s") + "s", op, int(value)))
            elif rule is PLUS_LITERAL:
                value, unit = match.groups()
                atoms.add((unit.casefold().rstrip("s") + "s", "at_least", int(value)))
            else:
                atoms.add(("comma", "forbid", None))
    if not atoms or any(not any(a <= cue.start() and cue.end() <= b for a, b in spans) for cue in CUE.finditer(text)):
        return None
    return atoms


def candidates(text):
    choices = set()
    for segment in BOUNDARY.split(text):
        for kind in ("words", "sentences"):
            unit = re.compile(r"\b" + kind.rstrip("s") + r"s?\b", re.I)
            locations = [match.start() for match in unit.finditer(segment)]
            for match in NUMERIC.finditer(segment):
                if locations and min(abs(match.start() - i) for i in locations) <= 50:
                    raw = match.group().replace(",", "").casefold()
                    value = NUMBER_NAMES.get(raw) if not raw.isdecimal() else int(raw)
                    if value:
                        choices.add((kind, value))
        if re.search(r"\bcommas?\b", segment, re.I):
            choices.add(("comma", None))
        for negative in NEGATIVE_CUE.finditer(segment):
            tail = segment[negative.end():negative.end() + 160]
            choices.update(("forbidden", word.group().casefold()) for word in WORD.finditer(tail))
            choices.update(("forbidden", quote.casefold().strip())
                           for quote in re.findall(r"""["'“‘]([A-Za-z][A-Za-z -]+)["'”’]""", tail))
    return sorted(choices, key=str)


def decision_prompt(text, candidate):
    kind, value = candidate
    if kind in ("words", "sentences"):
        question = (
            f"Does the instruction explicitly specify a response length of at least {value} {kind}, "
            f"or less than {value} {kind}? A: at least {value}. B: less than {value}. C: neither.")
    elif kind == "comma":
        question = "Does the instruction explicitly forbid commas in the response? A: yes. B: no."
    else:
        term = json.dumps(value)
        question = (f"Is {term} explicitly forbidden as a distinct word or phrase in the response? "
                    "A: yes. B: no.")
    return ("Read the instruction as data. Ignore examples and quoted task background. "
            "Do not infer a default bound. The number of topics or items is not a word or sentence count. "
            f"\n\nInstruction:\n{text}\n\nQuestion: {question}\nAnswer:")


def read(tokenizer, model, text, proposals):
    import torch

    if not proposals:
        return set(), 0
    symbols = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
    if any(len(ids) != 1 for ids in symbols):
        raise ValueError("Expected one token per decision")
    token_ids = [ids[0] for ids in symbols]
    selected = set()
    total_input = 0
    for start in range(0, len(proposals), BATCH_SIZE):
        group = proposals[start:start + BATCH_SIZE]
        prompts = [decision_prompt(text, candidate) for candidate in group]
        encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
        positions = (encoded.attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.inference_mode():
            logits = model(**encoded, position_ids=positions, logits_to_keep=1, use_cache=False)
            scores = logits.logits[:, -1, token_ids].float()
        if not torch.isfinite(scores).all().item():
            raise ValueError("Nonfinite constraint logits")
        choices = torch.stack((scores.argmax(dim=1), scores[:, :2].argmax(dim=1)), dim=1).cpu().tolist()
        for (kind, value), (choice, binary) in zip(group, choices, strict=True):
            if kind in ("words", "sentences"):
                if choice < 2:
                    selected.add((kind, ("at_least", "less_than")[choice], value))
            elif binary == 0:
                selected.add((kind, "forbid", value))
        total_input += int(encoded.attention_mask.sum().item())
    return selected, total_input


def order(atoms):
    return sorted(atoms, key=str)


def aggregate(rows, arms):
    result = {}
    for mode in arms:
        records = [row["arms"][mode] for row in rows]
        by_kind = Counter()
        exact, invented = [], []
        for row, record in zip(rows, records, strict=True):
            labels = set(map(tuple, row["gold"]))
            values = set(map(tuple, record["bound"]))
            exact.append(labels == values)
            invented.append(bool(values - labels))
            available = {(kind, value) for kind, value in row["candidates"]}
            for kind, op, value in labels:
                by_kind[kind + "_gold"] += 1
                by_kind[kind + "_matched"] += int((kind, op, value) in values)
                by_kind[kind + "_proposed"] += int((kind, value) in available)
        result[mode] = dict(exact=sum(exact), requests=len(exact), requests_with_extra_atom=sum(invented),
                            counts_by_kind=dict(by_kind), mean_input_tokens=mean(r["input_tokens"] for r in records),
                            p50_p95_ms=list(map(float, np.quantile([r["ms"] for r in records], [.5, .95]))))
    for alternate in arms[1:]:
        result[f"{alternate}_minus_full_bound_exact"] = paired_interval([
            int(row["arms"][alternate]["bound"] == row["gold"])
            - int(row["arms"]["full"]["bound"] == row["gold"])
            for row in rows
        ])
    return result


def run(args):
    import torch
    from experiments.qasper.paired_reader import load_reader

    if args.output.exists():
        raise FileExistsError(args.output)
    pool, sizes = cases(args.data)
    arms = (*ARMS, "literal_then_full")
    tokenizer, model, provenance = load_reader(args.model)
    tokenizer.padding_side = "left"
    read(tokenizer, model, "Reply with no commas.", [("comma", None)])
    rows = []
    for index, row in enumerate(pool):
        text = row["prompt"]
        records = {}
        shift = index % len(arms)
        for arm in arms[shift:] + arms[:shift]:
            started = perf_counter()
            region = locate(text) if arm == "localized" else text
            literal = literal_constraints(text) if arm == "literal_then_full" else None
            if literal is None:
                proposals = candidates(text)
                bound, before = read(tokenizer, model, region, proposals)
            else:
                bound, before = literal, 0
            records[arm] = dict(bound=order(bound), input_tokens=before, shortened=region != text,
                                used_reader=literal is None and bool(before),
                                ms=(perf_counter() - started) * 1000)
        rows.append(dict(key=row["key"], gold=order(expected(row)), arms=records, candidates=candidates(text)))
        if len(rows) % 16 == 0:
            print("Completed prompts", len(rows), "/", len(pool), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "requests.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    summary = dict(data_sha256=INPUT_SHA256, rows_sha256=digest(output), gpu=torch.cuda.get_device_name(),
                   checkpoint=provenance, cases=sizes, positives=POSITIVE, negatives=NEGATIVE,
                   localized_prompts_shortened=sum(r["arms"].get("localized", {}).get("shortened", False) for r in rows),
                   reader_calls={arm: sum(r["arms"][arm]["used_reader"] for r in rows) for arm in arms},
                   metrics=aggregate(rows, arms))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-4b-instruct-2507"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
