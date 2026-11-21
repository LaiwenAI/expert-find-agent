"""Train a small offset tagger and test provenance-preserving first-query retrieval."""

import argparse
from collections import Counter, defaultdict
from itertools import groupby
import json
import os
from hashlib import sha256
from pathlib import Path
import random
import re
from string import Formatter
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.budgeted_source_role_service import Service
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.engineering_cross_source_search import phrase_bank
from experiments.credit.engineering_one_role_recheck import RoleReader
from experiments.credit.engineering_role_banks import projections, queries
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_role_offset_lattice import best_paths
from experiments.credit.unquoted_budgeted_source_role import TEMPLATES, rule_parse, sample_families
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


TOKEN = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")
LABELS = ("a", "x", "b", "y")
TRAIN_FORMS = (
    "Please suggest an expert. For research on {a}, they must have handled {x}. For an article on {b}, look for evidence of {y}.",
    "For the investigation into {a}, we need evidence of {x}; for the investigation into {b}, evidence of {y}. Both should be by one author.",
    "I need a common author with documented experience of {x} in a publication on {a}, and of {y} in a publication on {b}.",
    "Who did {x} for the study of {a} and took responsibility for {y} in another study of {b}?",
    "Match a contributor across the papers dealing with {a} and {b}. The first needs {x}; the second needs {y}.",
    "Consider a manuscript on {a} and another on {b}. In the first manuscript look for {x}; in the second look for {y}.",
    "Start from two articles on {a} and {b}. Find someone who performed {y} in the second and {x} in the first.",
    "Two publications address {a} and {b}, respectively. In reverse order, their qualifying activities are {y} and {x}. Identify a shared author.",
    "I need someone involved in {x} and {y}, for work on {a} and for work on {b}, respectively.",
    "Those who carried out {x} in the manuscript discussing {a} should also be associated with {y} in the manuscript discussing {b}.",
    "Paper A covers {a}. Paper B covers {b}. Look for {y} under B and {x} under A, with a single contributor.",
    "Find a researcher with these credits: {a} — {x}; {b} — {y}. Two separate articles are required.",
    "Can you check whether an author's responsibility was {x} for their article about {a} and {y} for their article about {b}?",
    "Identify one person who led or helped with {x} during a project focusing on {a}, and who did {y} for a project that investigated {b}.",
    "If a scientist took part in {x} in research addressing {a}, see whether that scientist also had a role in {y} in work related to {b}.",
    "Give me somebody whose two publications had different subjects: {a} for the one where they did {x} and {b} for the one where they did {y}.",
)
CALIBRATION_FORMS = (
    "Find a contributor involved in {x} for research examining {a}; also verify {y} in a second article examining {b}.",
    "One study focuses on {a}; the other focuses on {b}. I need {y} for study two and {x} for study one, from the same person.",
    "Check for an article centered on {a} where an author contributed through {x} and a distinct article centered on {b} where that author contributed through {y}.",
)
OUTSIDE_FORMS = (
    "Please find a scientist who performed {x} when researching {a}.",
    "Suggest an expert in {a}; work on {b} would also be useful.",
    "Search for information comparing papers about {a} and {b}.",
)


def render(template, values):
    text, spans, offset = [], {}, 0
    for literal, field, _, _ in Formatter().parse(template):
        text.append(literal)
        offset += len(literal)
        if field:
            value = values[field]
            spans[field] = (offset, offset + len(value))
            text.append(value)
            offset += len(value)
    return "".join(text), spans


def synthetic(pool, activities, templates, count, seed, *, outside=False):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        a, b = rng.sample(pool, 2)
        x, y = rng.sample(activities, 2)
        choices = OUTSIDE_FORMS if outside and rng.random() < .25 else templates
        rows.append(render(rng.choice(choices), dict(a=a, b=b, x=x, y=y)))
    return rows


def tensors(rows, vocab):
    tokens = [list(TOKEN.finditer(text)) for text, _ in rows]
    width = max(map(len, tokens))
    ids = torch.zeros(len(tokens), width, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, (matches, (_, spans)) in enumerate(zip(tokens, rows, strict=True)):
        for j, match in enumerate(matches):
            ids[i, j] = vocab.get(match[0].casefold(), 1)
            labels[i, j] = next((k for k, field in enumerate(LABELS, 1) if field in spans and
                                 spans[field][0] <= match.start() < match.end() <= spans[field][1]), 0)
    return ids, labels


class Tagger(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.words = nn.Embedding(size, 48, padding_idx=0)
        self.memory = nn.GRU(48, 64, batch_first=True, bidirectional=True)
        self.readout = nn.Linear(128, len(LABELS) + 1)

    def forward(self, ids):
        if self.training:
            replace = (ids > 1) & (torch.rand(ids.shape, device=ids.device) < .25)
            ids = ids.masked_fill(replace, 1)
        # Packing gives the same right context during batched training and single-query inference.
        lengths = (ids != 0).sum(dim=1).cpu()
        sequence = nn.utils.rnn.pack_padded_sequence(self.words(ids), lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = self.memory(sequence)
        vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=ids.shape[1])
        return self.readout(vectors)


def fit(rows, seed):
    torch.manual_seed(seed)
    lexicon = sorted({m[0].casefold() for text, _ in rows for m in TOKEN.finditer(text)})
    vocab = {word: i for i, word in enumerate(lexicon, 2)}
    model = Tagger(len(vocab) + 2).cuda()
    ids, labels = (part.cuda() for part in tensors(rows, vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    losses = []
    for _ in range(14):
        total = 0.
        for batch in torch.randperm(len(ids), device="cuda").split(64):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(ids[batch]).transpose(1, 2)
            loss = F.cross_entropy(predictions, labels[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(ids))
    return model.eval().cpu(), vocab, losses


@torch.inference_mode()
def extract(model, vocab, text):
    matches = list(TOKEN.finditer(text))
    if not matches or len(matches) > 160:
        return (), 0.
    ids = torch.tensor([[vocab.get(m[0].casefold(), 1) for m in matches]])
    return decode(model(ids)[0], [match.span() for match in matches], text)


def decode(logits, windows, text):
    scores, tags = logits.softmax(-1).max(-1)
    tag_ids = tags.tolist()
    spans = decode_spans(tag_ids, windows, text)
    confidence = min((score for score, label in zip(scores.tolist(), tag_ids, strict=True) if label), default=0.)
    return spans, confidence if spans else 0.


def decode_spans(tags, windows, text):
    spans = []
    for label in range(1, len(LABELS) + 1):
        indices = [i for i, category in enumerate(tags) if category == label]
        if not indices or len(indices) != indices[-1] - indices[0] + 1:
            return ()
        positions = [windows[i] for i in indices if windows[i][0] < windows[i][1]]
        if not positions:
            return ()
        spans.append((positions[0][0], positions[-1][1]))
    ordered = sorted(spans)
    if spans[0][0] >= spans[2][0] or any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        return ()
    a, x, b, y = (text[start:end] for start, end in spans)
    return ((a, x), (b, y))


def trim_windows(text, offsets):
    spans = []
    for start, end in offsets:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        spans.append((start, end))
    return spans


class ContextTagger(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 48)
        self.memory = nn.GRU(48, 64, batch_first=True, bidirectional=True)
        self.readout = nn.Linear(128, len(LABELS) + 1)

    def forward(self, hidden, lengths):
        sequence = nn.utils.rnn.pack_padded_sequence(self.project(hidden.float()), lengths, batch_first=True,
                                                     enforce_sorted=False)
        outputs, _ = self.memory(sequence)
        vectors, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=hidden.shape[1])
        return self.readout(vectors)


@torch.no_grad()
def contextual_tokens(encoder, texts):
    tokenizer = encoder.tokenizer
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        batch = tokenizer(texts, padding=True, add_special_tokens=False, return_offsets_mapping=True,
                          return_tensors="pt")
    finally:
        tokenizer.padding_side = padding_side
    offsets = batch.pop("offset_mapping").tolist()
    windows = [trim_windows(text, spans) for text, spans in zip(texts, offsets, strict=True)]
    lengths = batch["attention_mask"].sum(dim=1)
    vectors = encoder.model(**batch.to("cuda")).last_hidden_state
    return vectors, lengths, windows


def fit_context(encoder, rows, seed):
    torch.manual_seed(seed)
    blocks = []
    for start in range(0, len(rows), 32):
        block = rows[start:start + 32]
        vectors, lengths, windows = contextual_tokens(encoder, [text for text, _ in block])
        labels = torch.full(vectors.shape[:2], -100, dtype=torch.long)
        for i, (_, fields) in enumerate(block):
            for j, (begin, end) in enumerate(windows[i][:lengths[i]]):
                labels[i, j] = next((k for k, field in enumerate(LABELS, 1) if field in fields and
                                    fields[field][0] <= begin < end <= fields[field][1]), 0)
        blocks.append((vectors, lengths, labels.cuda()))
    head = ContextTagger(blocks[0][0].shape[-1]).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    losses = []
    for _ in range(14):
        total = 0.
        for index in torch.randperm(len(blocks)).tolist():
            hidden, lengths, labels = blocks[index]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(hidden, lengths).transpose(1, 2), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        losses.append(total / len(rows))
    return head.eval(), losses


@torch.inference_mode()
def context_logits(head, encoder, text):
    vectors, lengths, windows = contextual_tokens(encoder, [text])
    return head(vectors, lengths)[0], windows[0]


def extract_context(head, encoder, text):
    logits, windows = context_logits(head, encoder, text)
    return decode(logits, windows, text)


@torch.inference_mode()
def joint_parse(head, encoder, text, partner=None):
    start = perf_counter()
    vectors, lengths, windows = contextual_tokens(encoder, [text])
    windows = windows[0]
    values = head(vectors, lengths)[0].float().cpu()
    encoder_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    greedy, _ = decode(values, windows, text)
    greedy_ms = encoder_ms + (perf_counter() - start) * 1000
    start = perf_counter()
    path, margins = best_paths(values.numpy())
    structured = decode_spans(path, windows, text) if path else ()
    if not structured:
        margins = {}
    structure_ms = encoder_ms + (perf_counter() - start) * 1000
    agreement, agreement_ms = (), structure_ms
    if partner is not None and structured:
        start = perf_counter()
        other_values = partner(vectors, lengths)[0].float().cpu().numpy()
        other_path, _ = best_paths(other_values)
        other = decode_spans(other_path, windows, text) if other_path else ()
        if other == structured:
            agreement = structured
        agreement_ms += (perf_counter() - start) * 1000
    return greedy, greedy_ms, structured, margins, structure_ms, agreement, agreement_ms


def expected_pairs(text, offsets):
    if set(offsets) != set(LABELS):
        return ()
    a, x, b, y = (text[slice(*offsets[field])] for field in LABELS)
    return ((a, x), (b, y))


def calibrate(parse, rows):
    observations = [(parse(text), expected_pairs(text, spans)) for text, spans in rows]
    stats = {}
    for threshold in (0., .25, .5, .65, .8, .9, .95, 1.):
        accepted = [(pred, gold) for (pred, probability), gold in observations if pred and probability >= threshold]
        stats[threshold] = dict(accepted=len(accepted), exact=sum(p == g for p, g in accepted),
                                outside=sum(not g for _, g in accepted))
    # A fixed selection rule uses only development sources and templates.
    eligible = [t for t, row in stats.items() if row["outside"] == 0 and row["exact"] >= .90 * row["accepted"]]
    return min(eligible), stats


def score_search(service, parsed, row):
    if not parsed:
        return evidence(dict(papers=None, owners=[], roles=[], calls=0, source_calls=0, role_calls=0), row)
    request = SimpleNamespace(payload=json.dumps({"requirements": [
        dict(topic=topic, role=role) for topic, role in parsed]}))
    return evidence(json.loads(service.search(request, "screened_one_call")), row)


def evaluate(service, test_rows, wording, parse, threshold, reverse):
    bank = phrase_bank(wording)
    rows = test_rows[::-1] if reverse else test_rows
    counts, times = defaultdict(Counter), defaultdict(list)
    for i, row in enumerate([rows[0], *rows]):
        a, b = [service.index.keywords(service.index.articles[p]["clue"]) for p in row["papers"]]
        x, y = [bank[r][int(row["family"][:16], 16) % len(bank[r])].strip().rstrip(".?!") for r in row["roles"]]
        expected = ((a, x), (b, y))
        for style, template in TEMPLATES.items():
            text = template.format(a=a, b=b, x=x, y=y)
            start = perf_counter()
            rules = rule_parse(text)
            rule_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            parsed, confidence = parse(text)
            extract_ms = (perf_counter() - start) * 1000
            selected = parsed if confidence >= threshold else ()
            routes = (
                ("supplied", expected, 0.),
                ("rules", rules, rule_ms),
                ("offsets", parsed, extract_ms),
                ("calibrated_offsets", selected, extract_ms),
                ("hybrid", rules, rule_ms) if rules else ("hybrid", selected, rule_ms + extract_ms),
            )
            for arm, prediction, parse_ms in routes:
                start = perf_counter()
                result = score_search(service, prediction, row)
                total_ms = parse_ms + (perf_counter() - start) * 1000
                if i:
                    key = f"{style}|{arm}"
                    counts[key].update(result, binding=prediction == expected, parsed=bool(prediction))
                    times[key].append((parse_ms, total_ms))
    return {key: dict(totals=dict(counts[key]),
                      parser_p95_ms=float(np.quantile([p for p, _ in values], .95)),
                      total_p95_ms=float(np.quantile([t for _, t in values], .95)))
            for key, values in sorted(times.items())}


def calibrate_joint(parse, rows):
    observations = []
    agreement_counts = Counter(accepted=0, exact=0, outside=0)
    for text, spans in rows:
        _, _, prediction, margins, _, agreement, _ = parse(text)
        gold = expected_pairs(text, spans)
        if prediction:
            observations.append((margins, prediction == gold, bool(gold)))
        if agreement:
            agreement_counts.update(accepted=1, exact=agreement == gold, outside=not gold)
    curves = {}
    for metric in ("presence", "alternative"):
        ordered = sorted(observations, key=lambda row: row[0][metric], reverse=True)
        points, accepted, exact, outside = [], 0, 0, 0
        for threshold, members in groupby(ordered, key=lambda row: row[0][metric]):
            for _, match, in_scope in members:
                accepted += 1
                exact += match
                outside += not in_scope
            points.append(dict(threshold=threshold, accepted=accepted, exact=exact, outside=outside))
        feasible = [p for p in points if not p["outside"] and p["exact"] >= .90 * p["accepted"]]
        selected = max(feasible, key=lambda point: point["accepted"]) if feasible else None
        # Record the strongest achievable precision with useful calibration coverage.
        comparable = [p for p in points if not p["outside"] and p["accepted"] >= 10]
        best = max(comparable, key=lambda p: (p["exact"] / p["accepted"], p["accepted"])) if comparable else None
        curves[metric] = dict(accepted_without_cut=len(ordered), selected=selected, best_n10=best)
    return curves, dict(agreement_counts)


def evaluate_joint(service, test_rows, wording, parse, calibration, reverse, ensemble):
    bank = phrase_bank(wording)
    rows = test_rows[::-1] if reverse else test_rows
    counts, times = defaultdict(Counter), defaultdict(list)
    for index, row in enumerate([rows[0], *rows]):
        topics = [service.index.keywords(service.index.articles[p]["clue"]) for p in row["papers"]]
        roles = [bank[r][int(row["family"][:16], 16) % len(bank[r])].strip().rstrip(".?!") for r in row["roles"]]
        expected = tuple(zip(topics, roles, strict=True))
        for style, template in TEMPLATES.items():
            text = template.format(a=topics[0], b=topics[1], x=roles[0], y=roles[1])
            greedy, greedy_ms, structured, margins, structure_ms, agreement, agreement_ms = parse(text)
            routes = [("supplied", expected, 0.), ("greedy", greedy, greedy_ms),
                      ("structured", structured, structure_ms)]
            if ensemble:
                routes.append(("agreement", agreement, agreement_ms))
            for metric, diagnostics in calibration.items():
                cut = diagnostics["selected"]
                accepted = structured and cut is not None and margins[metric] >= cut["threshold"]
                routes.append((metric, structured if accepted else (), structure_ms))
            if not greedy:
                routes.append(("structured_given_greedy_abstention", structured, structure_ms))
            for arm, prediction, parse_ms in routes:
                start = perf_counter()
                result = score_search(service, prediction, row)
                elapsed = parse_ms + (perf_counter() - start) * 1000
                if index:
                    key = f"{style}|{arm}"
                    counts[key].update(result, binding=prediction == expected, parsed=bool(prediction))
                    times[key].append((parse_ms, elapsed))
    return {key: dict(totals=dict(counts[key]), parser_p95_ms=float(np.quantile([p for p, _ in values], .95)),
                      total_p95_ms=float(np.quantile([t for _, t in values], .95)))
            for key, values in sorted(times.items())}


def pools(articles, index, test_rows):
    test_ids = {p for r in test_rows for p in r["papers"]}
    authors = {person for p in test_ids for person in articles[p]["roles"]}
    seen = {index.keywords(articles[p]["clue"]) for p in test_ids}
    fit_topics, held_topics = set(), set()
    for p, article in sorted(articles.items()):
        if p in test_ids or authors.intersection(article["roles"]):
            continue
        topic = index.keywords(article["clue"])
        if topic and topic not in seen:
            (held_topics if int(sha256(p.encode()).hexdigest()[:4], 16) % 5 == 0 else fit_topics).add(topic)
    return sorted(fit_topics - held_topics), sorted(held_topics)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    torch.set_num_threads(1)
    articles, rows, _ = build(args.projection, json.loads(args.manifest.read_bytes()))
    test_rows = sample_families(rows)
    definitions, training_phrases, wording = queries(args.niso, args.training)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    index = PostingCatalog(articles, encoder)
    fit_topics, held_topics = pools(articles, index, test_rows)
    activities = sorted({s.strip().rstrip(".?!") for s in training_phrases})
    random.Random(792).shuffle(activities)
    fit_phrases, held_phrases = activities[:64], activities[64:]
    fit_rows = synthetic(fit_topics, fit_phrases, TRAIN_FORMS, 2048, 805, outside=True)
    dev_rows = synthetic(held_topics, held_phrases, CALIBRATION_FORMS, 320, 807, outside=True)
    begin = perf_counter()
    if args.context:
        tagger, losses = fit_context(encoder, fit_rows, args.seed)

        def parse(text):
            return extract_context(tagger, encoder, text)
    else:
        tagger, vocab, losses = fit(fit_rows, args.seed)

        def parse(text):
            return extract(tagger, vocab, text)
    partner = None
    agreement_seed = None
    if args.ensemble:
        agreement_seed = 29 if args.seed == 17 else 17
        partner, _ = fit_context(encoder, fit_rows, agreement_seed)
    train_seconds = perf_counter() - begin
    threshold, calibration = calibrate(parse, dev_rows)
    joint_calibration = agreement_development = None
    if args.structured:
        def compare(text):
            return joint_parse(tagger, encoder, text, partner)

        joint_calibration, agreement_development = calibrate_joint(compare, dev_rows)
    print(f"Tagger fitted in {train_seconds:.1f}s, threshold={threshold}, loss={losses[-1]:.4f}", flush=True)
    banks = projections(encoder, definitions, training_phrases)
    source = Reranker(args.source_reader)
    role = RoleReader(args.role_reader, definitions)
    role.definitions = ROLES
    source_config = json.loads(args.source_frozen.read_bytes())["runs"][0]
    if source.provenance["revision"] != source_config["verifier"]["revision"]:
        raise ValueError("Source verifier revision changed")
    cutoffs = dict(source=source_config["thresholds"]["keywords"]["weakest"]["cutoff"],
                   role=json.loads(args.role_frozen.read_bytes())["runs"][0]["fit"]["cutoff"])
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    service = Service(index, encoder, banks, bands, source, role, cutoffs)
    if args.structured:
        results = evaluate_joint(service, test_rows, wording, compare, joint_calibration, args.reverse, args.ensemble)
    else:
        results = evaluate(service, test_rows, wording, parse, threshold, args.reverse)
    inputs = ("projection", "manifest", "niso", "training", "bands", "source_frozen", "role_frozen")
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), inputs={key: digest(getattr(args, key)) for key in inputs},
        embedding_revision=verify_checkpoint(args.embedding)["revision"],
        source_reader_revision=source.provenance["revision"], role_reader_revision=role.provenance["revision"],
        gpu=torch.cuda.get_device_name(), seed=args.seed, reverse=args.reverse, context=args.context,
        training_seconds=train_seconds, parameters=sum(p.numel() for p in tagger.parameters()),
        fit_topics=len(fit_topics), dev_topics=len(held_topics), fit_phrases=len(fit_phrases),
        dev_phrases=len(held_phrases), training_requests=len(fit_rows), development_requests=len(dev_rows),
        test_families=sorted({row["family"] for row in test_rows}),
        loss=losses, threshold=threshold, calibration=calibration, joint_calibration=joint_calibration,
        agreement_seed=agreement_seed, agreement_development=agreement_development,
        lattice_sha256=digest(Path(__file__).with_name("source_role_offset_lattice.py")), results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "niso", "training", "bands", "source_frozen", "role_frozen",
                 "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    options = parser.parse_args()
    if options.structured and not options.context:
        parser.error("--structured requires --context")
    if options.ensemble and not options.structured:
        parser.error("--ensemble requires --structured")
    run(options)
