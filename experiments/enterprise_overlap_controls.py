"""Measure windowed overlap sensitivity with known-source enterprise controls."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


THRESHOLDS = (.80, .85, .90, .95)
MODES = {"head_tail": (0, 1), "head_center": (0, 2), "head_tail_center": (0, 1, 2)}
EXCERPT = 180
MAX_HISTORICAL_ISSUERS = 80


def clip(text, width, center):
    start = min(max(center - width // 2, 0), max(len(text) - width, 0))
    return text[start:start + width].strip()


def prepare(args):
    from experiments.audit_enterprise_panel_neighbors import WINDOW, windows, write_private
    from experiments.enterprise_regional_overlap import CONTACT, checked_rows
    from experiments.external_enterprise_frame import digest_file
    from experiments.freeze_enterprise_panel import issuer_key

    current, history, earlier = checked_rows(args)
    rows = current + history
    masked = [CONTACT.sub("[contact]", row["zh"]) for row in rows]
    parts = [tuple(text.split("\n", 1)) for text in masked]
    stock = json.loads(args.baseline.read_bytes())
    head_tail = [list(windows(text)) for text in masked]
    if (stock["external"] != len(current) or stock["earlier"] != earlier
            or stock["frame_sha256"] != digest_file(args.frame) or stock["views"] != head_tail):
        raise ValueError("Stock overlap projection changed")

    corpus = []
    for (title, body), (head, tail) in zip(parts, head_tail, strict=True):
        center = clip(body, WINDOW, len(body) // 2)
        corpus.append([head, tail, f"{title}\n{center}".strip()])

    controls = {name: [] for name in (
        "regional_body_only", "regional_central_excerpt", "historical_head_excerpt",
        "historical_hidden_center", "historical_tail_excerpt",
    )}

    def add(name, index, query):
        if query.strip():
            controls[name].append([index, list(windows(query))])

    regional_long = 0
    for index in range(earlier, len(current)):
        _, body = parts[index]
        add("regional_body_only", index, f"\n{body}")
        if len(body) >= 200:
            add("regional_central_excerpt", index, f"\n{clip(body, EXCERPT, len(body) // 2)}")
        regional_long += len(body) >= 2 * WINDOW + EXCERPT

    valid = (index for index in range(len(current), len(rows))
             if len(parts[index][1]) >= 2 * WINDOW + EXCERPT
             and issuer_key(rows[index]["company"]))
    order = sorted(valid, key=lambda index: sha256(f"overlap-controls-v1:{rows[index]['id']}".encode()).digest())
    seen_issuers, seen_text = set(), set()
    for index in order:
        issuer = issuer_key(rows[index]["company"])
        _, body = parts[index]
        middle = clip(body, EXCERPT, len(body) // 2)
        if issuer in seen_issuers or middle in seen_text or middle in body[:WINDOW] or middle in body[-WINDOW:]:
            continue
        seen_issuers.add(issuer)
        seen_text.add(middle)
        add("historical_head_excerpt", index, f"\n{body[:EXCERPT]}")
        add("historical_hidden_center", index, f"\n{middle}")
        add("historical_tail_excerpt", index, f"\n{body[-EXCERPT:]}")
        if len(seen_issuers) == MAX_HISTORICAL_ISSUERS:
            break

    if any(not cohort for cohort in controls.values()):
        raise ValueError("A prespecified control cohort is empty")
    projection = dict(corpus=corpus, controls=controls, regional_long=regional_long,
                      historical_body_length_eligible=len(order), stock_projection_sha256=digest_file(args.baseline))
    write_private(args.output, [projection])
    print(json.dumps({key: len(cohort) for key, cohort in controls.items()}))


def summarize(scores, targets):
    n = len(targets)
    matched = scores[np.arange(n), targets]
    higher = (scores > matched[:, None]).sum(axis=1)
    ahead_on_ties = ((scores == matched[:, None]) & (np.arange(scores.shape[1]) < targets[:, None])).sum(axis=1)
    ranks = 1 + higher + ahead_on_ties
    scores[np.arange(n), targets] = -np.inf
    thresholds = {}
    for threshold in THRESHOLDS:
        passes = matched >= threshold
        other = (scores >= threshold).sum(axis=1)
        thresholds[str(threshold)] = dict(
            recovered=int(passes.sum()),
            recovered_first=int((passes & (ranks == 1)).sum()),
            controls_with_other_candidates=int((other > 0).sum()),
            additional_candidates=int(other.sum()),
        )
    return dict(n=n, top1=int((ranks == 1).sum()), top5=int((ranks <= 5).sum()),
                median_rank=float(np.median(ranks)), thresholds=thresholds)


def measure(data, encode, order):
    corpus = data["corpus"]
    unique = dict.fromkeys(window for views in corpus for window in views)
    unique.update(dict.fromkeys(window for cohort in data["controls"].values() for _, views in cohort for window in views))
    texts = list(unique)
    if order:
        texts.reverse()
    vectors = encode(texts)
    position = {text: index for index, text in enumerate(texts)}
    corpus_index = np.array([[position[text] for text in views] for views in corpus])

    result = {}
    for name, cohort in data["controls"].items():
        targets = np.array([target for target, _ in cohort])
        if len(set(targets)) != len(targets) or targets.min() < 0 or targets.max() >= len(corpus):
            raise ValueError("Invalid known-source control inventory")
        query_views = np.array([[position[text] for text in views] for _, views in cohort]).T
        scores = [
            np.maximum.reduce([vectors[query] @ vectors[corpus_index[:, view]].T for query in query_views])
            for view in range(corpus_index.shape[1])
        ]
        result[name] = {
            mode: summarize(np.maximum.reduce([scores[view] for view in views]), targets)
            for mode, views in MODES.items()
        }
    return dict(corpus=len(corpus), regional_long=data["regional_long"],
                historical_body_length_eligible=data["historical_body_length_eligible"],
                distinct_corpus_views={mode: len(set(corpus_index[:, views].ravel())) for mode, views in MODES.items()},
                controls=result)


def gpu(args):
    import os

    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_reranking import digest, write_json
    from experiments.litsearch.dense_cache import verify_checkpoint

    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm GPU")
    provenance = verify_checkpoint(args.checkpoint)
    encoder = QwenTorchEmbedding(args.checkpoint, max_length=384, batch_size=32)
    result = measure(json.loads(args.inputs.read_bytes()), encoder.encode, args.order)
    result.update(input_sha256=digest(args.inputs), encoder_revision=provenance["revision"],
                  gpu=torch.cuda.get_device_name(), order=args.order)
    write_json(args.output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    setup = modes.add_parser("prepare")
    for field in ("earlier", "municipal", "regional", "workbook", "previous_index", "report", "frame", "baseline", "output"):
        setup.add_argument(field, type=Path)
    setup.set_defaults(command=prepare)
    run = modes.add_parser("gpu")
    for field in ("inputs", "checkpoint", "output"):
        run.add_argument(field, type=Path)
    run.add_argument("--order", type=int, default=0, choices=(0, 1))
    run.set_defaults(command=gpu)
    args = parser.parse_args()
    args.command(args)
