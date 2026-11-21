"""Compare one role decision per request with per-author verification."""

import argparse
from collections import Counter
from itertools import groupby
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.calibrated_topic_evidence import margins
from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.engineering_contribution_verify import DESCRIPTIONS, TASK
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


NISO_SHA256 = "c107b4e780218e58aec4e2a13f2d7072533962a6a73b82d23e4273b43aafd528"
ARMS = ("shared", "scoped", "lexical", "dense")
POLICIES = ("shared", "scoped", "centered_scope", "lexical", "dense")
VIEWS = {
    "original": DESCRIPTIONS,
    "alternate": (
        "originated the research questions and framed the investigation",
        "documented and preserved the collection so others could reuse it",
        "quantified patterns in the observations by applying analytical techniques",
        "obtained a grant or other financial backing for the work",
        "carried out the practical work that generated observations",
        "devised the protocol used to run the investigation",
        "organized the schedule and managed the work needed to deliver the project",
        "made equipment, specimens, or computational infrastructure available",
        "programmed the tools the team used to carry out the work",
        "guided the research team and provided ongoing oversight",
        "verified that the procedures and findings could be reproduced",
        "made graphical displays that communicate the findings",
        "wrote the earliest version of the paper",
        "edited the manuscript after critically reviewing its contents",
    ),
}


class RoleGate:
    def __init__(self, examples, embedding):
        if digest(examples) != NISO_SHA256:
            raise ValueError("Unexpected glossary checksum")
        definitions = json.loads(examples.read_bytes())["definitions"]
        self.texts = [f"{role}: {definitions[role]}" for role in ROLES]
        self.encoder = QwenTorchEmbedding(embedding)
        self.encoder.model.config.use_cache = False
        self.vectors = self.encoder.encode(self.texts)
        frequencies = [Counter(normalize(text).split()) for text in self.texts]
        df = Counter(token for terms in frequencies for token in terms)
        mean = sum(sum(terms.values()) for terms in frequencies) / len(ROLES)
        self.postings = []
        for terms in frequencies:
            length = .3 + .9 * sum(terms.values()) / mean
            self.postings.append({
                term: math.log1p((len(ROLES) - df[term] + .5) / (df[term] + .5)) * 2.2 * n / (n + length)
                for term, n in terms.items()
            })

    def scores(self, query, arm):
        if arm == "dense":
            v = self.encoder.encode([f"Instruct: Match the research task to a contributor role.\nQuery:{query}"])[0]
            return self.vectors @ v
        terms = sorted(set(normalize(query).split()))
        return np.array([sum(postings.get(term, 0.) for term in terms) for postings in self.postings])


def compile_article(article):
    slots = range(len(article["aliases"]))
    raw = article["blocks"]
    visible = [[block for block in raw if i in block["owners"]] for i in slots]
    # Scope and fast memberships come from visible text, independently of the
    # JATS role fields that provide the evaluation labels.
    members = [[any(role in block["roles"] for block in scope) for role in ROLES] for scope in visible]
    contexts = ["\n".join(block["text"] for block in scope) for scope in visible]
    return dict(
        names=[article["aliases"][str(i)] for i in slots], memberships=np.asarray(members),
        labels=[[role in article["roles"][str(i)] for i in slots] for role in ROLES],
        shared="\n".join(block["text"] for block in raw), scoped=contexts,
    )


def observe(article, phrase, arm, reader, gate):
    start = perf_counter()
    if arm in ("lexical", "dense"):
        relevance = gate.scores(phrase, arm)
        membership = article["memberships"]
        member = np.where(membership, relevance, -np.inf).max(axis=1)
        absent = np.where(~membership, relevance, -np.inf).max(axis=1)
        # Fixed finite sentinels cover contributors with all glossary roles.
        scores = np.clip(member - absent, -100, 100).tolist()
        clipped = calls = 0
    else:
        passages = [article["shared"]] * len(article["names"]) if arm == "shared" else article["scoped"]
        prompts = [(f"Target author: {alias}. Requested contribution: {phrase}.", evidence)
                   for alias, evidence in zip(article["names"], passages, strict=True)]
        values, clipped = margins(reader, prompts, TASK)
        scores = values.tolist()
        calls = len(prompts)
    json.dumps(scores)
    return dict(scores=scores, clipped=clipped, reader_pairs=calls, ms=(perf_counter() - start) * 1000)


def collect(documents, reader, gate, offset):
    results = {}
    for pmc, source in sorted(documents.items()):
        article = compile_article(source)
        for name, phrases in VIEWS.items():
            for i, phrase in enumerate(phrases):
                shift = (offset + i) % len(ARMS)
                routes = {arm: observe(article, phrase, arm, reader, gate)
                          for arm in ARMS[shift:] + ARMS[:shift]}
                scope = routes["scoped"]
                median = float(np.median(scope["scores"]))
                routes["centered_scope"] = scope | {"scores": [v - median for v in scope["scores"]]}
                results.setdefault(source["journal"], {}).setdefault(name, []).append(dict(
                    article=pmc, role=ROLES[i], labels=article["labels"][i], routes=routes,
                ))
    return results


def evaluate(rows, arm, threshold):
    counts = Counter()
    clean, exact = [], []
    for row in rows:
        labels = row["labels"]
        accepted = [score > threshold for score in row["routes"][arm]["scores"]]
        good = sum(yes and known for yes, known in zip(accepted, labels, strict=True))
        bad = sum(yes and not known for yes, known in zip(accepted, labels, strict=True))
        scope = row["routes"][arm]
        counts.update(requests=1, expected_authors=sum(labels), accepted=good, unsupported=bad,
                      wrong_request=int(bool(bad)), no_declared=int(not any(labels)),
                      wrong_no_declared=int(bool(bad) and not any(labels)), any_declared=int(bool(good)),
                      exact=int(accepted == labels), all_shared=int(all(labels)),
                      retained_all_shared=int(all(labels) and all(accepted)), reader_pairs=scope["reader_pairs"],
                      clipped=scope["clipped"])
        clean.append(int(good > 0 and bad == 0))
        exact.append(int(accepted == labels))
    counts["clean_nonempty"] = sum(clean)
    return dict(counts), clean, exact


def calibrate(rows, arm):
    """Maximize credited acceptances subject to the source request risk ceiling."""
    floor = 0. if arm in ("lexical", "dense") else -np.inf
    values = [(v, i, label) for i, row in enumerate(rows)
              for v, label in zip(row["routes"][arm]["scores"], row["labels"], strict=True) if v > floor]
    best = (0, 0, max((value for value, _, _ in values), default=0.))
    unsupported_requests = set()
    good = 0
    cap = len(rows) // 20
    for value, ties in groupby(sorted(values, reverse=True), key=lambda item: item[0]):
        for _, i, label in ties:
            if label:
                good += 1
            else:
                unsupported_requests.add(i)
        if len(unsupported_requests) > cap:
            break
        # Strictly above a cutoff immediately below the last admitted score.
        cut = max(floor, float(np.nextafter(value, -np.inf)))
        best = max(best, (good, -len(unsupported_requests), cut))
    return best[2]


def report(panels):
    results = {}
    for target, fit in (("sensors", "frontiers"), ("frontiers", "sensors")):
        development = panels[fit]["original"]
        thresholds = {arm: calibrate(development, arm) for arm in POLICIES}
        results[target] = {}
        for wording, held in panels[target].items():
            baseline = evaluate(held, "scoped", thresholds["scoped"])[1]
            results[target][wording] = {}
            for arm, threshold in thresholds.items():
                measures, clean, _ = evaluate(held, arm, threshold)
                result = dict(
                    threshold=threshold, calibration=evaluate(development, arm, threshold)[0], held=measures,
                    p95_ms=float(np.quantile([r["routes"][arm]["ms"] for r in held], .95)),
                    clean_paired_vs_scoped=clustered_paired_interval(
                        np.asarray(clean) - baseline, [row["article"] for row in held],
                    ),
                    requests=[dict(article=row["article"], role=row["role"], labels=row["labels"],
                                   scores=row["routes"][arm]["scores"]) for row in held],
                )
                results[target][wording][arm] = result
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("panel", "examples", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    sources = json.loads(args.panel.read_bytes())
    embed = verify_checkpoint(args.embedding)
    gate = RoleGate(args.examples, args.embedding)
    reader = Reranker(args.reranker)
    # Warm each execution branch once before collecting request times.
    warm = compile_article(next(iter(sources.values())))
    for arm in ARMS:
        observe(warm, DESCRIPTIONS[0], arm, reader, gate)
    result = report(collect(sources, reader, gate, args.offset))
    write_json(args.output, dict(
        articles=len(sources), panel_sha256=digest(args.panel), glossary_sha256=digest(args.examples),
        script_sha256=digest(Path(__file__)), offset=args.offset, gpu=torch.cuda.get_device_name(),
        reranker_revision=reader.provenance["revision"], embedding_revision=embed["revision"], results=result,
    ))
