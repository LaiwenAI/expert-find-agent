"""Train a low-cost source gate and compare a two-candidate reranking cascade."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.source_confirmation import accepts
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
from experiments.litsearch.source_withholding import POOL, QUERY_SHA256, REVISION, TASK, calibrate, normalize, split
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


METHODS = ("cosine", "learned", "cosine_two", "reader_two", "reader_six", "cascade")
VIEWS = ("present", "withheld")
TWO = 2
CALIBRATION_BUDGETS = (.02, .05, .10, .15)


def split_training(questions):
    validation, test, _ = split(questions)
    excluded = validation + test
    targets = {doc for row in excluded for doc in row["corpusids"]}
    strings = {tuple(normalize(row["query"])) for row in excluded}
    available = [row for row in questions if row["query_set"].startswith("inline_")
                 and not set(row["corpusids"]) & targets
                 and tuple(normalize(row["query"])) not in strings]
    selected = sorted(available, key=lambda row: sha256(row["query"].encode()).hexdigest())[:128]
    if len(selected) != 128:
        raise ValueError("Insufficient training questions without target overlap")
    return selected, validation, test


def observation(pool, scores):
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    first = order[0]
    gap = scores[first] - scores[order[1]] if len(order) > 1 else 0.
    return (pool[first], float(scores[first]), float(gap)), [pool[i] for i in order]


def confirmation_features(cosine, sparse, title_hits, abstract_hits):
    """Candidate features shared by the training and transfer evaluations."""
    return np.column_stack((
        cosine, cosine - cosine[0], 1 / np.arange(1, len(cosine) + 1), np.log1p(sparse),
        sparse / max(1, sparse.max()), title_hits, abstract_hits,
    ))


class Endpoint:
    def __init__(self, papers, vectors, encoder, lexical, reranker):
        self.ids = [r["corpusid"] for r in papers]
        self.positions = {doc: i for i, doc in enumerate(self.ids)}
        self.docs = {r["corpusid"]: paper_text(r) for r in papers}
        self.vectors, self.encoder, self.lexical, self.reranker = vectors, encoder, lexical, reranker
        self.tokens = {}
        self.head = None

    @torch.inference_mode()
    def retrieve(self, question, view):
        query = self.encoder.encode([f"Instruct: {TASK}\nQuery: {question['query']}"])[0]
        scores = self.vectors @ torch.from_numpy(query).cuda()
        if view == "withheld":
            scores[[self.positions[doc] for doc in question["corpusids"]]] = -torch.inf
        chosen = torch.argsort(scores, descending=True, stable=True)[:POOL]
        indices = chosen.tolist()
        return [self.ids[i] for i in indices], indices, scores[chosen].cpu().numpy()

    def features(self, text, pool, positions, cosine):
        sparse, _ = self.lexical.document_arrays(text)
        selected = sparse[positions]
        terms = set(normalize(text))
        title_hits, abstract_hits = [], []
        for doc in pool:
            if doc not in self.tokens:
                title, abstract = self.docs[doc].split("\n", 1)
                self.tokens[doc] = set(normalize(title)), set(normalize(abstract))
            title, abstract = self.tokens[doc]
            title_hits.append(len(terms & title) / max(1, len(terms)))
            abstract_hits.append(len(terms & abstract) / max(1, len(terms)))
        return confirmation_features(cosine, selected, title_hits, abstract_hits)

    def train(self, rows):
        features, labels = [], []
        for question in rows:
            for view in VIEWS:
                pool, indices, cosine = self.retrieve(question, view)
                features.extend(self.features(question["query"], pool, indices, cosine))
                labels.extend(doc in question["corpusids"] for doc in pool)
        x, y = np.asarray(features), np.asarray(labels)
        center, scale = x.mean(axis=0), x.std(axis=0).clip(min=1e-6)
        inputs = torch.as_tensor((x - center) / scale, dtype=torch.float32, device="cuda")
        targets = torch.from_numpy(y).cuda()
        if not targets.any() or targets.all():
            raise ValueError("Insufficient positive or unjudged examples")
        layer = torch.nn.Linear(inputs.shape[1], 1).cuda()
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.03, weight_decay=.03)
        start = perf_counter()
        for _ in range(240):
            predictions = layer(inputs).flatten()
            loss = .5 * (F.softplus(-predictions[targets]).mean() + F.softplus(predictions[~targets]).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        coefficients = layer.weight.detach().cpu().numpy()[0]
        bias = float(layer.bias.item())
        self.head = (center, scale, coefficients, bias)
        return dict(training_requests=len(rows), labeled_candidates=int(y.sum()), candidates=len(y),
                    coefficients=coefficients.tolist(), center=center.tolist(), scale=scale.tolist(), bias=bias,
                    head_training_seconds=perf_counter() - start)

    def run(self, question, view, method, policy=None):
        start = perf_counter()
        pool, indices, cosine = self.retrieve(question, view)
        text = question["query"]
        called = False
        if method == "cosine":
            scored, _ = observation(pool, cosine)
        elif method in ("reader_six", "cosine_two"):
            candidates = pool if method == "reader_six" else pool[:TWO]
            values, _ = self.reranker.score(text, self.docs, candidates, task=TASK)
            scored, _ = observation(candidates, values)
            called = True
        else:
            center, scale, weights, bias = self.head
            values = ((self.features(text, pool, indices, cosine) - center) / scale) @ weights + bias
            scored, rank = observation(pool, values)
            if method == "reader_two" or (method == "cascade" and not accepts(scored, policy)):
                candidates = rank[:TWO]
                values, _ = self.reranker.score(text, self.docs, candidates, task=TASK)
                scored, _ = observation(candidates, values)
                called = True
        return dict(pool=pool, scored=scored, ms=(perf_counter() - start) * 1000, reader=called)


def evaluate(endpoint, questions, methods, first_policy=None):
    results = {method: [] for method in methods}
    for i, question in enumerate(questions):
        offset = i % len(methods)
        for method in methods[offset:] + methods[:offset]:
            arms = {}
            for view in VIEWS[i % 2:] + VIEWS[:i % 2]:
                arm = endpoint.run(question, view, method, first_policy)
                if view == "withheld" and set(arm["pool"]) & set(question["corpusids"]):
                    raise ValueError("Withheld target returned")
                arms[view] = arm
            results[method].append(dict(id=sha256(question["query"].encode()).hexdigest(),
                                        targets=question["corpusids"], views=arms))
    return results


def calibrate_method(rows, budget=.10):
    projected = [dict(targets=r["targets"], views={key: {"scored": v["scored"]} for key, v in r["views"].items()})
                 for r in rows]
    return calibrate(projected, "scored", budget)


def cascade_calibration(stage_one, stage_two, first_policy):
    provisional = []
    for a, b in zip(stage_one, stage_two, strict=True):
        views = {}
        for name in VIEWS:
            fast = a["views"][name]["scored"]
            views[name] = (fast, True) if accepts(fast, first_policy) else (b["views"][name]["scored"], False)
        provisional.append(dict(targets=a["targets"], views=views))
    fallback = [row["views"][view][0] for row in provisional for view in VIEWS if not row["views"][view][1]]
    if not fallback:
        return None
    thresholds = np.unique(np.quantile([obs[1] for obs in fallback], np.linspace(0, 1, 31))).tolist()
    margins = [0.] + np.unique(np.quantile([obs[2] for obs in fallback], np.linspace(0, 1, 16))).tolist()
    ceiling = int(.10 * len(provisional))
    best = None
    for score in [*thresholds, None]:
        for margin in margins:
            policy = None if score is None else dict(score=score, margin=margin)
            counts = Counter()
            for row in provisional:
                for view in VIEWS:
                    obs, fast = row["views"][view]
                    if fast or accepts(obs, policy):
                        key = "withheld" if view == "withheld" else "known" if obs[0] in row["targets"] else "unjudged"
                        counts[key] += 1
            if counts["withheld"] <= ceiling and counts["unjudged"] <= ceiling:
                objective = (counts["known"], -counts["withheld"], -counts["unjudged"])
                if best is None or objective > best[0]:
                    best = objective, policy
    return None if best is None else best[1]


def summarize(rows, method, threshold):
    report = {}
    for view in VIEWS:
        counts = Counter()
        times = []
        for row in rows:
            result = row["views"][view]
            decision = result["scored"]
            selected = ((method == "cascade" and not result["reader"]) or accepts(decision, threshold))
            counts["pool_hit"] += bool(set(result["pool"]) & set(row["targets"]))
            counts["top1_known"] += decision[0] in row["targets"]
            counts["reader_calls"] += result["reader"]
            counts["accepted"] += selected
            counts["known_accepted"] += selected and decision[0] in row["targets"]
            counts["unjudged_accepted"] += selected and decision[0] not in row["targets"]
            times.append(result["ms"])
        report[view] = dict(counts, requests=len(rows), mean_ms=mean(times), p95_ms=float(np.quantile(times, .95)))
    return report


def calibration_sweep(validation, test, fitted):
    sweep = {}
    for budget in CALIBRATION_BUDGETS:
        policies = fitted if budget == .10 else {
            method: calibrate_method(rows, budget) for method, rows in validation.items()
        }
        sweep[f"{budget:.2f}"] = {
            method: dict(calibration=fit, test=summarize(test[method], method, fit["policy"]))
            for method, fit in policies.items()
        }
    return sweep


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    with gzip.open(args.projection, "rb") as stream:
        data = json.load(stream)
    if data["revision"] != REVISION or data["query_sha256"] != QUERY_SHA256:
        raise ValueError("Unexpected LitSearch projection")
    papers = data["papers"]
    provenance = verify_checkpoint(args.embedding)
    projection_hash = digest(args.projection)
    with np.load(args.cache, allow_pickle=False) as cache:
        if (str(cache["projection_sha256"]) != projection_hash
                or json.loads(str(cache["provenance"])) != provenance
                or cache["corpus_ids"].tolist() != [r["corpusid"] for r in papers]):
            raise ValueError("Vector cache mismatch")
        vectors = torch.from_numpy(cache["vectors"]).cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    texts = {}
    for i, paper in enumerate(papers):
        text = paper_text(paper)
        while text in texts:
            text += "\n"
        texts[text] = i
    with TemporaryDirectory(dir=args.projection.parent) as temporary:
        lexical = build_lexical(texts, Path(temporary))
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    endpoint = Endpoint(papers, vectors, encoder, lexical, reader)
    training, validation, test = split_training(data["questions"])
    fitted = endpoint.train(training)
    print("Fitted", fitted, flush=True)
    for method in METHODS[:-1]:
        endpoint.run(validation[0], "present", method)
    validation_rows = evaluate(endpoint, validation, METHODS[:-1])
    policy_fit = {method: calibrate_method(rows) for method, rows in validation_rows.items()}
    first_policy = calibrate_method(validation_rows["learned"], .05)["policy"]
    threshold = cascade_calibration(validation_rows["learned"], validation_rows["reader_two"], first_policy)
    thresholds = {name: policy["policy"] for name, policy in policy_fit.items()}
    thresholds["cascade"] = threshold
    print("Frozen thresholds", thresholds, first_policy, flush=True)
    set_test = evaluate(endpoint, test, METHODS, first_policy)
    write_json(args.output, dict(
        projection_sha256=projection_hash, script_sha256=digest(Path(__file__)), corpus=len(papers),
        train_targets=sorted({d for row in training for d in row["corpusids"]}), fitted=fitted,
        calibrations=policy_fit, cascade_first=first_policy, thresholds=thresholds,
        validation=validation_rows, test=set_test, embedding=provenance, reranker=reader.provenance,
        calibration_sweep=calibration_sweep(validation_rows, set_test, policy_fit),
        gpu=torch.cuda.get_device_name(), report={
            method: summarize(rows, method, thresholds[method]) for method, rows in set_test.items()
        },
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "cache", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
