"""Distill a source-pair decision into an equivariant linear reranker head."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_confirmation import accepts
from experiments.docred.oracle_role_evidence import load_reader
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.contrastive_source_confirmation import ChoiceReader, STOPWORDS
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.litsearch.dense_retrieval import target_groups
from experiments.litsearch.learned_confirmation import observation, split_training
from experiments.litsearch.source_withholding import QUERY_SHA256, REVISION, TASK, calibrate, normalize
from experiments.litsearch.top_hit_confirmation import DenseConfirmation


ARMS = ("raw", "supervised", "distilled", "teacher")
STATES = ("present", "withheld")
BUDGETS = (.02, .05, .10)
TEMPERATURE = 2.


class Endpoint(DenseConfirmation):
    def __init__(self, papers, vectors, encoder, reranker, teacher):
        super().__init__(papers, vectors, encoder, None, reranker)
        self.teacher = teacher
        self.content = {}
        for doc, text in self.docs.items():
            title, _, abstract = text.partition("\n")
            self.content[doc] = (set(normalize(title)), set(normalize(abstract)))

    def features(self, text, selected, dense, ratings):
        terms = set(normalize(text)) - STOPWORDS
        size = max(1, len(terms))
        cover = [[len(tokens & terms) / size for tokens in self.content[d]] for d in selected]
        dense, ratings = np.asarray(dense), np.asarray(ratings)
        return np.column_stack((ratings, ratings - ratings[::-1], dense, dense - dense[::-1],
                                np.array(cover, dtype=np.float32))).astype(np.float32)

    def teacher_scores(self, text, selected, balanced):
        docs = [" ".join(self.docs[doc].split()[:160]) for doc in selected]
        return self.teacher.score(text, docs, balanced)

    def training_pairs(self, query):
        emitted = set()
        targets = set(query["corpusids"])
        for state in STATES:
            pool, _, scores = self.retrieve(query, state)
            dense = dict(zip(pool, map(float, scores), strict=True))
            choices = [tuple(pool[:2])]
            if state == "present":
                positive = next((d for d in pool if d in targets), None)
                near_misses = [d for d in pool if d not in targets]
                if positive is not None:
                    choices.extend(tuple(d for d in pool if d in {positive, near}) for near in near_misses[:2])
            for pair in choices:
                if pair not in emitted:
                    emitted.add(pair)
                    yield pair, [dense[d] for d in pair]

    def run(self, query, state, arm, predictors):
        start = perf_counter()
        pool, _, cosine = self.retrieve(query, state)
        selected = pool[:2]
        text = query["query"]
        if arm == "teacher":
            if sha256(text.encode()).hexdigest()[0] in "13579bdf":
                selected = selected[::-1]
            values, rows, _, _ = self.teacher_scores(text, selected, balanced=False)
            clipped = 0
        else:
            values, clipped = self.reranker.score(text, self.docs, selected, task=TASK)
            rows = 2
            if arm != "raw":
                values = predictors[arm](self.features(text, selected, cosine[:2], values))
        if not np.isfinite(values).all():
            raise ValueError("Invalid verification score")
        scored, _ = observation(selected, values)
        return scored, sorted(selected), rows, clipped, (perf_counter() - start) * 1000


def collect(endpoint, queries):
    features, labels, teacher, audit = [], [], [], []
    start = perf_counter()
    conflicts = clipped = disagreement = 0
    for query in queries:
        targets = set(query["corpusids"])
        text = query["query"]
        for pair, dense in endpoint.training_pairs(query):
            raw, cut = endpoint.reranker.score(text, endpoint.docs, pair, task=TASK)
            soft, _, _, flips = endpoint.teacher_scores(text, pair, balanced=True)
            active = [p in targets for p in pair]
            gold = [not any(active), *active]
            teacher_label = int(np.argmax([0., *soft]))
            conflicts += int(not gold[teacher_label])
            disagreement += flips
            clipped += cut
            features.append(endpoint.features(text, pair, dense, raw))
            labels.append(gold)
            teacher.append([0., *soft])
            audit.append(dict(query_sha256=sha256(text.encode()).hexdigest(), documents=pair,
                              known=gold, teacher_margins=soft.tolist()))
    return np.stack(features), np.asarray(labels), np.asarray(teacher), audit, dict(
        pairs=len(audit), candidate_order_disagreements=disagreement, teacher_label_proxy_disagreements=conflicts,
        truncated_student_rows=clipped, offline_label_seconds=perf_counter() - start)


def train(x, labels, teacher):
    center = x.reshape(-1, x.shape[-1]).mean(axis=0)
    scale = x.reshape(-1, x.shape[-1]).std(axis=0).clip(min=1e-5)
    values = torch.as_tensor((x - center) / scale, dtype=torch.float32, device="cuda")
    gold = torch.as_tensor(labels, dtype=torch.bool, device="cuda")
    distributions = torch.as_tensor(teacher, dtype=torch.float32, device="cuda")
    teacher_prob = torch.softmax(distributions / TEMPERATURE, dim=1)
    predictors, report = {}, {}
    for arm in ("supervised", "distilled"):
        layer = torch.nn.Linear(values.shape[-1], 1).cuda()
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.025, weight_decay=.03)
        started = perf_counter()
        for _ in range(240):
            optimizer.zero_grad(set_to_none=True)
            logits = F.pad(layer(values).squeeze(-1), (1, 0))
            gold_mass = logits.masked_fill(~gold, -torch.inf).logsumexp(dim=1)
            loss = (logits.logsumexp(dim=1) - gold_mass).mean()
            if arm == "distilled":
                loss += .5 * TEMPERATURE ** 2 * F.kl_div(
                    F.log_softmax(logits / TEMPERATURE, dim=1), teacher_prob, reduction="batchmean")
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        weights = layer.weight.detach().cpu().numpy().ravel().copy()
        bias = float(layer.bias.detach().item())
        predictors[arm] = lambda x, w=weights, b=bias: ((x - center) / scale) @ w + b
        report[arm] = dict(parameters=len(weights)+1, gpu_seconds=perf_counter() - started,
                           weights=weights.tolist(), bias=bias, final_loss=float(loss.detach()))
    return predictors, dict(center=center.tolist(), scale=scale.tolist(), models=report)


def evaluate(endpoint, queries, predictors):
    rows = []
    for i, query in enumerate(queries):
        values = {}
        for state in STATES[i % 2:] + STATES[:i % 2]:
            offset = (i + int(state == "withheld")) % len(ARMS)
            observations = {arm: endpoint.run(query, state, arm, predictors) for arm in ARMS[offset:] + ARMS[:offset]}
            if len({tuple(obs[1]) for obs in observations.values()}) != 1:
                raise ValueError("Rankers received different source pairs")
            values[state] = observations
        rows.append(dict(targets=query["corpusids"], views=values))
        if (i + 1) % 50 == 0:
            print(f"Completed {i+1}/{len(queries)} questions", flush=True)
    return rows


def summarize(validation, test):
    projected = [dict(targets=row["targets"], views={state: {a: x[0] for a, x in states.items()}
                                                   for state, states in row["views"].items()}) for row in validation]
    groups = target_groups(test)
    reports = {}
    for budget in BUDGETS:
        policies = {arm: calibrate(projected, arm, budget) for arm in ARMS}
        results, indicators = {}, {}
        for arm, fit in policies.items():
            results[arm] = dict(calibration=fit, test={})
            indicators[arm] = {}
            for state in STATES:
                observed = [r["views"][state][arm] for r in test]
                choices = [scored[0] if accepts(scored, fit["policy"]) else None for scored, *_ in observed]
                known = [int(source in r["targets"]) for source, r in zip(choices, test, strict=True)]
                unknown = [int(source is not None) - hit for source, hit in zip(choices, known, strict=True)]
                indicators[arm][state] = known, unknown
                results[arm]["test"][state] = dict(
                    known=sum(known), unjudged=sum(unknown), questions=len(known),
                    top_one_known=sum(row[0][0] in q["targets"] for row, q in zip(observed, test)),
                    reader_rows=sum(row[2] for row in observed), truncated=sum(row[3] for row in observed),
                    p95_ms=float(np.quantile([row[4] for row in observed], .95)))
        for arm in ARMS:
            for state in STATES:
                for reference in ("raw", "supervised"):
                    for i, metric in enumerate(("known", "unjudged")):
                        diff = np.subtract(indicators[arm][state][i], indicators[reference][state][i])
                        results[arm]["test"][state][f"paired_{metric}_vs_{reference}"] = clustered_paired_interval(diff, groups)
        reports[f"{budget:.2f}"] = results
    return reports


def run(projection, cache, embedding, reranker_path, teacher_path, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    with gzip.open(projection, "rb") as stream:
        corpus = json.load(stream)
    if corpus["revision"] != REVISION or corpus["query_sha256"] != QUERY_SHA256:
        raise ValueError("Unexpected LitSearch projection")
    provenance = verify_checkpoint(embedding)
    with np.load(cache, allow_pickle=False) as stored:
        if (str(stored["projection_sha256"].item()) != digest(projection)
                or json.loads(str(stored["provenance"].item())) != provenance
                or stored["corpus_ids"].tolist() != [p["corpusid"] for p in corpus["papers"]]):
            raise ValueError("Unexpected dense cache provenance")
        vectors = torch.from_numpy(stored["vectors"]).cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reranker = Reranker(reranker_path)
    tokenizer, model, info = load_reader(teacher_path)
    endpoint = Endpoint(corpus["papers"], vectors, encoder, reranker, ChoiceReader(tokenizer, model))
    training, calibration, test = split_training(corpus["questions"])
    x, labels, soft, audit, capture = collect(endpoint, training)
    predictors, training_details = train(x, labels, soft)
    print(f"Fitted students from {len(audit)} offline pairs", flush=True)
    for arm in ARMS:
        endpoint.run(calibration[0], "present", arm, predictors)
    validation = evaluate(endpoint, calibration, predictors)
    heldout = evaluate(endpoint, test, predictors)
    path = output.with_name("training-pairs.json")
    write_json(path, dict(projection_sha256=digest(projection), pairs=audit))
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        train_queries=len(training), calibration_queries=len(calibration), test_queries=len(test),
        code_sha256=digest(Path(__file__)), projection_sha256=digest(projection), cache_sha256=digest(cache),
        training_pairs_sha256=digest(path), target_sets=dict(Counter(len(row["corpusids"]) for row in test)),
        embedding=provenance["revision"], reranker=reranker.provenance["revision"], teacher=info["revision"],
        capture=capture, training=training_details, policies=summarize(validation, heldout)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("projection", "cache", "embedding", "reranker_path", "teacher_path", "output"):
        cli.add_argument(field, type=Path)
    run(**vars(cli.parse_args()))
