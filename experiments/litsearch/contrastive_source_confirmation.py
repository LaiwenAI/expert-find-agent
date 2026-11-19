"""Compare independent and contrastive two-document source decisions on LitSearch."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_confirmation import accepts
from experiments.docred.oracle_role_evidence import load_reader, reader_scorer
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.litsearch.dense_retrieval import target_groups
from experiments.litsearch.learned_confirmation import observation
from experiments.litsearch.source_withholding import QUERY_SHA256, REVISION, TASK, calibrate, normalize, split
from experiments.litsearch.top_hit_confirmation import DenseConfirmation


ARMS = ("reranker", "binary", "choice_full", "choice_short", "choice_balanced")
STATES = ("present", "withheld")
BUDGETS = (.02, .05, .10)
SENTENCES = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = set("a an and any are by for from i in is of on or paper papers that the to using which with".split())
INDEPENDENT_SYSTEM = ("Judge whether the title and abstract in the source directly satisfy the literature search "
                      "question. A broadly related topic is insufficient. Use only the supplied source. "
                      "Reply 1 for a direct match or 0 otherwise.")
CHOICE_SYSTEM = ("Choose a publication that directly satisfies the literature search question. "
                 "Use the supplied titles and abstracts. If both are directly relevant, choose the more directly "
                 "supported one. If neither is directly relevant, choose 0. Respond with one symbol only, "
                 "1 for the first publication, 2 for the second publication, or 0 for neither.")


class ChoiceReader:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        codes = [tokenizer.encode(x, add_special_tokens=False) for x in ("0", "1", "2")]
        if any(len(ids) != 1 for ids in codes) or len({ids[0] for ids in codes}) != 3:
            raise ValueError("Expected three distinct single-token decision labels")
        self.labels = [ids[0] for ids in codes]

    @torch.inference_mode()
    def score(self, text, documents, balanced):
        orders = ((0, 1), (1, 0)) if balanced else ((0, 1),)
        prompts = [self.tokenizer.apply_chat_template(
            [{"role": "system", "content": CHOICE_SYSTEM},
             {"role": "user", "content": f"Question: {text}\n\nPublication 1:\n{documents[i]}"
                                            f"\n\nPublication 2:\n{documents[j]}"}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        ) for i, j in orders]
        if max(map(len, prompts)) > self.model.config.max_position_embeddings:
            raise ValueError("Context overflow")
        inputs = self.tokenizer.pad([{"input_ids": prompt} for prompt in prompts], padding=True,
                                   padding_side="left", return_tensors="pt").to("cuda")
        values = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, self.labels].float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Invalid decision logits")
        margins = values[:, 1:] - values[:, :1]
        flip = int(balanced and (margins[0].argmax() != 1 - margins[1].argmax()))
        combined = (margins[0] + margins[1, ::-1]) / 2 if balanced else margins[0]
        return combined, len(orders), sum(map(len, prompts)), flip


class Endpoint(DenseConfirmation):
    def __init__(self, papers, vectors, encoder, reranker, reader, binary):
        super().__init__(papers, vectors, encoder, None, reranker)
        self.reader, self.binary = reader, binary
        self.sections = {}
        frequency = Counter()
        for doc, text in self.docs.items():
            title, _, abstract = text.partition("\n")
            sentences = [s for s in SENTENCES.split(abstract) if s.strip()]
            sections = [(s, set(normalize(s))) for s in sentences]
            self.sections[doc] = title, sections
            frequency.update(set(normalize(text)))
        self.idf = {word: math.log1p(len(self.ids) / n) for word, n in frequency.items()}

    def excerpt(self, doc, terms):
        title, sections = self.sections[doc]
        ranked = sorted(range(len(sections)), key=lambda i: (
            -sum(self.idf[word] for word in sorted(terms & sections[i][1])), i
        ))[:2]
        body = " ".join(sections[i][0] for i in sorted(ranked))
        return f"{title}\n{' '.join(body.split()[:64])}"

    def run(self, query, state, arm):
        started = perf_counter()
        pool, _, _ = self.retrieve(query, state)
        selected = pool[:2]
        # Rotate the front position deterministically independently of labels or visibility.
        if digest_bytes(query["query"].encode())[0] in "13579bdf" and arm.startswith("choice"):
            selected = selected[::-1]
        text = query["query"]
        terms = set(normalize(text)) - STOPWORDS
        clipped = 0
        if arm == "reranker":
            values, clipped = self.reranker.score(text, self.docs, selected, task=TASK)
            rows, tokens, flip = 2, None, 0
        elif arm == "binary":
            documents = [" ".join(self.docs[doc].split()[:160]) for doc in selected]
            values, clipped = self.binary([(text, doc) for doc in documents])
            if clipped:
                raise ValueError("Unexpected context truncation in independent scorer")
            rows, tokens, flip = 2, None, 0
        else:
            documents = ([" ".join(self.docs[doc].split()[:160]) for doc in selected] if arm == "choice_full"
                         else [self.excerpt(doc, terms) for doc in selected])
            values, rows, tokens, flip = self.reader.score(text, documents, arm == "choice_balanced")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite confirmation scores")
        scored, _ = observation(selected, values)
        return dict(scored=scored, pool=sorted(selected), reader_rows=rows, choice_tokens=tokens, flips=flip,
                    ms=(perf_counter() - started) * 1000, truncated=clipped)


def digest_bytes(text):
    return sha256(text).hexdigest()


def evaluate(endpoint, questions):
    out = []
    for i, query in enumerate(questions):
        views = {}
        for state in STATES[i % 2:] + STATES[:i % 2]:
            offset = (i + int(state == "withheld")) % len(ARMS)
            values = {arm: endpoint.run(query, state, arm) for arm in ARMS[offset:] + ARMS[:offset]}
            if len({tuple(r["pool"]) for r in values.values()}) != 1:
                raise ValueError("Methods received different candidate identities")
            views[state] = values
        out.append(dict(id=digest_bytes(query["query"].encode()), targets=query["corpusids"], views=views))
        if (i + 1) % 50 == 0:
            print(f"Completed {i+1}/{len(questions)} paired queries", flush=True)
    return out


def summarize(validation, test):
    projected = [dict(targets=r["targets"], views={state: {arm: d["scored"] for arm, d in vals.items()}
                                                  for state, vals in r["views"].items()}) for r in validation]
    groups = target_groups(test)
    result = {}
    for budget in BUDGETS:
        methods, outcomes = {}, {}
        for arm in ARMS:
            policy = calibrate(projected, arm, budget)
            methods[arm] = dict(calibration=policy, test={})
            outcomes[arm] = {}
            for state in STATES:
                records = [r["views"][state][arm] for r in test]
                accepted = [r["scored"][0] if accepts(r["scored"], policy["policy"]) else None for r in records]
                correct = [int(doc in row["targets"]) for doc, row in zip(accepted, test, strict=True)]
                unknown = [int(doc is not None) - hit for doc, hit in zip(accepted, correct, strict=True)]
                outcomes[arm][state] = correct, unknown
                methods[arm]["test"][state] = dict(
                    known=sum(correct), unjudged=sum(unknown), questions=len(records),
                    top_two_exposure=sum(bool(set(row["targets"]) & set(r["pool"])) for row, r in zip(test, records)),
                    top_one_known=sum(r["scored"][0] in row["targets"] for row, r in zip(test, records)),
                    reader_rows=sum(r["reader_rows"] for r in records), flips=sum(r["flips"] for r in records),
                    truncated=sum(r["truncated"] for r in records),
                    mean_choice_tokens=(None if records[0]["choice_tokens"] is None else
                                        float(np.mean([r["choice_tokens"] for r in records]))),
                    p95_ms=float(np.quantile([r["ms"] for r in records], .95)),
                )
        for arm in ARMS:
            for state in STATES:
                values = methods[arm]["test"][state]
                for key, col in (("known", 0), ("unjudged", 1)):
                    values[f"paired_{key}_delta_vs_binary"] = clustered_paired_interval(
                        np.subtract(outcomes[arm][state][col], outcomes["binary"][state][col]), groups
                    )
        result[f"{budget:.2f}"] = methods
    return result


def run(projection, cache, embedding, reranker, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    with gzip.open(projection, "rb") as stream:
        corpus = json.load(stream)
    if corpus["revision"] != REVISION or corpus["query_sha256"] != QUERY_SHA256:
        raise ValueError("Unexpected LitSearch projection")
    provenance = verify_checkpoint(embedding)
    with np.load(cache, allow_pickle=False) as stored:
        if (str(stored["projection_sha256"].item()) != digest(projection)
                or json.loads(str(stored["provenance"].item())) != provenance):
            raise ValueError("Unexpected dense cache provenance")
        ids = [paper["corpusid"] for paper in corpus["papers"]]
        if stored["corpus_ids"].tolist() != ids:
            raise ValueError("Dense cache source order changed")
        vectors = torch.from_numpy(stored["vectors"]).cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    repeated = Reranker(reranker)
    tokenizer, model, info = load_reader(checkpoint)
    choice = ChoiceReader(tokenizer, model)
    binary = reader_scorer(tokenizer, model, system=INDEPENDENT_SYSTEM)
    endpoint = Endpoint(corpus["papers"], vectors, encoder, repeated, choice, binary)
    calibration, test, _ = split(corpus["questions"])
    for arm in ARMS:
        endpoint.run(calibration[0], "present", arm)
    fit_data = evaluate(endpoint, calibration)
    test_data = evaluate(endpoint, test)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        corpus_papers=len(ids), calibration_queries=len(calibration), test_queries=len(test),
        projection_sha256=digest(projection), cache_sha256=digest(cache), code_sha256=digest(Path(__file__)),
        embedding=provenance["revision"], reranker=repeated.provenance["revision"], choice=info["revision"],
        policies=summarize(fit_data, test_data)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "cache", "embedding", "reranker", "checkpoint", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
