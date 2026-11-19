"""Transfer frozen source-pair heads to human-judged facet search."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_confirmation import accepts
from experiments.csfcube.frozen_confirmation import JudgedPoolEndpoint, load
from experiments.csfcube.grouped_calibration import fit, partition
from experiments.docred.oracle_role_evidence import load_reader
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.contrastive_source_confirmation import ChoiceReader
from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
from experiments.litsearch.distilled_pair_decision import ARMS, STATES, Endpoint
from experiments.litsearch.learned_confirmation import observation
from experiments.litsearch.source_withholding import TASK


STYLES = ("full", "matched_lead", "facet")
BUDGETS = (.05, .10, .20)
LABELS = {"background": {"background_label", "objective_label"},
          "method": {"method_label"}, "result": {"result_label", "conclusion_label"}}


def views(root, questions):
    with (root / "abstracts-csfcube-preds.jsonl").open() as stream:
        papers = {int(row["paper_id"]): row for row in map(json.loads, stream)}
    diagnostics = defaultdict(list)
    output = []
    for query in questions:
        row = papers[query["query_id"]]
        sentences = row["abstract"]
        tags = row["pred_labels"]
        marked = [text for text, tag in zip(sentences, tags)
                  if tag in LABELS[query["facet"]]]
        if len(sentences) != len(tags) and not row.get("pred_labels_truncated"):
            raise ValueError("Unexpected sentence labels")
        chosen = marked[:3] or sentences[:1]
        lead = sentences[:len(chosen)]
        header, _, _ = query["query"].partition("\n")
        requests = dict(full=query["query"])
        for method, spans in (("matched_lead", lead), ("facet", chosen)):
            requests[method] = f"{header}\n{row['title']}\n{' '.join(spans)}"
        diagnostics[query["facet"]].append(dict(
            label_fallback=not marked, selected_sentences=len(chosen),
            selected_equals_lead=chosen == lead, words={k: len(v.split()) for k, v in requests.items()}
        ))
        output.append(requests)
    summary = {facet: dict(
        questions=len(items), fallback=sum(x["label_fallback"] for x in items),
        same_as_first=sum(x["selected_equals_lead"] for x in items),
        mean_words={style: float(np.mean([x["words"][style] for x in items])) for style in STYLES})
        for facet, items in diagnostics.items()}
    return output, summary


class Transfer(Endpoint):
    retrieve = JudgedPoolEndpoint.retrieve

    def run(self, question, state, style, rendered, arm, predictors):
        began = perf_counter()
        pool, _, cosine = self.retrieve(question, state)
        selected = pool[:2]
        if len(selected) != 2 or (state == "withheld" and set(selected) & set(question["corpusids"])):
            raise ValueError("Invalid judged candidate pool")
        text = rendered[style]
        if arm == "teacher":
            if sha256(question["query"].encode()).hexdigest()[0] in "13579bdf":
                selected = selected[::-1]
            values, model_rows, _, _ = self.teacher_scores(text, selected, balanced=False)
            truncated = 0
        else:
            values, truncated = self.reranker.score(text, self.docs, selected, task=TASK)
            model_rows = 2
            if arm != "raw":
                values = predictors[arm](self.features(text, selected, cosine[:2], values))
        if not np.isfinite(values).all():
            raise ValueError("Invalid model scores")
        scored, _ = observation(selected, values)
        return dict(scored=scored, grade=question["grades"][scored[0]], pool=tuple(sorted(selected)),
                    rows=model_rows, truncated=truncated, ms=(perf_counter()-began)*1000)


def restore(reference):
    normal = reference["training"]
    center = np.asarray(normal["center"], dtype=np.float32)
    scale = np.asarray(normal["scale"], dtype=np.float32)
    predictors = {}
    for arm, model in normal["models"].items():
        weight = np.asarray(model["weights"], dtype=np.float32)
        bias = float(model["bias"])
        if weight.shape != center.shape or model["parameters"] != len(weight) + 1:
            raise ValueError("Invalid frozen head dimensions")
        predictors[arm] = lambda x, w=weight, b=bias: ((x-center)/scale) @ w + b
    return predictors


def evaluate(endpoint, questions, rendered, predictors):
    methods = [(style, arm) for style in STYLES for arm in ARMS]
    measured = {style: {arm: [] for arm in ARMS} for style in STYLES}
    exposure = dict.fromkeys(STATES, 0)
    for i, (question, query_views) in enumerate(zip(questions, rendered, strict=True)):
        local = {key: {} for key in methods}
        for state in STATES[i % 2:] + STATES[:i % 2]:
            offset = (i + int(state == "withheld")) % len(methods)
            for style, arm in methods[offset:] + methods[:offset]:
                local[style, arm][state] = endpoint.run(question, state, style, query_views, arm, predictors)
            candidate_pools = {item[state]["pool"] for item in local.values()}
            if len(candidate_pools) != 1:
                raise ValueError("Verification conditions received unequal pools")
            exposure[state] += bool(set(question["corpusids"]) & set(next(iter(candidate_pools))))
        for (style, arm), record in local.items():
            measured[style][arm].append(record)
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{len(questions)} judged queries", flush=True)
    return measured, exposure


def out_of_fold(rows, folds, budget):
    thresholds = [None] * len(rows)
    for test in folds:
        pending = set(test)
        policy = fit([r for i, r in enumerate(rows) if i not in pending], budget)
        for i in test:
            thresholds[i] = policy
    return thresholds


def summary(rows, thresholds, questions):
    groups = [q["query_id"] for q in questions]
    out = {}
    for state in STATES:
        outcomes = [r[state] for r in rows]
        returned = [r["grade"] if accepts(r["scored"], cut) else None
                    for r, cut in zip(outcomes, thresholds, strict=True)]
        grades = Counter(returned)
        out[state] = dict(
            grades=[grades[g] for g in range(4)], abstain=grades[None],
            top_one_grades=[sum(r["grade"] == g for r in outcomes) for g in range(4)],
            mean_returned_grade_mass=sum(g * grades[g] for g in range(4)) / len(rows),
            strong_cluster_interval=clustered_paired_interval([int(g in (2, 3)) for g in returned], groups),
            model_rows=sum(r["rows"] for r in outcomes), truncated=sum(r["truncated"] for r in outcomes),
            p95_ms=float(np.quantile([r["ms"] for r in outcomes], .95)),
        )
    return out


def run(data, reference, embedding, reranker_path, teacher_path, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    frozen = json.loads(reference.read_bytes())
    papers, questions, hashes = load(data)
    rendered, diagnostics = views(data, questions)
    provenance = verify_checkpoint(embedding)
    reranker = Reranker(reranker_path)
    tokenizer, model, info = load_reader(teacher_path)
    if (provenance["revision"], reranker.provenance["revision"], info["revision"]) != (
            frozen["embedding"], frozen["reranker"], frozen["teacher"]):
        raise ValueError("Frozen checkpoint mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(embedding, batch_size=32)
    encoder.model.config.use_cache = False
    vectors = torch.from_numpy(encoder.encode([paper_text(p) for p in papers])).cuda()
    endpoint = Transfer(papers, vectors, encoder, reranker, ChoiceReader(tokenizer, model))
    predictors = restore(frozen)
    folds = partition([(str(q["query_id"]), q["grades"]) for q in questions])
    for style in STYLES:
        for method in ARMS:
            endpoint.run(questions[0], "present", style, rendered[0], method, predictors)
    observations, exposure = evaluate(endpoint, questions, rendered, predictors)
    def compare(style, policies):
        groups = [q["query_id"] for q in questions]
        metrics = {}
        for arm, rows in observations[style].items():
            thresholds = policies[style][arm]
            report = metrics[arm] = summary(rows, thresholds, questions)
            raw_rows = observations["full"]["raw"]
            raw_cuts = policies["full"]["raw"]
            for state in STATES:
                def flags(items, cuts, grade_test):
                    return np.asarray([int(grade_test(d[state]["grade"])) if accepts(d[state]["scored"], cut) else 0
                                       for d, cut in zip(items, cuts, strict=True)])
                for metric, predicate in (("strong", lambda g: g >= 2), ("grade_zero", lambda g: g == 0)):
                    delta = flags(rows, thresholds, predicate) - flags(raw_rows, raw_cuts, predicate)
                    report[state][f"paired_{metric}_vs_raw_full"] = clustered_paired_interval(delta, groups)
        return metrics

    frozen_threshold = {}
    for ceiling, result in frozen["policies"].items():
        policies = {style: {arm: [result[arm]["calibration"]["policy"]] * len(questions)
                            for arm in ARMS} for style in STYLES}
        frozen_threshold[ceiling] = {style: compare(style, policies) for style in STYLES}
    local_threshold = {}
    for budget in BUDGETS:
        policies = {style: {arm: out_of_fold(rows, folds, budget) for arm, rows in by_arm.items()}
                    for style, by_arm in observations.items()}
        local_threshold[f"{budget:.2f}"] = {style: compare(style, policies) for style in STYLES}
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=digest(Path(__file__)), data_sha256=hashes, training_reference_sha256=digest(reference),
        queries=len(questions), query_papers=len({q["query_id"] for q in questions}), corpus_papers=len(papers),
        shared_top_two_exposure=exposure, query_styles=diagnostics, transferred=frozen_threshold,
        local_calibration=local_threshold))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "embedding", "reranker_path", "teacher_path", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
