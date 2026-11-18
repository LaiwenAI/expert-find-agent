"""Test whether source-state routing reduces the paragraph budget on QASPER."""

import argparse
from hashlib import file_digest, sha256
import importlib.util
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval


TASK = "Retrieve paragraphs from the research paper that answer the question."
ASSETS = {
    "qasper-dev-v0.3.json": "2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93",
    "qasper-test-v0.3.json": "6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5",
}
EVALUATOR_SHA256 = "781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe"
ACTIONS = ("stop", "dense", "diverse")
BASE_K, EXTENDED_K, STATIC_K = 2, 8, 5
CALIBRATION_TOKEN_RATIO = .90
STOP = frozenset("a an and are as at be by does for from how in is it of on or the this to was what which with".split())


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read_data(directory, split):
    name = f"qasper-{split}-v0.3.json"
    path = directory / name
    if digest(path) != ASSETS[name]:
        raise ValueError(f"Changed QASPER data: {name}")
    return json.loads(path.read_bytes())


def paper_inputs(record):
    paragraphs = {}
    for section in record["full_text"]:
        for paragraph in section["paragraphs"]:
            if paragraph.strip():
                paragraphs.setdefault(paragraph, section["section_name"] or "")
    return list(paragraphs), list(paragraphs.values())


def evidence_sets(qa, paragraphs):
    present = set(paragraphs)
    refs = []
    for annotation in qa["answers"]:
        answer = annotation["answer"]
        refs_text = {line for line in answer["evidence"] if "FLOAT SELECTED" not in line}
        if not answer["unanswerable"] and refs_text and refs_text <= present:
            refs.append(refs_text)
    return refs


def passage_orders(scores, sections):
    dense = np.argsort(-scores, kind="stable")[:EXTENDED_K].tolist()
    diverse = dense[:BASE_K]
    seen = {sections[i] for i in diverse}
    order = np.argsort(-scores, kind="stable")
    for index in order:
        if len(diverse) >= min(EXTENDED_K, len(order)):
            break
        section = sections[index]
        if section not in seen:
            diverse.append(int(index))
            seen.add(section)
    diverse.extend(int(i) for i in order if i not in diverse)
    return {"stop": dense[:BASE_K], "fixed5": dense[:STATIC_K], "dense": dense,
            "diverse": diverse[:EXTENDED_K]}


def features(question, paragraphs, sections, scores, orders, lengths):
    chosen = orders["stop"]
    anchors = set(tokens(question)) - STOP
    matched = [anchors & set(tokens(paragraphs[i])) for i in chosen]
    seen = set().union(*matched)
    first = scores[chosen[0]]
    second = scores[chosen[-1]]
    deep_order = orders["dense"]
    third = scores[deep_order[min(2, len(deep_order) - 1)]]
    last = scores[deep_order[-1]]
    same_section = len({sections[i] for i in chosen}) == 1
    return np.array([
        first, second, first - second, second - third, second - last, np.log1p(len(anchors)),
        len(seen) / max(1, len(anchors)), sum(map(len, matched)) / max(1, 2 * len(anchors)),
        same_section, np.log1p(len(paragraphs)), np.log1p(len(set(sections))),
        np.log1p(sum(lengths[i] for i in chosen)),
    ], dtype=np.float64)


def quality(predicted, references):
    if not references:
        return None
    selected = set(predicted)
    matches = [len(selected & gold) for gold in references]
    return {
        "f1": max(2 * n / (len(selected) + len(gold)) for n, gold in zip(matches, references)),
        "complete_evidence": int(any(n == len(gold) for n, gold in zip(matches, references))),
        "annotator_recall": max(n / len(gold) for n, gold in zip(matches, references)),
    }


def offline(dataset, encoder):
    metadata = {}
    texts = []
    started = perf_counter()
    for pid, paper in dataset.items():
        paragraphs, sections = paper_inputs(paper)
        start = len(texts)
        texts.extend([f"{section}\n{text}" for section, text in zip(sections, paragraphs, strict=True)])
        metadata[pid] = (paragraphs, sections, start, len(texts))
    count = encoder.tokenizer(texts, return_length=True, add_special_tokens=False)["length"]
    counted = perf_counter()
    vectors = encoder.encode(texts)
    return metadata, count, vectors, {"counting_seconds": counted - started, "encoding_seconds": perf_counter() - counted,
                                       "paragraphs": len(texts)}


def query_input(question):
    return f"Instruct: {TASK}\nQuery:{question}"


def run_rows(dataset, encoder, metadata, lengths, bank, decide=None):
    rows = []
    if decide is None:
        query_rows = [(pid, item) for pid, paper in dataset.items() for item in paper["qas"]]
        query_vectors = encoder.encode([query_input(qa["question"]) for _, qa in query_rows])
    else:
        query_vectors = None
    query_pos = 0
    for pid, paper in dataset.items():
        paragraphs, sections, start, end = metadata[pid]
        local_lengths = lengths[start:end]
        for qa in paper["qas"]:
            if not paragraphs:
                raise ValueError(f"Paper has no retrievable paragraphs: {pid}")
            began = perf_counter()
            if query_vectors is None:
                query = encoder.encode([query_input(qa["question"])])[0]
            else:
                query = query_vectors[query_pos]
            query_pos += 1
            scores = bank[start:end] @ query
            orders = passage_orders(scores, sections)
            state = features(qa["question"], paragraphs, sections, scores, orders, local_lengths)
            selected = dict(orders)
            if decide is not None:
                selected.update({method: orders[action] for method, action in decide(state).items()})
            # Materialize all output contexts. Reader/generator inference is outside the benchmark.
            contexts = {method: [paragraphs[i] for i in order] for method, order in selected.items()}
            elapsed = (perf_counter() - began) * 1000
            gold = evidence_sets(qa, paragraphs)
            arms = {method: dict(paragraph_indices=order, input_tokens=sum(local_lengths[i] for i in order),
                                passages=len(order), quality=quality(contexts[method], gold))
                    for method, order in selected.items()}
            rows.append(dict(paper_id=pid, question_id=qa["question_id"], state=state.tolist(),
                             timing_ms=elapsed, evidence_evaluable=bool(gold), arms=arms))
    return rows


def partition(pid):
    return int(sha256(pid.encode()).hexdigest(), 16) % 2


def fit(rows):
    fitted = [row for row in rows if partition(row["paper_id"]) == 0 and row["evidence_evaluable"]]
    x = np.asarray([row["state"] for row in fitted])
    location, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    scaled = (x - location) / scale
    design = np.column_stack((np.ones(len(scaled)), scaled))
    regularization = np.eye(design.shape[1]) * 8
    regularization[0, 0] = 0
    y = []
    for row in fitted:
        scores = {k: row["arms"][k]["quality"] for k in ACTIONS}
        stop = scores["stop"]
        y.append([scores[k]["complete_evidence"] - stop["complete_evidence"]
                  + .25 * (scores[k]["annotator_recall"] - stop["annotator_recall"]) for k in ACTIONS[1:]])
    beta = np.linalg.solve(design.T @ design + regularization, design.T @ np.asarray(y))
    return location, scale, beta, len(fitted)


def action_scores(state, model):
    location, scale, beta, _ = model
    design = np.r_[1, (state - location) / scale]
    return design @ beta


def standard_actions(state, model):
    scores = action_scores(state, model)
    best = int(np.argmax(scores))
    return {"gap": (-float(state[3]), "dense"), "learned": (float(scores[best]), ACTIONS[best + 1])}


def costs(rows, method):
    return sum(row["arms"][method]["input_tokens"] for row in rows)


def calibrate(rows, model):
    calibration = [row for row in rows if partition(row["paper_id"]) == 1]
    thresholds = {}
    target = CALIBRATION_TOKEN_RATIO * costs(calibration, "fixed5")
    start = costs(calibration, "stop")
    for name in ("gap", "learned"):
        proposals = []
        for row in calibration:
            score, action = standard_actions(np.asarray(row["state"]), model)[name]
            delta = row["arms"][action]["input_tokens"] - row["arms"]["stop"]["input_tokens"]
            proposals.append((score, delta))
        latest = max(score for score, _ in proposals)
        remaining = target - start
        for threshold in sorted({score for score, _ in proposals}, reverse=True):
            spent = sum(delta for score, delta in proposals if score == threshold)
            if spent > remaining:
                break
            remaining -= spent
            latest = float(np.nextafter(threshold, -np.inf))
        thresholds[name] = {"threshold": latest, "calibration_input_token_ratio":
                            (target - remaining) / costs(calibration, "fixed5")}

    def choose(state):
        return {name: action if score > thresholds[name]["threshold"] else "stop"
                for name, (score, action) in standard_actions(state, model).items()}

    return choose, thresholds, len(calibration)


def summarize(rows, evaluator, dataset):
    gold = evaluator.get_answers_and_evidence(dataset, True)
    answerable = [row for row in rows if row["evidence_evaluable"]]
    groups = [r["paper_id"] for r in answerable]
    metrics = {}
    for arm in rows[0]["arms"]:
        official = []
        for row in rows:
            paragraphs, _ = paper_inputs(dataset[row["paper_id"]])
            selected = [paragraphs[i] for i in row["arms"][arm]["paragraph_indices"]]
            official.append(max(evaluator.paragraph_f1_score(selected, r["evidence"])
                                for r in gold[row["question_id"]]))
        result = {"official_text_evidence_f1_all": mean(official),
                  "reader_input_tokens_mean_all": costs(rows, arm) / len(rows),
                  "reader_input_tokens_p95_all": float(np.quantile(
                      [row["arms"][arm]["input_tokens"] for row in rows], .95)),
                  "paragraphs_mean_all": mean(row["arms"][arm]["passages"] for row in rows)}
        for field in ("f1", "complete_evidence", "annotator_recall"):
            result[field] = mean(row["arms"][arm]["quality"][field] for row in answerable)
            result[f"{field}_delta_vs_fixed5"] = clustered_paired_interval([
                row["arms"][arm]["quality"][field] - row["arms"]["fixed5"]["quality"][field]
                for row in answerable], groups)
        metrics[arm] = result
    return metrics


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    dev, test = (read_data(args.data, split) for split in ("dev", "test"))
    if dev.keys() & test.keys():
        raise ValueError("Paper IDs overlap split")
    checkpoint = json.loads((args.model / "checkpoint-provenance.json").read_bytes())
    if checkpoint["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected encoder checkpoint")
    for name, checksum in checkpoint["files"].items():
        if digest(args.model / name) != checksum:
            raise ValueError(f"Changed checkpoint file: {name}")
    encoder = QwenTorchEmbedding(args.model, max_length=512, batch_size=32)
    encoder.model.config.use_cache = False
    encoder.encode([query_input("What method is used?")])
    meta, lengths, vectors, dev_timing = offline(dev, encoder)
    print("Development vectors built", dev_timing, flush=True)
    dev_rows = run_rows(dev, encoder, meta, lengths, vectors)
    model = fit(dev_rows)
    decide, thresholds, calibration_items = calibrate(dev_rows, model)
    meta, lengths, vectors, test_timing = offline(test, encoder)
    print("Test vectors built", test_timing, flush=True)
    rows = run_rows(test, encoder, meta, lengths, vectors, decide)
    evaluator_file = args.data / "qasper_evaluator.py"
    if digest(evaluator_file) != EVALUATOR_SHA256:
        raise ValueError("Changed evaluator")
    spec = importlib.util.spec_from_file_location("qasper_evaluator", evaluator_file)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    summary = dict(
        checkpoint=checkpoint, gpu=encoder.torch.cuda.get_device_name(),
        dev_papers=len(dev), test_papers=len(test), dev_questions=len(dev_rows), test_questions=len(rows),
        test_evidence_evaluable=sum(row["evidence_evaluable"] for row in rows),
        test_papers_evidence_evaluable=len({row["paper_id"] for row in rows if row["evidence_evaluable"]}),
        fitting_questions=model[-1], calibration_questions=calibration_items,
        thresholds=thresholds, development_offline_timing=dev_timing, test_offline_timing=test_timing,
        shared_ms_p50_p95=list(map(float, np.quantile([row["timing_ms"] for row in rows], [.50, .95]))),
        metrics=summarize(rows, evaluator, test), data_sha256=ASSETS, official_evaluator_sha256=digest(evaluator_file),
        code_sha256=digest(Path(__file__)),
    )

    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "test_rows.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    summary["test_rows_sha256"] = digest(path)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
