"""Compare cheap constraint features for judged expansion candidates."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from experiments.followir.core17_reranking import FILES, REVISION, digest, texts, write_json
from experiments.followir.decision_features import NEWS_FILES, REVISION as NEWS_REVISION
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.scifact.minilm_sufficiency_operating_points import operating_threshold
from experiments.scifact.minilm_sufficient_facts import bootstrap_delta, drops_auc


FEATURES = ("topic_similarity", "joint_similarity", "constraint_lift", "full_coverage", "constraint_coverage")
CAPS = (5, 10, 20)
TOP_K = 4
DEPTH = 24


def coverage(terms, document_terms, weights):
    total = sum(weights[w] for w in terms)
    return sum(weights[w] for w in terms & document_terms) / total if total else 0.0


def prepare(dataset_path, cache_path, provenance, encoder):
    import torch

    dataset = json.loads(dataset_path.read_text())
    valid = {REVISION: FILES, NEWS_REVISION: NEWS_FILES}
    if dataset["upstream_sha256"] != valid.get(dataset["revision"]):
        raise ValueError("Unexpected benchmark input")
    with np.load(cache_path, allow_pickle=False) as cached:
        ids = cached["doc_ids"].tolist()
        if (ids != sorted(dataset["docs"]) or str(cached["input_sha256"]) != digest(dataset_path)
                or json.loads(str(cached["checkpoint_provenance"])) != provenance):
            raise ValueError("Frozen vectors do not match the corpus")
        vectors = torch.from_numpy(cached["vectors"]).cuda()
    positions = {doc: i for i, doc in enumerate(ids)}
    terms = {doc: set(tokens(text)) for doc, text in dataset["docs"].items()}
    document_frequency = Counter(term for words in terms.values() for term in words)

    def inspect(pair):
        query, requirement = pair["query"], pair["instructions"][1]
        encoded = encoder.encode(texts("query_only", query, "") + texts("joint", query, requirement))
        ids = pair["pool"]
        similarities = (vectors[[positions[doc] for doc in ids]] @ torch.from_numpy(encoded.T.copy()).cuda()).cpu().numpy()
        order = np.argsort(-similarities[:, 0], kind="stable").tolist()
        chosen = order[TOP_K:DEPTH]
        topic_terms, instruction_terms = set(tokens(query)), set(tokens(requirement))
        words = topic_terms | instruction_terms
        weights = {term: float(np.log((1 + len(terms)) / (1 + document_frequency[term]))) for term in words}
        before, after = pair["qrels"]
        support = [after[ids[j]] for j in chosen if ids[j] in after and after[ids[j]] > 0]
        labeled = len([j for j in chosen if ids[j] in after])
        result = []
        for j in chosen:
            doc = ids[j]
            result.append(dict(
                topic=pair["topic"], grade=after.get(doc), previous=before.get(doc),
                invalidated=doc in pair["invalidated"],
                topic_similarity=float(similarities[j, 0]), joint_similarity=float(similarities[j, 1]),
                constraint_lift=float(similarities[j, 1] - similarities[j, 0]),
                full_coverage=coverage(words, terms[doc], weights),
                constraint_coverage=coverage(instruction_terms - topic_terms, terms[doc], weights),
            ))
        return result, dict(judged=labeled, positives=len(support),
                            positive_top4=sum(after.get(ids[j], 0) > 0 for j in order[:TOP_K]))

    inspect(dataset["pairs"][0])
    records, timings, inventory = [], [], Counter()
    for pair in dataset["pairs"]:
        start = perf_counter()
        scored, audit = inspect(pair)
        timings.append((perf_counter() - start) * 1000)
        records.extend(scored)
        inventory.update(audit)
    return records, dict(topics=len(dataset["pairs"]), expansion_candidates=len(records),
                         **inventory, shared_factored_p95_ms=float(np.quantile(timings, .95)))


def cross_evaluate(calibration, test):
    cutoffs = {}
    changed = [item for item in test if item["grade"] is not None]
    valid = np.asarray([row["grade"] > 0 for row in changed])
    invalidated = np.asarray([row["invalidated"] for row in changed])
    topics = np.asarray([row["topic"] for row in changed])
    unjudged = [row for row in test if row["grade"] is None]
    out = {}
    for feature in FEATURES:
        negatives = np.asarray([row[feature] for row in calibration if row["grade"] == 0])
        scores = np.asarray([row[feature] for row in changed])
        cutoffs[feature] = {}
        for cap in CAPS:
            threshold = operating_threshold(negatives, cap)
            accepts = scores > threshold
            cutoffs[feature][cap] = dict(
                threshold=threshold, calibration_negatives=len(negatives),
                calibration_accepted=int(np.sum(negatives > threshold)),
                positive_accepted=int(np.sum(accepts & valid)),
                negative_accepted=int(np.sum(accepts & ~valid)),
                invalidated_accepted=int(np.sum(accepts & invalidated)),
                unjudged_accepted=sum(row[feature] > threshold for row in unjudged),
            )
        out[feature] = dict(auc_all=drops_auc(scores[valid], scores[~valid]),
                            auc_retained_vs_invalidated=drops_auc(scores[valid], scores[invalidated]),
                            cutoffs=cutoffs[feature])
    for feature in FEATURES:
        if feature == "joint_similarity":
            continue
        for cap in CAPS:
            gate = out[feature]["cutoffs"][cap]
            baseline = out["joint_similarity"]["cutoffs"][cap]
            new = np.asarray([row[feature] > gate["threshold"] for row in changed])
            old = np.asarray([row["joint_similarity"] > baseline["threshold"] for row in changed])
            delta = new.astype(float) - old
            gate["positive_delta_ci95"] = bootstrap_delta(delta[valid], topics[valid])
            gate["negative_delta_ci95"] = bootstrap_delta(delta[~valid], topics[~valid])
    return dict(positives=int(valid.sum()), negatives=int((~valid).sum()), invalidated=int(invalidated.sum()),
                unjudged=len(unjudged), features=out)


def run(core, core_cache, news, news_cache, checkpoint, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(checkpoint)
    encoder = QwenTorchEmbedding(checkpoint, batch_size=24)
    encoder.model.config.use_cache = False
    core_rows, core_report = prepare(core, core_cache, provenance, encoder)
    print("Core17 complete", core_report, flush=True)
    news_rows, news_report = prepare(news, news_cache, provenance, encoder)
    print("News21 complete", news_report, flush=True)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            script_sha256=digest(Path(__file__)), dataset_sha256=[digest(core), digest(news)],
                            cache_sha256=[digest(core_cache), digest(news_cache)], embedding_revision=provenance["revision"],
                            core17=core_report, news21=news_report, caps=CAPS,
                            report={"core17_to_news21": cross_evaluate(core_rows, news_rows),
                                    "news21_to_core17": cross_evaluate(news_rows, core_rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("core", "core_cache", "news", "news_cache", "checkpoint", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
