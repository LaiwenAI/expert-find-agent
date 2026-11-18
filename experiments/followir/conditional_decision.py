"""Learn a small conditional ranking head and test its rejection policy on human qrels."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, ndcg10, pmrr, texts, write_json


TRAIN_SHA256 = "ca99cc3d1ab885eefa3674f980a498efa5f230770e24c55bcff8e784811fc4b6"
SEED = 9423
ARMS = ("joint", "topic_pair", "joint_binary", "joint_pair")
HEADS = {
    "topic_pair": ("query_only", True),
    "joint_binary": ("joint", False),
    "joint_pair": ("joint", True),
}


def words(text):
    return re.findall(r"\w+", text.casefold())


def normalized(text):
    return " ".join(words(text))


def shingles(text, width):
    terms = words(text)
    return set(zip(*(terms[i:] for i in range(width))))


def overlaps(left, right, threshold):
    return bool(left and right and len(left & right) / len(left | right) >= threshold)


def training_data(path, collections):
    if digest(path) != TRAIN_SHA256:
        raise ValueError("Unexpected training data")
    examples = [json.loads(line) for line in path.read_text().splitlines()]
    test_pairs = [pair for collection in collections for pair in collection["pairs"]]
    test_queries = [set(words(p["query"])) for p in test_pairs]
    test_narratives = [shingles(t, 5) for p in test_pairs for t in p["instructions"]]
    blocked = set()
    for row in examples:
        query = set(words(row["query"]))
        narrative = shingles(row["instruction"], 5)
        if (any(overlaps(query, target, .75) for target in test_queries)
                or any(overlaps(narrative, target, .70) for target in test_narratives)):
            blocked.add(normalized(row["query"]))
    kept = [r for r in examples if normalized(r["query"]) not in blocked]
    groups = defaultdict(set)
    for row in kept:
        groups[row["id"]].add(normalized(row["query"]))
    if any(len(keys) != 1 for keys in groups.values()):
        raise ValueError("A source ID spans multiple split groups")
    return kept, len(examples) - len(kept)


def validation_key(query):
    return int.from_bytes(sha256(normalized(query).encode()).digest()[:8], "big") % 5 == 0


class DecisionHead(nn.Module):
    """Residual bilinear interaction with an indexable document projection."""

    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, 16, bias=False)
        self.document = nn.Linear(width, 16, bias=False)
        self.offset = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.document.weight)

    def forward(self, q, d):
        return 10 * (q * d).sum(-1) + (self.query(q) * self.document(d)).sum(-1) + self.offset

    def score_index(self, q, documents, projected):
        return 10 * (documents @ q) + projected @ self.query(q) + self.offset


def fit(queries, documents, y, train, val, matched_pairs, pair_loss):
    torch.manual_seed(SEED)
    model = DecisionHead(queries.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.03)
    preferred, rejected = matched_pairs
    started = perf_counter()
    best = None
    for epoch in range(100):
        scores = model(queries, documents)
        loss = F.binary_cross_entropy_with_logits(scores[train], y[train])
        if pair_loss:
            loss = loss + .5 * F.softplus(scores[rejected] - scores[preferred]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            scores = model(queries[val], documents[val])
            held_loss = F.binary_cross_entropy_with_logits(scores, y[val]).item()
            if best is None or held_loss < best[0]:
                best = held_loss, epoch + 1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best[2])
    model.eval()
    with torch.inference_mode():
        scores = model(queries[val], documents[val]).cpu().numpy()
    positive = np.sort(scores[y[val].cpu().numpy() == 1])
    # Strict comparison yields at most 5% observed positive flags in calibration.
    threshold = float(positive[int(.05 * len(positive))])
    return model, {
        "epoch": best[1], "validation_bce": best[0], "threshold": threshold,
        "validation_positive_rejections": int((positive < threshold).sum()),
        "validation_positives": len(positive), "training_seconds": perf_counter() - started,
    }


def groups_for_fit(rows):
    training, validation = [], []
    candidates = defaultdict(dict)
    for i, row in enumerate(rows):
        is_validation = validation_key(row["query"])
        (validation if is_validation else training).append(i)
        if not is_validation:
            group = candidates[(row["id"], row["query"], row["instruction"])]
            if row["label"] in group:
                raise ValueError("Duplicate class within a training contrast")
            group[row["label"]] = i
    contrasts = [group for group in candidates.values() if set(group) == {"true", "false"}]
    return training, validation, ([group["true"] for group in contrasts], [group["false"] for group in contrasts])


def encoder_inputs(mode, query, narrative):
    return texts(mode, query, narrative)[0]


def rejection_stats(shortlist, scores, ids, before, after, invalidated, threshold):
    disposition = dict(invalidated=0, invalidated_flagged=0, retained=0, retained_flagged=0,
                       other_flagged=0, unjudged_flagged=0)
    for doc in shortlist:
        flagged = scores[ids[doc]] < threshold
        if doc in invalidated:
            disposition["invalidated"] += 1
            disposition["invalidated_flagged"] += int(flagged)
        elif before.get(doc, 0) > 0 and after.get(doc, 0) > 0:
            disposition["retained"] += 1
            disposition["retained_flagged"] += int(flagged)
        elif flagged:
            key = "other_flagged" if doc in after else "unjudged_flagged"
            disposition[key] += 1
    return disposition


@torch.inference_mode()
def evaluate(dataset, path, features, provenance, encoder, heads, fit_reports):
    with np.load(features, allow_pickle=False) as cache:
        if str(cache["input_sha256"]) != digest(path) or json.loads(str(cache["checkpoint_provenance"])) != provenance:
            raise ValueError("Cached vectors came from a different input or checkpoint")
        vector_ids = cache["doc_ids"].tolist()
        documents = torch.from_numpy(cache["vectors"]).cuda()
    if vector_ids != sorted(dataset["docs"]):
        raise ValueError("Cached source IDs do not match the dataset")
    positions = {doc: i for i, doc in enumerate(vector_ids)}
    projected = {arm: head.document(documents) for arm, head in heads.items()}
    rows = []
    for i, pair in enumerate(dataset["pairs"]):
        pool = pair["pool"]
        ids = {doc: pos for pos, doc in enumerate(pool)}
        indices = [positions[doc] for doc in pool]
        candidate_vectors = documents[indices]
        candidate_projection = {arm: matrix[indices] for arm, matrix in projected.items()}
        pool_grades = [{doc: grade for doc, grade in view.items() if doc in ids} for view in pair["qrels"]]
        invalidated = set(pair["invalidated"])
        torch.cuda.synchronize()

        def rank(arm, narrative):
            start = perf_counter()
            mode = HEADS[arm][0] if arm in heads else "joint"
            q = torch.from_numpy(encoder.encode([encoder_inputs(mode, pair["query"], narrative)])[0]).cuda()
            if arm in heads:
                scores = heads[arm].score_index(q, candidate_vectors, candidate_projection[arm])
            else:
                scores = candidate_vectors @ q
            scores = scores.cpu().numpy()
            ranking = [pool[j] for j in np.argsort(-scores, kind="stable")]
            return ranking, scores, (perf_counter() - start) * 1000

        results = {}
        candidate_scores = {}
        if i == 0:
            for arm in ARMS:
                rank(arm, pair["instructions"][0])
        offset = i % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            views = [None, None]
            for view in (i % 2, 1 - i % 2):
                views[view] = rank(arm, pair["instructions"][view])
            before, after = [entry[0] for entry in views]
            candidate_scores[arm] = views[1][1]
            results[arm] = dict(
                pmrr=pmrr(before, after, invalidated),
                changed_ndcg10=ndcg10(after, pool_grades[1]),
                invalidated_at10=[len(invalidated.intersection(ranking[:10])) for ranking in (before, after)],
                top10=[before[:10], after[:10]],
                ms=[entry[2] for entry in views],
            )
        if results["topic_pair"]["pmrr"] != 0:
            raise AssertionError("Topic-only control reacted to an unseen narrative")
        flags = {}
        for arm in heads:
            flags[arm] = rejection_stats(results["joint"]["top10"][1], candidate_scores[arm], ids,
                                         *pool_grades, invalidated, fit_reports[arm]["threshold"])
        rows.append(dict(topic=pair["topic"], arms=results, shortlist_flags=flags))

    report = {}
    for arm in ARMS:
        values = [row["arms"][arm] for row in rows]
        report[arm] = dict(
            pmrr=paired_interval([v["pmrr"] for v in values]),
            ndcg10=mean(v["changed_ndcg10"] for v in values),
            ndcg10_minus_joint=paired_interval([
                row["arms"][arm]["changed_ndcg10"] - row["arms"]["joint"]["changed_ndcg10"] for row in rows
            ]),
            invalidated_at10=[sum(v["invalidated_at10"][view] for v in values) for view in (0, 1)],
            p50_p95_ms=np.quantile([ms for v in values for ms in v["ms"]], [.5, .95]).tolist(),
        )
        if arm in heads:
            flags = [row["shortlist_flags"][arm] for row in rows]
            report[arm]["shortlist_flags"] = {key: sum(row[key] for row in flags) for key in flags[0]}
    return dict(input_sha256=digest(path), cache_sha256=digest(features), report=report, rows=rows)


def run(train_path, checkpoint, eval_paths, output):
    if output.exists():
        raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = json.loads((checkpoint / "checkpoint-provenance.json").read_bytes())
    for name, expected in provenance["files"].items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"Changed checkpoint artifact: {name}")
    collections = [json.loads(path.read_bytes()) for path, _ in eval_paths]
    rows, overlap_drops = training_data(train_path, collections)
    train, val, matched = groups_for_fit(rows)
    encoder = QwenTorchEmbedding(checkpoint)
    encoder.model.config.use_cache = False
    started = perf_counter()
    documents = torch.from_numpy(encoder.encode([r["document"] for r in rows])).cuda()
    q = {mode: torch.from_numpy(encoder.encode(
        [encoder_inputs(mode, r["query"], r["instruction"]) for r in rows])).cuda()
        for mode in ("joint", "query_only")}
    y = torch.tensor([int(r["label"] == "true") for r in rows], dtype=torch.float32, device="cuda")
    feature_seconds = perf_counter() - started
    heads, fit_reports = {}, {}
    for arm, (mode, pair_loss) in HEADS.items():
        heads[arm], fit_reports[arm] = fit(q[mode], documents, y, train, val, matched, pair_loss)
    print("Trained three decision heads", fit_reports, flush=True)
    results = {collection["revision"]: evaluate(collection, data, cache, provenance, encoder, heads, fit_reports)
               for collection, (data, cache) in zip(collections, eval_paths, strict=True)}
    write_json(output, dict(
        gpu=torch.cuda.get_device_name(), checkpoint=provenance, rows_train=len(train), rows_validation=len(val),
        overlap_rows_removed=overlap_drops, training_contrasts=len(matched[0]), train_sha256=TRAIN_SHA256,
        script_sha256=digest(Path(__file__)), fit=fit_reports, feature_seconds=feature_seconds, evaluations=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--evaluation", type=Path, nargs=2, action="append", required=True,
                        metavar=("PREPARED_JSON", "VECTORS"))
    args = parser.parse_args()
    run(args.training, args.checkpoint, args.evaluation, args.output)
