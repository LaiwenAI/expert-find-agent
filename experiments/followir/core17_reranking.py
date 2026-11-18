"""Replay paired FollowIR Core17 narratives over the same released candidate pools."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter


REVISION = "7030c7efc3585d9020f243b12862997889243b78"
FILES = {
    "corpus": "f668289b7b57be7a3c8ea46b9ee3764661dca311b2eee16e204e28f9350e5bb9",
    "default": "ce159b0ffc14b798d938572bbfcf7ffa3e924fc1644f40452b033be8dc352bf0",
    "instruction": "88adf1f7422eccd9d849a5d57454b854ff7bc7be7d412376c72f0269a0170f1e",
    "qrel_diff": "4cd756a4601700425871197fff95ee5b9ccdbe313c84488e7f1430635ef7673f",
    "queries": "f0a0a46e154f5c1a57ef4b670dda35af0e38653c8f09e5380f78b556fc549450",
    "top_ranked": "90bf35b34f0a8bc4a550f8f7d5f4dea8877f89da843cb30ebacf38586361e4cf",
}
VIEWS = ("og", "changed")
ARMS = ("query_only", "joint", "late_fusion")
TASK = "Given a search query, retrieve relevant passages that meet its relevance criteria."


def digest(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    result = sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False)
        stream.write("\n")


def prepare(root, output):
    from pyarrow import parquet

    def load(name):
        path = root / f"{name}.parquet"
        if digest(path) != FILES[name]:
            raise ValueError(f"Unexpected upstream file: {path}")
        return parquet.read_table(path).to_pylist()

    queries = {r["_id"]: r["text"] for r in load("queries")}
    instructions = {r["query-id"]: r["instruction"] for r in load("instruction")}
    pools = {r["query-id"]: set(r["corpus-ids"]) for r in load("top_ranked")}
    changed = {r["query-id"]: set(r["corpus-ids"]) for r in load("qrel_diff")}
    qrels = defaultdict(dict)
    for row in load("default"):
        qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    pairs = []
    used = set()
    for topic, invalidated in sorted(changed.items()):
        old, new = (f"{topic}-{view}" for view in VIEWS)
        pool = pools[old]
        if len(pool) != 1000 or pool != pools[new] or queries[old] != queries[new]:
            raise ValueError(f"Invalid paired candidate pool: {topic}")
        before, after = qrels[old], qrels[new]
        diff = {doc for doc in before.keys() | after.keys() if before.get(doc) != after.get(doc)}
        if diff != invalidated or not invalidated <= pool:
            raise ValueError(f"Unexpected qrel diff or coverage: {topic}")
        if any(before[doc] <= 0 or after[doc] != 0 for doc in invalidated):
            raise ValueError(f"Invalidated document did not go from relevant to nonrelevant: {topic}")
        pairs.append(dict(topic=topic, query=queries[old], instructions=[instructions[old], instructions[new]],
                          pool=sorted(pool), qrels=[before, after], invalidated=sorted(invalidated)))
        used.update(pool)
    docs = {r["_id"]: f'{r["title"]}\n{r["text"]}' for r in load("corpus") if r["_id"] in used}
    if set(docs) != used or len(pairs) != 20 or sum(len(p["invalidated"]) for p in pairs) != 436:
        raise ValueError("Unexpected benchmark inventory")
    write_json(output, dict(revision=REVISION, upstream_sha256=FILES, pairs=pairs, docs=docs))


def pmrr(before, after, invalidated):
    old = {doc: rank for rank, doc in enumerate(before, 1)}
    new = {doc: rank for rank, doc in enumerate(after, 1)}
    return mean(
        math.copysign(1 - min(old[d], new[d]) / max(old[d], new[d]), new[d] - old[d])
        for d in invalidated
    )


def ndcg10(ranking, grades):
    def gain(values):
        return sum(value / math.log2(i + 2) for i, value in enumerate(values[:10]))

    return gain([grades.get(doc, 0) for doc in ranking[:10]]) / gain(sorted(grades.values(), reverse=True))


def texts(arm, query, instruction):
    if arm == "joint":
        inputs = [f"{query}\nRelevance criteria: {instruction}"]
    elif arm == "late_fusion":
        inputs = [query, instruction]
    else:
        inputs = [query]
    return [f"Instruct: {TASK}\nQuery: {value}" for value in inputs]


def run(data, model_path, output):
    import numpy as np
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from expert_search.statistics import paired_interval

    dataset = json.loads(data.read_bytes())
    if (dataset["revision"], dataset["upstream_sha256"]) != (REVISION, FILES):
        raise ValueError("Unexpected prepared dataset")
    provenance = json.loads((model_path / "checkpoint-provenance.json").read_bytes())
    for name, check in provenance["files"].items():
        if digest(model_path / name) != check:
            raise ValueError(f"Changed embedding artifact: {name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    embedder = QwenTorchEmbedding(model_path)
    embedder.model.config.use_cache = False
    ids = sorted(dataset["docs"])
    positions = {doc: i for i, doc in enumerate(ids)}
    start = perf_counter()
    vectors = torch.as_tensor(embedder.encode([dataset["docs"][doc] for doc in ids]),
                              device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    offline = perf_counter() - start

    @torch.inference_mode()
    def rank(arm, query, instruction, candidates, pool):
        start = perf_counter()
        embedding = torch.as_tensor(embedder.encode(texts(arm, query, instruction)),
                                    device="cuda", dtype=torch.float32)
        scores = (candidates @ embedding.T).mean(dim=1).cpu().numpy()
        permutation = np.argsort(-scores, kind="stable")
        ranking = [pool[i] for i in permutation]
        return ranking, (perf_counter() - start) * 1000

    for arm in ARMS:
        rank(arm, "search topic", "required subject", vectors[:1000], ids[:1000])
    rows = []
    for index, pair in enumerate(dataset["pairs"]):
        results = {}
        pool = pair["pool"]
        candidates = vectors[[positions[doc] for doc in pool]]
        torch.cuda.synchronize()
        invalidated = set(pair["invalidated"])
        offset = index % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            ranked = [None, None]
            for i in (index % 2, 1 - index % 2):
                ranked[i] = rank(arm, pair["query"], pair["instructions"][i], candidates, pool)
            before, after = (result[0] for result in ranked)
            results[arm] = dict(
                pmrr=pmrr(before, after, invalidated),
                current_ndcg10=[ndcg10(ranking, {d: g for d, g in grades.items() if d in pool})
                                for ranking, grades in zip((before, after), pair["qrels"])],
                invalidated_at10=[len(invalidated.intersection(ranking[:10])) for ranking in (before, after)],
                ms=[result[1] for result in ranked],
                top10=[before[:10], after[:10]],
                invalidated_ranks=[{d: rank for rank, d in enumerate(ranking, 1) if d in invalidated}
                                   for ranking in (before, after)],
            )
        rows.append(dict(topic=pair["topic"], invalidated=len(invalidated), arms=results))
    if any(row["arms"]["query_only"]["pmrr"] != 0 for row in rows):
        raise AssertionError("Query-only ranking changed for identical inputs")
    report = {}
    for arm in ARMS:
        records = [row["arms"][arm] for row in rows]
        report[arm] = dict(
            pmrr=paired_interval([row["pmrr"] for row in records]),
            changed_ndcg10=mean(row["current_ndcg10"][1] for row in records),
            invalidated_at10_total=[sum(row["invalidated_at10"][i] for row in records) for i in range(2)],
            latency_p50_p95_ms=np.quantile([ms for row in records for ms in row["ms"]], [.5, .95]).tolist(),
            changed_ndcg10_minus_query_only=paired_interval([
                row["arms"][arm]["current_ndcg10"][1] - row["arms"]["query_only"]["current_ndcg10"][1]
                for row in rows
            ]),
        )
    write_json(output, dict(
        report=report, rows=rows, revision=REVISION, upstream_sha256=FILES, input_sha256=digest(data),
        script_sha256=digest(Path(__file__)), checkpoint=provenance, gpu=torch.cuda.get_device_name(),
        unique_candidates=len(ids), offline_document_seconds=offline,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("parquet_dir", type=Path)
    prep.add_argument("output", type=Path)
    replay = sub.add_parser("run")
    replay.add_argument("data", type=Path)
    replay.add_argument("model", type=Path)
    replay.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.parquet_dir, args.output)
    else:
        run(args.data, args.model, args.output)
