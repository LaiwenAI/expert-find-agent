"""Compare two bounded cross-encoder reranking routes on FollowIR Core17."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from expert_search.qwen_reranker import PREFIX, SUFFIX, pack_tokens
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, ndcg10, write_json


TASK = "Find a document that answers the search query within the stated relevance criteria."
ARM_NAMES = ("topic_only", "with_criteria")


class Reranker:
    def __init__(self, checkpoint):
        self.provenance = json.loads((checkpoint / "checkpoint-provenance.json").read_bytes())
        for name, check in self.provenance["files"].items():
            if digest(checkpoint / name) != check:
                raise ValueError(f"Changed reranker artifact: {name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
        encode = self.tokenizer.encode
        self.prefix = encode(PREFIX, add_special_tokens=False)
        self.suffix = encode(SUFFIX, add_special_tokens=False)
        labels = [encode(word, add_special_tokens=False) for word in ("yes", "no")]
        if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
            raise ValueError("Expected distinct single-token yes/no answers")
        self.labels = [ids[0] for ids in labels]

    def score(self, query, docs, pool, *, task=TASK):
        return self.score_pairs([(query, docs[doc]) for doc in pool], task=task)

    def score_pairs(self, pairs, *, task=TASK, count_tokens=False):
        return self.score_tasks(((task, query, document) for query, document in pairs), count_tokens=count_tokens)

    @torch.inference_mode()
    def score_tasks(self, rows, *, count_tokens=False):
        sequences = []
        truncated = 0
        for task, query, document in rows:
            body = self.tokenizer.encode(f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {document}",
                                         add_special_tokens=False)
            packed, raw = pack_tokens(self.prefix, body, self.suffix, 1024)
            sequences.append(dict(input_ids=packed))
            truncated += raw > len(packed)
        inputs = self.tokenizer.pad(sequences, padding=True, return_tensors="pt").to("cuda")
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, self.labels].float()
        scores = (logits[:, 0] - logits[:, 1]).cpu().tolist()
        if count_tokens:
            return scores, truncated, sum(len(sequence["input_ids"]) for sequence in sequences)
        return scores, truncated

    def rank(self, query, docs, pool):
        start = perf_counter()
        margins, truncated = self.score(query, docs, pool)
        ranking = [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-margins[i], i))]
        return ranking, (perf_counter() - start) * 1000, truncated


def run(data, ranking_path, checkpoint, output):
    dataset = json.loads(data.read_bytes())
    baseline = json.loads(ranking_path.read_bytes())
    if baseline["input_sha256"] != digest(data):
        raise ValueError("Baseline and reranker inputs differ")
    reranker = Reranker(checkpoint)
    rows = []
    for i, (pair, base) in enumerate(zip(dataset["pairs"], baseline["rows"], strict=True)):
        if pair["topic"] != base["topic"]:
            raise ValueError("Unaligned topics")
        fixed = base["arms"]["query_only"]
        candidates = fixed["top10"][0]
        if candidates != fixed["top10"][1] or fixed["invalidated_ranks"][0] != fixed["invalidated_ranks"][1]:
            raise ValueError("Query-only candidates or ranks changed")
        invalidated = set(pair["invalidated"])
        grades = [{d: g for d, g in view.items() if d in pair["pool"]} for view in pair["qrels"]]
        results = {}
        if i == 0:
            reranker.rank(pair["query"], dataset["docs"], candidates)
        offset = i % len(ARM_NAMES)
        for arm in ARM_NAMES[offset:] + ARM_NAMES[:offset]:
            views = [None, None]
            for v in (i % 2, 1 - i % 2):
                query = pair["query"]
                if arm == "with_criteria":
                    query += "\nRelevance criteria: " + pair["instructions"][v]
                ranking, ms, truncated = reranker.rank(query, dataset["docs"], candidates)
                ranks = fixed["invalidated_ranks"][v].copy()
                ranks.update({doc: j for j, doc in enumerate(ranking, 1) if doc in invalidated})
                views[v] = dict(ms=ms, ndcg10=ndcg10(ranking, grades[v]), truncated=truncated,
                                invalidated_at5=len(invalidated.intersection(ranking[:5])), ranks=ranks,
                                top10=ranking)
            old, new = [view["ranks"] for view in views]
            pmrr = mean(math.copysign(1 - min(old[d], new[d]) / max(old[d], new[d]), new[d] - old[d])
                        for d in invalidated)
            results[arm] = dict(views=views, pmrr=pmrr)
        if results["topic_only"]["pmrr"] != 0:
            raise AssertionError("Identical inputs yielded different ranks")
        rows.append(dict(topic=pair["topic"], invalidated_in_pool=len(invalidated.intersection(candidates)),
                         invalidated=len(invalidated), arms=results))
    report = {}
    for name in ARM_NAMES:
        records = [row["arms"][name] for row in rows]
        report[name] = dict(
            pmrr=paired_interval([r["pmrr"] for r in records]),
            changed_ndcg10=mean(r["views"][1]["ndcg10"] for r in records),
            invalidated_at5=[sum(r["views"][v]["invalidated_at5"] for r in records) for v in range(2)],
            p50_p95_ms=np.quantile([v["ms"] for r in records for v in r["views"]], [.5, .95]).tolist(),
            truncated=sum(v["truncated"] for r in records for v in r["views"]),
        )
    write_json(output, dict(report=report, rows=rows, input_sha256=digest(data),
                            baseline_sha256=digest(ranking_path), script_sha256=digest(Path(__file__)),
                            checkpoint=reranker.provenance, gpu=torch.cuda.get_device_name()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("data", "baseline", "checkpoint", "output"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    run(args.data, args.baseline, args.checkpoint, args.output)
