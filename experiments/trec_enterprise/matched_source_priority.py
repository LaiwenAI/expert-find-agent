"""Compare top-two source priority with the published learned scheduler."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.expert_evidence_budget import FILES, load
from experiments.trec_enterprise.joint_topic_allocation import HistoricalIndex, TOPIC_SHA256, read_topics
from experiments.trec_enterprise.nomination_coverage_scheduler import estimate, fit, observations_for, schedule
from experiments.trec_enterprise.nomination_read_budget import allocate
from experiments.trec_enterprise.nomination_waterfall import PERSON_CAP, expert_votes, lexical_vectors, nominations


def source_priority(people, scores, count, method):
    groups = defaultdict(list)
    rank = {owner: i for i, owner in enumerate(people, 1)}
    for (owner, doc), score in scores.items():
        groups[doc].append((rank[owner], score))
    priorities = {}
    for doc, pairs in groups.items():
        if method == "rank_sum":
            priorities[doc] = sum(1 / index for index, _ in pairs)
        else:
            strengths = sorted((score for _, score in pairs), reverse=True)
            priorities[doc] = sum(strengths[:2])
    return set(sorted(groups, key=lambda doc: (-priorities[doc], doc))[:count])


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a Slurm allocation and fresh results")
    archive = json.loads(args.baseline.read_text())["reverse_order_replicas"][0]["both_caps_matched"]
    experts, documents = load(args.qrels)
    topics = sorted(experts, key=int)
    vectors = lexical_vectors(topics, read_topics(args.qrels / "ent06.expert.topics"))
    records = {}
    for held in reversed(topics) if args.reverse else topics:
        history = HistoricalIndex({t: documents[t] for t in topics if t != held}, topics, vectors)
        design, truth = [], []
        for topic in topics:
            if topic == held:
                continue
            people, owned = expert_votes(history, history.positions[topic])
            _, rows, labels = observations_for(history, topic, people[:PERSON_CAP], owned, experts, documents)
            design.append(rows)
            truth.append(labels)
        model = fit(np.concatenate(design), np.concatenate(truth))
        masks = nominations(history, held)
        people, owned = masks["expert_first"]
        people = people[:PERSON_CAP]
        doc_order, doc_sources = masks["document_first"]
        evidence, relevant = documents[held], experts[held]
        generic_cap = allocate(doc_order[:PERSON_CAP], doc_sources, evidence, relevant, 3, "rank_first")["reads"]
        reuse = allocate(people, owned, evidence, relevant, generic_cap, "reuse_first")
        keys, rows, _ = observations_for(history, held, people, owned, experts, documents)
        scores = dict(zip(keys, map(float, estimate(model, rows)), strict=True))
        reference = schedule(people, scores, evidence, relevant, reuse["reads"], False, reuse["checks"])
        caps = {"reads": reference["reads"], "checks": reference["checks"]}
        results = {"learned_scheduler": reference}
        for method in ("rank_sum", "top_two"):
            shortlist = source_priority(people, scores, caps["reads"], method)
            selected = {key: score for key, score in scores.items() if key[1] in shortlist}
            result = schedule(people, selected, evidence, relevant, caps["reads"], False, caps["checks"])
            if any(result[key] > cap for key, cap in caps.items()):
                raise ValueError(f"{method} exceeded the realized reference budget")
            results[method] = result
            matched = schedule(people, scores, evidence, relevant, result["reads"], False, result["checks"])
            if matched["checks"] > result["checks"] or matched["reads"] > result["reads"]:
                raise ValueError(f"Reference exceeded the {method} budget")
            results[f"scheduler_at_{method}_work"] = matched
        records[held] = results

    summary = {}
    for arm in results:
        rows = [records[t][arm] for t in topics]
        totals = dict(sum((Counter(row) for row in rows), Counter()))
        totals["p_at_5_delta_pp"] = paired_interval([
            20 * (row["supported"] - records[t]["learned_scheduler"]["supported"])
            for t, row in zip(topics, rows, strict=True)
        ])
        if arm in ("rank_sum", "top_two"):
            totals["p_at_5_delta_vs_work_matched_pp"] = paired_interval([
                20 * (row["supported"] - records[t][f"scheduler_at_{arm}_work"]["supported"])
                for t, row in zip(topics, rows, strict=True)
            ])
        summary[arm] = totals
    for k in ("reads", "checks", "unknown", "issued", "supported"):
        if summary["learned_scheduler"][k] != archive["fitted_greedy"][k]:
            raise ValueError(f"Reference replay did not reproduce {k}")
    write_json(args.output, dict(
        summary=summary, topics=len(topics), reverse=args.reverse, job=os.environ["SLURM_JOB_ID"],
        inputs=FILES | {"ent06.expert.topics": TOPIC_SHA256}, baseline_sha256=digest(args.baseline),
        script_sha256=digest(Path(__file__)),
        signature=sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("qrels", "baseline", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
