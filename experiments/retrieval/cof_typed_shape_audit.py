"""Localize shape-dependent verdicts in immediate and deferred verification."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval import cof_typed_evidence as previous

SCHEDULES = ("selective", "deferred_typed")
SHAPES = {"dynamic": (False, False), "fixed": (True, True)}
ABLATIONS = {"generic_fixed": (True, False), "typed_fixed": (False, True)}


def threshold_state(margin):
    return 0 if margin < 0. else 1 if margin < previous.BAND else 2


def differences(immediate, deferred):
    a, b = immediate["generic_trace"], deferred["generic_trace"]
    c, d = immediate["typed_trace"], deferred["typed_trace"]
    common_generic, common_typed = a.keys() & b.keys(), c.keys() & d.keys()
    return dict(common_generic=len(common_generic), common_typed=len(common_typed),
                generic_score_drift=sum(a[key] != b[key] for key in common_generic),
                generic_threshold_drift=sum(threshold_state(a[key]) != threshold_state(b[key])
                                            for key in common_generic),
                typed_decision_drift=sum(c[key] != d[key] for key in common_typed),
                exact=immediate["ranked"] == deferred["ranked"])


def run(dataset, embedding, reranker, reader, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and a Slurm CUDA allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = previous.verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = previous.read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = previous.profiles.ProfileIndex(reference)
    encoder = previous.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    cheap = previous.Reranker(reranker)
    expensive = previous.TypedReader(reader)
    chosen = previous.profiles.select(dataset, ids, texts)
    warmup = next(qid for qid in ids if qid not in chosen)

    def measure(qid, shape, schedule):
        fixed_generic, fixed_typed = (SHAPES | ABLATIONS)[shape]
        return previous.request(index, encoder, cheap, expensive, texts, texts[qid], labels[qid], schedule,
                                fixed_generic=fixed_generic, fixed_typed=fixed_typed, audit=True)

    arms = [(shape, schedule) for shape in SHAPES for schedule in SCHEDULES]
    for shape, schedule in arms:
        measure(warmup, shape, schedule)
    rows = []
    for i, qid in enumerate(chosen):
        offset = i % len(arms)
        observations = {arm: measure(qid, *arm) for arm in arms[offset:] + arms[:offset]}
        if any(row["candidate_ids"] != observations[arms[0]]["candidate_ids"] for row in observations.values()):
            raise AssertionError("Candidate order changed")
        comparisons = {shape: differences(*(observations[shape, schedule] for schedule in SCHEDULES))
                       for shape in SHAPES}
        if not comparisons["dynamic"]["exact"]:
            for shape in ABLATIONS:
                comparisons[shape] = differences(*(measure(qid, shape, schedule) for schedule in SCHEDULES))
        rows.append((observations, comparisons))
        if (i+1) % 30 == 0:
            print(dataset, f"completed {i+1} of {len(chosen)} queries", flush=True)

    report = {}
    for shape in SHAPES | ABLATIONS:
        comparisons = [values[shape] for _, values in rows if shape in values]
        if not comparisons:
            continue
        report[shape] = {"queries": len(comparisons), **{key: sum(r[key] for r in comparisons) for key in comparisons[0]},
                         "queries_with_generic_threshold_drift": sum(r["generic_threshold_drift"] > 0 for r in comparisons),
                         "queries_with_typed_drift": sum(r["typed_decision_drift"] > 0 for r in comparisons)}
        if shape in SHAPES:
            report[shape]["dynamic_disagreements_repaired"] = sum(
                not values["dynamic"]["exact"] and values[shape]["exact"] for _, values in rows)
            report[shape]["timing"] = {}
            for arm in SCHEDULES:
                data = [r[shape, arm] for r, _ in rows]
                report[shape]["timing"][arm] = dict(
                    latency_p50_p95_ms=np.quantile([r["elapsed"] for r in data], [.5,.95]).tolist(),
                    typed_forwards=mean(r["typed_forwards"] for r in data),
                    generic_forwards=mean(r["generic_forwards"] for r in data),
                    exact_vs_dynamic_immediate=sum(
                        r["ranked"] == all_arms["dynamic","selective"]["ranked"]
                        for r,(all_arms,_) in zip(data,rows,strict=True)),
                )
            report[shape]["paired_ms_saved"] = paired_interval(
                [r[shape,"selective"]["elapsed"] - r[shape,"deferred_typed"]["elapsed"] for r,_ in rows])
    write_json(output, dict(
        dataset=dataset, queries=len(chosen), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        script_sha256=digest(Path(__file__)), runner_sha256=digest(Path(previous.__file__)),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        hard_label_sha256=digest(root / f"{dataset}_queries_test_hard.json"),
        embedding_revision=provenance["revision"], reranker_revision=cheap.provenance["revision"],
        typed_revision=expensive.provenance["revision"], report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD","SciRepEval"))
    for key in ("embedding","reranker","reader","output"):
        parser.add_argument(key,type=Path)
    run(**vars(parser.parse_args()))
