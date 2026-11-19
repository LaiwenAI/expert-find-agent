"""Test whether lexical and community agreement can avoid first-query GPU encoding."""

import argparse
from collections import defaultdict
from hashlib import blake2b
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.retrieval.sepef_prior_fusion import DEPTH, fuse, quality


MAX_VALIDATION_DENSE_RATE = .5
AGREEMENT_CUTOFFS = range(7)
LEXICAL_PRIOR_WEIGHT = 1.


def agreement(lexical, prior):
    return len({owner for owner, _ in lexical[:5]} & {owner for owner, _ in prior[:5]})


def choose_cutoff(rows):
    trials = {}
    for cutoff in AGREEMENT_CUTOFFS:
        escalated = [row["agreement"] < cutoff for row in rows]
        rate = float(np.mean(escalated))
        chosen = [row["dense"] if use_dense else row["cheap"]
                  for row, use_dense in zip(rows, escalated, strict=True)]
        trials[cutoff] = dict(
            dense_rate=rate,
            **{key: float(np.mean([row[key] for row in chosen])) for key in ("mrr10", "hit5")},
        )
    eligible = [cutoff for cutoff, result in trials.items()
                if result["dense_rate"] <= MAX_VALIDATION_DENSE_RATE]
    cutoff = max(eligible, key=lambda k: (trials[k]["mrr10"], trials[k]["hit5"], -trials[k]["dense_rate"], -k))
    return cutoff, trials


def run(panel, prior, fusion_report, checkpoint, scratch, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval import sepef_first_query as baseline

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Require a fresh output and Slurm CUDA")
    owners, ids, texts, positions, metadata = baseline.payload(panel)
    data = json.loads(prior.read_bytes())
    previous = json.loads(fusion_report.read_bytes())
    if (previous["manifest_sha256"] != baseline.digest(panel / "manifest.json")
            or previous["prior_sha256"] != baseline.digest(prior)
            or data["history_sha256"] != metadata["source_sha256"]["SE-PEF/SE-PEF/expert_test_data.json"]
            or any(baseline.digest(panel / f"{name}.json") != digest
                   for name, digest in data["query_sha256"].items())):
        raise ValueError("Prior and dense fusion must use the same panel")
    model = verify_checkpoint(checkpoint)
    if model != previous["model"]:
        raise ValueError("Changed dense embedding checkpoint")
    encoder = QwenTorchEmbedding(checkpoint, max_length=256, batch_size=64)
    encoder.model.config.use_cache = False
    encoder.encode(["Warm-up query."])
    started = perf_counter()
    vectors = torch.as_tensor(encoder.encode(texts), device="cuda")
    torch.cuda.synchronize()
    encode_seconds = perf_counter() - started
    global_bank = baseline.SourceBank(vectors, np.arange(len(ids)), positions)
    communities = defaultdict(list)
    for i, source in enumerate(ids):
        communities[source.split("_", 1)[0]].append(i)
    banks = {site: baseline.SourceBank(vectors, np.array(indices), positions)
             for site, indices in communities.items()}
    masks = {site: set(indices) for site, indices in communities.items()}
    rankings = data["ranking"]
    dense_weight = previous["weight"]

    with TemporaryDirectory(dir=scratch, prefix="sepef-router-") as tmp:
        started = perf_counter()
        lexical_index = baseline.lexical_index(texts, Path(tmp))
        lexical_seconds = perf_counter() - started

        def cheap(text, scope, leaders):
            matches = baseline.bm25(lexical_index, text, positions, masks.get(scope), k=DEPTH)
            lexical = [(owners[o], ids[d]) for o, d in matches]
            return fuse(lexical, leaders, LEXICAL_PRIOR_WEIGHT), agreement(lexical, leaders)

        def dense(text, scope, leaders):
            query = torch.as_tensor(encoder.encode([f"Instruct: {baseline.TASK}\nQuery: {text}"])[0], device="cuda")
            matches = [(owners[o], ids[d]) for o, d in banks.get(scope, global_bank).dense(query, k=DEPTH)]
            return fuse(matches, leaders, dense_weight)

        def inputs(row):
            scope = row["community"]
            return baseline.clean(row["text"]), scope, rankings.get(scope, rankings[""])

        val = json.loads((panel / "val.json").read_bytes())
        calibration = []
        for row in val:
            text, scope, leaders = inputs(row)
            lexical, signal = cheap(text, scope, leaders)
            calibration.append(dict(agreement=signal, cheap=quality(row, lexical),
                                    dense=quality(row, dense(text, scope, leaders))))
        cutoff, trials = choose_cutoff(calibration)
        rate = trials[cutoff]["dense_rate"]
        print("Calibration complete;", len(val), "queries, cutoff", cutoff, flush=True)

        def evaluate(row, arm):
            start = perf_counter()
            use_dense = False
            if arm == "prior":
                matches = rankings.get(row["community"], rankings[""])[:baseline.TOP]
            else:
                text, scope, leaders = inputs(row)
                if arm == "agreement_router":
                    matches, signal = cheap(text, scope, leaders)
                    use_dense = signal < cutoff
                elif arm == "random_router":
                    value = int.from_bytes(blake2b(row["id"].encode(), digest_size=8).digest(), "big")
                    use_dense = value / 2**64 < rate
                    if not use_dense:
                        matches, _ = cheap(text, scope, leaders)
                elif arm == "dense_fused":
                    use_dense = True
                else:
                    matches, _ = cheap(text, scope, leaders)
                if use_dense:
                    matches = dense(text, scope, leaders)
            json.dumps(matches, separators=(",", ":"))
            ms = (perf_counter() - start) * 1000
            return dict(**quality(row, matches), ms=ms, used_dense=use_dense)

        arms = ("prior", "lexical_fused", "dense_fused", "agreement_router", "random_router")
        test = json.loads((panel / "test.json").read_bytes())
        measurements = {arm: [] for arm in arms}
        for i, row in enumerate(test):
            offset = i % len(arms)
            for arm in arms[offset:] + arms[:offset]:
                measurements[arm].append(evaluate(row, arm))
        groups = [row["community"] for row in test]
        metrics = {
            arm: {k: float(np.mean([row[k] for row in results])) for k in ("hit5", "mrr10", "used_dense")}
            | {"p95_ms": float(np.quantile([row["ms"] for row in results], .95))}
            for arm, results in measurements.items()
        }
        paired = {
            control: {
                field: clustered_paired_interval(
                    [float(a[field]) - float(b[field])
                     for a, b in zip(measurements["agreement_router"], measurements[control], strict=True)], groups,
                ) | {"sampling_unit": "whole community; query-weighted mean"}
                for field in ("hit5", "mrr10")
            }
            for control in ("prior", "lexical_fused", "dense_fused", "random_router")
        }
        baseline.write_json(output, dict(
            job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=model,
            script_sha256=baseline.digest(Path(__file__)), panel_sha256=previous["manifest_sha256"],
            fusion_report_sha256=baseline.digest(fusion_report), prior_sha256=previous["prior_sha256"],
            validation_dense_cap=MAX_VALIDATION_DENSE_RATE, validation_queries=len(val),
            validation_trials=trials, agreement_cutoff=cutoff, dense_prior_weight=dense_weight,
            lexical_prior_weight=LEXICAL_PRIOR_WEIGHT, test_queries=len(test), test=metrics,
            router_minus_controls=paired, encode_seconds=encode_seconds, lexical_seconds=lexical_seconds,
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("panel", "prior", "fusion_report", "checkpoint", "scratch", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
