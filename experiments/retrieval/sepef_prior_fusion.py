"""Match full-history priors to the time-frozen SE-PEF text panel."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter


HISTORY_SHA256 = "9e2856bb4bcf09dbe8e565c278b28da3737d915f2f7273a6919c51fad3e6ae51"
SNAPSHOT = 1577836799
WEIGHTS = (0., .25, .5, 1., 2., 4.)
DEPTH = 50
TOP = 10


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def build_prior(history):
    by_site = defaultdict(Counter)
    latest = {}
    for owner, item in history.items():
        for stamp, source in zip(item["answer_timestamps"], item["answers"], strict=True):
            if stamp >= SNAPSHOT:
                raise ValueError("History crosses the official profile boundary")
            site = source.split("_", 1)[0]
            for scope in (site, ""):
                by_site[scope][owner] += 1
                latest[scope, owner] = max((stamp, source), latest.get((scope, owner), (-1, "")))
    rankings = {}
    for site, counts in by_site.items():
        leaders = sorted(counts, key=lambda owner: (-counts[owner], owner))[:DEPTH]
        rankings[site] = [[owner, latest[site, owner][1]] for owner in leaders]
    return rankings


def quality(rows, matches):
    labels = set(rows["expert_ids"])
    rank = next((i for i, (owner, _) in enumerate(matches[:TOP], 1) if owner in labels), None)
    return dict(hit5=bool(rank and rank <= 5), mrr10=1 / rank if rank else 0.0)


def averages(values):
    return {field: sum(row[field] for row in values) / len(values) for field in ("hit5", "mrr10")}


def prepare(history_file, panel, output, report):
    if output.exists() or report.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh outputs on a Slurm compute node")
    if digest(history_file) != HISTORY_SHA256:
        raise ValueError("Changed upstream expert history")
    ranking = build_prior(json.loads(history_file.read_bytes()))
    samples = {name: json.loads((panel / f"{name}.json").read_bytes()) for name in ("val", "test")}
    queries = {name: digest(panel / f"{name}.json") for name in samples}
    data = dict(history_sha256=HISTORY_SHA256, query_sha256=queries, ranking=ranking)
    with output.open("x") as stream:
        json.dump(data, stream, separators=(",", ":"))
        stream.write("\n")
    metrics = {}
    for split, rows in samples.items():
        metrics[split] = dict(queries=len(rows), metrics={
            arm: averages([quality(row, ranking.get(row["community"], ranking[""]) if arm == "community"
                                   else ranking[""]) for row in rows])
            for arm in ("global", "community")
        })
    with report.open("x") as stream:
        json.dump(dict(job=os.environ["SLURM_JOB_ID"], script_sha256=digest(Path(__file__)),
                       history_sha256=HISTORY_SHA256, prior_sha256=digest(output), query_sha256=queries,
                       metrics=metrics), stream, sort_keys=True, indent=2)
        stream.write("\n")


def fuse(dense, prior, weight):
    scores = {}
    citations = {}
    for records, mass in ((dense, 1.), (prior, weight)):
        if not mass:
            continue
        for rank, (owner, source) in enumerate(records, 1):
            scores[owner] = scores.get(owner, 0.) + mass / (60 + rank)
            citations.setdefault(owner, source)
    return [(owner, citations[owner]) for owner in sorted(scores, key=lambda o: (-scores[o], o))[:TOP]]


def run(panel, prior, checkpoint, output):
    import numpy as np
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from expert_search.statistics import clustered_paired_interval
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval import sepef_first_query as baseline

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Require a fresh output and a Slurm CUDA allocation")
    owners, ids, texts, positions, metadata = baseline.payload(panel)
    data = json.loads(prior.read_bytes())
    if (metadata["source_sha256"]["SE-PEF/SE-PEF/expert_test_data.json"] != data["history_sha256"]
            or data["history_sha256"] != HISTORY_SHA256
            or any(digest(panel / f"{name}.json") != checksum for name, checksum in data["query_sha256"].items())):
        raise ValueError("Prior and retrieval panel use different inputs")
    ranking = data["ranking"]
    known = set(owners)
    if any(owner not in known for entries in ranking.values() for owner, _ in entries):
        raise ValueError("Prior contains an owner absent from the text panel")
    model = verify_checkpoint(checkpoint)
    if model["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected frozen embedding checkpoint")
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
    banks = {site: baseline.SourceBank(vectors, np.array(selected), positions)
             for site, selected in communities.items()}

    def rows(split):
        return json.loads((panel / f"{split}.json").read_bytes())

    def dense(row, k):
        text = baseline.clean(row["text"])
        query = torch.as_tensor(encoder.encode([f"Instruct: {baseline.TASK}\nQuery: {text}"])[0], device="cuda")
        bank = banks.get(row["community"], global_bank)
        return [(owners[owner], ids[doc]) for owner, doc in bank.dense(query, k=k)]

    val = rows("val")
    trial = {weight: [] for weight in WEIGHTS}
    for row in val:
        candidates = dense(row, DEPTH)
        older = ranking.get(row["community"], ranking[""])
        for weight in WEIGHTS:
            trial[weight].append(quality(row, fuse(candidates, older, weight)))
    fit = {weight: averages(outcomes) for weight, outcomes in trial.items()}
    weight = max(WEIGHTS, key=lambda w: (fit[w]["mrr10"], fit[w]["hit5"], -w))

    test = rows("test")
    arms = ("prior", "dense", "fused")
    measurements = {arm: [] for arm in arms}
    for i, row in enumerate(test):
        offset = i % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            started = perf_counter()
            previous = ranking.get(row["community"], ranking[""])
            if arm == "prior":
                matches = previous[:TOP]
            elif arm == "dense":
                matches = dense(row, TOP)
            else:
                matches = fuse(dense(row, DEPTH), previous, weight)
            json.dumps(matches, separators=(",", ":"))
            elapsed = (perf_counter() - started) * 1000
            measurements[arm].append(dict(**quality(row, matches), ms=elapsed))
    results = {arm: dict(**averages(values), p95_ms=float(np.quantile([r["ms"] for r in values], .95)))
               for arm, values in measurements.items()}
    groups = [row["community"] for row in test]
    paired = {control: {field: clustered_paired_interval(
        [float(a[field]) - float(b[field]) for a, b in zip(measurements["fused"], measurements[control], strict=True)],
        groups) for field in ("hit5", "mrr10")} for control in ("prior", "dense")}
    for comparison in paired.values():
        for summary in comparison.values():
            summary["sampling_unit"] = "whole community; query-weighted mean"
    baseline.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=model,
        script_sha256=digest(Path(__file__)), baseline_sha256=digest(Path(baseline.__file__)),
        manifest_sha256=digest(panel / "manifest.json"), prior_sha256=digest(prior),
        history_sha256=HISTORY_SHA256, fit=fit, weight=weight, val_queries=len(val),
        test_queries=len(test), test=results, fused_minus_controls=paired, encode_seconds=encode_seconds,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    choices = cli.add_subparsers(dest="mode", required=True)
    for mode, names in (("prepare", ("history_file", "panel", "output", "report")),
                        ("run", ("panel", "prior", "checkpoint", "output"))):
        parser = choices.add_parser(mode)
        for name in names:
            parser.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    {"prepare": prepare, "run": run}[args.pop("mode")](**args)
