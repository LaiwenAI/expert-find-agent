"""Run and package a blinded development comparison of expert-search routes."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from time import perf_counter

from experiments.retrieval import build_exact_enterprise_review as packet


ARMS = ("whole", "criterion_rrf", "criterion_cover")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write(path, content, lines=False):
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        if lines:
            for row in content:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(content, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def run(queries, corpus, models, output):
    import numpy as np
    import torch

    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval import enterprise_criterion_recall as routes
    from experiments.retrieval.enterprise_cross_precision_search import Parser
    from experiments.retrieval.enterprise_first_query import (
        CUTOFF, CRITERION_CHARS, EMBED_TASK, K, ROLES, spans, verify_candidates,
    )
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Changed source inventory or development cohort")
    requests = rows(queries)
    sources = packet.index(rows(corpus), lambda item: item["id"])
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder_path = models / "qwen3-embedding-0.6b"
    checkpoint = verify_checkpoint(encoder_path)
    encoder = QwenTorchEmbedding(encoder_path)
    parser = Parser(models / "qwen3-4b-instruct-2507")
    verifier = Reranker(models / "qwen3-reranker-0.6b")
    start = perf_counter()
    embeddings = encoder.encode([source["text"] for source in sources.values()])
    ref = ExactOwnedVectorIndex(embeddings, list(sources), [[r["expert_id"]] for r in sources.values()])
    index = GpuOwnedVectorIndex(ref)
    index_seconds = perf_counter() - start
    print(f"Indexed {len(sources)} owned sources", flush=True)

    def one(text, arm):
        started = perf_counter()
        clauses = spans(text)
        labels = parser.evaluate(text, clauses, "consensus", CUTOFF)["labels"]
        qualifications = [(part["start"], part["end"]) for part, role in zip(clauses, labels, strict=True)
                          if ROLES[int(role)] == "prior_qualification"]
        objective = text.strip().splitlines()[0][:CRITERION_CHARS]
        criteria = [objective, *(text[a:b] for a, b in qualifications)]
        prompt = f"Instruct: {EMBED_TASK}\nQuery: "
        vector = encoder.encode([prompt + text])[0]
        hits = index.search_vector(vector, k=K)["hits"]
        if arm != "whole" and len(criteria) > 1:
            variants = [prompt + c + (f"\nProject objective: {objective}" if i else "")
                        for i, c in enumerate(criteria)]
            encoded = encoder.encode(variants)
            with torch.inference_mode():
                whole = torch.as_tensor(vector, device=index.device, dtype=torch.float32)
                contrasts = torch.as_tensor(encoded, device=index.device, dtype=torch.float32)
                scores = torch.cat((torch.mv(index.vectors, whole)[:, None],
                                    torch.mm(index.vectors, contrasts.T)), dim=1).cpu().numpy()
            options = routes.assemble(ref, scores, routes.orders(ref, scores))
            hits = options["rrf_openings" if arm == "criterion_rrf" else "rrf_cover"]
        decisions, checks, clipped, withheld = verify_candidates(hits, sources, criteria, verifier)
        matches = []
        for expert, supported, citations in decisions:
            if supported:
                matches.append(dict(expert=expert, citations=list(dict.fromkeys(citations))))
        return dict(qualifications=qualifications, ranked=[row[0] for row in decisions],
                    recommendations=matches, checks=checks, clipped=clipped, withheld=withheld,
                    milliseconds=(perf_counter() - started) * 1000)

    for arm in ARMS:
        one("Find an expert in optical inspection. Prior experience testing polymer films is required.", arm)
    results = []
    for i, query in enumerate(requests):
        offset = i % len(ARMS)
        item = {arm: one(query["text"], arm) for arm in ARMS[offset:] + ARMS[:offset]}
        if any(value["qualifications"] != item["whole"]["qualifications"] for value in item.values()):
            raise ValueError("The shared reader changed the qualification frame")
        results.append(dict(id=query["id"], arms=item))
        if (i+1) % 10 == 0:
            print(f"Ran {i+1} development requests", flush=True)
    metrics = {}
    for arm in ARMS:
        values = [record["arms"][arm] for record in results]
        metrics[arm] = dict(
            issued=sum(bool(x["recommendations"]) for x in values),
            model_supported_slots=sum(len(x["recommendations"]) for x in values),
            **{field: sum(x[field] for x in values) for field in ("checks", "clipped", "withheld")},
            p50_p95_ms=np.quantile([x["milliseconds"] for x in values], [.5, .95]).tolist(),
            changed_ranked_lists=sum(x["ranked"] != r["arms"]["whole"]["ranked"]
                                     for x, r in zip(values, results, strict=True)),
        )
    write(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries_sha256=DEVELOPMENT,
        corpus_sha256=SOURCE_INVENTORY, script_sha256=digest(Path(__file__)), sources=len(sources),
        experts=len(ref.owners), index_seconds=index_seconds, requests=len(requests), metrics=metrics,
        models=dict(encoder=checkpoint["revision"], parser=parser.readers["bf16"].provenance["revision"],
                    verifier=verifier.provenance["revision"]), results=results,
    ))


def build(dataset, trace, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    inputs = {name: digest(dataset / name) for name in ("queries.jsonl", "corpus.jsonl", "experts.jsonl")}
    if any(manifest["files_sha256"][name] != checksum for name, checksum in inputs.items()):
        raise ValueError("Changed private snapshot")
    data = json.loads(trace.read_bytes())
    if data["script_sha256"] != digest(Path(__file__)):
        raise ValueError("Changed evaluation runner")
    tables = {name: packet.index((item for item in rows(dataset/name) if item["split"] == "development"),
                                 lambda item: item["id"]) for name in inputs}
    queries, corpus, experts = (tables[name] for name in inputs)
    traces = packet.index(data["results"], lambda item: item["id"])
    if set(traces) != set(queries):
        raise ValueError("Changed development frame")
    methods = {arm: {} for arm in ARMS}
    for qid, record in traces.items():
        for arm, result in record["arms"].items():
            methods[arm][qid] = [
                (r["expert"], packet.sources(corpus, r["expert"], [
                    dict(source_id=source_id, start=a, end=b) for source_id,a,b in r["citations"]
                ])) for r in result["recommendations"]
            ]
    forms, mapping, predictions, counts = packet.packet_rows(queries, experts, methods)
    for form, private in zip(forms, mapping, strict=True):
        qid = private["query_id"]
        text = queries[qid]["text"]
        form["prior_qualifications"] = [text[a:b] for a,b in traces[qid]["arms"]["whole"]["qualifications"]]
    output.mkdir(mode=0o700)
    form = output/"blind-review.jsonl"
    mapping_file = output/"private-mapping.jsonl"
    write(form, forms, lines=True)
    write(mapping_file, mapping, lines=True)
    metadata = output/"manifest.json"
    write(metadata, dict(queries=len(queries), candidate_views=len(forms), human_judgments=0,
                         outputs_sha256={f.name:digest(f) for f in (form,mapping_file)},
                         inputs_sha256=inputs | {"private_live_trace":digest(trace)}))
    packet.check(output, form)
    frame = output/"frame.jsonl"
    write(frame, [dict(id=qid, group=qid, text_sha256=sha256(row["text"].encode()).hexdigest())
                  for qid,row in sorted(queries.items())], lines=True)
    for arm, recommendations in predictions.items():
        protocol = output/f"{arm}-protocol.json"
        write(protocol, dict(frame_sha256=digest(frame), policy=arm, evaluation_role="Development only; query groups are placeholders",
                             top_k=packet.TOP, required_fields=packet.FIELDS, coverage_floor=.60, risk_ceiling=.05, confidence=.95))
        prediction = output/f"{arm}-outputs.jsonl"
        write(prediction, recommendations, lines=True)
        write(output/f"{arm}-execution.json", dict(outputs_sha256=digest(prediction), protocol_sha256=digest(protocol),
                                                   packet_sha256=digest(metadata)))
    summary = {key:value for key,value in data.items() if key != "results"}
    summary.update(counts, review_cases=len(forms), human_judgments=0,
                   request_status_counts={arm:dict(Counter(row["status"] for row in predictions[arm])) for arm in ARMS})
    write(output/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for mode, args in {"run":("queries", "corpus", "models", "output"),
                       "build":("dataset", "trace", "output")}.items():
        command = commands.add_parser(mode)
        for arg in args:
            command.add_argument(arg, type=Path)
    args = vars(parser.parse_args())
    {"run":run, "build":build}[args.pop("command")](**args)
