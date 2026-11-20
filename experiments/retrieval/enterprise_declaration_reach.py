"""Audit role-declaration coverage on actual enterprise first-query candidates."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from experiments.retrieval import enterprise_project_role_rescue as project
from experiments.retrieval.enterprise_routed_review import digest, rows, write


DUTY_CUE = re.compile(
    r"\u4e3b\u6301|\u7275\u5934|\u9879\u76ee\u8d1f\u8d23\u4eba|\u8bfe\u9898\u8d1f\u8d23\u4eba|"
    r"\u53c2\u4e0e|\u53c2\u52a0|\u53d1\u660e\u4eba|"
    r"project lead|principal investigator|project participant|named inventor", re.I,
)
FIELD_KIND = {project.PROJ: "project", project.PATENT: "patent", project.FIELDS[4]: "paper_list"}


def role_spans(source, name):
    found = []
    for span in project.passages(source, name):
        role = span["roles"][0]
        if role == "profile_listed_inventor":
            label = "named_inventor"
        elif name and name in project.name_key(span["text"]):
            label = "role_and_exact_name"
        else:
            label = "profile_role_label"
        found.append([span["start"], span["end"], label])
    return found


def prepare(dataset, live_sources, output):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    profiles = project.profile_subset(dataset, "experts", manifest)
    inventory = project.profile_subset(dataset, "corpus", manifest)
    live = rows(live_sources)
    if len(live) != 52667 or {row["id"] for row in live} != set(inventory):
        raise ValueError("Unexpected live source inventory")
    names = {key: project.name_key(row["name"] or "") for key, row in profiles.items()}
    record_types = Counter()
    flags = {}
    for row in live:
        source = inventory[row["id"]]
        expected = row["expert_id"], row["text"]
        if expected != (source["expert_id"], source["text"]):
            raise ValueError("Changed source projection or associated candidate")
        field = FIELD_KIND.get(source.get("field"), "other_profile") if source["kind"] == "profile" else "paper"
        spans = role_spans(source, names[source["expert_id"]])
        value = source.get("original_value")
        record_types[f"{field}:{type(value).__name__}"] += 1
        flags[row["id"]] = [field, spans]
    # Source identifiers belong to the private dataset. This projection is used
    # only on the authorized compute node; publication uses aggregate counts.
    write(output, dict(source_sha256=digest(live_sources), data_manifest_sha256=digest(dataset/"manifest.json"),
                       flags=flags, original_value_types=dict(record_types)))


class CapturedIndex:
    def __init__(self, index):
        self.index = index
        self.hits = []

    def search_vector(self, *args, **kwargs):
        result = self.index.search_vector(*args, **kwargs)
        self.hits = result["hits"]
        return result


def applicable(citation, flags):
    key, begin, end = citation
    kind, postings = flags[key]
    return kind, {label for start, stop, label in postings if begin <= start <= stop <= end}


def record_coverage(citations, flags):
    output = Counter()
    for citation in citations:
        kind, usable = applicable(citation, flags)
        output.update((kind,))
        output.update(usable)
        output.update(("any_role_cue" if usable else "no_role_cue",))
    return output


def run(queries, corpus, roles, flag_file, vectors, vector_meta, models, output):
    import torch
    from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
    from expert_search.owned_vector_index import ExactOwnedVectorIndex
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval import enterprise_shared_qualification as search
    from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
    from experiments.retrieval.enterprise_prefix_fork import ComparisonParser

    search.role.require_cuda(output)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Unexpected development inventory")
    metadata = json.loads(flag_file.read_bytes())
    flags = metadata["flags"]
    sources = {row["id"]: row for row in rows(corpus)}
    if metadata["source_sha256"] != SOURCE_INVENTORY or flags.keys() != sources.keys():
        raise ValueError("Role audit belongs to a different source inventory")
    sidecar = json.loads(vector_meta.read_bytes())
    model_path = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(model_path)
    if (sidecar["corpus_sha256"], sidecar["checkpoint"], sidecar["vectors_sha256"], sidecar["rows"]) != (
            SOURCE_INVENTORY, provenance["revision"], digest(vectors), len(sources)):
        raise ValueError("Index embedding provenance mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    index = CapturedIndex(GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[row["expert_id"]] for row in sources.values()])))
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    reader = search.EncodedReader(Reranker(models / "qwen3-reranker-0.6b"))
    encoder = QwenTorchEmbedding(model_path)
    role_index = search.role.load_role_index(roles, SOURCE_INVENTORY, sources)
    all_sources = [(key, 0, len(row["text"])) for key, row in sources.items()]
    full_counts = record_coverage(all_sources, flags)
    full_owners = defaultdict(set)
    for key, (_, spans) in flags.items():
        for _, _, label in spans:
            full_owners[sources[key]["expert_id"]].add(label)

    primary_excerpts = Counter()
    primary_pairs = Counter()
    witness_counts = Counter()
    cohort = Counter()
    candidates_by_query = Counter()
    elapsed = []
    request_ms = []
    search.one("Find expertise in optical calibration and past sensor testing.", "automatic",
               parser, encoder, index, sources, role_index, reader)
    for item in rows(queries):
        text = item["text"]
        started = perf_counter()
        result = search.one(text, "automatic", parser, encoder, index, sources, role_index, reader)
        request_ms.append(result["elapsed_ms"])
        criteria = [text.strip().splitlines()[0][:search.base.CRITERION_CHARS],
                    *(text[a:b] for a, b in result["qualifications"])]
        planned = search.plan(index.hits, sources, criteria)
        if len(planned) != result["metadata"]["original_pairs"]:
            raise ValueError("Recorded primary plan differs from live slots")
        citations = list(dict.fromkeys(row.citation for row in planned))
        paths = record_coverage(citations, flags)
        primary_excerpts.update(paths)
        primary_pairs.update(record_coverage([row.citation for row in planned], flags))
        cohort.update(requests=1,
                      total_candidate_slots=len(index.hits),
                      queries_with_any_primary_role=bool(paths["any_role_cue"]),
                      parsed_prior_qualifications=len(result["qualifications"]),
                      parsed_qualifications_with_duty_cue=sum(bool(DUTY_CUE.search(text[a:b]))
                                                             for a, b in result["qualifications"]),
                      requests_with_duty_cue_in_qualifications=any(DUTY_CUE.search(text[a:b])
                                                                  for a, b in result["qualifications"]))
        for hit in index.hits:
            for label in full_owners[hit["id"]]:
                candidates_by_query[label] += 1
            candidates_by_query["any_role_cue"] += bool(full_owners[hit["id"]])
        issued = [*result["initial"], *result["extra"]]
        cohort["requests_with_provisional_recommendation"] += bool(issued)
        for item in issued:
            points = record_coverage(item["citations"], flags)
            witness_counts.update(points)
            cohort.update(provisional_recommendations=1,
                          provisional_with_any_uncued_witness=bool(points["no_role_cue"]),
                          provisional_with_all_witnesses_uncued=not bool(points["any_role_cue"]))
        elapsed.append((perf_counter()-started)*1000)
        if cohort["requests"] % 20 == 0:
            print(f"Audited {cohort['requests']} development requests", flush=True)
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                       script_sha256=digest(Path(__file__)), frozen_sources=len(sources),
                       role_projection_sha256=digest(flag_file), source_record_shapes=metadata["original_value_types"],
                       full_corpus=dict(full_counts), candidates_with_cue_somewhere=dict(candidates_by_query),
                       cohort=dict(cohort), primary_excerpts=dict(primary_excerpts), planned_pairs=dict(primary_pairs),
                       provisional_witnesses=dict(witness_counts), observed_p95_ms=float(np.quantile(request_ms,.95)),
                       audit_p95_ms=float(np.quantile(elapsed,.95))))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    for name in ("dataset", "live_sources", "output"):
        prep.add_argument(name, type=Path)
    replay = modes.add_parser("run")
    for name in ("queries", "corpus", "roles", "flag_file", "vectors", "vector_meta", "models", "output"):
        replay.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    (prepare if args.pop("mode") == "prepare" else run)(**args)
