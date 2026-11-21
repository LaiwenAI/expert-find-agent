"""Run the shared verification budget under single-GPU HTTP load."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from random import Random

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.budgeted_source_role_search import check_source, role_check
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.elife_role_prompt_service import Request, replay, summarize
from experiments.credit.engineering_cross_source_search import (
    TOPIC_TASK, candidates, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_one_role_recheck import RoleReader
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.keyword_author_rescue import UNGATED, PostingCatalog, at_gate
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.source_confirmation_transfer import VIEWS, render
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("static", "role_only", "both", "one_call", "screened_one_call")
RATES = (45, 75)


def workload(index, rows, wording):
    bank = phrase_bank(wording)
    requests, labels = [], {}
    for row in rows:
        text = [index.articles[paper]["clue"] for paper in row["papers"]]
        roles = [bank[role][int(row["family"][:16], 16) % len(bank[role])] for role in row["roles"]]
        for view in VIEWS:
            clues = [render(index, raw, view) for raw in text]
            request = Request("elife", json.dumps({"requirements": [
                dict(topic=clue, role=role) for clue, role in zip(clues, roles, strict=True)]}))
            if request in labels and any(labels[request][key] != row[key] for key in ("papers", "roles", "gold")):
                raise ValueError("Conflicting reference for identical input")
            requests.append(request)
            labels[request] = row
    return requests, labels


def arrivals_for(requests, rate):
    rng = Random(20261122)
    ordered = requests.copy()
    rng.shuffle(ordered)
    arrivals = [0.]
    for _ in ordered[1:]:
        arrivals.append(arrivals[-1] + rng.expovariate(rate))
    return ordered, arrivals


class Service:
    def __init__(self, index, encoder, banks, bands, source, role, cutoffs):
        self.index = index
        self.encoder = encoder
        self.banks = banks
        self.bands = bands
        self.source = source
        self.role = role
        self.cutoffs = cutoffs

    def search(self, request, arm):
        requirements = json.loads(request.payload)["requirements"]
        clues = [item["topic"] for item in requirements]
        phrases = [item["role"] for item in requirements]
        vectors = self.encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues] +
                                      [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in phrases])
        scores = {bank: vectors[2:] @ self.banks[bank].T for bank in self.bands}
        topic = self.index.full_rank(vectors[:2], clues, None)
        pair, _ = select(self.index, [side | {"fused": side["fused"][:6]} for side in topic],
                         {}, [set(), set()], "anchored", None)
        roles = required_roles({}, scores, "prototype_band", self.bands)
        source_calls = role_calls = tokens = 0
        if pair is None and arm in ("both", "one_call", "screened_one_call"):
            proposal = at_gate(self.index.linked_options(topic)[0], *UNGATED)
            if proposal and (arm != "screened_one_call" or candidates(self.index, proposal, roles)):
                pair, _ = check_source(self.index, self.source, topic, proposal, clues, self.cutoffs["source"])
                source_calls = 1
        if pair and arm != "static" and (not source_calls or arm == "both"):
            roles, role_calls, tokens, _ = role_check(self.index, pair, scores, self.bands, self.role,
                                                      phrases, self.cutoffs["role"])
        if arm in ("one_call", "screened_one_call") and source_calls + role_calls > 1:
            raise ValueError("Exceeded the verification budget")
        owners = candidates(self.index, pair, roles) if pair else []
        return json.dumps(dict(
            papers=pair if owners else None, owners=owners, roles=[sorted(x) for x in roles] if owners else [],
            calls=source_calls + role_calls, tokens=tokens, source_calls=source_calls, role_calls=role_calls,
        ))


def service_stats(trace, arrivals, records, elapsed, labels):
    stats, signatures = summarize(trace, arrivals, records, elapsed, labels, 1)
    calls = Counter()
    for row in records:
        if "response" not in row:
            continue
        response = json.loads(row["response"])
        calls.update(source_calls=response["source_calls"], role_calls=response["role_calls"],
                     issued_with_source_call=bool(response["owners"] and response["source_calls"]))
    stats["verification"] = dict(calls)
    return stats, signatures


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    articles, rows, census = build(args.projection, json.loads(args.manifest.read_bytes()))
    definitions, training, wording = queries(args.niso, args.training)
    ckpt = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    source = Reranker(args.source_reader)
    role = RoleReader(args.role_reader, definitions)
    role.definitions = ROLES
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    frozen = json.loads(args.source_frozen.read_bytes())["runs"][0]
    if frozen["verifier"]["revision"] != source.provenance["revision"]:
        raise ValueError("Source model revision differs from the frozen cutoff")
    role_cutoff = json.loads(args.role_frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    cutoffs = {"source": frozen["thresholds"]["keywords"]["weakest"]["cutoff"], "role": role_cutoff}
    index = PostingCatalog(articles, encoder)
    service = Service(index, encoder, banks, bands, source, role, cutoffs)
    requests, labels = workload(index, rows, wording)
    arms = ARMS[::-1] if args.reverse else ARMS
    results, reference = {}, {}
    for offset, rate in enumerate(RATES):
        trace, arrivals = arrivals_for(requests, rate)
        observations = {}
        results[str(rate)] = {}
        for arm in arms[offset:] + arms[:offset]:
            values, elapsed = replay(lambda req: service.search(req, arm), trace, arrivals, trace[:32])
            metrics, responses = service_stats(trace, arrivals, values, elapsed, labels)
            if arm in reference:
                earlier = reference[arm]
                metrics["changed_from_first_rate"] = sum(earlier[req] != response for req, response in responses.items()
                                                        if req in earlier)
            else:
                reference[arm] = responses
            results[str(rate)][arm] = metrics
            observations[arm] = responses
            print(rate, arm, metrics["p95_ms"], flush=True)
        a, b = observations["one_call"], observations["screened_one_call"]
        shared = a.keys() & b.keys()
        results[str(rate)]["screen_comparison"] = {
            "paired": len(shared), "changed_evidence": sum(a[req] != b[req] for req in shared)}
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), projection_sha256=digest(args.projection), manifest_sha256=digest(args.manifest),
        embedding_revision=ckpt["revision"], source_reader_revision=source.provenance["revision"],
        role_reader_revision=role.provenance["revision"], gpu=torch.cuda.get_device_name(), cutoffs=cutoffs, reverse=args.reverse,
        panel=census, distinct_requests=len(labels), trace_requests=len(requests), results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "manifest", "niso", "training", "bands", "source_frozen", "role_frozen",
                "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
