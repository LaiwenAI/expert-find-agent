"""Compare full-menu and resident role classifiers in a matched HTTP search service."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.elife_prompt_holdout import build, sample_families
from experiments.credit.elife_role_feedback_reads import GUARDS, menus
from experiments.credit.elife_role_prompt_service import arrivals_for, replay, summarize
from experiments.credit.engineering_cross_source_search import candidates, phrase_bank
from experiments.credit.engineering_role_banks import projections, queries
from experiments.credit.joint_request_queue import Request
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.paired_role_prospective_service import FROZEN_SHA, clusters, request_inputs
from experiments.credit.paired_role_verification import PairedReader
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_contrast_head import RoleContrastHeads
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint

ARMS = ("full_menu", "ridge", "resident_ce", "resident_ce_fused")
RATES = (18, 35)
REPEATS = 2


def build_requests(catalog, cases, wording):
    glossary = phrase_bank(wording)
    lookup = {}
    for row in cases:
        topics = [catalog.articles[paper]["clue"] for paper in row["papers"]]
        phrases = [glossary[role][int(row["family"][:16], 16) % len(glossary[role])] for role in row["roles"]]
        for view, clues in (("keyword", list(map(catalog.keywords, topics))), ("excerpt", topics)):
            request = Request("elife", json.dumps({"requirements": [
                {"topic": clue, "role": phrase} for clue, phrase in zip(clues, phrases, strict=True)
            ]}))
            if request in lookup:
                raise ValueError("Unexpected duplicate request")
            lookup[request] = row | {"view": view}
    return lookup


class Search:
    def __init__(self, catalog, encoder, reader, cutoff, weights):
        self.catalog, self.encoder, self.reader, self.cutoff = catalog, encoder, reader, cutoff
        self.weights = weights
        self.byline = {paper: set(record["roles"]) for paper, record in catalog.articles.items()}

    def execute(self, request, arm):
        clues, phrases, prompts = request_inputs(request)
        # A fused encoder forward is a separate arm, since batch shape can alter ranking.
        fused = arm == "resident_ce_fused"
        embeddings = self.encoder.encode(prompts if fused else prompts[:2])
        ranking = self.catalog.full_rank(embeddings[:2], clues, None)
        pairs = menus(ranking, self.byline, GUARDS[1])[3]
        pair = pairs[0] if pairs else None
        owners, options, forwards, tokens = [], [], 0, 0
        n_encodes = 1
        if pair and arm == "full_menu":
            labels, margin, tokens = self.reader.batched(phrases, [ROLES, ROLES])
            forwards = 1
            if margin > self.cutoff and None not in labels:
                options = [{role} for role in labels]
        elif pair:
            points = embeddings[2:] if fused else self.encoder.encode(prompts[2:])
            n_encodes += int(not fused)
            weights = self.weights["ridge" if arm == "ridge" else "resident_ce"]
            options = [{ROLES[i]} for i in (points @ weights.T).argmax(axis=1)]
        if options:
            owners = sorted(candidates(self.catalog, pair, options))
        return json.dumps(dict(papers=pair, owners=owners, roles=[sorted(choice) for choice in options],
                               calls=forwards, tokens=tokens, encoder_forwards=n_encodes, selected_pair=pair))


def quality(proofs, labels):
    outcomes = {}
    for request, (papers, owners, labels_found) in proofs.items():
        truth = labels[request]
        strict = (list(truth["papers"]) == papers and labels_found == [[role] for role in truth["roles"]]
                  and bool(set(owners) & set(truth["gold"])))
        outcomes[request] = (strict, bool(owners) and not strict)
    return outcomes


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require Slurm CUDA and a fresh output")
    if digest(args.reference) != FROZEN_SHA:
        raise ValueError("Unexpected frozen role reference")
    reference = json.loads(args.reference.read_bytes())["runs"][0]
    if digest(args.training) != reference["inputs"]["training"]:
        raise ValueError("Unexpected original training bank")
    embedding = verify_checkpoint(args.embedding)["revision"]
    if embedding != reference["embedding_revision"]:
        raise ValueError("Unexpected frozen encoder")
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads(args.manifest.read_bytes())
    articles, cases, census = build(args.projection, manifest)
    cases = sample_families(cases)
    if census != manifest["panel"] or sorted({row["family"] for row in cases}) != manifest["chosen_families"]:
        raise ValueError("Changed frozen evaluation cohort")
    encoder = QwenTorchEmbedding(args.embedding)
    definitions, phrases, wording = queries(args.niso, args.training)
    banks = projections(encoder, definitions, phrases)
    fit = RoleContrastHeads(args.contrasts, encoder, phrases, wording)
    reader = PairedReader(args.reader, ROLES)
    if reader.provenance["revision"] != reference["role_reader_revision"]:
        raise ValueError("Unexpected frozen role reader")
    catalog = PostingCatalog(articles, encoder)
    search = Search(catalog, encoder, reader, reference["fit"]["two_batched"]["cutoff"],
                    {"ridge": banks["ridge"], "resident_ce": fit.weights["original_ce"]})
    labels = build_requests(catalog, cases, wording)
    timings, outcomes, ranking, signatures = {}, {}, {}, {}
    ordered = ARMS[::-1] if args.reverse else ARMS
    for offset, rate in enumerate(args.rates):
        timings[rate] = {}
        trace, arrivals = arrivals_for(labels, rate, REPEATS)
        for arm in ordered[offset:] + ordered[:offset]:
            rows, elapsed = replay(lambda req: search.execute(req, arm), trace, arrivals, trace[:16])
            detail, proofs = summarize(trace, arrivals, rows, elapsed, labels, REPEATS)
            if detail["failures"] or len(proofs) != len(labels):
                raise ValueError(f"Incomplete service replay: {arm}, {detail['failures']}")
            counts = Counter()
            stable = {}
            for req, row in zip(trace, rows, strict=True):
                response = json.loads(row["response"])
                counts.update(encoder_forwards=response["encoder_forwards"], role_forwards=response["calls"],
                              role_tokens=response["tokens"], logical_reads=2 * bool(response["selected_pair"]))
                selected = response["selected_pair"]
                if req in stable and stable[req] != selected:
                    raise ValueError("Source pair varied on identical requests")
                stable[req] = selected
            if arm in ranking and ranking[arm] != stable:
                raise ValueError("Source pair changed under load")
            signature = sha256(json.dumps(list(proofs[req] for req in labels), sort_keys=True).encode()).hexdigest()
            if arm in signatures and signatures[arm] != signature:
                raise ValueError("Evidence changed under load")
            ranking[arm], signatures[arm] = stable, signature
            outcomes[arm] = quality(proofs, labels)
            timings[rate][arm] = {**detail, "model_and_reads": dict(counts)}
            print(rate, arm, round(detail["p95_ms"]["total_ms"], 2), flush=True)
    if ranking["full_menu"] != ranking["ridge"] or ranking["full_menu"] != ranking["resident_ce"]:
        raise ValueError("Unfused arms must have identical sources")
    grouped = clusters(cases, articles)
    results = {}
    for view in ("keyword", "excerpt"):
        requests = [req for req, row in labels.items() if row["view"] == view]
        positives = [req for req in requests if labels[req]["gold"]]
        results[view] = {
            arm: dict(positives=len(positives), requests=len(requests),
                      strict=sum(outcomes[arm][req][0] for req in requests),
                      adverse=sum(outcomes[arm][req][1] for req in requests),
                      source_changes=sum(ranking[arm][req] != ranking["full_menu"][req] for req in requests),
                      paired_vs_full_menu=clustered_paired_interval([
                          int(outcomes[arm][req][0]) - int(outcomes["full_menu"][req][0]) for req in positives
                      ], [grouped[labels[req]["family"]] for req in positives]))
            for arm in ARMS
        }
    write_json(args.output, dict(
        quality=results, timings=timings, signatures=signatures, reverse=args.reverse, head=fit.provenance,
        slurm=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), encoder=embedding,
        reader=reader.provenance["revision"], manifest=digest(args.manifest), projection=digest(args.projection),
        runner=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "embedding", "reader", "niso", "training", "reference", "contrasts", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    cli.add_argument("--rates", nargs=2, type=int, default=RATES)
    run(cli.parse_args())
