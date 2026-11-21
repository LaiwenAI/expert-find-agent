"""Test the resident search policies on a fixed engineering publisher panel."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.elife_resident_role_service import ARMS, Search
from experiments.credit.engineering_role_banks import projections, queries
from experiments.credit.joint_request_queue import Request
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.paired_role_prospective_service import FROZEN_SHA, clusters
from experiments.credit.paired_role_verification import PairedReader
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_contrast_head import RoleContrastHeads
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint

PANEL_SHA = "53f4cda52f54d2982ff748c0e9fbbff627cbc1c9eeef024cd35e8539422137fb"


class MaskedCatalog(PostingCatalog):
    withheld = None

    def full_rank(self, vectors, clues, withheld):
        return super().full_rank(vectors, clues, self.withheld)


def evaluate(search, cases, wording):
    catalog = search.catalog
    results = {}
    for bank in ("original", "alternate"):
        phrases = [text for _, _, text in wording[bank]]
        results[bank] = {}
        for view in ("keyword", "excerpt"):
            for state in ("both_present", "one_withheld"):
                results[bank][f"{view}/{state}"] = totals = {arm: Counter() for arm in ARMS}
                for case in cases:
                    papers, roles, gold = case["papers"], case["roles"], set(case["gold"])
                    topics = [catalog.articles[doc]["clue"] for doc in papers]
                    if view == "keyword":
                        topics = list(map(catalog.keywords, topics))
                    request = Request("elife", json.dumps({"requirements": [
                        {"topic": topic, "role": phrases[ROLES.index(role)]}
                        for topic, role in zip(topics, roles, strict=True)
                    ]}))
                    for document in ((None,) if state == "both_present" else papers):
                        catalog.withheld = document
                        for arm in ARMS:
                            response = json.loads(search.execute(request, arm))
                            pair, owners = response["papers"], set(response["owners"])
                            if pair and document in pair:
                                raise ValueError("Withheld paper entered the result")
                            strict = (papers == pair and response["roles"] == [[role] for role in roles]
                                      and bool(owners & gold))
                            totals[arm].update(requests=1, strict=strict, adverse=bool(owners) and not strict,
                                               exact_sources=pair == papers, positive=bool(gold))
    return results


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require Slurm CUDA and a fresh output")
    reference_sha = digest(args.reference)
    if reference_sha != FROZEN_SHA or digest(args.panel) != PANEL_SHA:
        raise ValueError("Unexpected role calibration or publisher panel")
    reference = json.loads(args.reference.read_bytes())["runs"][0]
    if digest(args.training) != reference["inputs"]["training"]:
        raise ValueError("Unexpected training bank")
    panel = json.loads(args.panel.read_bytes())
    articles, cases = panel["articles"], panel["queries"]
    definition, phrases, wording = queries(args.niso, args.training)
    embed_rev = verify_checkpoint(args.embedding)["revision"]
    if embed_rev != reference["embedding_revision"]:
        raise ValueError("Unexpected frozen embedding")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    banks = projections(encoder, definition, phrases)
    head = RoleContrastHeads(args.contrasts, encoder, phrases, wording)
    reader = PairedReader(args.reader, ROLES)
    if reader.provenance["revision"] != reference["role_reader_revision"]:
        raise ValueError("Unexpected frozen reader")
    search = Search(MaskedCatalog(articles, encoder), encoder, reader, reference["fit"]["two_batched"]["cutoff"],
                    {"ridge": banks["ridge"], "resident_ce": head.weights["original_ce"]})
    outcomes = evaluate(search, cases[::-1] if args.reverse else cases, wording)
    write_json(args.output, dict(results=outcomes, training=head.provenance, panel=PANEL_SHA,
                                articles=len(articles), groups=len(set(clusters(cases, articles).values())),
                                cases=len(cases), gpu=torch.cuda.get_device_name(), encoder=embed_rev,
                                reader=reader.provenance["revision"], slurm=os.environ["SLURM_JOB_ID"],
                                reverse=args.reverse))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "embedding", "reader", "niso", "training", "reference", "contrasts", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
