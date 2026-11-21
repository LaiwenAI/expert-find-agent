"""Compare two-window same-owner evidence recovery on original enterprise requests."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.retrieval import enterprise_packed_role_search as packed
from experiments.retrieval.enterprise_first_query import CRITERION_CHARS, EMBED_TASK, K, TASK, evaluate
from experiments.retrieval.enterprise_packed_role_load import init_fixture
from experiments.retrieval.enterprise_role_first_query import recover, recovery_options, require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("global_on_abstain", "owned_bundle", "guarded_bundle")
DEPTH = 24
GUARD_DEPTH = 3


class BundleIndex:
    def __init__(self, windows, encoder, depth):
        self.windows = windows
        self.encoder = encoder
        self.depth = depth
        self.vectors = torch.from_numpy(encoder.encode([item["view"]["text"] for item in windows])).to("cuda")

    def select(self, criteria):
        prompts = [f"Instruct: {EMBED_TASK}\nQuery: {criterion}" for criterion in criteria]
        queries = torch.from_numpy(self.encoder.encode(prompts)).to("cuda")
        similarity = (queries @ self.vectors.T).cpu().numpy()
        hot = np.argsort(-similarity, axis=1, kind="stable")[:, :self.depth]
        ranks = [{int(i): rank + 1 for rank, i in enumerate(row)} for row in hot]
        owned = defaultdict(list)
        for i in sorted(set(int(x) for row in hot for x in row)):
            owned[self.windows[i]["expert"]].append(i)
        proposals = []
        for candidates in owned.values():
            for a, b in combinations(candidates, 2):
                if self.windows[a]["view"]["id"] == self.windows[b]["view"]["id"]:
                    continue
                covered = [min(r.get(a, self.depth + 1), r.get(b, self.depth + 1)) for r in ranks]
                if max(covered) > self.depth:
                    continue
                proposals.append((max(covered), sum(covered), a, b))
        if not proposals:
            return None
        _, _, a, b = min(proposals)
        best = tuple(self.windows[i] for i in (a, b))
        guaranteed_candidates = [[r.get(i, self.depth + 1) <= GUARD_DEPTH for i in (a, b)] for r in ranks]
        return best, guaranteed_candidates


def verify_bundle(options, prior_ranks, criteria, reader, guarded):
    inputs = [(criterion, item["view"]["text"]) for item in options for criterion in criteria]
    margins, clipped = reader.score_pairs(inputs, task=TASK)
    if clipped:
        return [], clipped
    if not np.isfinite(margins).all():
        raise ValueError("Nonfinite bundle margin")
    scores = np.asarray(margins).reshape((2, len(criteria)))
    complete = np.flatnonzero((scores > 0).all(axis=1))
    if len(complete):
        best = max(complete, key=lambda i: float(scores[i].min()))
        other = 1-int(best)
        scores[other, :] = -np.inf
    elif guarded:
        scores = np.where(np.asarray(prior_ranks, dtype=bool).T, scores, -np.inf)
    citations = []
    for k in range(len(criteria)):
        winner = int(scores[:, k].argmax())
        if scores[winner, k] <= 0:
            return [], 0
        view = options[winner]["view"]
        citations.append((view["id"], view["start"], view["end"]))
    expert = options[0]["expert"]
    if expert != options[1]["expert"]:
        raise ValueError("The selected sources do not share an owner")
    return [dict(expert=expert, citations=[list(c) for c in dict.fromkeys(citations)])], 0


def act(text, arm, fixture, bundles):
    parser, encoder, index, sources, role_index, reader = fixture
    reader.change("guarded_rescue")
    started = perf_counter()
    initial = evaluate(text, "broadcast_prefix", parser, encoder, index, sources, reader)
    primary = [dict(expert=eid, citations=[list(c) for c in dict.fromkeys(citations)])
               for eid, passed, citations in initial["decisions"] if passed]
    criteria = [text.strip().splitlines()[0][:CRITERION_CHARS],
                *(text[a:b] for a, b in initial["qualifications"])]
    issued = {row["expert"] for row in primary}
    ranked = [decision[0] for decision in initial["decisions"]]
    reader.counts.clear()
    added, clipped, allocated, attempted, top_three_complete = [], 0, False, False, False
    if not issued and all(len(c) <= CRITERION_CHARS for c in criteria):
        if arm != "global_on_abstain" and len(criteria) >= 2:
            original = recovery_options(criteria, ranked, issued, role_index, "global_on_abstain")
            if len(original) == 2:
                proposal = bundles.select(criteria)
                if proposal is not None:
                    allocated = attempted = True
                    options, prior_ranks = proposal
                    top_three_complete = all(map(any, prior_ranks))
                    added, clipped = verify_bundle(options, prior_ranks, criteria, reader,
                                                   guarded=arm == "guarded_bundle")
        if not allocated:
            added, _, attempted, clipped = recover(criteria, ranked, issued, role_index, reader,
                                                    "global_on_abstain", TASK, K-len(primary))
    return dict(qualifications=initial["qualifications"], ranked=initial["retrieved"],
                primary=primary, added=added, attempted=attempted, allocated=allocated, work=dict(reader.counts),
                top_three_complete=top_three_complete,
                clipped=initial["clipped_pairs"] + clipped, ms=(perf_counter()-started)*1000)


def run(queries, corpus, roles, vectors, meta, models, serial, output, private, reverse, proposal_depth):
    require_cuda(output, private)
    texts, fixture, fingerprint = init_fixture(queries, corpus, roles, vectors, meta, models, serial)
    windows, _ = fixture[4]
    start = perf_counter()
    bundles = BundleIndex(windows, fixture[1], proposal_depth)
    index_ms = (perf_counter()-start)*1000
    arms = ARMS[::-1] if reverse else ARMS
    observations = []
    for i, text in enumerate(texts):
        order = arms[i % len(arms):] + arms[:i % len(arms)]
        frozen = packed.one(text, "guarded_rescue", *fixture)
        records = {arm: act(text, arm, fixture, bundles) for arm in order}
        assert all((r["qualifications"], r["ranked"], r["primary"]) ==
                   (frozen["qualifications"], frozen["ranked"], frozen["initial"]) for r in records.values())
        baseline = records[ARMS[0]]
        assert (baseline["added"], baseline["attempted"], baseline["clipped"]) == (
            frozen["extra"], frozen["invoked"], frozen["clipped"])
        assert all(r["work"].get("logical_pairs", 0) == baseline["work"].get("logical_pairs", 0)
                   for r in records.values())
        observations.append(records)
        if (i+1) % 20 == 0:
            print(i+1, "original queries replayed", flush=True)
    ids = [row["id"] for row in rows(queries)]
    trace = [dict(query=id_, arms=row) for id_, row in zip(ids, observations, strict=True)]
    write(private, dict(observations=trace, **fingerprint))
    totals = {}
    for name in ARMS:
        current = [r[name] for r in observations]
        reference = [r[ARMS[0]] for r in observations]
        totals[name] = dict(
            allocated=sum(x["allocated"] for x in current),
            top_three_complete=sum(x["top_three_complete"] for x in current),
            issued=sum(bool(x["primary"] or x["added"]) for x in current),
            added_slots=sum(len(x["added"]) for x in current),
            multi_source_additions=sum(len({c[0] for c in item["citations"]})>1
                                       for x in current for item in x["added"]),
            changed_additions=sum(x["added"]!=y["added"] for x,y in zip(current,reference,strict=True)),
            verifier_work=dict(sum((Counter(row["work"]) for row in current), Counter())),
            serial_p95_ms=float(np.quantile([r["ms"] for r in current], .95)),
        )
    write(output, dict(slurm_job=os.environ["SLURM_JOB_ID"], runner_sha256=digest(Path(__file__)),
                       index_ms=index_ms, reverse=reverse, proposal_depth=proposal_depth,
                       arms=totals, **fingerprint))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "meta", "models", "serial", "output", "private"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    cli.add_argument("--proposal-depth", type=int, default=DEPTH, choices=(DEPTH, 256))
    run(**vars(cli.parse_args()))
