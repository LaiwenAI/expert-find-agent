"""Compare first-query source allocation with identical labels and declaration budgets."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import permutations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit import contrastive_rescue_identity as shared
from experiments.credit import joint_byline_depth as byline_plan
from experiments.credit import live_joint_role_postings as role_plan

route, base = shared.route, shared.base
ARMS = ("topical", "byline_sequential", "byline_joint", "role_sequential", "role_joint")
DEPTH = 64


def facets_for(ranks, roles, owners):
    return tuple({doc: {person for person, declared in owners[doc].items() if role in declared}
                  for doc in side} for role, side in zip(roles, ranks, strict=True))


def read_plan(store, ranks, roles, arm, byline, owners):
    typed = arm.startswith("role_")
    facets = facets_for(ranks, roles, owners) if typed else tuple(
        {doc: byline[doc] for doc in lane} for lane in ranks
    )
    pool = set().union(*ranks)
    key = lambda doc: (min(lane.get(doc, DEPTH + 1) for lane in ranks),
                       sum(lane.get(doc, DEPTH + 1) for lane in ranks), doc)
    selected = []
    loaded = {}

    def read(docs):
        for doc in docs:
            if doc in loaded:
                raise ValueError("Repeated declaration read")
            selected.append(doc)
            loaded[doc] = json.loads(store[doc].read_bytes())

    for side, lane in enumerate(ranks):
        eligible = (doc for doc in lane if (not typed or facets[side][doc]) and doc not in selected)
        # Reserve distinct original anchors even when the available role
        # metadata fail to admit a source for this side.
        winner = next(eligible, None)
        if winner is None:
            winner = next(doc for doc in lane if doc not in selected)
        selected.append(winner)
    initial = selected.copy()
    selected.clear()
    extras = sorted(pool - set(initial), key=key)
    fallback = [*initial, extras[0]]
    if arm == "role_joint":
        known = tuple({doc: (people, set()) for doc, people in lane.items()} for lane in facets)
        read(role_plan.source_plan(ranks, known, fallback, DEPTH))
    elif arm == "byline_joint":
        read(byline_plan.posting_plan(byline, pool, ranks, fallback, DEPTH))
    elif arm == "topical":
        read(fallback)
    else:
        read(initial)

        def gain(doc):
            value = 0.
            for slot, role in enumerate(roles):
                other = 1 - slot
                if doc not in ranks[other]:
                    continue
                for anchor in initial:
                    if anchor not in ranks[slot]:
                        continue
                    people = {person for person, declared in loaded[anchor].items() if role in declared}
                    value += len(people & facets[other][doc]) / ranks[slot][anchor]
            return value

        read([max(extras, key=gain)])
    if len(loaded) != 3:
        raise ValueError("Every route must read three distinct declarations")
    options = []
    for left, right in permutations(selected, 2):
        if left not in ranks[0] or right not in ranks[1]:
            continue
        a, b = loaded[left], loaded[right]
        ranks_pair = ranks[0][left], ranks[1][right]
        score = (max(ranks_pair), sum(ranks_pair), left, right)
        for person in sorted(a.keys() & b.keys()):
            if roles[0] in a[person] and roles[1] in b[person]:
                options.append((score, person))
    return selected, sorted(options)


def run_trial(encoder, catalog, store, owners, byline, clues, roles, missing, arm):
    started = perf_counter()
    embedded = encoder.encode(base.search.topic_prompts(clues))
    topic = catalog.rank(embedded, clues, missing, depth=DEPTH)
    ranks = tuple({doc: i for i, doc in enumerate(side["fused"][:DEPTH], 1)} for side in topic)
    selected, proofs = read_plan(store, ranks, roles, arm, byline, owners)
    output = json.dumps(proofs[:1], separators=(",", ":"))
    return (perf_counter() - started) * 1000, selected, proofs, json.loads(output)


def summary(records):
    by_state = {}
    for state, data in records.items():
        data = sorted(data, key=lambda row: (row["group"], row["request_key"]))
        quality = {}
        n = len(data)
        for arm in ARMS:
            rows = [item["arms"][arm] for item in data]
            quality[arm] = dict(
                issued=sum(x["issued"] for x in rows), exact=sum(x["exact"] for x in rows),
                wrong=sum(x["issued"] - x["exact"] for x in rows), reachable=sum(x["reachable"] for x in rows),
                requests=n, reads=3*n, p95_ms=float(np.quantile([x["latency"] for x in rows], .95)),
            )
        matched = {}
        for rival in ARMS[:-1]:
            names = (rival, "role_joint")
            quota = min(quality[name]["issued"] for name in names)
            selected = {
                arm: sorted((row for row in data if row["arms"][arm]["issued"]),
                            key=lambda row: (row["arms"][arm]["priority"], row["request_key"]))[:quota]
                for arm in names
            }
            common = [row for row in data if all(row["arms"][arm]["issued"] for arm in names)]
            delta = [row["arms"][names[1]]["exact"] - row["arms"][names[0]]["exact"] for row in data]
            matched[rival] = dict(
                quota=quota, exact={arm: sum(row["arms"][arm]["exact"] for row in selected[arm]) for arm in names},
                common_issuance=len(common),
                common_exact={arm: sum(row["arms"][arm]["exact"] for row in common) for arm in names},
                paired_exact=clustered_paired_interval(np.asarray(delta), [row["group"] for row in data]),
            )
        by_state[state] = dict(methods=quality, matched=matched)
    return by_state


def run(args):
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output on a Slurm GPU")
    base.verify_checkpoint(args.embedding)
    elife = args.confirmation, args.extended
    base_manifest = shared.role.read(args.base_manifest)
    old, new = (shared.role.read(path) for path in (args.old_manifest, args.fresh_manifest))
    if new["prior_digests"] != {"base": base.digest(args.base_manifest), "holdout": base.digest(args.old_manifest)}:
        raise ValueError("Changed prior catalog locks")
    older = shared.prior.restore(base_manifest, old, args.old_xml, args.base_projection, *elife)
    fresh = shared.prior.restore(base_manifest, new, args.fresh_xml, args.base_projection, *elife)
    articles = older | fresh
    if len(older) != 414 or len(articles) != 440 or len(new["test"]["pairs"]) != 13:
        raise ValueError("Expected the frozen 440-source catalog and 13 groups")
    _, _, wording = base.queries(args.glossary, args.training)
    plan = shared.prior.transfer.plos_requests(articles, new["test"], wording)
    encoder = base.QwenTorchEmbedding(args.embedding)
    catalog = route.Catalog(articles, encoder)
    owners = {doc: data["roles"] for doc, data in articles.items()}
    byline = {doc: set(persons) for doc, persons in owners.items()}
    records = defaultdict(list)
    details = []
    with TemporaryDirectory(dir=args.output.parent) as dir_name:
        store = {doc: Path(dir_name) / f"{i}.json" for i, doc in enumerate(sorted(articles))}
        for doc, path in store.items():
            path.write_text(json.dumps({person: sorted(roles) for person, roles in owners[doc].items()}))
        warm_pair = next(iter(plan))
        warm_clues = [catalog.keywords(articles[doc]["clue"]) for doc in warm_pair]
        for arm in ARMS:
            run_trial(encoder, catalog, store, owners, byline, warm_clues, plan[warm_pair][0][1], None, arm)
        counter = 0
        for intended, cases in sorted(plan.items(), reverse=args.reverse):
            clues = [catalog.keywords(articles[doc]["clue"]) for doc in intended]
            for missing in (None, *intended):
                state = "present" if missing is None else "withdrawn"
                for _, pair_roles, _ in cases:
                    expected = base.known_owners(articles, intended, pair_roles)
                    arms = ARMS[::-1] if args.reverse else ARMS
                    offset = counter % len(ARMS)
                    outcome = {}
                    for arm in arms[offset:] + arms[:offset]:
                        ms, reads, proofs, returned = run_trial(encoder, catalog, store, owners, byline,
                                                               clues, pair_roles, missing, arm)
                        def valid(proof):
                            score, person = proof
                            return tuple(score[2:]) == intended and person in expected

                        outcome[arm] = dict(issued=int(bool(returned)), exact=int(bool(returned) and valid(returned[0])),
                                            reachable=int(any(map(valid, proofs))), latency=ms,
                                            priority=tuple(returned[0][0][:2]) if returned else None)
                        details.append((intended, missing, pair_roles, arm, reads, returned))
                    key = sha256(json.dumps((clues, pair_roles, missing)).encode()).hexdigest()
                    records[state].append(dict(group="|".join(intended), request_key=key, arms=outcome))
                    counter += 1
    signature = sha256(json.dumps(sorted(details, key=lambda row: json.dumps(row))).encode()).hexdigest()
    base.write_json(args.output, dict(outcomes=summary(records), signature=signature, reverse=args.reverse,
                    panel_sha256=base.digest(args.fresh_manifest), script_sha256=base.digest(Path(__file__)),
                    gpu=torch.cuda.get_device_name()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("confirmation", "extended", "base_manifest", "base_projection", "old_manifest", "old_xml",
                  "fresh_manifest", "fresh_xml", "glossary", "training", "embedding", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
