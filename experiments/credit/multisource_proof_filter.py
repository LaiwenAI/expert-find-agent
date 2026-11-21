"""Verify multi-source contributor proofs with complete and incomplete bylines."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.credit.linked_witness_dispatch import EXAMPLES, PROJECTION, stable
from experiments.credit.live_linked_witness_verification import TASK
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker

ELIFE = "ee6715866d53d973a437d13c679f8ef2efa8a995d721782843c5b1944859f2a3"
ARMS = ("scoped_dense", "complete_filter", "stale_filter", "guarded_filter", "guarded_staged")
LIMIT = 12
BATCH = 8


def load(path, expected, publisher):
    if digest(path) != expected:
        raise ValueError(f"Changed {publisher} projection")
    projection = json.loads(path.read_bytes())
    if publisher == "plos":
        rows = ((doi, item["owners"], "owner") for doi, item in projection["documents"].items())
    else:
        rows = ((item["doi"], item["authors"], "person") for item in projection["sources"]
                if item["declaration_complete"])
    records = {}
    for doi, authors, field in rows:
        candidates = {author[field]: set(author["roles"]) for author in authors if author[field] and author["roles"]}
        if len(candidates) != sum(bool(author[field] and author["roles"]) for author in authors):
            raise ValueError("Repeated linked authors in an article")
        if candidates:
            records[doi] = candidates
    return records


def owners_for(records, papers):
    return set.intersection(*(set(records[paper]) for paper in papers))


def gold(records, papers, clauses):
    return {person for person in owners_for(records, papers)
            if all(role in records[papers[side]][person] for side, role in clauses)}


def cases(records, n):
    by_owner = defaultdict(list)
    for paper, owners in records.items():
        for owner in owners:
            by_owner[owner].append(paper)
    linked = {group for papers in by_owner.values() for group in combinations(sorted(papers), n)}
    occupied = set()
    chosen = []
    for papers in sorted(linked, key=lambda ids: stable("publisher-multisource", ids)):
        if occupied.intersection(papers):
            continue
        available = [set.union(*records[paper].values()) for paper in papers]
        proposals = []
        for owner in sorted(owners_for(records, papers)):
            roles = [sorted(records[paper][owner], key=lambda role: stable(papers, side, role))
                     for side, paper in enumerate(papers)]
            clauses = tuple((side, role) for side, options in enumerate(roles) for role in options[:1 + (side == 0)])
            if len(clauses) != n + 1:
                continue
            for i, (side, old) in enumerate(clauses):
                for role in sorted(available[side]):
                    if role == old or (side, role) in clauses:
                        continue
                    edited = (*clauses[:i], (side, role), *clauses[i+1:])
                    if not gold(records, papers, edited):
                        proposals.append((clauses, edited))
        if proposals:
            chosen.append((papers, *min(proposals, key=lambda p: stable(papers, p))))
            occupied.update(papers)
        if len(chosen) == LIMIT:
            break
    return chosen, len(linked)


def schedules(records, papers, clauses):
    owners = [set(records[paper]) for paper in papers]
    common = set.intersection(*owners)
    dropped = min(common, key=lambda person: stable("omit-link", papers, person))
    dirty = int.from_bytes(stable("dirty-source", papers)[:4]) % len(papers)
    observed = [people - {dropped} if i == dirty else people for i, people in enumerate(owners)]
    certain = set.intersection(*(people for i, people in enumerate(observed) if i != dirty))
    sets = {
        "scoped_dense": owners,
        "complete_filter": [common for _ in papers],
        "stale_filter": [set.intersection(*observed) for _ in papers],
        "guarded_filter": [people if i == dirty and certain else certain for i, people in enumerate(owners)],
    }
    plans = {arm: [(clause, person) for clause, (side, _) in enumerate(clauses) for person in sorted(eligible[side])]
             for arm, eligible in sets.items()}
    return plans, dirty


def read_plan(reader, plan, texts, queries, clauses):
    margins, tokens = {}, 0
    start = perf_counter()
    for offset in range(0, len(plan), BATCH):
        group = plan[offset:offset+BATCH]
        values, clipped, cost = reader.score_pairs(
            [(queries[i], texts[clauses[i][0]][person]) for i, person in group], task=TASK, count_tokens=True)
        if clipped or not np.isfinite(values).all():
            raise ValueError("Truncated or invalid declaration verdict")
        margins.update(zip(group, values, strict=True))
        tokens += cost
    return margins, tokens, (perf_counter() - start) * 1000


def answer(plan, margins, clauses):
    accepted = Counter(person for i, person in plan if margins[i, person] >= 0)
    return {person for person, count in accepted.items() if count == len(clauses)}


def staged_guard(plan, dirty, clauses, check):
    clean = [unit for unit in plan if clauses[unit[0]][0] != dirty]
    good = check(clean)
    eligible = answer(clean, good, [clause for clause in clauses if clause[0] != dirty])
    pending = [unit for unit in plan if clauses[unit[0]][0] == dirty and unit[1] in eligible]
    return clean + pending, good | check(pending)


def evaluate(reader, records, n, glossary, reverse):
    families, linked = cases(records, n)
    if not families:
        raise ValueError(f"No {n}-source contrast families")
    counts = defaultdict(Counter)
    timings = defaultdict(list)
    outcomes = {}
    phrases = defaultdict(set)
    for row in glossary["examples"]:
        phrases[row["role"]].add(row["text"])
    for i, (papers, positive, negative) in enumerate(reversed(families) if reverse else families):
        texts = [{owner: " ".join(glossary["definitions"][role] for role in sorted(roles))
                  for owner, roles in records[paper].items()} for paper in papers]
        for variant, clauses in (("positive", positive), ("negative", negative)):
            target = gold(records, papers, clauses)
            queries = [min(phrases[role], key=lambda phrase: stable(papers, role, phrase)) for _, role in clauses]
            plans, dirty = schedules(records, papers, clauses)
            scores = {}
            shift = (i + int(reverse)) % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                if arm == "guarded_staged":
                    work = Counter()

                    def check(units):
                        values, tokens, elapsed = read_plan(reader, units, texts, queries, clauses)
                        work.update(tokens=tokens, elapsed=elapsed, forwards=(len(units)+BATCH-1)//BATCH)
                        return values

                    plan, values = staged_guard(plans["guarded_filter"], dirty, clauses, check)
                    plans[arm] = plan
                    tokens, elapsed, forwards = (work[key] for key in ("tokens", "elapsed", "forwards"))
                else:
                    values, tokens, elapsed = read_plan(reader, plans[arm], texts, queries, clauses)
                    forwards = (len(plans[arm])+BATCH-1)//BATCH
                scores[arm] = values
                counts[arm].update(queries=1, checks=len(plans[arm]), tokens=tokens, forwards=forwards)
                timings[arm].append(elapsed)
            reference = answer(plans["scoped_dense"], scores["scoped_dense"], clauses)
            responses = {}
            frozen_staged, _ = staged_guard(
                plans["guarded_filter"], dirty, clauses, lambda units: {unit: scores["scoped_dense"][unit] for unit in units}
            )
            for arm in ARMS:
                found = answer(plans[arm], scores[arm], clauses)
                frozen_plan = frozen_staged if arm == "guarded_staged" else plans[arm]
                frozen = answer(frozen_plan, scores["scoped_dense"], clauses)
                if arm in ("complete_filter", "guarded_filter", "guarded_staged") and frozen != reference:
                    raise ValueError("A certified schedule dropped a feasible proof")
                rows = counts[arm]
                covered = Counter(person for _, person in plans[arm])
                rows.update(exact=found == target, sufficient=bool(found & target), issued=bool(found),
                            adverse=bool(found - target), negative_issued=variant == "negative" and bool(found),
                            live_changes=found != reference, frozen_changes=frozen != reference,
                            frozen_lost_gold=len((reference & target) - frozen),
                            unchecked_gold=sum(covered[person] < len(clauses) for person in target))
                responses[arm] = [sorted(found), sorted(frozen)]
            outcomes[stable(papers, variant).hex()] = responses
    return dict(
        families=len(families), linked_source_sets=linked, checks={
            arm: dict(counts[arm], verifier_p95_ms=float(np.quantile(timings[arm], .95))) for arm in ARMS
        }, signature=stable(outcomes).hex(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "elife", "glossary", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a Slurm GPU and fresh output")
    if digest(args.glossary) != EXAMPLES:
        raise ValueError("Changed CRediT glossary")
    glossary = json.loads(args.glossary.read_bytes())
    corpora = {"plos": load(args.plos, PROJECTION, "plos"), "elife": load(args.elife, ELIFE, "elife")}
    reader = Reranker(args.reranker)
    reader.score_pairs([("Design the experiments", glossary["definitions"]["Methodology"])], task=TASK)
    panels = {}
    for publisher, records in corpora.items():
        for n in (2, 3):
            panel = f"{publisher}_{n}_sources"
            panels[panel] = evaluate(reader, records, n, glossary, args.reverse)
            print(panel, panels[panel], flush=True)
    write_json(args.output, dict(
        inputs=dict(plos=PROJECTION, elife=ELIFE, glossary=EXAMPLES), panels=panels, gpu=torch.cuda.get_device_name(),
        reranker=reader.provenance["revision"], script_sha256=digest(Path(__file__)),
        job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
