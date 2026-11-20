"""Test a budgeted declaration index on the independent eLife catalog."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.credit import budgeted_declaration_index as budgeted
from experiments.credit import minimal_evidence_bundles as bundles
from experiments.followir.core17_reranking import digest, write_json


def inventory(path, root, bits, *, complete_only):
    original = json.loads(path.read_bytes())["sources"]
    sources = [article for article in original if not complete_only or article["declaration_complete"]]
    authors, declared = bundles.owner_data(sources)
    labels = {key: bundles.role_mask(roles, bits) for key, roles in declared.items()}
    caps = {article["doi"]: bundles.role_mask(
        {role for record in article["authors"] for role in record["roles"]}, bits,
    ) for article in sources}
    records = {source: {} for source in caps}
    for (owner, source), mask in labels.items():
        records[source][owner] = mask
    files = {doc: root / f"{i}.json" for i, doc in enumerate(records)}
    for doc, file in files.items():
        file.write_text(json.dumps(records[doc], separators=(",", ":")))
    costs = {doc: file.stat().st_size for doc, file in files.items()}
    return authors, labels, caps, files, costs, len(original)


def features(owners, caps, costs, asks):
    ranks = defaultdict(list)
    linked = Counter()
    for rank, (_, docs) in enumerate(owners):
        for doc in docs:
            ranks[doc].append(rank + 1)
            linked[doc] += len(docs) > 1
    optimistic = Counter()
    for ask in asks:
        optimistic.update(budgeted.optimistic_sources(owners, caps, ask))
    # Keep the scale comparable for catalogues with different owner counts.
    rank_scale = np.log1p(len(owners))
    return np.asarray([
        (np.log1p(min(ranks[doc])) / rank_scale,
         np.log1p(float(np.mean(ranks[doc]))) / rank_scale,
         np.log1p(len(ranks[doc])),
         linked[doc] / len(ranks[doc]),
         caps[doc].bit_count(),
         optimistic[doc] / len(asks),
         np.log1p(costs[doc]))
        for doc in costs
    ])


def train_transfer(source, target, training, cold):
    s_owners, _, s_caps, _, s_costs, _ = source
    t_owners, _, t_caps, _, t_costs, _ = target
    train = features(s_owners, s_caps, s_costs, training)
    test = features(t_owners, t_caps, t_costs, training)
    response = Counter()
    for ask in training:
        response.update(cold[ask])
    rates = np.asarray([response[doc] / len(training) for doc in s_costs])
    center = train.mean(axis=0)
    width = train.std(axis=0)
    width[width == 0] = 1
    standardized = (train - center) / width
    coef = np.linalg.solve(standardized.T @ standardized + 8 * np.eye(train.shape[1]),
                           standardized.T @ (rates - rates.mean()))
    estimates = np.maximum(0, rates.mean() + ((test - center) / width) @ coef)
    return dict(zip(t_costs, estimates, strict=True))


def choose_on_development(owners, labels, caps, costs, training, policies, budgets):
    options = ("author_prior", "training_read_frequency_per_byte")
    chosen = {}
    volumes = {}
    for budget in budgets:
        counts = {}
        for name in options:
            selected = budgeted.pack(policies[name], [budget], costs)[budget]
            cached = budgeted.state(owners, labels, caps, selected)
            counts[name] = sum(len(budgeted.cold_sources(owners, cached, labels, ask)[0])
                               for ask in training)
        volumes[budget] = counts
        chosen[budget] = min(options, key=counts.__getitem__)
    return chosen, volumes


def run(plos, elife, manifest, scratch, output):
    from experiments.credit.plos_role_inventory import ROLES

    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a new output on a Slurm compute node")
    anchor = json.loads(manifest.read_bytes())
    if digest(plos) != bundles.FIXTURE_SHA256 or digest(elife) != anchor["projection_sha256"]:
        raise ValueError("Publisher snapshot changed")
    bits = {role: 1 << i for i, role in enumerate(sorted(ROLES))}
    requests = [sum(triple) for triple in combinations(bits.values(), 3)]
    result = {}
    with TemporaryDirectory(dir=scratch, prefix="publisher-transfer-") as directory:
        root = Path(directory)
        (root / "plos").mkdir()
        (root / "elife").mkdir()
        source = inventory(plos, root / "plos", bits, complete_only=False)
        target = inventory(elife, root / "elife", bits, complete_only=True)
        s_owners, s_labels, s_caps, *_ = source
        owners, labels, caps, files, costs, target_total = target
        s_state = budgeted.state(s_owners, s_labels, s_caps, set())
        target_state = budgeted.state(owners, labels, caps, set())
        budgets = [sum(costs.values()) * level // 100 for level in budgeted.LEVELS]
        for fold in budgeted.FOLDS:
            role = 1 << fold
            training = [ask for ask in requests if not ask & role]
            held = [ask for ask in requests if ask & role]
            plos_cold, _ = budgeted.cold_workload(dict.fromkeys(training, s_owners), s_state, s_labels)
            elicold, _ = budgeted.cold_workload(dict.fromkeys(training, owners), target_state, labels)
            predicted = train_transfer(source, target, training, plos_cold)
            policies = budgeted.placements(owners, caps, costs, training, elicold)
            policies["plos_model_per_byte"] = sorted(
                costs, key=lambda doc: (-predicted[doc] / costs[doc], costs[doc], doc),
            )
            chosen, development_reads = choose_on_development(
                owners, labels, caps, costs, training, policies, budgets,
            )
            held_order = dict.fromkeys(held, owners)
            cold_test, references = budgeted.cold_workload(held_order, target_state, labels)
            measured = budgeted.execute(
                files, owners, labels, caps, costs, held, held_order, references, policies, budgets,
            )
            result[fold] = dict(
                held_queries=len(held), calibration_queries=len(training),
                cold_reads=sum(map(len, cold_test.values())),
                chosen_by_development=chosen, development_reads=development_reads, by_budget=measured,
            )
            print(f"eLife held role {fold}: {len(held)} requests", flush=True)
        counts = dict(target_articles=len(files), original_target_articles=target_total,
                      target_authors=len(owners), target_bytes=sum(costs.values()), source_articles=len(source[3]))
    write_json(output, dict(
        folds=result, counts=counts, budgets=dict(zip(budgeted.LEVELS, budgets, strict=True)),
        plos_sha256=digest(plos), elife_sha256=digest(elife), elife_manifest_sha256=digest(manifest),
        script_sha256=digest(Path(__file__)), placement_script_sha256=digest(Path(budgeted.__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "elife", "manifest", "scratch", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
