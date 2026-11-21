"""Compare fixed-budget near-duplicate review schedules on enterprise requests."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.followir.core17_reranking import digest, write_json


SCOPES = ("earlier_frame", "new_groups", "historical")
BUDGET = 8
POOL = 4


def gpu(args):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.external_enterprise_overlap import signal

    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm GPU")
    prior = json.loads(args.baseline.read_bytes())
    controls = json.loads(args.controls.read_bytes())
    if controls["stock_projection_sha256"] != digest(args.baseline):
        raise ValueError("Expected controls built from the same natural corpus")
    provenance = verify_checkpoint(args.checkpoint)
    encoder = QwenTorchEmbedding(args.checkpoint, max_length=384, batch_size=32)
    body_only = controls["controls"]["regional_body_only"]
    shorts = controls["controls"]["regional_central_excerpt"]
    focus = np.asarray(prior["focus"])
    n = prior["external"]
    earlier = prior["earlier"]
    if [i for i, _ in body_only] != list(focus) or not np.array_equal(focus, np.arange(earlier, n)):
        raise ValueError("Mismatched focal groups")

    views = prior["views"]
    texts = list(dict.fromkeys(
        [text for pair in views for text in pair] + [text for _, pair in body_only + shorts for text in pair]
    ))
    if args.order:
        texts.reverse()
    vectors = encoder.encode(texts)
    lookup = {text: i for i, text in enumerate(texts)}
    bank = [vectors[[lookup[pair[side]] for pair in views]] for side in (0, 1)]

    def scores(queries):
        query = [vectors[[lookup[pair[side]] for pair in queries]] for side in (0, 1)]
        return np.maximum.reduce([q @ d.T for q in query for d in bank])

    natural = scores([views[i] for i in focus])
    natural[np.arange(len(focus)), focus] = -np.inf
    padded = np.full((n, len(views)), -np.inf, dtype=np.float32)
    padded[focus] = natural
    counts, baseline_pairs = signal(padded, n, earlier)
    pool = {}
    for scope, left, right in zip(SCOPES, (0, earlier, n), (earlier, n, len(views)), strict=True):
        part = natural[:, left:right]
        order = np.argsort(-part, axis=1, kind="stable")[:, :POOL]
        pool[scope] = [
            [[int(j + left), round(float(part[row, j]), 6)] for j in order[row]]
            for row in range(len(focus))
        ]

    # Interleave easy full-body probes and harder short excerpts in one budget queue.
    short_by_id = dict(shorts)
    query = [(i, short_by_id[i] if i in short_by_id else views) for i, views in body_only]
    stress = scores([pair for _, pair in query])
    stress_rows = []
    for index, (target, _) in enumerate(query):
        best = np.argsort(-stress[index], kind="stable")[:POOL]
        stress_rows.append(dict(short=target in short_by_id, target=target,
                                choices=[[int(j), round(float(stress[index, j]), 6)] for j in best]))
    result = dict(focus=focus.tolist(), n=n, earlier=earlier, counts=counts,
                  baseline_pairs=baseline_pairs, pool=pool, stress=stress_rows, order=args.order,
                  encoder_revision=provenance["revision"], gpu=torch.cuda.get_device_name(),
                  baseline_sha256=digest(args.baseline), controls_sha256=digest(args.controls))
    write_json(args.output, result)
    args.output.chmod(0o600)


def pair(a, b):
    return tuple(sorted((a, b)))


def distributed(focus, sources, pool, budget, already):
    covered = set(already)
    selected = set()
    by_source = {name: 0 for name in set(sources.values())}
    for _ in range(budget):
        candidates = [
            (i, j, score) for i, choices in zip(focus, pool, strict=True)
            for j, score in choices if pair(i, j) not in selected
        ]
        if not candidates:
            break

        def priority(edge):
            i, j, score = edge
            return (i in covered, by_source[sources[i]], j in covered or j not in sources, -score, i, j)

        i, j, _ = min(candidates, key=priority)
        selected.add(pair(i, j))
        covered.update({i, j} & sources.keys())
        by_source[sources[i]] += 1
    return sorted(selected)


def choose(data, sources):
    focus = data["focus"]
    baseline = {name: [tuple(p) for p in items[:BUDGET]] for name, items in data["baseline_pairs"].items()}
    result = {}
    covered = set()
    # Each policy has eight pair slots per destination; the coverage policy
    # tracks which focal groups will already be reviewed in earlier strata.
    for name in SCOPES:
        result[name] = distributed(focus, sources, data["pool"][name], BUDGET, covered)
        covered.update({value for p in result[name] for value in p} & sources.keys())
    return {"stock": baseline, "distributed": result}


def stress_results(rows, sources):
    result = {}
    short = {row["target"] for row in rows if row["short"]}
    for budget in (8, 24, 48):
        scores = sorted(((row["choices"][0][1], row["target"], row["choices"][0][0]) for row in rows),
                        key=lambda item: (-item[0], item[1], item[2]))
        ranked = [(i, j) for _, i, j in scores[:budget]]
        # Keep rank-one candidates fixed; only their order in the review queue changes.
        visits = dict.fromkeys(set(sources.values()), 0)
        waiting = {i: (j, score) for score, i, j in scores}
        spread = []
        for _ in range(budget):
            i = min(waiting, key=lambda key: (visits[sources[key]], -waiting[key][1], key))
            j, _ = waiting.pop(i)
            spread.append((i, j))
            visits[sources[i]] += 1
        result[str(budget)] = {
            name: dict(known_source=sum(i == j for i, j in selected),
                       short_source=sum(i == j and i in short for i, j in selected))
            for name, selected in (("highest_global_score", ranked), ("source_balanced", spread))
        }
    return result


def packet(args):
    from hashlib import sha256
    from experiments.audit_enterprise_panel_neighbors import write_private
    from experiments.enterprise_regional_overlap import checked_rows

    external, history, earlier = checked_rows(args)
    records = external + history
    focus = list(range(earlier, len(external)))
    sources = {i: external[i]["batch"] for i in focus}
    results = [json.loads(path.read_bytes()) for path in (args.first, args.second)]
    if ([result["order"] for result in results] != [0, 1]
            or any(result["focus"] != focus or result["earlier"] != earlier or result["n"] != len(external)
                   or result["baseline_sha256"] != digest(args.baseline)
                   or result["controls_sha256"] != digest(args.controls) for result in results)):
        raise ValueError("Unexpected frame inventory or GPU job configuration")
    policies = [choose(result, sources) for result in results]
    private_mapping = {}
    report = []
    for result, alternatives in zip(results, policies, strict=True):
        summary = {}
        for policy, scopes in alternatives.items():
            covered = set()
            for scope, pairs in scopes.items():
                for i, j in pairs:
                    covered.update({i, j} & sources.keys())
                    private_mapping.setdefault((i, j), set()).add(f"{policy}/{scope}/{result['order']}")
            summary[policy] = dict(pairs=sum(map(len, scopes.values())), focal_groups=len(covered),
                                   covered_by_source={name: sum(sources[i] == name for i in covered)
                                                      for name in sorted(set(sources.values()))})
        report.append(dict(natural=summary, stress=stress_results(result["stress"], sources),
                           threshold_counts=result["counts"], order=result["order"]))
    visible, mapping = [], []
    for (i, j), routes in sorted(private_mapping.items()):
        a, b = records[i], records[j]
        case = sha256(f"enterprise-overlap-budget:{a['id']}:{b['id']}".encode()).hexdigest()[:24]
        if int(case,16) % 2:
            a, b = b, a
        visible.append(dict(case_id=case, left={"issuer": a["company"], "request": a["zh"]},
                            right={"issuer": b["company"], "request": b["zh"]}, intent=None,
                            organizational_link=None, reviewer=None, rationale=None))
        mapping.append(dict(case_id=case, left_id=a["id"], right_id=b["id"], routes=sorted(routes)))
    visible.sort(key=lambda row: row["case_id"])
    args.output.mkdir(mode=0o700)
    for name in ("blind-review.jsonl", "reviewer-a.jsonl", "reviewer-b.jsonl"):
        write_private(args.output / name, visible)
    write_private(args.output / "private-mapping.jsonl", mapping)
    summary = dict(blind_form_sha256=digest(args.output / "blind-review.jsonl"), blind_review_rows=len(visible),
                   human_reviews=0, focal_groups=len(focus), encoder_revision=results[0]["encoder_revision"],
                   gpu=results[0]["gpu"], script_sha256=digest(Path(__file__)), runs=report)
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("gpu")
    for field in ("baseline", "controls", "checkpoint", "output"):
        run.add_argument(field, type=Path)
    run.add_argument("--order", type=int, choices=(0, 1), default=0)
    run.set_defaults(command=gpu)
    audit = modes.add_parser("packet")
    for field in ("earlier", "municipal", "regional", "workbook", "previous_index", "report", "frame",
                  "baseline", "controls", "first", "second", "output"):
        audit.add_argument(field, type=Path)
    audit.set_defaults(command=packet)
    args = parser.parse_args()
    args.command(args)
