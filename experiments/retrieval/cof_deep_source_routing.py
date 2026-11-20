"""Compare fixed and state-dependent use of the same deeper source budget."""

import argparse
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("dense_fixed", "dense_state", "lexical_state", "learned_fixed", "learned_state")
BUDGETS = (1, 3, 5, 10)
TOKEN_CAPS = (1536, 2560)


def initial_reads(row):
    return {doc for profile in row["candidates"] for doc in profile[:3]}


def train(source, target_titles):
    rows = []
    outcomes = []
    queries = 0
    for query in source["data"]:
        if query["title"] in target_titles:
            continue
        shallow = initial_reads(query)
        deep = [doc for doc in range(len(query["accepts"])) if doc not in shallow]
        if deep:
            queries += 1
        rows.extend([1., *query["features"][doc]] for doc in deep)
        outcomes.extend(query["accepts"][doc] for doc in deep)
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    weights = np.zeros(x.shape[1])
    penalty = 2 * np.eye(x.shape[1])
    penalty[0, 0] = 0.
    for _ in range(18):
        estimate = logistic(x @ weights)
        hessian = (x.T * (estimate * (1 - estimate))) @ x + penalty
        gradient = x.T @ (estimate - y) + penalty @ weights
        delta = np.linalg.solve(hessian, gradient)
        weights -= delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    return weights, dict(queries=queries, deeper_sources=len(y), accepted_sources=int(y.sum()))


def logistic(values):
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def status_map(profiles, observations):
    return [min(2, sum(observations.get(doc, False) for doc in profile)) for profile in profiles]


def actions(profiles, observations, probabilities):
    counts = status_map(profiles, observations)
    pending = {}
    for person, profile in enumerate(profiles):
        needed = 2 - counts[person]
        sources = [doc for doc in profile if doc not in observations]
        if len(sources) < needed or not needed:
            continue
        if needed == 1:
            for doc in sources:
                pending[doc] = pending.get(doc, 0.) + probabilities[doc]
            continue
        best, second = sorted(sources, key=lambda doc: (-probabilities[doc], doc))[:2]
        for doc in sources:
            partner = second if doc == best else best
            pending[doc] = pending.get(doc, 0.) + probabilities[doc] * probabilities[partner]
    return sorted(pending, key=lambda doc: (-pending[doc], doc))


def simulate(row, policy, weights, budget, token_cap=None):
    start = perf_counter()
    profiles = row["candidates"]
    truth = row["truth"]
    initial = initial_reads(row)
    observations = {doc: row["accepts"][doc] for doc in initial}
    matrix = np.asarray(row["features"], dtype=np.float64)
    if policy.startswith("dense"):
        probabilities = matrix[:, 0]
    elif policy.startswith("lexical"):
        probabilities = .10 + .90 * matrix[:, 3]
    else:
        probabilities = logistic(weights[0] + matrix @ weights[1:])
    baseline = actions(profiles, observations, probabilities)
    # The fixed controls roll out the order formed from the shallow observations
    # without reacting to individual second-wave verifier decisions.
    ordered = sorted(set(range(len(matrix))) - initial, key=lambda doc: (-probabilities[doc], doc))
    static = list(dict.fromkeys([*baseline, *ordered]))
    purchased = []
    tokens = 0
    cap = min(budget, len(static))
    for _ in range(cap):
        priorities = static
        if policy.endswith("_state"):
            ranked = actions(profiles, observations, probabilities)
            priorities = list(dict.fromkeys([*ranked, *ordered]))
        eligible = (
            doc for doc in priorities if doc not in observations
            and (token_cap is None or tokens + row["token_lengths"][doc] <= token_cap)
        )
        doc = next(eligible, None)
        if doc is None:
            break
        observations[doc] = row["accepts"][doc]
        purchased.append(doc)
        tokens += row["token_lengths"][doc]
    if token_cap is not None and tokens > token_cap:
        raise AssertionError("Exceeded source verification token cap")
    elapsed = (perf_counter() - start) * 1000
    result = attachments(profiles, initial, observations)
    return dict(
        read_count=len(purchased), initial_reads=len(initial), route_ms=elapsed,
        cited=sum(bool(docs) for docs in result), citations=result,
        known=sum(bool(docs) and grade is not None and grade > 0 for docs, grade in zip(result, truth, strict=True)),
        negative=sum(bool(docs) and grade is not None and grade <= 0 for docs, grade in zip(result, truth, strict=True)),
        unknown=sum(bool(docs) and grade is None for docs, grade in zip(result, truth, strict=True)),
        tokens=tokens,
    )


def attachments(profiles, initial, observations):
    result = []
    for profile in profiles:
        shallow = [doc for doc in profile if doc in initial and observations.get(doc, False)]
        if len(shallow) >= 2:
            result.append(shallow[:2])
            continue
        evidence = [doc for doc in profile if observations.get(doc, False)]
        result.append(evidence[:2] if len(evidence) >= 2 else [])
    return result


def matched(current, reference, rows):
    cap_positives = []
    cap_negatives = []
    lower = []
    upper = []
    for value, baseline, row in zip(current, reference, rows, strict=True):
        cap = min(value["cited"], baseline["cited"])
        issued = [set(i for i, docs in enumerate(arm["citations"]) if docs)
                  for arm in (value, baseline)]
        current_ids, baseline_ids = [set(sorted(indices)[:cap]) for indices in issued]
        added = [row["truth"][i] for i in current_ids - baseline_ids]
        dropped = [row["truth"][i] for i in baseline_ids - current_ids]
        positive = sum(grade is not None and grade > 0 for grade in added)
        positive -= sum(grade is not None and grade > 0 for grade in dropped)
        negative = sum(grade is not None and grade <= 0 for grade in added)
        negative -= sum(grade is not None and grade <= 0 for grade in dropped)
        cap_positives.append(positive)
        cap_negatives.append(negative)
        lower.append(positive - sum(grade is None for grade in dropped))
        upper.append(positive + sum(grade is None for grade in added))
    return dict(known_positive_delta=sum(cap_positives), judged_negative_delta=sum(cap_negatives),
                positive_delta_p5=paired_interval([v / 5 for v in cap_positives]),
                partial_label_delta_bounds=[sum(lower)/(5*len(rows)), sum(upper)/(5*len(rows))])


def summarize(rows, weights, budget, token_cap=None):
    data = {arm: [simulate(row, arm, weights, budget, token_cap) for row in rows] for arm in ARMS}
    if token_cap is None:
        for row_result in zip(*(data[a] for a in ARMS), strict=True):
            if len({value["read_count"] for value in row_result}) != 1:
                raise AssertionError("The arms must share each realized source count")
    reference = data["dense_state"]
    result = {}
    for arm, observations in data.items():
        counts = {
            k: sum(row[k] for row in observations)
            for k in ("read_count", "initial_reads", "cited", "known", "negative", "unknown", "tokens")
        }
        if counts["cited"] != counts["known"] + counts["negative"] + counts["unknown"]:
            raise AssertionError("Citation counts differ from judgment partition")
        comparison = paired_interval([
            (candidate["known"] - baseline["known"]) / 5
            for candidate, baseline in zip(observations, reference, strict=True)
        ])
        result[arm] = dict(
            **counts, route_p95_ms=float(np.quantile([r["route_ms"] for r in observations], .95)),
            delta_vs_dense_state=comparison,
            matched_issuance_vs_dense_state=matched(observations, reference, rows),
        )
    return result


def evaluate(source, target):
    rows = target["data"]
    weights, fit = train(source, {row["title"] for row in rows})
    full = [simulate(row, "dense_fixed", weights, target["depth"] * 5) for row in rows]
    initial = [simulate(row, "dense_fixed", weights, 0) for row in rows]
    return dict(train=source["dataset"], target=target["dataset"], queries=len(rows), fit=fit,
                baseline_depth3={k:sum(r[k] for r in initial) for k in ("known", "negative", "unknown", "cited")},
                full_depth12={k:sum(r[k] for r in full) for k in ("known", "negative", "unknown", "cited")},
                equal_reads={str(budget): summarize(rows, weights, budget) for budget in BUDGETS},
                capped_tokens={str(tokens): summarize(rows, weights, 5, tokens) for tokens in TOKEN_CAPS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kdd", "scirepeval", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    first, second = (json.loads(path.read_bytes()) for path in (args.kdd, args.scirepeval))
    for field in ("embedding", "reranker", "features", "depth", "score_batch", "token_cap"):
        if first[field] != second[field]:
            raise ValueError(f"Capture protocol differs: {field}")
    write_json(args.output, dict(embedding=first["embedding"], reranker=first["reranker"],
                                capture_sha256={x["dataset"]: digest(path) for x, path in
                                                zip((first, second), (args.kdd, args.scirepeval), strict=True)},
                                directions=[evaluate(first, second), evaluate(second, first)]))


if __name__ == "__main__":
    main()
