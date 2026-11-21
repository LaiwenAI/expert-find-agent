"""Filter role rechecks whose returned proof would be structurally impossible."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from experiments.credit import plos_frozen_role_recheck_service as frozen

base, resident = frozen.base, frozen.resident
ARMS = ("baseline", "frozen_source_score_4b", "exact_feasibility", "top2_feasibility")


class Engine(frozen.Engine):
    def should_recheck(self, pair, labels, uncertain, request):
        policy = request["policy"]
        if policy == "frozen_source_score_4b":
            return True

        articles = self.index.catalog.articles
        a, b = (articles[paper]["roles"] for paper in pair)
        if policy == "exact_feasibility":
            pools = [None if side in uncertain else {label} for side, label in enumerate(labels)]
        else:
            # Use the second-ranked label from the same small-model forward.
            pools = [self.top_two[side] if side in uncertain else {label} for side, label in enumerate(labels)]
        return any(all(bool(side[owner]) if pool is None else bool(pool.intersection(side[owner]))
                       for side, pool in zip((a, b), pools, strict=True))
                   for owner in a.keys() & b.keys())

    @property
    def top_two(self):
        return [{base.ROLES[i] for i in row} for row in self.role_order]

    def execute(self, request):
        reply = super().execute(request)
        reply["ridge_priority"] = self.request_margin
        return reply

    # Expose the ranked roles without rerunning the embedding model.
    def encode(self, clues, duties):
        prompts = base.search.topic_prompts(clues) + [f"Instruct: {base.ROLE_TASK}\nQuery:{text}" for text in duties]
        topic, roles = np.split(self.encoder.encode(prompts), [2])
        values = frozen.shared.role.normalize_scores(roles @ self.ridge.T)
        self.role_order = np.argsort(-values, axis=1, kind="stable")[:, :2]
        labels = tuple(base.ROLES[i] for i in self.role_order[:, 0])
        margins = [float(values[i, a] - values[i, b]) for i, (a, b) in enumerate(self.role_order)]
        self.current = (topic, labels, margins)
        self.request_margin = min(margins)
        return topic, labels, self.request_margin


def pack(result, exact, wrong):
    return dict(correct=int(exact), wrong=int(wrong), calls=result["role_calls"], tokens=result["role_tokens"])


def matched_budget(records, budget):
    predicted = [row for row in records if row[1]["frozen_source_score_4b"]["role_calls"]]
    ranked = sorted(predicted, key=lambda row: (-row[1]["frozen_source_score_4b"]["ridge_priority"], row[0]))
    admitted = {row[0] for row in ranked[:budget]}
    if len(admitted) != budget:
        raise ValueError("Matched budget needs distinct requests")
    score = Counter()
    for key, _, facts in records:
        score.update(facts["frozen_source_score_4b" if key in admitted else "baseline"])
    return dict(score)


def exercise(engine, rows, reverse):
    service = frozen.HTTPServer(("127.0.0.1", 0), resident.Handler)
    service.engines = {"plos": engine}
    thread = frozen.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    client = frozen.HTTPConnection("127.0.0.1", service.server_address[1], timeout=90)
    articles = engine.index.catalog.articles
    records, stats, latency, all_proofs = defaultdict(list), defaultdict(Counter), defaultdict(list), []
    aliases = Counter()
    try:
        for _, _, _, _, request in rows[::max(1, len(rows)//12)]:
            for arm in ARMS:
                resident.calculate(client, "/search", request | {"policy": arm})
        for i, (intended, absent, kind, gold, request) in enumerate(rows):
            order = ARMS[::-1] if reverse else ARMS
            offset = i % len(order)
            replies = {arm: resident.calculate(client, "/search", request | {"policy": arm})
                       for arm in order[offset:] + order[:offset]}
            if replies["exact_feasibility"]["proof"] != replies["frozen_source_score_4b"]["proof"]:
                raise ValueError("Feasibility gate changed a proof")
            if replies["baseline"]["proof"] and any(
                result["proof"] != replies["baseline"]["proof"] for result in replies.values()
            ):
                raise ValueError("A policy changed an issued baseline proof")
            identity = json.dumps((intended, absent, kind, gold, request["duties"]), sort_keys=True)
            route_key = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            key = (route_key, aliases[route_key])
            aliases[route_key] += 1
            state = "present" if absent is None else "withdrawn"
            facts = {}
            for arm, result in replies.items():
                exact, wrong = frozen.checked_proof(articles, intended, gold, result)
                facts[arm] = pack(result, exact, wrong)
                stats[state, arm].update(facts[arm])
                latency[state, arm].append(result["client_ms"])
                all_proofs.append((identity, arm, result["proof"]))
            records[state].append((key, replies, facts))
    finally:
        client.close()
        service.shutdown()
        thread.join()
        service.server_close()
    reports = {}
    for state in records:
        arms = {arm: dict(stats[state, arm], **resident.aggregate({"client": latency[state, arm]})["client"])
                for arm in ARMS}
        reports[state] = dict(arms=arms, score_matched={
            gate: matched_budget(records[state], arms[gate]["calls"])
            for gate in ("exact_feasibility", "top2_feasibility")
        })
    signature = sha256(json.dumps(sorted(all_proofs, key=lambda row: json.dumps(row, sort_keys=True)),
                                  sort_keys=True).encode()).hexdigest()
    return reports, signature


def run(args):
    engine, data = frozen.prepare(args, Engine)
    results, signature = exercise(engine, data, args.reverse)
    base.write_json(args.output, dict(
        holdout=base.digest(args.fresh_manifest), frozen_reference=frozen.REFERENCE,
        reverse=args.reverse, outcomes=results, signature=signature,
        script_sha256=base.digest(Path(__file__)), service_sha256=base.digest(Path(frozen.__file__)),
        gpu=frozen.policy.torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    run(frozen.command(__doc__).parse_args())
