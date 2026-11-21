"""Replay a locked role recheck policy on new sources through loopback HTTP."""

from collections import Counter, defaultdict
from hashlib import sha256
from http.client import HTTPConnection
from http.server import HTTPServer
import json
from pathlib import Path
from threading import Thread

import numpy as np

from experiments.credit import contrastive_rescue_identity as shared
from experiments.credit import plos_source_first_role_recheck as policy
from experiments.credit import resident_directed_source_timing as resident

route, base = shared.route, shared.base
REFERENCE = "7ee0997e56a8d58dc227ccd6a3b0e41ee0478856c0e94fc9d5df8d779c68d2c9"
ROLE_MARGIN = 26.125
ARMS = ("baseline", "frozen_source_score_4b")


class Engine(resident.Engine):
    def __init__(self, index, encoder, banks, cutoff, model, source_cutoff, reader):
        super().__init__(index, encoder, banks["ridge"], cutoff, {"keywords": model}, {"keywords": source_cutoff})
        self.reader = reader

    def encode(self, clues, duties):
        prompts = base.search.topic_prompts(clues) + [f"Instruct: {base.ROLE_TASK}\nQuery:{text}" for text in duties]
        topic, roles = np.split(self.encoder.encode(prompts), [2])
        values = shared.role.normalize_scores(roles @ self.ridge.T)
        first, second = np.argsort(-values, axis=1, kind="stable")[:, :2].T
        margins = [float(values[i, a] - values[i, b]) for i, (a, b) in enumerate(zip(first, second, strict=True))]
        labels = tuple(base.ROLES[i] for i in first)
        self.current = (topic, labels, margins)
        return topic, labels, min(margins)

    def execute(self, request):
        result = super().execute(request | {"arm": "vectorized_author_role"})
        result.update(role_calls=0, role_tokens=0)
        try:
            topic, labels, margins = self.current
            if request["policy"] == "baseline" or result["proof"] or min(margins) > self.role_cut:
                return result
            clues, duties, absent = (request[key] for key in ("clues", "duties", "exclude"))
            ranked = self.vectorized.rank(topic, resident.lexical_terms(clues), absent)
            pair, verified, _ = policy.source_proposal(
                ranked, clues, self.footprints, self.models["keywords"], self.thresholds["keywords"]["author_role"]
            )
            if verified:
                uncertain = [i for i, margin in enumerate(margins) if margin <= self.role_cut]
                roles, margin, tokens, ms = policy.measured_recheck(self.reader, duties, labels, uncertain)
                result.update(role_calls=1, role_tokens=tokens, recheck_ms=ms)
                if margin > ROLE_MARGIN and None not in roles:
                    owners = sorted(base.known_owners(self.index.catalog.articles, pair, roles))
                    if owners:
                        result["proof"] = dict(sources=pair, roles=roles, owners=owners)
            return result
        finally:
            del self.current


def requests(articles, catalog, plan, reverse):
    for intended, cases in sorted(plan.items(), reverse=reverse):
        clues = [catalog.keywords(articles[doc]["clue"]) for doc in intended]
        for absent in (None, *intended):
            for kind, gold, phrases in cases:
                payload = dict(clues=clues, duties=phrases, mode="keywords", exclude=absent, publisher="plos")
                yield intended, absent, kind, gold, payload


def checked_proof(articles, intended, gold, result):
    proof = result["proof"]
    if proof is None:
        return False, False
    pair, roles, owners = (proof[key] for key in ("sources", "roles", "owners"))
    if not owners or set(owners) != base.known_owners(articles, pair, roles):
        raise ValueError("Inconsistent returned attribution")
    exact = route.result_is_correct(articles, intended, tuple(pair), tuple(gold), tuple(roles))
    return exact, not exact


def exercise(engine, rows, reverse):
    service = HTTPServer(("127.0.0.1", 0), resident.Handler)
    service.engines = {"plos": engine}
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection("127.0.0.1", service.server_address[1], timeout=90)
    articles = engine.index.catalog.articles
    stats, latency, by_group, outcomes = defaultdict(Counter), defaultdict(list), defaultdict(Counter), []
    try:
        for _, _, _, _, request in rows[::max(1, len(rows) // 12)]:
            for arm in ARMS:
                resident.calculate(client, "/search", request | {"policy": arm})
        check = rows[0][-1]
        engine.check_rank(check["clues"], check["duties"], check["exclude"])
        for i, (intended, absent, kind, gold, request) in enumerate(rows):
            position = (i + reverse) % 2
            arms = ARMS[position:] + ARMS[:position]
            returned = {arm: resident.calculate(client, "/search", request | {"policy": arm}) for arm in arms}
            if returned["baseline"]["proof"] is not None and returned["baseline"]["proof"] != returned[ARMS[1]]["proof"]:
                raise ValueError("Frozen update changed a previously issued proof")
            context = "present" if absent is None else "withdrawn"
            for name, row in returned.items():
                exact, wrong = checked_proof(articles, intended, gold, row)
                stats[context, name].update(requests=1, correct=int(exact), wrong=int(wrong), calls=row["role_calls"],
                                            tokens=row["role_tokens"])
                latency[context, name].append(row["client_ms"])
                by_group[intended, context, name].update(correct=int(exact), wrong=int(wrong))
                outcomes.append((intended, absent, kind, gold, request["duties"], name, row["proof"]))
    finally:
        client.close()
        service.shutdown()
        thread.join()
        service.server_close()
    reports = {}
    for context in ("present", "withdrawn"):
        clusters = sorted({pair for pair, state, _ in by_group if state == context})
        difference = np.array([[by_group[pair, context, ARMS[1]][key] - by_group[pair, context, "baseline"][key]
                                for key in ("correct", "wrong")] for pair in clusters])
        rng = np.random.default_rng(12)
        draws = rng.integers(len(clusters), size=(4000, len(clusters)))
        per_group = difference[draws].mean(axis=1)
        reports[context] = dict(
            arms={arm: dict(stats[context, arm], **resident.aggregate({"client": latency[context, arm]})["client"])
                  for arm in ARMS},
            groups=len(clusters), groups_with_new_correct=int((difference[:, 0] > 0).sum()),
            groups_with_added_wrong=int((difference[:, 1] > 0).sum()),
            paired_mean_count_per_group_95=np.quantile(per_group, [.025, .975], axis=0).tolist(),
        )
    # Request order may differ, but signatures include every proof and abstention.
    signature = sha256(json.dumps(sorted(outcomes, key=lambda row: json.dumps(row, sort_keys=True)),
                                  sort_keys=True).encode()).hexdigest()
    return reports, signature


def run(args):
    if base.digest(args.lock) != REFERENCE:
        raise ValueError("Changed frozen role policy")
    locked = shared.role.read(args.lock)["runs"][0]
    if locked["outcomes"]["keywords"]["source_score_4b"]["cutoff"] != ROLE_MARGIN:
        raise ValueError("Frozen margin mismatch")
    definitions, phrases, wording = base.queries(args.glossary, args.training)
    reader = policy.PairedReader(args.reader, definitions)
    if reader.provenance["revision"] != locked["reader"]:
        raise ValueError("Changed reader")
    new = shared.role.read(args.fresh_manifest)
    if new["prior_digests"] != {"base": base.digest(args.base_plos_manifest),
                                 "holdout": base.digest(args.prospective_plos)}:
        raise ValueError("Changed original exposure lock")
    fresh_with_base = shared.prior.restore(shared.role.read(args.base_plos_manifest), new, args.fresh_xml,
                                           args.base_plos_projection, args.confirmation, args.extended)
    for entrance, articles, _, encoder, old, _, _, role_cut, _, model, source_cutoff in shared.prepared(args):
        if entrance != "keywords":
            continue
        if set(old["source_sha256"]) & set(new["source_sha256"]) or len(new["test"]["pairs"]) < 12:
            raise ValueError("Insufficient independent holdout")
        catalog = route.Catalog(articles | fresh_with_base, encoder)
        banks, _ = shared.role.train(encoder, definitions, phrases)
        plan = shared.prior.transfer.plos_requests(catalog.articles, new["test"], wording)
        engine = Engine(route.PostingIndex(catalog), encoder, banks, role_cut, model, source_cutoff, reader)
        results, signature = exercise(engine, list(requests(catalog.articles, catalog, plan, args.reverse)), args.reverse)
        base.write_json(args.output, dict(
            holdout=base.digest(args.fresh_manifest), frozen_reference=REFERENCE,
            reader=reader.provenance["revision"], statements=len(catalog.ids), outcomes=results, signature=signature,
            reverse=args.reverse, script_sha256=base.digest(Path(__file__)), gpu=policy.torch.cuda.get_device_name(),
        ))


if __name__ == "__main__":
    parser = shared.command(__doc__)
    for name in ("fresh-manifest", "fresh-xml", "lock", "reader"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
