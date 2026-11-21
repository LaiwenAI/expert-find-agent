"""Time equivalent resident source and contributor lookup paths over loopback."""

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from hashlib import sha256
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import product
import json
import os
from pathlib import Path
from threading import Thread
from time import perf_counter

import numpy as np
import torch

from experiments.credit import plos_prospective_calibration as prospective

route, base, role = prospective.route, prospective.base, prospective.role
ARMS = {
    "legacy_author": (False, "author"),
    "legacy_author_role": (False, "author_role"),
    "vectorized_author": (True, "author"),
    "vectorized_author_role": (True, "author_role"),
}


class DenseScores(Mapping):
    def __init__(self, positions, values, absent):
        self.positions, self.values_array, self.absent = positions, values, absent

    def __getitem__(self, doc):
        if doc == self.absent:
            raise KeyError(doc)
        return float(self.values_array[self.positions[doc]])

    def __iter__(self):
        return (doc for doc in self.positions if doc != self.absent)

    def __len__(self):
        return len(self.positions) - (self.absent in self.positions)


class VectorRanks:
    def __init__(self, index):
        self.catalog, self.positions = index.catalog, index.position
        posting = defaultdict(list)
        for position, terms in enumerate(self.catalog.lexical):
            for word, score in terms.items():
                posting[word].append((position, score))
        self.lexical = {
            term: (np.asarray([i for i, _ in values], dtype=np.int32),
                   np.asarray([s for _, s in values], dtype=np.float64))
            for term, values in posting.items()
        }
        self.bytes = sum(indices.nbytes + weights.nbytes for indices, weights in self.lexical.values())

    def rank(self, vectors, terms, absent):
        catalog = self.catalog
        positions = np.arange(len(catalog.ids))
        kept = positions if absent is None else positions[positions != self.positions[absent]]
        output = []
        for scores, words in zip(vectors @ catalog.vectors.T, terms, strict=True):
            sparse = np.zeros(len(positions), dtype=np.float64)
            for term in words:
                if term in self.lexical:
                    hits, weights = self.lexical[term]
                    sparse[hits] += weights
            d = kept[np.argsort(-scores[kept], kind="stable")]
            s = kept[np.argsort(-sparse[kept], kind="stable")]
            # Reproduce Python's summation in near ties, where low-order
            # differences introduced by array addition could change the rank.
            high, low = sparse[s[:-1]], sparse[s[1:]]
            close = (low > 0) & (high - low <= high * (len(words) + 2) * np.finfo(np.float64).eps)
            if close.any():
                for i in np.union1d(s[:-1][close], s[1:][close]):
                    row = catalog.lexical[i]
                    sparse[i] = sum(row.get(term, 0.) for term in words)
                s = kept[np.argsort(-sparse[kept], kind="stable")]
            fused = np.zeros(len(positions), dtype=np.float64)
            reciprocal = 1 / (61 + positions[:len(kept)])
            fused[d] = reciprocal
            fused[s] += reciprocal
            ranking = kept[np.argsort(-fused[kept], kind="stable")]
            output.append(dict(fused=[catalog.ids[i] for i in ranking[:route.DEEP]],
                               dense=[catalog.ids[i] for i in d[:route.DEEP]],
                               best=float(scores[d[0]]), cosine=DenseScores(self.positions, scores, absent)))
        return output


def lexical_terms(clues):
    return [sorted(set(route.normalize(clue).split()) - route.KEYWORD_STOP) for clue in clues]


class Engine:
    def __init__(self, index, encoder, ridge, role_cut, models, thresholds):
        self.index, self.encoder, self.ridge, self.role_cut = index, encoder, ridge, role_cut
        self.models, self.thresholds = models, thresholds
        self.footprints = route.nested.previous.Fingerprints(index.catalog)
        self.vectorized = VectorRanks(index)

    def encode(self, clues, duties):
        prompts = base.search.topic_prompts(clues) + [
            f"Instruct: {base.ROLE_TASK}\nQuery:{text}" for text in duties
        ]
        topic, roles = np.split(self.encoder.encode(prompts), [2])
        values = role.normalize_scores(roles @ self.ridge.T)
        first, second = np.argsort(-values, axis=1, kind="stable")[:, :2].T
        margin = min(float(values[i, a] - values[i, b]) for i, (a, b) in enumerate(zip(first, second, strict=True)))
        return topic, tuple(base.ROLES[i] for i in first), margin

    def check_rank(self, clues, duties, absent):
        vectors, _, _ = self.encode(clues, duties)
        legacy = self.index.catalog.rank(vectors, clues, absent, depth=route.DEEP)
        fast = self.vectorized.rank(vectors, lexical_terms(clues), absent)
        for left, right in zip(legacy, fast, strict=True):
            if (left["fused"] != right["fused"] or left["dense"] != right["dense"] or left["best"] != right["best"]
                    or left["cosine"] != dict(right["cosine"])):
                raise ValueError("Vectorized retrieval differs from exact Python retrieval")

    def execute(self, request):
        clock = perf_counter()
        clues, duties, mode, absent = (request[key] for key in ("clues", "duties", "mode", "exclude"))
        fast, arm = ARMS[request["arm"]]
        topic_vectors, labels, confidence = self.encode(clues, duties)
        after_encoding = perf_counter()
        terms = lexical_terms(clues) if fast else None
        if confidence <= self.role_cut:
            return dict(proof=None, encode_ms=(after_encoding-clock)*1000, rank_ms=0.,
                        fallback_ms=None, posting_reads=0, source_scans=0)
        ranked = (self.vectorized.rank(topic_vectors, terms, absent) if fast else
                  self.index.catalog.rank(topic_vectors, clues, absent, depth=route.DEEP))
        after_ranking = perf_counter()
        articles = self.index.catalog.articles
        a, b = (side["fused"][:route.SHALLOW] for side in ranked)
        shallow = min(((i+j, (left, right)) for i, left in enumerate(a) for j, right in enumerate(b)
                       if base.valid_pair(self.index.catalog, left, right)), default=(0, None))[1]
        pair, fallback_ms, read, work = shallow, None, 0, 0
        if pair is None or not route.applicable(articles, pair, labels):
            started = perf_counter()
            terms = terms if terms is not None else lexical_terms(clues)
            candidates = set()
            for side in (0, 1):
                anchor = ranked[side]["fused"][0]
                neighbors, visited = self.index.target(anchor, labels[side], labels[1-side], arm)
                neighbors -= {anchor, absent}
                read += visited
                work += len(neighbors)
                for doc in route.rrf(self.index, ranked[1-side], terms[1-side], neighbors):
                    candidates.add((anchor, doc) if side == 0 else (doc, anchor))
            features = {}
            for pair in sorted(candidates):
                if route.applicable(articles, pair, labels):
                    x, y = [route.source_features(ranked, clues, self.footprints, doc, side)
                            for side, doc in enumerate(pair)]
                    features[pair] = np.r_[np.minimum(x, y), np.maximum(x, y)]
            pair, _ = max(((pair, value) for pair, value in route.scored(features, self.models[mode])
                           if value > self.thresholds[mode][arm]), key=lambda item: item[1], default=(None, None))
            fallback_ms = (perf_counter() - started) * 1000
        proof = None if pair is None else dict(sources=pair, roles=labels, owners=sorted(
            base.known_owners(articles, pair, labels)
        ))
        return dict(proof=proof, encode_ms=(after_encoding-clock)*1000, rank_ms=(after_ranking-after_encoding)*1000,
                    fallback_ms=fallback_ms, posting_reads=read, source_scans=work)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        engine = self.server.engines[request["publisher"]]
        result = engine.execute(request)
        payload = json.dumps(result, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        return


def calculate(client, uri, payload):
    body = json.dumps(payload, separators=(",", ":")).encode()
    started = perf_counter()
    client.request("POST", uri, body, {"Content-Type": "application/json"})
    response = client.getresponse()
    data = response.read()
    if response.status != 200:
        raise RuntimeError(f"HTTP timing failed with {response.status}")
    result = json.loads(data)
    result["client_ms"] = (perf_counter() - started) * 1000
    return result


def requests_for(publisher, engine, plan, reverse):
    for pair in sorted(plan, reverse=reverse):
        cases = plan[pair]
        by_kind = [
            next(row for row in cases if row[0] == "positive"),
            next(row for row in cases if row[0] != "positive"),
        ]
        for mode in ("excerpt", "keywords"):
            clues = [engine.index.catalog.articles[paper]["clue"] for paper in pair]
            if mode == "keywords":
                clues = [engine.index.catalog.keywords(clue) for clue in clues]
            for removed in (None, *pair):
                for variant, gold, duties in by_kind:
                    trial = (publisher, pair, mode, removed or "", variant)
                    request = dict(publisher=publisher, clues=clues, duties=duties, mode=mode, exclude=removed)
                    yield trial, request, gold


def percentile(series, p):
    return float(np.percentile(series, p)) if series else None


def aggregate(times):
    return {key: dict(n=len(values), p50_ms=percentile(values, 50), p95_ms=percentile(values, 95))
            for key, values in times.items()}


def profile(engines, requests, reverse, rounds):
    service = HTTPServer(("127.0.0.1", 0), Handler)
    service.engines = engines
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection("127.0.0.1", service.server_address[1], timeout=30)
    times, counts, signatures = defaultdict(lambda: defaultdict(list)), defaultdict(Counter), {}
    try:
        # Keep the model and index resident. Every timed request encodes its
        # text afresh even when another arm has seen the same request.
        for _, request, _ in requests[::max(1, len(requests)//12)]:
            for arm in ARMS:
                calculate(client, "/search", request | dict(arm=arm))
        order = tuple(reversed(ARMS)) if reverse else tuple(ARMS)
        for iteration in range(rounds):
            for i, (ident, request, gold) in enumerate(requests):
                publisher, papers, entrance, removed, _ = ident
                for step in range(len(order)):
                    arm = order[(i+iteration+step) % len(order)]
                    report = calculate(client, "/search", request | dict(arm=arm))
                    proof = report["proof"]
                    # Compare exhaustive per-case proofs, including claims on
                    # the actual selected papers and their declared owners.
                    if signatures.setdefault(ident, proof) != proof:
                        raise ValueError(f"Timed paths changed proof for {publisher}/{entrance}/{arm}")
                    key = publisher, arm
                    for stage in ("client_ms", "encode_ms", "rank_ms", "fallback_ms"):
                        if report[stage] is not None:
                            times[key][stage].append(report[stage])
                        if report["fallback_ms"] is not None and stage == "client_ms":
                            times[key]["client_during_fallback_ms"].append(report[stage])
                    ok = (proof is not None and tuple(proof["sources"]) == papers and tuple(proof["roles"]) == tuple(gold)
                          and bool(base.known_owners(engines[publisher].index.catalog.articles, papers, gold)))
                    counts[key].update(requests=1, fallbacks=report["fallback_ms"] is not None,
                                       correct=ok and not removed, wrong=proof is not None and (not ok or bool(removed)),
                                       posting_reads=report["posting_reads"], source_scans=report["source_scans"])
    finally:
        client.close()
        service.shutdown()
        service.server_close()
        thread.join()
    summaries = {name: {arm: dict(work=dict(counts[name, arm]), stages=aggregate(times[name, arm]))
                        for arm in ARMS}
                 for name in engines}
    serialized = sorted((*ident, proof) for ident, proof in signatures.items())
    return summaries, sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "elife_groups", "elife_held", "role_archive", "elife_policy",
                 "base_plos_manifest", "base_plos_projection", "new_plos_manifest", "plos_xml",
                 "glossary", "training", "embedding", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--rounds", type=int, default=2)
    args = p.parse_args()
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID") or args.rounds < 1:
        raise ValueError("Require fresh output, Slurm GPU, and positive round count")
    e_reference, r_reference = role.read(args.elife_policy), role.read(args.role_archive)
    embedding = base.verify_checkpoint(args.embedding)["revision"]
    if (base.digest(args.elife_held) != e_reference["panel_sha256"]
            or base.digest(args.role_archive) != e_reference["archive_sha256"]
            or base.digest(args.elife_groups) != r_reference["frozen_sha256"]
            or base.digest(Path(route.__file__)) != e_reference["script_sha256"]
            or embedding != r_reference["embedding"]):
        raise ValueError("Changed archived evaluation inputs")
    e_catalog = base.load_catalog((args.confirmation, args.extended))
    p_manifest = role.read(args.new_plos_manifest)
    p_catalog = prospective.restore(role.read(args.base_plos_manifest), p_manifest, args.plos_xml,
                                    args.base_plos_projection, args.confirmation, args.extended)
    elife = role.read(args.elife_groups)["panels"]
    held = role.read(args.elife_held)
    encoder = base.QwenTorchEmbedding(args.embedding)
    definitions, phrases, wording = base.queries(args.glossary, args.training)
    banks, _ = role.train(encoder, definitions, phrases)
    cut = r_reference["role_cutoffs"]["query_ridge_calibrated"]
    dev, old, e_plan = (base.phrase_cases(panel, wording) for panel in (elife["calibration"], elife["held"], held))
    _, decoded = role.embed(encoder, banks, r_reference["training"]["weights"], (dev, old, e_plan))
    e_index = route.PostingIndex(route.Catalog(e_catalog, encoder))
    p_index = route.PostingIndex(route.Catalog(p_catalog, encoder))
    e_footprint = route.nested.previous.Fingerprints(e_index.catalog)
    e_order = sorted(elife["calibration"]["pairs"], key=lambda row: base.stable("open-proof", tuple(row["papers"])))
    models, thresholds = {}, {}
    for mode in ("excerpt", "keywords"):
        def rows(groups, train=False):
            return route.data(e_index, encoder, groups, dev | old, mode,
                              decoded["ridge"], cut, e_footprint, args.reverse, training=train)
        models[mode], fit = route.fit(rows(e_order[:18], train=True))
        thresholds[mode], adverse = route.calibrate(rows(e_order[18:] + elife["held"]["pairs"]), models[mode], e_catalog)
        prospective.transfer.check_calibration(fit, thresholds[mode], adverse, e_reference["outcomes"][mode])
    engines = {name: Engine(index, encoder, banks["ridge"], cut, models, thresholds)
               for name, index in (("elife", e_index), ("plos", p_index))}
    plans = dict(elife=e_plan, plos=prospective.transfer.plos_requests(p_catalog, p_manifest["test"], wording))
    trials = [row for name, engine in engines.items() for row in requests_for(name, engine, plans[name], args.reverse)]
    # Verify ordered ranks and numerical program equivalence on one full
    # paper group's cases per publisher before timing all held requests.
    groups_to_check = {name: next(iter(sorted(plan))) for name, plan in plans.items()}
    for ident, trial, _ in trials:
        if ident[1] == groups_to_check[ident[0]]:
            engines[ident[0]].check_rank(trial["clues"], trial["duties"], trial["exclude"])
    results, signatures = profile(engines, trials, args.reverse, args.rounds)
    base.write_json(args.output, dict(
        results=results, outcomes_sha256=signatures, unique_trials=len(trials), rounds=args.rounds,
        lexical_payload_bytes={name: engine.vectorized.bytes for name, engine in engines.items()},
        catalog_articles={name: len(engine.index.catalog.ids) for name, engine in engines.items()},
        embedding_revision=embedding, script_sha256=base.digest(Path(__file__)),
        elife_panel_sha256=base.digest(args.elife_held), plos_panel_sha256=base.digest(args.new_plos_manifest),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
