"""Time certified batching with fixed-shape live CoF source verification."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.deferred_verification import _extreme, defer_and_rank, resolve_shared_frontier
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.qwen_reranker import pack_tokens
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_live_prefix_stop import TASK, TOP, verify_inputs


WIDTH = 4
LENGTH_BUCKET = 64
CAP = 1024
POOL = 10
SUPPORT = (1, 2)
METHODS = ("exhaustive", "one_per_expert", "fill_idle_slots")
PREFIX_CONTROL = ("exhaustive", "one_per_expert", "prefix_two", "prefix_three")
DYNAMIC_METHODS = (*METHODS, "dynamic_one", "dynamic_fill", "dynamic_length_fit")
GUARD_METHODS = (*METHODS, "dynamic_one", "guarded_dynamic")
DEFERRED_METHODS = ("exhaustive", "guarded_dynamic", "deferred_dynamic")
STRONG_FIXED_METHODS = ("exhaustive", "guarded_bulk", "deferred_dynamic", "deferred_bulk")
BUCKET_METHODS = ("exhaustive", "guarded_cached", "guarded_bucketed", "deferred_dynamic", "deferred_cached")
BUCKET_BATCH = 8
GUARD_MARGIN = .25
SCHEDULE_FOR = {"dynamic_one": "one_per_expert", "dynamic_fill": "fill_idle_slots",
                "dynamic_length_fit": "fill_same_length", "guarded_dynamic": "one_per_expert",
                "guarded_cached": "one_per_expert"}
PER_EXPERT = {"one_per_expert": 1, "prefix_two": 2, "prefix_three": 3}


@torch.inference_mode()
def forward_tokens(reader, rows, width=None):
    padding = True if width is None else "max_length"
    tensors = reader.tokenizer.pad(rows, padding=padding, max_length=width, return_tensors="pt")
    inputs = tensors.to("cuda")
    values = reader.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, reader.labels].float()
    return (values[:, 0] - values[:, 1]).cpu().tolist(), tensors["input_ids"].shape[1]


class WaveReader:
    def __init__(self, reader, query, documents, sources, *, dynamic=False, guard=None, cache=False,
                 broker=None):
        self.reader = reader
        self.dynamic = dynamic
        self.guard = guard
        self.broker = broker
        self.score_calls = 0
        self.scored_documents = 0
        self.recheck_calls = 0
        self.rechecked_pairs = 0
        self.padded_tokens = 0
        self.score_cache = {} if cache else None
        self.recheck_cache = {} if cache else None
        self.inputs = {}
        self.truncated = 0
        self.maximum = 0
        for source in sorted(sources):
            message = f"<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {documents[source]}"
            body = reader.tokenizer.encode(message, add_special_tokens=False)
            tokens, raw = pack_tokens(reader.prefix, body, reader.suffix, CAP)
            self.inputs[source] = dict(input_ids=tokens)
            self.truncated += raw > len(tokens)
            self.maximum = max(self.maximum, len(tokens))
        self.maximum = ((self.maximum + LENGTH_BUCKET - 1) // LENGTH_BUCKET) * LENGTH_BUCKET

    def length(self, doc):
        return len(self.inputs[doc]["input_ids"])

    def score(self, jobs):
        return self._score_with_cache(jobs, self.score_cache, self._score)

    def _score(self, jobs):
        self.score_calls += 1
        self.scored_documents += len(jobs)
        values = self._forward([self.inputs[doc] for _, doc in jobs], self.dynamic)
        if self.guard is not None:
            indices = [i for i, value in enumerate(values) if abs(value) <= self.guard]
            if indices:
                corrected = self.recheck([jobs[i] for i in indices])
                for i, fixed in zip(indices, corrected, strict=True):
                    values[i] = fixed
        return values

    def recheck(self, jobs):
        return self._score_with_cache(jobs, self.recheck_cache, self._recheck)

    def _recheck(self, jobs):
        self.recheck_calls += 1
        self.rechecked_pairs += len(jobs)
        return self._forward([self.inputs[doc] for _, doc in jobs], dynamic=False)

    @staticmethod
    def _score_with_cache(jobs, cache, score):
        if cache is None:
            return score(jobs)
        missing = {}
        for expert, doc in jobs:
            if doc not in cache:
                missing.setdefault(doc, expert)
        if missing:
            fresh = [(expert, doc) for doc, expert in missing.items()]
            cache.update(zip(missing, score(fresh), strict=True))
        return [cache[doc] for _, doc in jobs]

    def _forward(self, batch, dynamic):
        rows = batch if dynamic else batch + [batch[0]] * (WIDTH - len(batch))
        width = None if dynamic else self.maximum
        if self.broker is None:
            values, charged_width = forward_tokens(self.reader, rows, width)
        elif dynamic:
            values, charged_width = self.broker.score(rows)
        else:
            values, charged_width = self.broker.fixed(lambda: forward_tokens(self.reader, rows, width))
        self.padded_tokens += len(rows) * charged_width
        return values[:len(batch)]


def bulk_verify(ordered, reader, support, deferred, batch_size=None):
    """Score every distinct document and selectively stabilize tentative verdicts."""
    pairs = [(expert, doc) for expert, docs in ordered for doc in docs]
    unique = {doc: expert for expert, doc in reversed(pairs)}
    jobs = [(unique[doc], doc) for doc in dict.fromkeys(doc for _, doc in pairs)]
    if batch_size is None:
        batch_size = len(jobs)
    else:
        jobs.sort(key=lambda pair: reader.length(pair[1]))
    scores = {}
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start+batch_size]
        scores.update((doc, value) for (_, doc), value in zip(batch, reader.score(batch), strict=True))
    trusted = {doc: value for doc, value in scores.items() if abs(value) > GUARD_MARGIN}

    def mapping(values):
        return {pair: values[pair[1]] for pair in pairs if pair[1] in values}

    def correct(documents):
        for start in range(0, len(documents), WIDTH):
            batch = [(unique[doc], doc) for doc in documents[start:start+WIDTH]]
            for (_, doc), score in zip(batch, reader.recheck(batch), strict=True):
                trusted[doc] = scores[doc] = score

    if not deferred:
        correct([doc for doc in scores if doc not in trusted])
    while True:
        fixed = mapping(trusted)
        lower, _ = _extreme(ordered, fixed, support, TOP, False)
        upper, _ = _extreme(ordered, fixed, support, TOP, True)
        if lower == upper:
            return dict(selected=lower, checks=len(jobs), calls=1, unique_reads=len(jobs),
                        margins=mapping(scores))
        pending = list(dict.fromkeys(doc for _, docs in upper for doc in docs if doc not in trusted))
        if not pending:
            raise AssertionError("Unequal bulk bounds with no tentative citation")
        correct(pending[:WIDTH])


def schedule(ordered, reader, count, method):
    pending = dict(ordered)
    seen = {e: 0 for e in pending}
    accepted = {e: [] for e in pending}
    rank = {e: i for i, e in enumerate(pending)}
    margins = {}
    reads = set()
    calls = 0
    while True:
        confirmed = [e for e in pending if len(accepted[e]) == count]
        if method == "exhaustive":
            jobs = [(e, doc) for e in pending for doc in pending[e][seen[e]:]][:WIDTH]
        else:
            boundary = rank[confirmed[TOP - 1]] if len(confirmed) >= TOP else len(rank)
            eligible = [e for e in pending if rank[e] < boundary and len(accepted[e]) < count
                        and len(accepted[e]) + len(pending[e]) - seen[e] >= count]
            if method in ("fill_idle_slots", "fill_same_length"):
                jobs = [(e, pending[e][seen[e]]) for e in eligible[:WIDTH]]
                if 0 < len(jobs) < WIDTH:
                    limit = max(reader.length(doc) for _, doc in jobs) if method == "fill_same_length" else None
                    suffix = []
                    for expert in eligible:
                        for doc in pending[expert][seen[expert] + 1:]:
                            if limit is not None and reader.length(doc) > limit:
                                break
                            suffix.append((expert, doc))
                    jobs.extend(suffix[:WIDTH - len(jobs)])
            else:
                jobs = [(e, doc) for e in eligible
                        for doc in pending[e][seen[e]:seen[e] + PER_EXPERT[method]]][:WIDTH]
        if not jobs:
            order = confirmed + [e for e in pending if len(accepted[e]) < count]
            selected = [[e, accepted[e] if e in confirmed else []] for e in order[:TOP]]
            return dict(selected=selected, checks=len(margins), calls=calls,
                        unique_reads=len(reads), margins=margins)
        calls += 1
        for (expert, source), score in zip(jobs, reader.score(jobs), strict=True):
            margins[expert, source] = score
            seen[expert] += 1
            reads.add(source)
            if len(accepted[expert]) < count and score >= 0.:
                accepted[expert].append(source)


def summarize(rows, methods):
    report = {}
    baseline = next((name for name in ("one_per_expert", "guarded_cached", "guarded_bulk", "guarded_dynamic")
                     if name in methods), "exhaustive")
    for count in SUPPORT:
        values = [row["support"][count] for row in rows]
        reference = [row["exhaustive"] for row in values]
        report[count] = {}
        for method in methods:
            arms = [row[method] for row in values]
            report[count][method] = dict(
                exact_experts_and_sources=sum(a["selected"] == b["selected"]
                                              for a, b in zip(arms, reference, strict=True)),
                score_differences=sum(a["score_differences"] for a in arms),
                decision_differences=sum(a["decision_differences"] for a in arms),
                p95_ms=float(np.quantile([a["ms"] for a in arms], .95)),
                p50_ms=float(np.median([a["ms"] for a in arms])),
                average={key: mean(arm[key] for arm in arms)
                         for key in ("calls", "checks", "unique_reads", "padded_tokens",
                                     "recheck_calls", "rechecked_pairs", "hard_p5", "confirmed_hard_p5",
                                     "score_calls", "scored_documents")
                         if key in arms[0]},
                **{f"latency_saved_vs_{baseline}": paired_interval([
                    row[baseline]["ms"] - row[method]["ms"] for row in values
                ])},
            )
    return report


def prepare_runner(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    provenance = verify_checkpoint(args.embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.dataset == "SIGIR":
        root, papers, reviewers, labels = load("SIGIR")
        ids = verify_inputs(args.manifest)
        if len(ids) != 73 or any(set(ids) != grades.keys() for grades in labels.values()):
            raise ValueError("Unexpected SIGIR query set")
        ownership = defaultdict(list)
        for owner, profile in reviewers.items():
            for doc in set(profile["papers"]):
                ownership[doc].append(owner)
        source_ids = sorted(ownership)
        cache = root / "qwen-embedding-torch-v1"
        vectors = np.load(cache / "vectors.npy", allow_pickle=False)
        if (vectors.shape != (len(source_ids) + len(ids), 1024)
                or provenance != json.loads((cache / "protocol.json").read_bytes())["checkpoint"]
                or set(ids).intersection(ownership)):
            raise ValueError("Changed source-vector snapshot")
        source_index = ExactOwnedVectorIndex(vectors[:len(source_ids)], source_ids,
                                             [ownership[doc] for doc in source_ids])
        hard_labels = labels["hard"]
        inputs = digest(args.manifest)
    else:
        from experiments.retrieval.cof_compact_profile_shortlist import read_data, title_tokens

        root, papers, _, source_index, ids, _, hard_labels = read_data(args.dataset, provenance)
        ownership = {doc: set() for doc in source_index.source_ids}
        for owner, positions in source_index.owners.items():
            for pos in positions:
                ownership[source_index.source_ids[pos]].add(owner)
        inputs = {
            "protocol": digest(root / "qwen-embedding-torch-v1" / "protocol.json"),
            "vectors": digest(root / "qwen-embedding-torch-v1" / "vectors.npy"),
            "hard_labels": digest(root / f"{args.dataset}_queries_test_hard.json"),
        }
        if args.exclude_nips_titles:
            if args.dataset != "SciRepEval":
                raise ValueError("NIPS title exclusion is only defined for the SciRepEval panel")
            _, nips_papers, _, _, nips_ids, _, _ = read_data("NIPS", provenance)
            titles = {tuple(title_tokens(nips_papers[qid]["title"])) for qid in nips_ids}
            ids = [qid for qid in ids if tuple(title_tokens(papers[qid]["title"])) not in titles]
            inputs["nips_title_panel"] = digest(root.parent / "NIPS" / "qwen-embedding-torch-v1" / "protocol.json")
    index = GpuOwnedVectorIndex(source_index)
    encoder = QwenTorchEmbedding(args.embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reranker = Reranker(args.reranker)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}

    def retrieve(query):
        return index.search_vector(encoder.encode([query], query=True)[0], k=POOL)["hits"]

    def measure(query, count, method, broker=None):
        start = perf_counter()
        hits = retrieve(query) if broker is None else broker.serial(lambda: retrieve(query))
        ordered = [(row["id"], [w["id"] for w in row["evidence"]]) for row in hits]
        if len(ordered) != POOL or any(e not in ownership[d] for e, docs in ordered for d in docs):
            raise ValueError("Unexpected source ownership")
        bulk = method in ("guarded_bulk", "deferred_bulk", "guarded_bucketed")
        deferred = method in ("deferred_dynamic", "deferred_cached")
        shared_frontier = method in ("shared_frontier", "cost_aware_frontier")
        def prepare_prompts():
            return WaveReader(
                reranker, query, texts, {d for _, docs in ordered for d in docs},
                dynamic=method in SCHEDULE_FOR or bulk or deferred or shared_frontier,
                guard=GUARD_MARGIN if method in ("guarded_dynamic", "guarded_cached") else None,
                cache=method in ("guarded_cached", "deferred_cached") or shared_frontier,
                broker=broker,
            )

        reader = prepare_prompts() if broker is None else broker.serial(prepare_prompts)
        if bulk:
            batch_size = BUCKET_BATCH if method == "guarded_bucketed" else None
            result = bulk_verify(ordered, reader, count, method == "deferred_bulk", batch_size)
        elif shared_frontier:
            result = resolve_shared_frontier(
                ordered, reader.score, reader.recheck, reader.length, support=count,
                top=TOP, width=WIDTH, band=GUARD_MARGIN, cost_aware=method == "cost_aware_frontier",
            )
        elif deferred:
            result = defer_and_rank(ordered, reader.score, reader.recheck,
                                    support=count, top=TOP, width=WIDTH, band=GUARD_MARGIN)
        else:
            result = schedule(ordered, reader, count, SCHEDULE_FOR.get(method, method))
        result["padded_tokens"] = reader.padded_tokens
        result["calls"] = reader.score_calls + reader.recheck_calls
        result["score_calls"] = reader.score_calls
        result["scored_documents"] = reader.scored_documents
        result["recheck_calls"] = reader.recheck_calls
        result["rechecked_pairs"] = reader.rechecked_pairs
        result["ms"] = (perf_counter() - start) * 1000
        result["truncated"] = reader.truncated
        return ordered, result

    measure.forward = lambda rows: forward_tokens(reranker, rows)
    return measure, ids, texts, hard_labels, inputs, provenance, reranker.provenance


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    measure, ids, texts, hard_labels, inputs, provenance, reranker_provenance = prepare_runner(args)
    methods = METHODS
    if args.prefix_control:
        methods = PREFIX_CONTROL
    elif args.compare_dynamic:
        methods = DYNAMIC_METHODS
    elif args.guard_shape:
        methods = GUARD_METHODS
    elif args.defer_rechecks:
        methods = DEFERRED_METHODS
    elif args.strong_fixed:
        methods = STRONG_FIXED_METHODS
    elif args.compare_buckets:
        methods = BUCKET_METHODS
    for count in SUPPORT:
        for method in methods:
            measure(texts[ids[0]], count, method)
    rows = []
    for i, qid in enumerate(ids):
        by_count = {}
        offset = i % len(methods)
        for count in SUPPORT[::1 if i % 2 == 0 else -1]:
            ordered = None
            arms = {}
            for method in methods[offset:] + methods[:offset]:
                pool, arm = measure(texts[qid], count, method)
                if ordered is not None and pool != ordered:
                    raise ValueError("Candidate inventory drift")
                ordered = pool
                arms[method] = arm
            expected = arms["exhaustive"]["margins"].copy()
            for arm in arms.values():
                actual = arm.pop("margins")
                arm["score_differences"] = sum(value != expected[key] for key, value in actual.items())
                arm["decision_differences"] = sum((value >= 0.) != (expected[key] >= 0.)
                                                  for key, value in actual.items())
                supported = hard_labels[qid]
                arm["hard_p5"] = sum(supported.get(e, 0) > 0 for e, _ in arm["selected"]) / TOP
                arm["confirmed_hard_p5"] = sum(supported.get(e, 0) > 0 and bool(docs)
                                                 for e, docs in arm["selected"]) / TOP
            by_count[count] = arms
        rows.append(dict(id=qid, support=by_count))
        if (i + 1) % 20 == 0:
            print(f"Completed {i+1}/{len(ids)}", flush=True)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(rows),
        dataset=args.dataset,
        guard_margin=GUARD_MARGIN if args.guard_shape or args.defer_rechecks or args.strong_fixed or args.compare_buckets else None,
        bucket_batch=BUCKET_BATCH if args.compare_buckets else None,
        script_sha256=digest(Path(__file__)), input_sha256=inputs,
        deferred_code_sha256=digest(Path(defer_and_rank.__code__.co_filename)),
        embedding=provenance, reranker=reranker_provenance, report=summarize(rows, methods), rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prefix-control", action="store_true")
    modes.add_argument("--compare-dynamic", action="store_true")
    modes.add_argument("--guard-shape", action="store_true")
    modes.add_argument("--defer-rechecks", action="store_true")
    modes.add_argument("--strong-fixed", action="store_true")
    modes.add_argument("--compare-buckets", action="store_true")
    parser.add_argument("--dataset", choices=("SIGIR", "SciRepEval", "NIPS"), default="SIGIR")
    parser.add_argument("--exclude-nips-titles", action="store_true")
    run(parser.parse_args())
