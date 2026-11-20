"""Nominate one reviewer using the query component absent from the incumbent's best source."""

import argparse
from collections import Counter
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.retrieval import cof_global_profile_pooling as base
from experiments.retrieval.cof_competing_profile_evidence import ASSOCIATION_PENALTY, SUPPORT_MARGIN
from experiments.retrieval.cof_contrastive_swap import change_stats, score_sources


TOP = 5
POOL = 10
DEPTH = 3
ARMS = ("dense_five", "sixth", "raw_best", "unshared_best", "residual_nomination", "residual_evidence")


class ResidualIndex(base.ProfileIndex):
    def __init__(self, reference):
        super().__init__(reference)
        self.owner_count = np.bincount(reference.owner_rows, minlength=len(reference.source_ids))

    def primary(self, query, encoder):
        import torch

        vector = encoder.encode([query], query=True)[0]
        scores = torch.mv(self.gpu.vectors, torch.as_tensor(vector, device="cuda")).cpu().numpy()
        reference = self.reference
        centroids = self.centroids @ vector
        best = np.maximum.reduceat(scores[reference.owner_rows], reference.owner_starts)
        ranks = np.argsort(-.5 * (centroids + best), kind="stable")[:POOL]
        reviews = [reference.owner_ids[i] for i in ranks]
        source_order = {
            owner: sorted(reference.owners[owner], key=lambda doc: (-float(scores[doc]), reference.source_ids[doc]))
            for owner in reviews
        }
        return vector, scores, reviews, source_order

    def nominate(self, primary, arm):
        vector, scores, reviews, source_order = primary
        incumbent = reviews[TOP - 1]
        first_five = [(owner, source_order[owner][:DEPTH]) for owner in reviews[:TOP]]
        if arm == "dense_five":
            return self.as_ids(first_five)
        depth = len(first_five[-1][1])
        viable = [owner for owner in reviews[TOP:] if len(source_order[owner]) >= depth]
        if not viable:
            return self.as_ids(first_five)
        if arm == "sixth":
            owner = viable[0]
        elif arm == "raw_best":
            owner = max(viable, key=lambda name: float(scores[source_order[name][0]]))
        elif arm == "unshared_best":
            owner = max(viable, key=lambda name: max(
                scores[doc] - ASSOCIATION_PENALTY * np.log2(self.owner_count[doc])
                for doc in source_order[name]
            ))
        else:
            anchor = base.unit(self.reference.vectors[source_order[incumbent][0]])
            remainder = base.unit(vector - max(0., float(vector @ anchor)) * anchor)
            weighted = {}
            for name in viable:
                columns = np.asarray(source_order[name])
                residual = self.reference.vectors[columns] @ remainder
                weighted[name] = .5 * (residual + scores[columns])
            owner = max(viable, key=lambda name: float(weighted[name].max()))
        docs = source_order[owner][:depth]
        if arm == "residual_evidence" and depth >= 2:
            order = source_order[owner]
            second = order[1 + int(np.argmax(weighted[owner][1:]))]
            docs = [order[0], second] + [doc for doc in order if doc not in {order[0], second}][:depth - 2]
        return self.as_ids(first_five + [(owner, docs)])

    def as_ids(self, hits):
        return [(owner, [self.reference.source_ids[doc] for doc in documents]) for owner, documents in hits]


def evaluate(index, primary, query, texts, verifier, arm, common_ms):
    start = perf_counter()
    hits = index.nominate(primary, arm)
    nominee = hits[TOP][0] if len(hits) > TOP else None
    values, clipped, reader_pairs = score_sources(verifier, query, hits, texts)
    swapped = nominee is not None and max(values[TOP][:2]) > max(values[TOP - 1][:2])
    selected = list(range(TOP - 1)) + [TOP if swapped else TOP - 1]
    ranking = tuple(hits[position][0] for position in selected)
    cited = {}
    for pos in selected:
        owner, sources = hits[pos]
        available = [source for source, margin in zip(sources, values[pos], strict=True) if margin > SUPPORT_MARGIN]
        if len(available) >= 2:
            cited[owner] = tuple(available[:2])
    return dict(ranking=ranking, cited=cited, nominee=nominee, swapped=swapped,
                reader_pairs=reader_pairs, clipped=clipped,
                challenger_sources=tuple(hits[TOP][1]) if nominee else None,
                complete_ms=common_ms+(perf_counter()-start)*1000)


def summarize(records, labels):
    output = {}
    for arm in ARMS:
        totals = Counter()
        deltas, cited_deltas = [], []
        lower, upper = [], []
        timings = []
        for row, truth in zip(records, labels, strict=True):
            reference, trial = row["dense_five"], row[arm]
            cutoff = reference["ranking"][-1]
            nominee = trial["nominee"]
            if len(trial["ranking"]) != TOP or reference["ranking"][:-1] != trial["ranking"][:-1]:
                raise AssertionError("Fixed recommendation slots changed")
            if arm != "dense_five" and trial["reader_pairs"] != row["sixth"]["reader_pairs"]:
                raise AssertionError("Unequal source verification budget")
            known, new, lost, bounds = change_stats(reference, trial, truth, with_citations=False)
            cited, cited_new, cited_lost, _ = change_stats(reference, trial, truth, with_citations=True)
            totals.update(
                known_positive=known, p5_gained=new, p5_lost=lost,
                cited_positive=cited, cited_gained=cited_new, cited_lost=cited_lost,
                swapped=int(trial["swapped"]), both_labeled=int(nominee in truth and cutoff in truth),
                possible_positive_swap=int(nominee is not None and truth.get(nominee) == 1 and truth.get(cutoff) == 0),
                known_challenger_positive=int(nominee is not None and truth.get(nominee) == 1),
                changed_nominee=int(nominee != row["sixth"]["nominee"]) if nominee is not None else 0,
                changed_sources_vs_residual=int(
                    arm == "residual_evidence"
                    and trial["challenger_sources"] != row["residual_nomination"]["challenger_sources"]
                ),
                reader_pairs=trial["reader_pairs"], clipped=trial["clipped"],
            )
            deltas.append((new-lost)/TOP)
            cited_deltas.append((cited_new-cited_lost)/TOP)
            lower.append(bounds[0])
            upper.append(bounds[1])
            timings.append(trial["complete_ms"])
        output[arm] = dict(**totals, p95_ms=float(np.quantile(timings, .95)),
                           known_p5_delta=base.paired_interval(deltas),
                           known_cited_p5_delta=base.paired_interval(cited_deltas),
                           missing_label_p5_delta_bounds=[float(np.mean(lower)),float(np.mean(upper))])
    return output


def run(dataset, embedding, reranker, output, offset):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output in a Slurm CUDA allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = base.verify_checkpoint(embedding)
    root, papers, _, ref, ids, _, truth = base.read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = ResidualIndex(ref)
    encoder = base.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reader = base.Reranker(reranker)
    selected = ids if dataset == "SIGIR" else base.select(dataset, ids, texts)
    warm = index.primary(texts[selected[0]], encoder)
    for arm in ARMS:
        evaluate(index, warm, texts[selected[0]], texts, reader, arm, 0)
    traces = []
    for i, qid in enumerate(selected):
        start = perf_counter()
        primary = index.primary(texts[qid], encoder)
        common = (perf_counter()-start)*1000
        shift = (offset + i) % len(ARMS)
        traces.append({arm: evaluate(index, primary, texts[qid], texts, reader, arm, common)
                       for arm in ARMS[shift:] + ARMS[:shift]})
        if (i+1) % 30 == 0:
            print(f"{dataset}: {i+1}/{len(selected)}", flush=True)
    base.write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), order_offset=offset,
        code_sha256=base.digest(Path(__file__)), embedding_revision=provenance["revision"],
        reranker_revision=reader.provenance["revision"], queries=len(selected),
        cohort_sha256=sha256("\n".join(selected).encode()).hexdigest(),
        hard_label_sha256=base.digest(root/f"{dataset}_queries_test_hard.json"),
        vector_protocol_sha256=base.digest(root/"qwen-embedding-torch-v1/protocol.json"),
        arms=summarize(traces,[truth[qid] for qid in selected]),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dataset", choices=("SIGIR", "KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--order-offset", type=int, default=0)
    args = cli.parse_args()
    run(args.dataset, args.embedding, args.reranker, args.output, args.order_offset)
