"""Measure whether a fixed source is accepted only for its judged facet."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.calibrated_topic_evidence import margins, operating_point
from experiments.csfcube.distilled_facet_transfer import LABELS
from experiments.csfcube.frozen_confirmation import FACETS, load
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


TASK = "Decide whether the candidate paper matches the specific research aspect requested in the query."
GENERIC_TASK = "Decide whether the candidate paper is broadly related to the query paper."
FOLDS = 4
WEIGHTS = (0., .25, .5, 1.)


def key(*parts):
    return sha256(":".join(map(str, parts)).encode()).digest()


def pairs(questions):
    cases = defaultdict(dict)
    for row in questions:
        for doc, grade in row["grades"].items():
            cases[row["query_id"], doc][row["facet"]] = grade
    by_query = defaultdict(list)
    for (qid, doc), grades in cases.items():
        if len(grades) != 2:
            continue
        low, high = sorted(grades.values())
        if low == high == 0:
            kind = "both_zero"
        elif high < 2:
            continue
        elif low >= 2:
            kind = "both_strong"
        else:
            kind = "switch_zero" if low == 0 else "switch_weak"
        by_query[qid].append((doc, kind, grades))
    observed = []
    for qid, items in sorted(by_query.items()):
        negatives = sorted((x for x in items if x[1] == "both_zero"), key=lambda x: key(qid, x[0]))
        hold = {doc for doc, *_ in negatives[:12]}
        observed.extend((qid, doc, kind, tuple(sorted(grades.items())))
                        for doc, kind, grades in items if kind != "both_zero" or doc in hold)
    return observed


def render(row, facet=None, *, lead=False):
    body = row["abstract"]
    if facet is not None:
        labels = row["pred_labels"]
        if len(body) != len(labels) and not row.get("pred_labels_truncated"):
            raise ValueError("Unexpected sentence tag alignment")
        matching = [sentence for sentence, label in zip(body, labels) if label in LABELS[facet]]
        selected = matching[:3] or body[:1]
        body = body[:len(selected)] if lead else selected
    return f"{row['title']}\n{' '.join(body)}"


def collect(root, cases, encoder, reader):
    with (root / "abstracts-csfcube-preds.jsonl").open() as stream:
        papers = {int(row["paper_id"]): row for row in map(json.loads, stream)}
    rendered = {pid: render(papers[pid]) for case in cases for pid in case[:2]}
    focal, leads = {}, {}
    rows = {name: [] for name in ("explicit", "query_excerpt", "dual_excerpt", "dual_lead")}
    generic, qfeatures, dfeatures = [], [], []
    for qid, doc, _, grades in cases:
        generic.append((rendered[qid], rendered[doc]))
        for facet, _ in grades:
            for pid in (qid, doc):
                if (pid, facet) not in focal:
                    focal[pid, facet] = render(papers[pid], facet)
                    leads[pid, facet] = render(papers[pid], facet, lead=True)
            header = f"Find a paper with a similar {FACETS[facet]}.\n"
            qf = header + focal[qid, facet]
            df = focal[doc, facet]
            rows["explicit"].append((header + rendered[qid], rendered[doc]))
            rows["query_excerpt"].append((qf, rendered[doc]))
            rows["dual_excerpt"].append((qf, df))
            rows["dual_lead"].append((header + leads[qid, facet], leads[doc, facet]))
            qfeatures.append(qf)
            dfeatures.append(df)
    pair_scores = {}
    for name, items in rows.items():
        values, clipped = margins(reader, items, TASK)
        pair_scores[name] = (values.reshape((-1, 2)), clipped)
    values, clipped = margins(reader, generic, GENERIC_TASK)
    pair_scores["generic"] = (np.repeat(values[:, None], 2, axis=1), clipped)

    def encode_unique(texts, is_query):
        index = dict.fromkeys(texts)
        vectors = encoder.encode(list(index), query=is_query)
        positions = {text: i for i, text in enumerate(index)}
        return vectors[[positions[text] for text in texts]]

    similarity = np.einsum("ij,ij->i", encode_unique(qfeatures, True), encode_unique(dfeatures, False))
    pair_scores["facet_cosine"] = (similarity.reshape((-1, 2)), 0)
    return pair_scores


def grade_counts(scores, grades, kind):
    passes = scores.astype(bool)
    matches = Counter()
    for row, tags, mode in zip(passes, grades, kind, strict=True):
        matches["grade0_pass"] += sum(row & (tags == 0))
        matches["grade0_total"] += sum(tags == 0)
        matches["grade1_pass"] += sum(row & (tags == 1))
        matches["grade1_total"] += sum(tags == 1)
        matches["strong_pass"] += sum(row & (tags >= 2))
        matches["strong_total"] += sum(tags >= 2)
        if mode == "switch_zero":
            matches["switch_zero_correct"] += bool(np.array_equal(row, tags >= 2))
        if mode == "both_strong":
            matches["both_strong_correct"] += bool(row.all())
        if mode == "both_zero":
            matches["both_zero_correct"] += not row.any()
    return {k: int(v) for k, v in matches.items()}


def crossfit(cases, pair_scores):
    query = np.array([row[0] for row in cases])
    candidate = np.array([row[1] for row in cases])
    strata = [row[2] for row in cases]
    grades = np.array([[grade for _, grade in row[3]] for row in cases])
    fold_for_query = {q: i % FOLDS for i, q in enumerate(sorted(set(query), key=key))}
    folds = np.array([fold_for_query[q] for q in query])
    report, paired = {}, {}
    all_routes = {**pair_scores, "contrastive": pair_scores["explicit"]}
    for name, (raw, clipped) in all_routes.items():
        weights = WEIGHTS if name == "contrastive" else (0.,)
        verdicts = np.zeros(grades.shape, dtype=bool)
        contrasts = np.zeros_like(raw, dtype=np.float64)
        configurations = []
        for fold in range(FOLDS):
            test = folds == fold
            # Neither the query nor the candidate articles can cross the fold boundary.
            held_articles = np.union1d(query[test], candidate[test])
            train = ~test & ~np.isin(query, held_articles) & ~np.isin(candidate, held_articles)
            if not len(np.unique(query[train])) or not (grades[train] == 0).any():
                raise ValueError("Independent calibration fold empty")
            trials = []
            for weight in weights:
                contrast = raw - weight * raw[:, ::-1]
                cutoff = operating_point(contrast[train].ravel(), grades[train].ravel())
                accepted = contrast[train] > cutoff
                target = grades[train] >= 2
                switched = target.sum(axis=1) == 1
                primary = switched & (grades[train] == 0).any(axis=1)
                correct = np.sum(primary & (accepted == target).all(axis=1))
                strong = np.sum(accepted & target)
                trials.append((correct, strong, -weight, cutoff))
            _, _, negative_weight, cutoff = max(trials)
            weight = -negative_weight
            contrasts[test] = raw[test] - weight * raw[test, ::-1]
            verdicts[test] = contrasts[test] > cutoff
            configurations.append(dict(fold=fold, fit_pairs=int(train.sum()), test_pairs=int(test.sum()),
                                       cutoff=cutoff, contrast_weight=weight))
        strong = grades >= 2
        positions = np.flatnonzero(np.array(strata) == "switch_zero")
        diffs = [float(contrasts[i, strong[i]][0] > contrasts[i, ~strong[i]][0]) for i in positions]
        ties = [float(contrasts[i, strong[i]][0] == contrasts[i, ~strong[i]][0]) for i in positions]
        paired[name] = [int(np.array_equal(verdicts[i], strong[i])) for i in positions]
        report[name] = dict(
            fits=configurations, clipped=clipped,
            **grade_counts(verdicts, grades, strata),
            paired_strict_win=clustered_paired_interval(diffs, query[positions]),
            paired_ties=int(sum(ties)),
            paired_switch_correct=clustered_paired_interval(
                paired[name], query[positions],
            ),
        )
    for name, decisions in paired.items():
        report[name]["paired_switch_delta_vs_explicit"] = clustered_paired_interval(
            np.asarray(decisions) - paired["explicit"], query[positions],
        )
        report[name]["paired_switch_delta_vs_dual_lead"] = clustered_paired_interval(
            np.asarray(decisions) - paired["dual_lead"], query[positions],
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "provenance", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    _, questions, sources = load(args.data)
    receipt = json.loads(args.provenance.read_bytes())
    embedding = verify_checkpoint(args.embedding)
    reader = Reranker(args.reranker)
    if (sources != receipt["source_hashes"] or embedding["revision"] != receipt["embedding"]["revision"]
        or reader.provenance["revision"] != receipt["reranker"]["revision"]):
        raise ValueError("Unexpected CSFCube source or model revision")
    cases = pairs(questions)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    report = crossfit(cases, collect(args.data, cases, encoder, reader))
    write_json(args.output, dict(
        cases=len(cases), query_articles=len({qid for qid, *_ in cases}),
        strata=dict(Counter(row[2] for row in cases)), methods=report, source_hashes=sources,
        embedding_revision=embedding["revision"], reranker_revision=reader.provenance["revision"],
        gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
