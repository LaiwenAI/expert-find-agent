"""Compare fixed-budget, annotation-free sentence selectors on SciFact-Open."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import tarfile
from time import perf_counter

from expert_search.retrieval import tokens
from experiments.scifact.compact_open_holdout import CUTS, MAX_CHARACTERS, PINS, components, digest
from experiments.scifact.compact_open_rationale import PROJECTION as EARLIER_PROJECTION


PROJECTION = "ea308f920097d46ab0c88edb8f577b01467460c163fa147c04f156693e9bdceb"
ARMS = ("lexical_windows", "bm25_sentences", "weighted_coverage")


def prepare(archive, earlier_projection, output):
    if digest(archive) != PINS["archive"] or digest(earlier_projection) != EARLIER_PROJECTION:
        raise ValueError("Changed SciFact-Open archive")
    previously_audited = {(row["case"], row["doc"]) for row in
                          map(json.loads, earlier_projection.read_text().splitlines())}
    pairs = []
    corpus = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name == "data/claims.jsonl":
                for line in stream.extractfile(member):
                    case = json.loads(line)
                    for doc, annotation in case["evidence"].items():
                        pairs.append(dict(case=case["id"], doc=doc, criterion=case["claim"],
                                          grade=annotation["label"].lower(), gold=annotation["sentences"],
                                          provenance=annotation["provenance"],
                                          audited=(case["id"], doc) in previously_audited))
            if member.name == "data/corpus.jsonl":
                needed = {row["doc"] for row in pairs}
                for line in stream.extractfile(member):
                    row = json.loads(line)
                    if str(row["doc_id"]) in needed:
                        corpus[str(row["doc_id"])] = row["abstract"]
                break
    rows = []
    for row in pairs:
        sentences = corpus[row["doc"]]
        gold = sorted(set(row["gold"]))
        if not gold:
            continue
        if row["grade"] not in {"support", "contradict"} or not 0 <= gold[0] <= gold[-1] < len(sentences):
            raise ValueError("Unexpected annotated pair")
        if len(" ".join(sentences)) <= MAX_CHARACTERS:
            rows.append(row | dict(sentences=sentences, gold=gold))
    with output.open("x", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (item["case"], item["doc"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(len(rows), sum(row["audited"] for row in rows), digest(output), flush=True)


def candidate_indices(claim, sentences, k, arm):
    if arm not in ARMS[1:] or not 0 < k <= len(sentences):
        raise ValueError("Invalid selector or sentence budget")
    tokenized = [tokens(sentence) for sentence in sentences]
    query = tokens(claim)
    wanted = sorted(set(query))
    counts = [Counter(words) for words in tokenized]
    df = {term: sum(term in sentence for sentence in counts) for term in wanted}
    weights = {term: math.log1p((len(sentences) - freq + .5) / (freq + .5)) for term, freq in df.items()}
    average = max(1, sum(map(len, tokenized)) / len(sentences))
    relevance = [
        math.fsum(weights[term] * words[term] * 2.2 /
                  (words[term] + 1.2 * (.25 + .75 * len(terms) / average))
                  for term in wanted if term in words)
        for terms, words in zip(tokenized, counts, strict=True)
    ]
    order = sorted(range(len(sentences)), key=lambda i: (-relevance[i], i))
    if arm == "bm25_sentences":
        return tuple(sorted(order[:k]))

    # Preserve a high-relevance anchor, then cover distinct rare terms and exact adjacent query terms.
    phrases = sorted(set(zip(query, query[1:])))
    present = [set(zip(words, words[1:])) for words in tokenized]
    features = [
        {term: weight for term, weight in weights.items() if term in words} |
        {pair: math.sqrt(weights[pair[0]] * weights[pair[1]]) for pair in phrases if pair in bigrams}
        for words, bigrams in zip(counts, present, strict=True)
    ]
    selected = [order[0]]
    covered = set(features[selected[0]])
    while len(selected) < k:
        best = max((i for i in order if i not in selected), key=lambda i: (
            math.fsum(weight for key, weight in features[i].items() if key not in covered), relevance[i], -i,
        ))
        selected.append(best)
        covered.update(features[best])
    return tuple(sorted(selected))


def summarize(audits, baseline, arm):
    import numpy as np

    from expert_search.statistics import clustered_paired_interval

    full = np.array([row[baseline]["complete"] for row in audits], dtype=int)
    test = np.array([row[arm]["complete"] for row in audits], dtype=int)
    labels = Counter(row["grade"] for row in audits)
    hits = Counter(row["grade"] for row in audits if row[arm]["support"])
    groups = components(audits)
    return dict(
        pairs=len(audits), labels=dict(labels), complete_rationale=int(test.sum()),
        complete_gained=int((test > full).sum()), complete_lost=int((test < full).sum()),
        paired_delta=clustered_paired_interval(test - full, groups),
        accepted_as_support={label: hits[label] for label in labels},
        evidence_words_mean=float(np.mean([row[arm]["words"] for row in audits])),
        evidence_tokens_mean=float(np.mean([row[arm]["tokens"] for row in audits])),
        p50_p95_ms=np.quantile([row[arm]["ms"] for row in audits], [.5, .95]).tolist(),
    )


def run(projection, checkpoint, output):
    import torch

    from experiments.followir.core17_reranking import write_json
    from experiments.qasper.paired_reader import load_reader
    from experiments.scifact.criterion_reader import measure
    from experiments.scifact.nli_evidence_windows import choose_windows
    from experiments.scifact.selective_evidence_verifier import support_margin

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a new output path on a Slurm GPU")
    if digest(projection) != PROJECTION:
        raise ValueError("Changed benchmark projection")
    rows = list(map(json.loads, projection.read_text().splitlines()))
    tokenizer, reader, model = load_reader(checkpoint)
    first = rows[0]
    measure(tokenizer, reader, [dict(criterion=first["criterion"], source=" ".join(first["sentences"]))])
    arms = ("full", *ARMS)
    audits = []
    for offset, row in enumerate(rows):
        claim, sentences = row["criterion"], row["sentences"]
        # Determine the shared sentence budget for the methods compared against lexical windows.
        _, base = choose_windows(claim, sentences)
        report = {name: row[name] for name in ("case", "doc", "grade", "audited", "provenance")}
        for arm in arms[offset % len(arms):] + arms[:offset % len(arms)]:
            started = perf_counter()
            if arm == "full":
                chosen = tuple(range(len(sentences)))
            elif arm == "lexical_windows":
                _, chosen = choose_windows(claim, sentences)
            else:
                _, window = choose_windows(claim, sentences)
                chosen = candidate_indices(claim, sentences, len(window), arm)
            snippet = " ".join(sentences[i] for i in chosen)
            logits, _ = measure(tokenizer, reader, [dict(criterion=claim, source=snippet)])
            elapsed = (perf_counter() - started) * 1000
            cutoff = CUTS["full"] if arm == "full" else CUTS["two_lexical_windows"]
            if arm != "full" and len(set(chosen)) != len(base):
                raise ValueError("Different sentence budget in compact arm")
            report[arm] = dict(support=bool(support_margin(logits)[0] > cutoff),
                               complete=set(row["gold"]) <= set(chosen),
                               words=len(snippet.split()), tokens=len(tokenizer.encode(snippet, add_special_tokens=False)),
                               ms=elapsed)
        audits.append(report)
    slices = {
        "all": audits,
        "prior_37": [row for row in audits if row["audited"]],
        "other_open_pairs": [row for row in audits if not row["audited"]],
        "pooling": [row for row in audits if row["provenance"] == "pooling"],
        "citation": [row for row in audits if row["provenance"] == "citation"],
    }
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), revision=model["revision"],
        projection_sha256=digest(projection), script_sha256=digest(Path(__file__)), thresholds=CUTS,
        results={key: {arm: summarize(part, "lexical_windows", arm) for arm in arms}
                 for key, part in slices.items()},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    for method, arguments in (
        (prepare, ("archive", "earlier_projection", "output")),
        (run, ("projection", "checkpoint", "output")),
    ):
        sub = subparsers.add_parser(method.__name__)
        for arg in arguments:
            sub.add_argument(arg, type=Path)
        sub.set_defaults(method=method)
    values = vars(parser.parse_args())
    values.pop("method")(**values)
