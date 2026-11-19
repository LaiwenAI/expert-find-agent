"""Test transfer of a fixed-budget evidence repair policy to cross-document facts."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.hotpotqa.second_document_complement import DATA_SHA256
from experiments.qasper import certified_reranking as base
from experiments.qasper import set_repair as repair
from experiments.qasper.evidence_gap_audit import choose, score
from experiments.qasper.evidence_shortlist import online_scores, terms


SALT = "hotpot-sentence-set-transfer-v1"
SIZES = {"test": 256, "calibration": 100, "fit": 220}
SOURCE_THRESHOLD = .10
ARMS = ("pointwise", "two_doc_rule", "qasper_transfer", "target_score_only", "target_joint")


def project(row):
    context = row["context"]
    keys = [(title, i) for title, sentences in context for i, sent in enumerate(sentences) if sent.strip()]
    texts = [sent for _, sentences in context for sent in sentences if sent.strip()]
    facts = set(map(tuple, row["supporting_facts"]))
    if (len(keys) != len(set(keys)) or not 2 <= len(facts) <= repair.WIDTH or
            len({title for title, _ in facts}) < 2 or not facts <= set(keys)):
        return None
    position = {key: i for i, key in enumerate(keys)}
    return row["_id"], row["question"], texts, [title for title, _ in keys], {position[key] for key in facts}


def split_hotpot(rows):
    def normalize(value):
        return " ".join(value.casefold().split())

    seen_titles = set()
    seen_questions = set()
    seen_ids = set()
    selection = {}
    ordered = sorted(rows, key=lambda row: sha256(f"{SALT}:{row['_id']}".encode()).digest())
    for split, limit in SIZES.items():
        picked = []
        for row in ordered:
            titles = {normalize(title) for title, _ in row["context"]}
            question = normalize(row["question"])
            if titles & seen_titles or question in seen_questions or row["_id"] in seen_ids:
                continue
            case = project(row)
            if case is None:
                continue
            picked.append(case)
            seen_ids.add(row["_id"])
            seen_questions.add(question)
            seen_titles.update(titles)
            if len(picked) == limit:
                break
        if len(picked) != limit:
            raise ValueError(f"Could not obtain a disjoint {split} panel")
        selection[split] = picked
    return selection


def observe(cases, encoder, reader):
    # Each input sentence keeps its document title in both model views.
    contexts = [[f"{title}\n{text}" for title, text in zip(titles,texts,strict=True)]
                for _, _, texts, titles, _ in cases]
    unique = sorted({view for views in contexts for view in views})
    begin = perf_counter()
    encode_rows = dict(zip(unique,encoder.encode(unique),strict=True))
    index_seconds = perf_counter()-begin
    rows = []
    for (_, question, texts, titles, gold), context in zip(cases,contexts,strict=True):
        vectors = np.asarray([encode_rows[text] for text in context], dtype=np.float64)
        counts = [Counter(terms(text)) for text in texts]
        frequency = Counter(word for count in counts for word in count)
        begin = perf_counter()
        dense = online_scores(encoder,question,vectors)
        pool = np.argsort(-dense,kind="stable")[:repair.POOL].tolist()
        ranked, _ = score(reader,question,context,dense,pool)
        ranked.sort(key=lambda row:(-row[0],row[1]))
        fused, order = zip(*ranked,strict=True)
        prefix = (perf_counter()-begin)*1000
        begin = perf_counter()
        alternatives, view, one_swap = repair.features(question,order,np.asarray(fused),dense,titles,vectors,(counts,frequency))
        compute = (perf_counter()-begin)*1000
        labels = repair.gold_sets(list(range(len(texts))),order,alternatives,[gold])
        begin = perf_counter()
        # Only intervene when all baseline positions are from one document.
        preferred = list(range(min(repair.WIDTH,len(order))))
        if len({titles[order[slot]] for slot in preferred}) == 1:
            complement = next((slot for slot in range(len(preferred),len(order))
                               if titles[order[slot]] != titles[order[0]]),None)
            if complement is not None:
                preferred[-1] = complement
        rule_ms = (perf_counter()-begin)*1000
        rows.append(dict(view=view,labels=labels,one_swap=one_swap,prefix_ms=prefix,
                         feature_ms=compute,reachable=gold <= set(pool),rule_ms=rule_ms,
                         rule_correct=gold <= {order[slot] for slot in preferred},
                         rule_changed=preferred != list(range(len(preferred)))))
    return rows, dict(sentences=len(unique), encoding_seconds=index_seconds)


def evaluate(rows, models, thresholds):
    result = {}
    original = [bool(row["labels"][0]) for row in rows]
    for arm in ARMS:
        observed, edits, latency, overhead = [], [], [], []
        for row in rows:
            if arm == "pointwise":
                correct, changed, extra = bool(row["labels"][0]), False, 0.
            elif arm == "two_doc_rule":
                correct, changed, extra = row["rule_correct"], row["rule_changed"], row["rule_ms"]
            else:
                threshold = thresholds[arm]
                begin = perf_counter()
                candidate, gain = (0,0.) if threshold is None else repair.preference(row,models[arm],"joint_two")
                correct, changed = repair.outcome(row,candidate,gain,threshold)
                extra = (perf_counter()-begin)*1000 + (0. if threshold is None else row["feature_ms"])
            observed.append(correct)
            edits.append(changed)
            overhead.append(extra)
            latency.append(row["prefix_ms"]+extra)
        delta = [int(a)-int(b) for a,b in zip(observed,original,strict=True)]
        result[arm] = dict(complete=sum(observed),changed=sum(edits),gained=delta.count(1),lost=delta.count(-1),
                           paired_delta=paired_interval(delta),p95_ms=float(np.quantile(latency,.95)),
                           selector_p95_ms=float(np.quantile(overhead,.95)))
    return result


def run(train_archive, hotpot, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    if base.digest(hotpot) != DATA_SHA256:
        raise ValueError("Changed HotpotQA source")
    provenance = base.verify_checkpoint(embedding)
    encoder = base.QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = base.Reranker(reranker)
    source = repair.read_train(train_archive)
    qasper_panel = choose(source,multi=True)
    train, offline = repair.observe(source,qasper_panel,encoder,reader)
    source_model, comparable = repair.fit(train,train[0]["view"].shape[1])
    print("Trained source head",comparable,flush=True)
    cases = split_hotpot(json.loads(hotpot.read_bytes()))
    rows, summaries = {}, {}
    for split in ("fit","calibration","test"):
        panel = cases[split]
        rows[split], preparation = observe(panel,encoder,reader)
        summaries[split] = dict(questions=len(panel), offline=preparation,
                                reachable=sum(row["reachable"] for row in rows[split]),
                                two_swap_ceiling=sum(int(row["labels"].any()) for row in rows[split]),
                                id_sha256=sha256("\n".join(row[0] for row in panel).encode()).hexdigest())
        print("Completed HotpotQA",split,len(panel),flush=True)
    full, target_pairs = repair.fit(rows["fit"],rows["fit"][0]["view"].shape[1])
    additive, _ = repair.fit(rows["fit"],repair.ADDITIVE_FEATURES)
    adapted = {"target_score_only":additive,"target_joint":full}
    settings = repair.calibrate(rows["calibration"],adapted)
    thresholds = {"qasper_transfer":SOURCE_THRESHOLD} | {arm:state["threshold"] for arm,state in settings.items()}
    base.write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
        script_sha256=base.digest(Path(__file__)),parent_script_sha256=base.digest(Path(repair.__file__)),
        embedding_revision=provenance["revision"],reranker_revision=reader.provenance["revision"],
        hotpot_sha256=DATA_SHA256,qasper_train_sha256=repair.TRAIN_SHA,qasper_train_questions=len(qasper_panel),
        qasper_train_offline=offline,qasper_comparable=comparable,target_fit_comparable=target_pairs,
        hotpot_panels=summaries,calibration=settings,source_threshold=SOURCE_THRESHOLD,
        results=evaluate(rows["test"],adapted|{"qasper_transfer":source_model},thresholds),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train_archive","hotpot","embedding","reranker","output"):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
