"""Train compact evidence verification on annotated and matched short windows."""

import argparse
from collections import Counter
import os
from pathlib import Path
import tarfile

import numpy as np

from expert_search.evidence_windows import DOC_SENTENCES
from expert_search.retrieval import tokens
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.minilm_transfer import LABELS, cutoff, summary
from experiments.scifact.vitaminc_contrastive_minilm import SEED, fit

EPOCHS = 4
ARMS = ("rationale_with_other_unknown", "rationale_with_same_unknown", "rationale_pair")
VITAMINC = dict(zip(LABELS, ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"), strict=True))


def rank_windows(claim, sentences, rationale):
    terms = set(tokens(claim))
    width = min(DOC_SENTENCES, len(sentences))
    windows = [tuple(range(i,i+width)) for i in range(len(sentences)-width+1)]
    covered = {indices:len(terms.intersection(tokens(" ".join(sentences[i] for i in indices)))) for indices in windows}
    ordered = sorted(windows, key=lambda w:(-covered[w], w[0]))
    forbidden = set().union(*rationale)
    positive = next((w for w in ordered if any(r <= set(w) for r in rationale)), None)
    control = next((w for w in ordered if forbidden.isdisjoint(w)), None)
    return positive, control


def construct(rows, abstracts, claims):
    pairs = []
    for row in rows:
        if row["label"]=="unknown":
            continue
        sentences = abstracts[row["doc"]]
        rationale = [set(g["sentences"]) for g in claims[row["case"]]["evidence"][row["doc"]]
                     if g["label"].lower()==row["label"]]
        first, second = rank_windows(row["criterion"], sentences, rationale)
        if first is not None and second is not None:
            base = dict(claim=row["criterion"], doc=row["doc"])
            pairs.append((base | {"evidence":" ".join(sentences[i] for i in first), "label":VITAMINC[row["label"]]},
                          base | {"evidence":" ".join(sentences[i] for i in second), "label":VITAMINC["unknown"]}))
    return pairs


def run(archive_path, checkpoint, output, weights):
    import torch
    from experiments.scifact.criterion_verifier_calibration import ARCHIVE, prepare, read_jsonl
    from experiments.scifact.finetune_nli_verifier import lexical, split
    from experiments.scifact.small_nli_verifier import load, measure

    if weights.exists() or output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh outputs and Slurm CUDA")
    if digest(archive_path)!=ARCHIVE:
        raise ValueError("Changed SciFact archive")
    with tarfile.open(archive_path,"r:gz") as archive:
        abstracts = {str(row["doc_id"]):row["abstract"] for row in read_jsonl(archive,"corpus")}
        train_claims = read_jsonl(archive,"claims_train")
        dev_claims = read_jsonl(archive,"claims_dev")
    docs = {doc:" ".join(lines) for doc,lines in abstracts.items() if lines}
    blocked = {str(doc) for row in dev_claims for doc in (*row["cited_doc_ids"],*row["evidence"])}
    training, controls = split(prepare(train_claims,docs,blocked))
    test = prepare(dev_claims,docs,set())
    if (len(training),len(controls),len(test))!=(233,76,226):
        raise ValueError("Changed public panels")
    groups = [{row["doc"] for row in q} for q in (training,controls,test)]
    if sum(map(len,groups)) != len(set.union(*groups)):
        raise ValueError("Publication overlap")
    same = construct(training, abstracts, {r["id"]:r for r in train_claims})
    unknown = [lexical(r,abstracts) for r in training if r["label"]=="unknown"]
    if not same or not unknown:
        raise ValueError("Insufficient training groups")
    other = [(a,dict(claim=b["criterion"], evidence=b["source"],label=VITAMINC["unknown"]))
             for (a,_),b in zip(same, (unknown[i%len(unknown)] for i in range(len(same))), strict=True)]
    views = {name:[lexical(row,abstracts) for row in rows] for name,rows in
             (("dev",controls),("test",test))}
    targets = {name:np.array([row["label"] for row in rows]) for name,rows in views.items()}

    # Hold the window size and premise pair constant while testing rationale removal.
    matched = construct(test,abstracts,{r["id"]:r for r in dev_claims})
    matched_texts = [dict(source=r["evidence"],criterion=r["claim"]) for pair in matched for r in pair]
    matched_targets = [(LABELS[list(VITAMINC.values()).index(a["label"])], "unknown") for a,_ in matched]
    matched_targets = [("entailment" if a=="support" else "contradiction","neutral") for a,_ in matched_targets]

    weights.mkdir()
    results = {}
    frozen = None
    arms = [("frozen",SEED)] + [(arm,seed) for seed in (SEED,SEED+1) for arm in ARMS]
    for arm, seed in arms:
        name = f"{arm}_{seed}" if arm!="frozen" else arm
        tokenizer,model,positions,provenance=load(checkpoint,"minilm")
        if arm == "frozen":
            fit_result = {}
        else:
            pairs = other if arm == "rationale_with_other_unknown" else same
            fit_result = fit(tokenizer,model,positions,pairs,
                             "rationale_pair" if arm=="rationale_pair" else "classification",
                             weights, seed=seed, artifact_name=name, epochs=EPOCHS)
        measured = {}
        measure(tokenizer,model,positions,views["test"][:4],batch_size=1)
        for key,q in views.items():
            values,times=measure(tokenizer,model,positions,q,batch_size=1)
            measured[key]=(np.array([v[1] if v else -np.inf for v in values]),times)
        threshold=cutoff(measured["dev"][0],targets["dev"])
        if frozen is None:
            frozen=measured["test"][0]>threshold
        checks,_=measure(tokenizer,model,positions,matched_texts,batch_size=1)
        decisions=[v[0] if v else "skipped" for v in checks]
        results[name]=dict(
            training=fit_result,threshold=threshold,
            p95_ms=float(np.quantile(measured["test"][1],.95)),
            calibration=summary(measured["dev"][0],targets["dev"],threshold),
            test=summary(measured["test"][0],targets["test"],threshold,frozen),
            matched_rationale_and_missing_correct=sum(
                tuple(decisions[i:i+2])==gold for i,gold in zip(range(0,len(decisions),2),matched_targets,strict=True)),
            matched_rationale_direction_correct=sum(a==gold[0] for a,gold in zip(decisions[::2],matched_targets,strict=True)),
            matched_missing_neutral=sum(a=="neutral" for a in decisions[1::2]),
        )
        print(name,"complete",flush=True)

    write_json(output,dict(job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
                           archive_sha256=ARCHIVE,script_sha256=digest(Path(__file__)), model_revision=provenance["revision"],
                           train_pairs=len(same), train_directions=dict(Counter(a["label"] for a,_ in same)),
                           train_unknown_pool=len(unknown), matched_test_cases=len(matched),
                           panels={k:dict(Counter(r["label"] for r in q)) for k,q in views.items()}, report=results))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("archive_path","checkpoint","output","weights"):
        parser.add_argument(key,type=Path)
    run(**vars(parser.parse_args()))
