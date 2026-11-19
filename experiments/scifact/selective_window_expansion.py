"""Compare bounded second-window checks with single-input evidence verification."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import tarfile

import numpy as np

from expert_search.evidence_windows import choose_windows
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.minilm_transfer import LABELS, apply_weights, cutoff, summary

SEEDS = (20261119, 20261120)
REVIEW_CAP = .25
STEPS = (-6., -3., -1.5, -.5, 0., .5, 1.5, 3., 6.)


def build(archive_path):
    from experiments.scifact.criterion_verifier_calibration import ARCHIVE, prepare, read_jsonl
    from experiments.scifact.finetune_nli_verifier import split

    if digest(archive_path)!=ARCHIVE:
        raise ValueError("Changed public archive")
    with tarfile.open(archive_path,"r:gz") as archive:
        abstracts={str(r["doc_id"]):r["abstract"] for r in read_jsonl(archive,"corpus")}
        training=read_jsonl(archive,"claims_train")
        test=read_jsonl(archive,"claims_dev")
    sources={k:" ".join(v) for k,v in abstracts.items() if v}
    blocked={str(x) for r in test for x in (*r["cited_doc_ids"],*r["evidence"])}
    _, calibration=split(prepare(training,sources,blocked))
    held=prepare(test,sources,set())
    if (len(calibration),len(held))!=(76,226):
        raise ValueError("Changed public cohorts")
    if {r["doc"] for r in calibration}&{r["doc"] for r in held}:
        raise ValueError("Overlapping calibration and evaluation sources")
    panels={}
    for name,rows,claims in (("calibration",calibration,training),("test",held,test)):
        by_id={r["id"]:r for r in claims}
        group=[]
        for row in rows:
            sections=abstracts[row["doc"]]
            first,combined=choose_windows(row["criterion"],sections)
            second=tuple(i for i in combined if i not in first)
            rationales=[set(g["sentences"]) for g in by_id[row["case"]]["evidence"].get(row["doc"],[])]
            views={view:" ".join(sections[i] for i in indices) for view,indices in
                   (("first",first),("second",second),("combined",combined))}
            group.append(dict(row, views=views, has_second=bool(second),
                              first_complete=any(g <= set(first) for g in rationales),
                              combined_complete=any(g <= set(combined) for g in rationales)))
        panels[name]=group
    return panels


def simulate(first, extra, threshold, lower, upper):
    requested=(first>=lower)&(first<=upper)&np.isfinite(extra)
    result=np.where(requested,extra>threshold,first>threshold)
    return result, requested


def calibrate(first,extra,labels):
    base=cutoff(first,labels)
    caps=[math.floor(.05*np.sum(labels==label)) for label in LABELS[1:]]
    candidates=[]
    for t in (*STEPS,base):
        for low in STEPS:
            for high in STEPS:
                if low>high:
                    continue
                pred,requests=simulate(first,extra,t,low,high)
                false=[int(np.sum(pred&(labels==label))) for label in LABELS[1:]]
                if all(n<=cap for n,cap in zip(false,caps,strict=True)) and requests.mean()<=REVIEW_CAP:
                    correct=int(np.sum(pred&(labels=="support")))
                    candidates.append((-correct,int(requests.sum()),sum(false),float(t),low,high))
    original=int(np.sum((first>base)&(labels=="support")))
    # The no-expansion first-window rule is always an admissible alternative.
    false=int(np.sum((first>base)&(labels!="support")))
    candidates.append((-original,0,false,base,1.,-1.))
    _,_,_,threshold,lo,hi=min(candidates)
    return threshold,lo,hi


def assess(rows, observations):
    labels={k:np.asarray([r["label"] for r in q]) for k,q in rows.items()}
    cal,test=observations["calibration"],observations["test"]
    first_threshold=cutoff(cal["first"],labels["calibration"])
    control=test["first"]>first_threshold
    combined_cal=np.where(np.isfinite(cal["combined"]),cal["combined"],cal["first"])
    combined_test=np.where(np.isfinite(test["combined"]),test["combined"],test["first"])
    fixed_threshold=cutoff(combined_cal,labels["calibration"])
    max_cal=np.maximum(cal["first"],cal["second"])
    max_test=np.maximum(test["first"],test["second"])
    max_threshold=cutoff(max_cal,labels["calibration"])
    args=calibrate(cal["first"],cal["combined"],labels["calibration"])
    pred,requested=simulate(test["first"],test["combined"],*args)
    attempts=(test["first"]>=args[1])&(test["first"]<=args[2])
    y=labels["test"]
    support=y=="support"
    variants={
        "first_only":(control,np.zeros(len(y),dtype=bool),test["first_ms"]),
        "direct_combined":(combined_test>fixed_threshold,np.zeros(len(y),dtype=bool),
                           test["combined_ms"]+test["first_ms"]*~np.isfinite(test["combined"])),
        "separate_max":(max_test>max_threshold,np.isfinite(test["second"]),test["first_ms"]+test["second_ms"]),
        "selective_combined":(pred,requested,test["first_ms"]+test["combined_ms"]*attempts),
    }
    answers={}
    for key,(accepted,calls,times) in variants.items():
        val=(accepted.astype(int)-control.astype(int))
        answers[key]={
            **summary(accepted.astype(float),y,.5),
            "second_forwards":int(np.sum(calls)), "factored_p95_ms":float(np.quantile(times,.95)),
            "paired_support_delta":paired_interval(val[support]),
        }
    calibrated, calibration_calls = simulate(cal["first"], cal["combined"], *args)
    return dict(thresholds=dict(first=first_threshold,direct_combined=fixed_threshold,max_separate=max_threshold,
                                selective=dict(cutoff=args[0],lower=args[1],upper=args[2])),
                calibration_selective=summary(calibrated.astype(float),labels["calibration"],.5),
                calibration_second_forwards=int(calibration_calls.sum()),
                over_limit_combined=int(np.sum((test["combined_ms"]>0)&~np.isfinite(test["combined"]))),
                test=answers)


def run(archive_path, training_results, weights, checkpoint, output):
    import torch
    from experiments.scifact.small_nli_verifier import load,measure

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    panels=build(archive_path)
    previous=json.loads(training_results.read_text())
    arms=("frozen",*(f"rationale_pair_{seed}" for seed in SEEDS))
    results={}
    for arm in arms:
        tokenizer,model,positions,provenance=load(checkpoint,"minilm")
        if provenance["revision"]!=previous["model_revision"]:
            raise ValueError("Changed model checkpoint")
        if arm!="frozen":
            apply_weights(model,weights/f"{arm}.safetensors",previous["report"][arm]["training"]["weight_sha256"])
        measure(tokenizer,model,positions,panels["test"][:3],batch_size=1)
        observed={}
        for name,rows in panels.items():
            record={}
            for view in ("first","second","combined"):
                selected=[i for i,r in enumerate(rows) if view=="first" or r["has_second"]]
                inputs=[dict(source=rows[i]["views"][view],criterion=rows[i]["criterion"]) for i in selected]
                logits,costs=measure(tokenizer,model,positions,inputs,batch_size=1)
                matrix=np.full(len(rows),-np.inf)
                ms=np.zeros(len(rows))
                for i,score,cost in zip(selected,logits,costs,strict=True):
                    if score is None and view == "first":
                        raise ValueError("The first window must fit the model context")
                    if score is not None:
                        matrix[i]=score[1]
                    ms[i]=cost
                record[view]=matrix
                record[view+"_ms"]=ms
            observed[name]=record
        results[arm]=assess(panels,observed)
        print(arm,"complete",flush=True)
    write_json(output,dict(job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
                           archive_sha256=digest(archive_path),script_sha256=digest(Path(__file__)),
                           training_sha256=digest(training_results), model_revision=provenance["revision"],
                           panels={name:dict(Counter(r["label"] for r in rows)) for name,rows in panels.items()},
                           test_expansion_available=sum(r["has_second"] for r in panels["test"]),
                           test_rationales={k:sum(r[k] for r in panels["test"])
                                            for k in ("first_complete","combined_complete")},report=results))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("archive_path","training_results","weights","checkpoint","output"):
        parser.add_argument(key,type=Path)
    run(**vars(parser.parse_args()))
