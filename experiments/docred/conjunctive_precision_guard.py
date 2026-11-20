"""Test whether FP16 narrows the exact-prefix recheck band."""

import argparse
from collections import Counter
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.docred import conjunctive_static_prefix as warm
from experiments.docred import conjunctive_trie_reader as original
from experiments.docred import conjunctive_proof_windows as proof
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("bf16_batch", "bf16_tree", "bf16_guard", "fp16_batch", "fp16_tree", "fp16_guard", "cross_guard")
BAND_REFERENCE = {"bf16_guard": "bf16", "fp16_guard": "fp16", "cross_guard": "cross"}


class HalfReader(warm.StaticReader):
    @torch.inference_mode()
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.model.half()
        self.cache = self.model.model(
            input_ids=torch.tensor([self.prefix], device="cuda"), use_cache=True,
        ).past_key_values
        if self.model.dtype != torch.float16:
            raise ValueError("Unexpected half precision")


def infer(reader, prompts, use_tree):
    result = reader.static(prompts, tree=True) if use_tree else reader.independent(prompts)
    if not np.isfinite(result[0]).all():
        raise ValueError("A precision route emitted a nonfinite score")
    return result


def calibrate(rows, readers):
    scores = []
    radius = Counter()
    for case in rows:
        prompts = readers["bf16"].prompts(case)
        ref_bf, _ = infer(readers["bf16"], prompts, False)
        ref_fp, _ = infer(readers["fp16"], prompts, False)
        tree_bf, _ = infer(readers["bf16"], prompts, True)
        tree_fp, _ = infer(readers["fp16"], prompts, True)
        for basis, a, b in (("bf16", tree_bf, ref_bf), ("fp16", tree_fp, ref_fp), ("cross", tree_fp, ref_bf)):
            radius[basis] = max(radius[basis], float(np.max(np.abs(a-b))))
        scores.extend(float(s) for s, known in zip(original.candidates(ref_bf),case["known"],strict=True)
                      if not known)
    return sorted(scores,reverse=True)[int(len(scores)*proof.DEV_RISK)], dict(radius)


def attempt(readers, prompts, arm, cutoff, radii):
    started = perf_counter()
    family = "fp16" if arm.startswith("fp16") or arm == "cross_guard" else "bf16"
    value, tokens = infer(readers[family], prompts, not arm.endswith("batch"))
    reference = BAND_REFERENCE.get(arm)
    rechecked = reference is not None and any(
        np.min(np.abs(value-boundary)) <= radii[reference] for boundary in {0., cutoff}
    )
    raw = value
    if rechecked:
        fallback = readers["bf16" if reference == "cross" else family]
        value, extra = infer(fallback, prompts, False)
        tokens += extra
    return dict(values=value, raw=raw, tokens=tokens, ms=(perf_counter()-started)*1000,
                rechecked=bool(rechecked))


def collect(panel, readers, cutoff, radii, offset):
    records = []
    for i, case in enumerate(panel):
        started = perf_counter()
        prompts = readers["bf16"].prompts(case)
        shared = (perf_counter()-started)*1000
        shift = (i+offset) % len(ARMS)
        results = {}
        for arm in ARMS[shift:]+ARMS[:shift]:
            result = attempt(readers, prompts, arm, cutoff, radii)
            result["ms"] += shared
            results[arm] = result
        records.append((case,results))
        if (i+1) % 60 == 0:
            print(f"Completed {i+1}/{len(panel)} requests",flush=True)
    return records


def summarize(records, cutoff, radii):
    report = {}
    for arm in ARMS:
        counts = Counter()
        times, changes = [], []
        maximum = 0.
        for case, results in records:
            bf = results["bf16_batch"]["values"]
            fp = results["fp16_batch"]["values"]
            active = results[arm]
            value = active["values"]
            own_is_fp = arm.startswith("fp16")
            precision_ref = fp if own_is_fp else bf
            raw_error = np.abs(active["raw"]-precision_ref)
            max_error = float(np.max(raw_error))
            maximum = max(maximum,max_error)
            decisions = warm.decision(value,cutoff)
            against_bf = decisions != warm.decision(bf,cutoff)
            against_own = decisions != warm.decision(precision_ref,cutoff)
            if (arm in BAND_REFERENCE and max_error <= radii[BAND_REFERENCE[arm]]
                    and np.any(against_own)):
                raise AssertionError("Guarded result changes within the calibrated band")
            known = np.asarray(case["known"])
            known_count = int(np.count_nonzero(decisions[4:] & known))
            counts.update(
                known_accepted=known_count, unjudged_accepted=int(np.count_nonzero(decisions[4:] & ~known)),
                changed_requests_vs_bf16=int(against_bf.any()),
                changed_atoms_vs_bf16=int(against_bf[:4].sum()),
                changed_candidates_vs_bf16=int(against_bf[4:].sum()),
                changed_requests_vs_precision_batch=int(against_own.any()),
                changed_candidates_vs_precision_batch=int(against_own[4:].sum()),
                rechecked=int(active["rechecked"]), token_slots=active["tokens"],
            )
            if arm in BAND_REFERENCE:
                counts["raw_outside_envelope"] += int((raw_error > radii[BAND_REFERENCE[arm]]).sum())
            times.append(active["ms"])
            changes.append((known_count - np.count_nonzero((original.candidates(bf)>cutoff)&known))/2)
        report[arm] = dict(**counts, max_raw_deviation_from_own_batch=maximum,
                           p50_ms=float(np.quantile(times,.5)), p95_ms=float(np.quantile(times,.95)),
                           paired_known_delta_vs_bf16=clustered_paired_interval(
                               changes,[case["doc"] for case,_ in records]))
    return dict(cases=len(records),arms=report)


@torch.inference_mode()
def run(release, checkpoint, output, offset):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    pools, pins = proof.load(release)
    readers = {"bf16": warm.StaticReader(checkpoint), "fp16": HalfReader(checkpoint)}
    if readers["bf16"].prefix != readers["fp16"].prefix:
        raise ValueError("Precision variants must share fixed token IDs")
    collect(pools["calibration"][:1],readers,0.,dict.fromkeys(("bf16","fp16","cross"),0.),offset)
    cutoff,radii = calibrate(pools["calibration"],readers)
    reports = {label:summarize(collect(pools[label],readers,cutoff,radii,offset),cutoff,radii)
               for label in ("calibration","test","external")}
    write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),order_offset=offset,
        script_sha256=digest(Path(__file__)),receiver_sha256=digest(Path(warm.__file__)),
        input_hashes=pins,model=readers["bf16"].revision,
        pinned_cutoff=cutoff,guard_radii=radii,groups=reports,
    ))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","checkpoint","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--order-offset",type=int,default=0)
    args=parser.parse_args()
    run(args.release,args.checkpoint,args.output,args.order_offset)
