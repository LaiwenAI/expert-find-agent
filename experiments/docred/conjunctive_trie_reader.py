"""Benchmark exact-prefix attention with unchanged atomic proof prompts."""

import argparse
from collections import Counter
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.exact_prompt_trie import pack_prompts, prefix_length
from expert_search.statistics import clustered_paired_interval
from experiments.docred import conjunctive_joint_reader as joint
from experiments.docred import conjunctive_proof_injection as injection
from experiments.docred import conjunctive_proof_windows as proof
from experiments.docred import oracle_role_evidence as readers
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.enterprise_isolated_parsing import padded
from experiments.retrieval.enterprise_prefix_fork import fork_prefix


ROUTES = ("independent", "root_cache", "root_trie", "full_trie", "guarded_trie")


class Reader:
    def __init__(self, checkpoint):
        self.tokenizer, self.model, spec = readers.load_reader(checkpoint)
        self.revision = spec["revision"]
        digit = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in (1, 0)]
        if any(len(ids) != 1 for ids in digit) or digit[0] == digit[1]:
            raise ValueError("Expected distinct single-token decisions")
        self.labels = [ids[0] for ids in digit]

    def prompts(self, case):
        windows, _ = injection.baseline(case)
        return [self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        ) for system, user in joint.messages(case, windows, "isolated")]

    def project(self, last):
        scores = self.model.lm_head(last.contiguous())[:, self.labels].float()
        return (scores[:, 0] - scores[:, 1]).cpu().numpy()

    def independent(self, prompts):
        payload = self.tokenizer.pad([{"input_ids": prompt} for prompt in prompts], padding=True,
                                     padding_side="left", return_tensors="pt").to("cuda")
        logits = self.model(**payload, use_cache=False, logits_to_keep=1).logits[:, -1, self.labels].float()
        return (logits[:, 0] - logits[:, 1]).cpu().numpy(), payload["input_ids"].numel()

    def root_cache(self, prompts):
        width = prefix_length(min(prompts), max(prompts))
        if width == 0 or any(len(prompt) == width for prompt in prompts):
            return self.independent(prompts)
        front = torch.tensor([prompts[0][:width]], device="cuda")
        cache = self.model.model(input_ids=front, use_cache=True).past_key_values
        suffixes = [prompt[width:] for prompt in prompts]
        ids, mask, positions = padded(suffixes, self.tokenizer.pad_token_id, width)
        front_mask = torch.ones((len(prompts), width), dtype=torch.bool, device="cuda")
        batch = self.model.model(
            input_ids=ids, attention_mask=torch.cat((front_mask, mask), dim=1),
            position_ids=positions, past_key_values=fork_prefix(cache, len(prompts)), use_cache=True,
        )
        if cache.get_seq_length() != width:
            raise AssertionError("Prefix changed after a read")
        return self.project(batch.last_hidden_state[:, -1]), width + ids.numel()

    def packed(self, prompts, common_only):
        tree = pack_prompts(prompts, common_only=common_only)
        nodes = torch.arange(len(tree.tokens), device="cuda")
        subtree_end = torch.tensor(tree.subtree_end, device="cuda")
        allowed = (nodes[:, None] >= nodes[None, :]) & (nodes[:, None] < subtree_end[None, :])
        dtype = self.model.dtype
        mask = torch.zeros(allowed.shape, device="cuda", dtype=dtype).masked_fill_(
            ~allowed, torch.finfo(dtype).min,
        )
        states = self.model.model(
            input_ids=torch.tensor([tree.tokens], device="cuda"),
            position_ids=torch.tensor([tree.depth], device="cuda"),
            attention_mask=mask[None, None], use_cache=False,
        ).last_hidden_state[0]
        return self.project(states[list(tree.leaves)]), len(tree.tokens)

    @torch.inference_mode()
    def execute(self, prompts, arm, cutoff, radius):
        begin = perf_counter()
        if arm == "independent":
            values, tokens = self.independent(prompts)
        elif arm == "root_cache":
            values, tokens = self.root_cache(prompts)
        else:
            values, tokens = self.packed(prompts, arm == "root_trie")
        recheck = arm == "guarded_trie" and any(
            abs(values - boundary).min() <= radius for boundary in {0., cutoff}
        )
        raw = values
        if recheck:
            values, extra = self.independent(prompts)
            tokens += extra
        if not np.isfinite(values).all() or values.shape != (4,):
            raise ValueError("Invalid atomic margin vector")
        return dict(values=values, approximate=raw, token_slots=tokens, rechecked=recheck,
                    ms=(perf_counter()-begin)*1000)


def candidates(scores):
    return scores[list(np.ravel(proof.PAIR))].reshape(2, 2).min(axis=1)


def calibrate(rows, reader):
    scores = []
    errors = 0.
    for case in rows:
        prompts = reader.prompts(case)
        ref = reader.execute(prompts, "independent", 0., 0.)["values"]
        packed = reader.execute(prompts, "full_trie", 0., 0.)["values"]
        errors = max(errors, float(np.max(np.abs(ref-packed))))
        scores.extend(float(v) for v, known in zip(candidates(ref), case["known"], strict=True) if not known)
    cutoff = sorted(scores, reverse=True)[int(len(scores) * proof.DEV_RISK)]
    return float(cutoff), errors


def collect(panel, reader, cutoff, radius, offset):
    result = []
    for i, case in enumerate(panel):
        begin = perf_counter()
        prompts = reader.prompts(case)
        prep = (perf_counter()-begin)*1000
        shift = (i + offset) % len(ROUTES)
        measurements = {arm: reader.execute(prompts, arm, cutoff, radius)
                        for arm in ROUTES[shift:]+ROUTES[:shift]}
        for obs in measurements.values():
            obs["ms"] += prep
        result.append((case,measurements))
        if (i+1) % 60 == 0:
            print(f"{i+1}/{len(panel)}", flush=True)
    return result


def summarize(rows, cutoff, radius):
    report = {}
    for arm in ROUTES:
        totals = Counter()
        durations, known_differences = [], []
        maximum = 0.
        for case, obs in rows:
            reference = obs["independent"]["values"]
            current = obs[arm]
            value = current["values"]
            error = np.abs(current["approximate"] - reference)
            maximum = max(maximum, float(error.max()))
            atom_same = np.array_equal(value > 0, reference > 0)
            decisions = candidates(value) > cutoff
            ref = candidates(reference) > cutoff
            known = np.asarray(case["known"])
            positive = int(np.count_nonzero(decisions & known))
            unchanged = bool(atom_same and np.array_equal(ref, decisions))
            if arm == "guarded_trie" and error.max() <= radius and not unchanged:
                raise AssertionError("An in-envelope request changed under the guard")
            totals.update(
                accepted_known=positive, accepted_unjudged=int(np.count_nonzero(decisions & ~known)),
                known=int(known.sum()), unjudged=2-int(known.sum()), unchanged_requests=int(unchanged),
                changed_atomic_labels=int(np.count_nonzero((value > 0) != (reference > 0))),
                changed_candidate_labels=int(np.count_nonzero(decisions != ref)),
                raw_radius_exceedances=int(np.count_nonzero(error > radius)),
                rechecked=int(current["rechecked"]), token_slots=current["token_slots"],
            )
            known_differences.append((positive - np.count_nonzero(ref & known))/2)
            durations.append(current["ms"])
        report[arm] = dict(
            **totals, max_raw_margin_error=maximum, p95_ms=float(np.quantile(durations,.95)),
            paired_known_delta=clustered_paired_interval(known_differences,[case["doc"] for case,_ in rows]),
        )
    return dict(cases=len(rows), arms=report)


def run(release, checkpoint, output, offset):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    pools, pins = proof.load(release)
    reader = Reader(checkpoint)
    collect(pools["calibration"][:1], reader, 0., 0., offset)
    cutoff, radius = calibrate(pools["calibration"], reader)
    reports = {group: summarize(collect(pools[group], reader, cutoff, radius, offset), cutoff, radius)
               for group in ("calibration", "test", "external")}
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), order_offset=offset, input_hashes=pins,
        model=reader.revision, cutoff=cutoff, margin_radius=radius, script_sha256=digest(Path(__file__)),
        trie_sha256=digest(Path(pack_prompts.__code__.co_filename)), groups=reports,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    run(args.release,args.checkpoint,args.output,args.order_offset)
