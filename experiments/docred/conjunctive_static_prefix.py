"""Amortize the fixed system prefix before scoring independent proof claims."""

import argparse
from collections import Counter
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.exact_prompt_trie import pack_prompts, prefix_length
from experiments.docred import conjunctive_trie_reader as base
from experiments.docred import conjunctive_proof_windows as proof
from experiments.docred import work_owner_contrast as owners
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("independent", "static_cache", "static_trie", "guarded_static_trie")


class StaticReader(base.Reader):
    @torch.inference_mode()
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        examples = [self.tokenizer.apply_chat_template(
            [{"role": "system", "content": owners.SYSTEM},
             {"role": "user", "content": f"Source:\n{source}\n\nQuestion: {source}"}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        ) for source in ("a", "z")]
        width = prefix_length(*examples)
        if not width:
            raise ValueError("The fixed instruction prefix is empty")
        self.prefix = examples[0][:width]
        start = perf_counter()
        self.cache = self.model.model(
            input_ids=torch.tensor([self.prefix], device="cuda"), use_cache=True,
        ).past_key_values
        torch.cuda.synchronize()
        self.initialization_ms = (perf_counter()-start)*1000

    def static(self, prompts, tree):
        width = len(self.prefix)
        if any(len(prompt) <= width or prompt[:width] != self.prefix for prompt in prompts):
            raise ValueError("Prompt outside the frozen prefix contract")
        suffixes = [row[width:] for row in prompts]
        mask = None
        if tree:
            packed = pack_prompts(suffixes)
            nodes = torch.arange(len(packed.tokens), device="cuda")
            ends = torch.tensor(packed.subtree_end, device="cuda")
            ancestor = (nodes[:, None] >= nodes[None, :]) & (nodes[:, None] < ends[None, :])
            suffix_mask = torch.zeros(ancestor.shape, device="cuda", dtype=self.model.dtype).masked_fill_(
                ~ancestor, torch.finfo(self.model.dtype).min,
            )
            prefix_mask = torch.zeros((len(nodes), width), device="cuda", dtype=self.model.dtype)
            mask = torch.cat((prefix_mask, suffix_mask), dim=1)[None, None]
            ids = torch.tensor([packed.tokens], device="cuda")
            positions = torch.tensor([packed.depth], device="cuda") + width
        else:
            ids, suffix_mask, positions = base.padded(suffixes, self.tokenizer.pad_token_id, width)
            prefix_mask = torch.ones((len(prompts), width), dtype=torch.bool, device="cuda")
            mask = torch.cat((prefix_mask, suffix_mask), dim=1)
        cache = base.fork_prefix(self.cache, len(ids))
        states = self.model.model(
            input_ids=ids, position_ids=positions, attention_mask=mask, past_key_values=cache,
            use_cache=True,
        ).last_hidden_state
        if self.cache.get_seq_length() != width:
            raise AssertionError("Persistent prefix was changed")
        last = states[0, list(packed.leaves)] if tree else states[:, -1]
        return self.project(last), ids.numel()


def decision(values, cutoff):
    return np.r_[values > 0, base.candidates(values) > cutoff]


@torch.inference_mode()
def run(release, checkpoint, output, offset):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    pools, pins = proof.load(release)
    reader = StaticReader(checkpoint)
    cutoff, _ = base.calibrate(pools["calibration"], reader)
    radius = 0.
    for case in pools["calibration"]:
        prompts = reader.prompts(case)
        canonical, _ = reader.independent(prompts)
        combined, _ = reader.static(prompts, tree=True)
        radius = max(radius, float(np.max(np.abs(canonical-combined))))

    report = {}
    for label in ("test", "external"):
        counts = {arm: Counter() for arm in ARMS}
        times = {arm: [] for arm in ARMS}
        for i, case in enumerate(pools[label]):
            started = perf_counter()
            prompts = reader.prompts(case)
            shared = (perf_counter()-started)*1000
            per_arm = {}
            shift = (i+offset) % len(ARMS)
            for arm in ARMS[shift:]+ARMS[:shift]:
                started = perf_counter()
                if arm == "independent":
                    raw, consumed = reader.independent(prompts)
                else:
                    raw, consumed = reader.static(prompts, tree=arm != "static_cache")
                rechecked = arm == "guarded_static_trie" and any(
                    np.min(np.abs(raw-boundary)) <= radius for boundary in (0.,cutoff)
                )
                values = raw
                if rechecked:
                    values, extra = reader.independent(prompts)
                    consumed += extra
                times[arm].append(shared+(perf_counter()-started)*1000)
                per_arm[arm] = (values, raw, consumed, rechecked)
            reference = per_arm["independent"][0]
            for arm, (values, raw, consumed, rechecked) in per_arm.items():
                mismatch = decision(values,cutoff) != decision(reference,cutoff)
                known = np.asarray(case["known"])
                approved = base.candidates(values)>cutoff
                counts[arm].update(
                    known_accepted=int((approved&known).sum()), unjudged_accepted=int((approved&~known).sum()),
                    changed_atoms=int(mismatch[:4].sum()), changed_conjunctions=int(mismatch[4:].sum()),
                    changed_requests=int(mismatch.any()), rechecked=int(rechecked), tokens=consumed,
                    outside_calibration=int((np.abs(raw-reference)>radius).sum()),
                )
        report[label] = {arm: dict(**counts[arm],p95_ms=float(np.quantile(times[arm],.95))) for arm in ARMS}
    write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), order_offset=offset,
        inputs=pins, model=reader.revision, cutoff=cutoff, margin_radius=radius,
        static_prefix_tokens=len(reader.prefix), static_prefill_ms=reader.initialization_ms,
        script_sha256=digest(Path(__file__)), groups=report,
    ))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("release","checkpoint","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--order-offset",type=int,default=0)
    args=parser.parse_args()
    run(args.release,args.checkpoint,args.output,args.order_offset)
