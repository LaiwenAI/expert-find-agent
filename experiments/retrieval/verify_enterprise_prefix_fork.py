"""Check bitwise BF16 cached-clause agreement on public requirement controls."""

import argparse
import json
import os
from pathlib import Path

import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.enterprise_cross_precision import CONTROLS, prefix_cache
from experiments.retrieval.enterprise_isolated_parsing import blocks, shared_prompt, spans
from experiments.retrieval.enterprise_prefix_fork import ForkReader


@torch.inference_mode()
def run(checkpoint, controls, output):
    if output.exists() or digest(controls) != CONTROLS or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require the pinned public controls and a fresh Slurm output")
    torch.backends.cuda.matmul.allow_tf32 = False
    reader = ForkReader(checkpoint)
    counts = dict(requests=0, blocks=0, exact_blocks=0, label_changes=0, max_logit_delta=0.)
    for item in json.loads(controls.read_bytes()):
        text = item["text"]
        front, suffixes = shared_prompt(reader.tokenizer, text, spans(text))
        cache = prefix_cache(reader, front)
        counts["requests"] += 1
        for block in blocks(front, suffixes):
            suffix_batch = tuple(seq for _, seq in block)
            reader.broadcast = False
            original, original_tokens = reader.cached(front, suffix_batch, cache)
            reader.broadcast = True
            forked, forked_tokens = reader.cached(front, suffix_batch, cache)
            if original_tokens != forked_tokens:
                raise AssertionError("Cached token geometry changed")
            counts["blocks"] += 1
            counts["exact_blocks"] += int(torch.equal(original, forked))
            counts["label_changes"] += int((original.argmax(1) != forked.argmax(1)).sum().item())
            counts["max_logit_delta"] = max(counts["max_logit_delta"],
                                           float((original.float() - forked.float()).abs().max().item()))
    write_json(output, dict(counts, job=os.environ["SLURM_JOB_ID"], controls_sha256=digest(controls),
                            code_sha256=digest(Path(__file__)), parser_revision=reader.provenance["revision"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "controls", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
