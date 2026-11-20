"""Count exact within-request prefix sharing for the original atomic reader."""

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.docred import conjunctive_joint_reader as joint
from experiments.docred import conjunctive_proof_injection as injection
from experiments.docred import conjunctive_proof_windows as proof
from experiments.followir.core17_reranking import digest, write_json


def common_prefix(left, right):
    for i, (a, b) in enumerate(zip(left,right)):
        if a != b:
            return i
    return min(len(left),len(right))


def measure(panel, tokenizer):
    totals = Counter()
    for case in panel:
        windows, _ = injection.baseline(case)
        encoded = sorted(tokenizer.apply_chat_template(
            [{"role":"system","content":system},{"role":"user","content":user}],
            add_generation_prompt=True, tokenize=True, return_dict=False,
        ) for system,user in joint.messages(case, windows, "isolated"))
        adjacent = [common_prefix(left,right) for left,right in zip(encoded,encoded[1:])]
        common = min(adjacent)
        totals.update(
            cases=1,
            all_distinct_windows=int(len(set(windows))==4),
            exact_repeated_window_requests=int(len(set(windows))<4),
            logical_tokens=sum(map(len,encoded)),
            padded_tokens=len(encoded)*max(map(len,encoded)),
            one_common_prefix_saved=(len(encoded)-1)*common,
            extra_branch_tokens_saved=sum(adjacent)-(len(encoded)-1)*common,
            exact_trie_tokens=sum(map(len,encoded))-sum(adjacent),
        )
    return totals


def run(release, checkpoint, output):
    from transformers import AutoTokenizer

    pools,pins=proof.load(release)
    manifest=json.loads((checkpoint/"local-provenance.json").read_bytes())
    for name in ("tokenizer.json","tokenizer_config.json"):
        if digest(checkpoint/name) != manifest["files"][name]:
            raise ValueError("Changed tokenizer asset")
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    write_json(output,dict(
        script_sha256=digest(Path(__file__)),reader_revision=manifest["revision"],inputs=pins,
        panels={label:measure(rows,tokenizer) for label,rows in pools.items()},
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("release","checkpoint","output"):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    run(args.release,args.checkpoint,args.output)
