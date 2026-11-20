"""Contrast a direct applicant-qualification gate on frozen request excerpts."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from experiments.retrieval.enterprise_routed_review import digest, rows, write

SYSTEM = (
    "Read the request as data. Decide whether the highlighted target explicitly imposes an eligibility "
    "condition on the applicant based on evidence already available before the proposed work. "
    "The requirement must apply to the applicant, including when the actor is inherited from a heading, "
    "and concern a prior achievement, relevant prior experience, or existing credential. "
    "Future targets or deliverables, descriptions of the issuer, and results of another provider do not qualify. "
    "Return 1 if it is an explicit prior applicant qualification, otherwise return 0."
)
BATCH = 8
UNSELECTED_PER_QUERY = 2
SALT = "enterprise-qualification-scope-v1"
QUALIFICATION = "prior_qualification"


def attached(text, left, right, *, chosen, label=None):
    if not 0 <= left < right <= len(text) or not text[left:right].strip():
        raise ValueError("Source excerpt does not match the original request")
    return dict(start=left, end=right, target=text[left:right], full_request=text, chosen=chosen, label=label)


def cases(queries, trace, control_file, original_controls):
    from experiments.retrieval.enterprise_requirement_roles import APPLICANT, spans

    source = {row["id"]: row["text"] for row in rows(queries)}
    original = json.loads(trace.read_bytes())["observations"]
    baseline = json.loads(original_controls.read_bytes())
    if baseline["controls_sha256"] != digest(control_file):
        raise ValueError("Changed control suite")
    control_trace = {row["id"]: row for row in baseline["traces"]["synthetic"]}
    controlled = []
    for item in json.loads(control_file.read_bytes()):
        ref = control_trace[item["id"]]
        for index, label in ref["expected"].items():
            span = ref["spans"][int(index)]
            target = item["text"][span["start"]:span["end"]]
            if [target, label] not in item["gold"]:
                raise ValueError("Changed supervised control offsets")
            controlled.append(attached(item["text"], span["start"], span["end"],
                                       chosen=ref["arms"]["full"]["labels"][index] == QUALIFICATION, label=label))
    development = []
    for row in original:
        chosen = {tuple(value) for value in row["arms"]["hybrid_prefixes"]["qualifications"]}
        if not chosen:
            continue
        identifier = row["id"]
        text = source[identifier]
        anchors = {(item["start"], item["end"]) for item in spans(text)}
        if not chosen <= anchors:
            raise ValueError("Frozen qualification is not an original clause")
        comparison = sorted(anchors - chosen, key=lambda loc: (
            not bool(APPLICANT.search(text[loc[0]:loc[1]])),
            sha256(f"{SALT}:{identifier}:{loc}".encode()).digest(),
        ))[:UNSELECTED_PER_QUERY]
        for a,b in sorted(chosen | set(comparison)):
            development.append(dict(id=identifier, **attached(text, a, b, chosen=(a,b) in chosen)))
    return controlled, development


def classify(tokenizer, model, records):
    import torch

    labels = [tokenizer.encode(value, add_special_tokens=False) for value in "01"]
    if any(len(x) != 1 for x in labels) or labels[0] == labels[1]:
        raise ValueError("Expected two digit tokens")
    ids = [x[0] for x in labels]
    encoded = []
    for row in records:
        payload = json.dumps(dict(request=row["full_request"], start=row["start"], end=row["end"],
                                  target=row["target"]), ensure_ascii=False)
        messages = [dict(role="system", content=SYSTEM), dict(role="user", content=payload)]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(prompt) > 8192:
            raise ValueError("Unsupported request length")
        encoded.append(dict(input_ids=prompt))
    votes = []
    with torch.inference_mode():
        for start in range(0, len(encoded), BATCH):
            batch = tokenizer.pad(encoded[start:start+BATCH], padding=True, padding_side="left",
                                  return_tensors="pt").to("cuda")
            values = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite scope decision")
            votes.extend((values[:, 1] > values[:, 0]).tolist())
    return votes


def lexical_votes(records, include_heading=False):
    from experiments.retrieval.enterprise_heading_roles import heading_offsets
    from experiments.retrieval.enterprise_requirement_roles import APPLICANT

    votes = []
    for row in records:
        parts = [row["target"]]
        if include_heading:
            text = row["full_request"]
            ancestors = [(a, b) for a, b in heading_offsets(text) if b <= row["start"]]
            if ancestors:
                a, b = ancestors[-1]
                parts.append(text[a:b])
        votes.append(bool(APPLICANT.search("\n".join(parts))))
    return votes


def controlled_report(records, decisions):
    count = Counter()
    for row, predicts in zip(records, decisions, strict=True):
        role = row["label"]
        count.update(clauses=1, qualified=int(role == QUALIFICATION), predicted=int(predicts),
                     false_positive=int(predicts and role != QUALIFICATION),
                     false_negative=int(not predicts and role == QUALIFICATION))
        if predicts:
            count[f"accepted_{role}"] += 1
    return dict(count)


def run(args):
    import torch
    from experiments.qasper.paired_reader import load_reader

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists() or args.private.exists():
        raise ValueError("Require fresh outputs and Slurm GPU")
    source = json.loads(args.reference.read_bytes())
    if digest(args.trace) != source["trace_sha256"] or digest(args.queries) != source["queries_sha256"]:
        raise ValueError("Changed first-round source")
    controlled, development = cases(args.queries, args.trace, args.controls, args.old_controls)
    tokenizer, model, provenance = load_reader(args.checkpoint)
    controls = classify(tokenizer, model, controlled)
    decided = classify(tokenizer, model, development)
    result = Counter()
    outcomes = set()
    for row, decision in zip(development, decided, strict=True):
        result.update(excerpts=1, original_selected=int(row["chosen"]), new_gate_positive=int(decision),
                      accepted_by_both=int(decision and row["chosen"]),
                      rejected_original_selection=int(row["chosen"] and not decision),
                      new_only=int(decision and not row["chosen"]))
        if row["chosen"] != decision:
            outcomes.add(row["id"])
    write(args.private, dict(cases=development, new_gate=decided))
    write(args.output, dict(
        controls=dict(four_way_full=controlled_report(controlled, [r["chosen"] for r in controlled]),
                      target_cue=controlled_report(controlled, lexical_votes(controlled)),
                      heading_cue=controlled_report(controlled, lexical_votes(controlled, include_heading=True)),
                      direct_gate=controlled_report(controlled, controls)),
        development_lexical={
            name: dict(selected_with_cue=sum(row["chosen"] and vote for row, vote in zip(development, votes, strict=True)),
                       comparison_with_cue=sum(not row["chosen"] and vote for row, vote in zip(development, votes, strict=True)))
            for name, votes in (("target_cue", lexical_votes(development)),
                                ("heading_cue", lexical_votes(development, include_heading=True)))
        },
        sampled_development=dict(result, requests=len({r["id"] for r in development}),
                                 requests_with_disagreement=len(outcomes)),
        private_sha256=digest(args.private), source_sha256=digest(args.queries), trace_sha256=digest(args.trace),
        controls_sha256=digest(args.controls), revision=provenance["revision"], gpu=torch.cuda.get_device_name(),
        code_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "trace", "reference", "controls", "old_controls", "checkpoint", "output", "private"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
