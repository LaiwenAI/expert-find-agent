"""Isolate batch-shape drift during selective BF16 qualification rechecks."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import linear

from experiments.retrieval.enterprise_cross_precision import CONTROLS, NARROW, QUALIFICATION, evaluate
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT
from experiments.retrieval.enterprise_guarded_parsing import target
from experiments.retrieval.enterprise_isolated_parsing import padded, shared_prompt, spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_routed_review import digest, rows, write

ARMS = ("full", "compact", "head_shape", "four_rows", "eight_rows", "guarded_head_shape")
BAND = .5
LAYER_PROBES = (
    "input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.q_norm", "self_attn.k_norm", "self_attn.o_proj.input", "self_attn.o_proj",
    "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj.input", "mlp.down_proj",
)


def qualification(scores):
    return scores.argmax(dim=1) == QUALIFICATION


def qualification_margin(scores):
    other = [i for i in range(scores.shape[1]) if i != QUALIFICATION]
    return (scores[:, QUALIFICATION].float() - scores[:, other].float().amax(dim=1)).abs()


def take(tensors, indices):
    return tuple(tensor.index_select(0, indices) for tensor in tensors)


def hidden(reader, tensors):
    ids, mask, positions = tensors
    return reader.backbone(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False).last_hidden_state[:, -1].contiguous()


def preserve_head(reader, partial, selected, total):
    packed = partial.new_zeros((total, partial.shape[1]))
    packed.index_copy_(0, selected, partial)
    return linear(packed, reader.head).index_select(0, selected)


def enough_rows(selected, total, minimum):
    present = set(selected.tolist())
    for i in range(total):
        if len(present) >= minimum:
            break
        present.add(i)
    return torch.tensor(sorted(present), dtype=selected.dtype, device=selected.device)


def local_rows(carrier, selected, total):
    offsets = torch.full((total,), -1, dtype=carrier.dtype, device=carrier.device)
    offsets[carrier] = torch.arange(len(carrier), device=carrier.device)
    return offsets[selected]


def timed(callback):
    torch.cuda.synchronize()
    begin = perf_counter()
    result = callback()
    torch.cuda.synchronize()
    return result, (perf_counter() - begin) * 1000


def layer_trace(reader, tensors, watch):
    log = []

    def capture(_, args, kwargs):
        values = args[0] if args else kwargs["hidden_states"]
        log.append(values.index_select(0, watch)[:, -1].detach().clone())

    hooks = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in reader.backbone.layers]
    try:
        result = hidden(reader, tensors)
    finally:
        for hook in hooks:
            hook.remove()
    return [*log, result.index_select(0, watch)]


def layer_probes(reader, tensors, watch):
    first = reader.backbone.layers[0]
    traces, hooks = {}, []

    def register(key):
        def record(value):
            traces[key] = value.index_select(0, watch).detach().clone()
        if key.endswith(".input"):
            path = key.removesuffix(".input")
            return first.get_submodule(path).register_forward_pre_hook(lambda _, args: record(args[0]))
        return first.get_submodule(key).register_forward_hook(lambda _, args, output: record(output))

    hooks.extend(register(key) for key in LAYER_PROBES)
    try:
        hidden(reader, tensors)
    finally:
        for hook in hooks:
            hook.remove()
    return traces


class Study:
    def __init__(self, reverse):
        self.orders = defaultdict(list)
        self.reverse = reverse

    def run_block(self, reader, front, suffixes, cached, alternate):
        tensors = padded([front + suffix for suffix in suffixes], reader.tokenizer.pad_token_id)
        total, width = tensors[0].shape
        needed = (qualification(cached) != qualification(alternate)) | (qualification_margin(alternate) <= NARROW)
        selected = torch.where(needed)[0]
        if not selected.numel():
            raise ValueError("The sparse selection must contain the original fallback trigger")
        compact_inputs = take(tensors, selected)
        sequence = ("full", "compact", "four_rows", "eight_rows")
        if self.reverse ^ bool(len(self.orders["full"]) % 2):
            sequence = sequence[::-1]
        states, outputs, samples, work = {}, {}, {}, {}
        for variant in sequence:
            if variant == "full":
                carrier = torch.arange(total, device=selected.device)
                scoped = tensors
            elif variant == "compact":
                carrier = selected
                scoped = compact_inputs
            else:
                carrier = enough_rows(selected, total, 4 if variant == "four_rows" else 8)
                scoped = take(tensors, carrier)
            def execute():
                last = hidden(reader, scoped)
                logits = linear(last, reader.head)
                return last, logits
            state, samples[variant] = timed(execute)
            states[variant], outputs[variant] = state
            work[variant] = len(carrier) * width
            if variant in ("four_rows", "eight_rows"):
                outputs[variant] = outputs[variant].index_select(0, local_rows(carrier, selected, total))
        # Swap the final head geometry holding the hidden vectors fixed.
        # Both crossovers measure the same rows at the same in-block slots.
        reference = outputs["full"]
        wanted = reference.index_select(0, selected)
        # Measure the implementable path without executing or charging the
        # compact head. Each sparse variant includes input rows gathered above.
        def head_shape():
            state = hidden(reader, compact_inputs)
            return preserve_head(reader, state, selected, total)
        outputs["head_shape"], samples["head_shape"] = timed(head_shape)
        work["head_shape"] = work["compact"]
        def guarded_head_shape():
            state = hidden(reader, compact_inputs)
            scores = preserve_head(reader, state, selected, total)
            native = linear(state, reader.head)
            if (qualification_margin(scores) <= BAND).any() or (qualification(scores) != qualification(native)).any():
                return reader.independent(front, suffixes)[0], True
            return scores, False
        (outputs["guarded_head_shape"], guard), samples["guarded_head_shape"] = timed(guarded_head_shape)
        work["guarded_head_shape"] = work["compact"] + (work["full"] if guard else 0)
        full_hidden_compact_head = linear(states["full"].index_select(0, selected), reader.head)
        compact_changes = qualification(outputs["compact"]) != qualification(wanted)
        if compact_changes.any():
            compact_order = torch.arange(len(selected), device=selected.device)
            old = layer_trace(reader, tensors, selected)
            new = layer_trace(reader, compact_inputs, compact_order)
            original_trace = layer_probes(reader, tensors, selected)
            compact_trace = layer_probes(reader, compact_inputs, compact_order)
            for row in torch.where(compact_changes)[0].tolist():
                divergence = next((i for i, (a, b) in enumerate(zip(old, new, strict=True))
                                   if not torch.equal(a[row], b[row])), len(old))
                self.orders["trace"].append(divergence)
                for key in LAYER_PROBES:
                    first, second = original_trace[key][row], compact_trace[key][row]
                    self.orders["probes"].append((key, not torch.equal(first, second),
                                                  not torch.equal(first[-1], second[-1])))
        results = {"full": reference}
        for variant in ARMS:
            if variant == "full":
                continue
            values = outputs[variant]
            if variant == "guarded_head_shape" and guard:
                results[variant] = values
            else:
                result = cached.clone()
                result.index_copy_(0, selected, values)
                results[variant] = result
        for variant, result in results.items():
            drift = qualification(result) != qualification(reference)
            self.orders[variant].append(dict(
                ms=samples[variant], tensor_slots=work[variant], selected_drift=int((drift & needed).sum()),
                omitted_drift=int((drift & ~needed).sum()),
                selected_score_drift=int((result.index_select(0, selected) != wanted).any(dim=1).sum()),
            ))
        self.orders["mechanism"].append(dict(
            selected=len(selected), candidate_rows=total,
            head_shape_changes_with_full_backbone=int(
                (qualification(full_hidden_compact_head) != qualification(wanted)).sum()),
            native_compact_decision_changes=int(compact_changes.sum()),
            compact_hidden_changes=int((states["compact"] != states["full"].index_select(0, selected)).any(dim=1).sum()),
            compact_selected_logits_linf=float((outputs["compact"].float()-wanted.float()).abs().amax()),
            full_prefix_fallback=bool(guard),
        ))
        self.block_outputs.append(([self.coordinates[tuple(item)] for item in suffixes], results))
        return reference, work["full"]

    @torch.inference_mode()
    def request(self, parser, text, clauses):
        self.block_outputs = []
        _, suffixes = shared_prompt(parser.readers["bf16"].tokenizer, text, clauses)
        self.coordinates = {tuple(suffix): i for i, suffix in enumerate(suffixes)}
        if len(self.coordinates) != len(clauses):
            raise ValueError("Unexpected identical clause targets in one request")
        for reader in parser.readers.values():
            reader.broadcast = True
            reader.forks = reader.prefix_bytes = 0
        full = evaluate(text, clauses, parser.readers, "consensus", recheck=self.run_block)
        labels = {arm: full["labels"].copy() for arm in ARMS}
        for coordinates, results in self.block_outputs:
            for arm, result in results.items():
                labels[arm][coordinates] = result.argmax(dim=1).cpu().numpy()
        return labels

    def summary(self):
        out = {}
        for variant in ARMS:
            reports = self.orders[variant]
            out[variant] = {field: sum(row[field] for row in reports) for field in
                            ("tensor_slots", "selected_drift", "omitted_drift", "selected_score_drift")}
            out[variant]["recheck_p50_p95_ms"] = (
                np.quantile([row["ms"] for row in reports], [.5, .95]).tolist() if reports else []
            )
        stats = self.orders["mechanism"]
        return dict(variants=out, rechecked_blocks=len(stats), mechanism={
            field: sum(row[field] for row in stats) for field in
            ("selected", "candidate_rows", "head_shape_changes_with_full_backbone", "native_compact_decision_changes",
             "compact_hidden_changes", "full_prefix_fallback")
        }, max_compact_logit_drift=max((row["compact_selected_logits_linf"] for row in stats), default=0.),
                    first_nonmatching_last_token_layer=dict(Counter(self.orders["trace"])),
                    first_layer_probes={key: {
                        "changed_rows_any_token": sum(any_drift for name, any_drift, _ in self.orders["probes"] if name == key),
                        "changed_rows_last_token": sum(last_drift for name, _, last_drift in self.orders["probes"] if name == key),
                    } for key in LAYER_PROBES if self.orders["probes"]})


def run(args):
    if (args.output.exists() or digest(args.queries) != DEVELOPMENT or digest(args.controls) != CONTROLS
            or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise ValueError("Require pinned projects, Slurm CUDA and new output")
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = json.loads(args.reference.read_bytes())
    parser = ComparisonParser(args.checkpoint)
    if parser.readers["bf16"].provenance["revision"] != reference["models"]["parser"]:
        raise ValueError("Changed model")
    public = json.loads(args.controls.read_bytes())
    development = rows(args.queries)
    if len(public) != 72 or len(development) != 80 or any(row["split"] != "development" for row in development):
        raise ValueError("Unexpected cohort")
    outcomes = {}
    for name, cohort in (("development", development), ("public", public)):
        experiment = Study(args.reverse)
        stats = {arm: Counter() for arm in ARMS}
        order = reversed(cohort) if args.reverse else cohort
        for row in order:
            text = row["text"]
            clauses = spans(text)
            labels = experiment.request(parser, text, clauses)
            qualified = labels["full"] == QUALIFICATION
            index, gold = target(text, clauses, row) if name == "public" else (None, None)
            for arm, values in labels.items():
                proposed = values == QUALIFICATION
                stats[arm].update(
                    requests=1, qualification_changes=bool(np.any(proposed != qualified)),
                    role_changes=int(np.count_nonzero(values != labels["full"])),
                    qualifications=int(proposed.sum()),
                )
                if index is not None:
                    stats[arm].update(
                        targets=1, target_correct=bool(values[index] == gold),
                        false_qualification=bool(proposed[index] and gold != QUALIFICATION),
                        missed_qualification=bool(not proposed[index] and gold == QUALIFICATION),
                    )
        outcomes[name] = experiment.summary() | {"output_counts": stats}
        print(name, outcomes[name]["rechecked_blocks"], flush=True)
    write(args.output, dict(
        outcomes=outcomes, queries_sha256=digest(args.queries), controls_sha256=digest(args.controls),
        model=parser.readers["bf16"].provenance["revision"], script_sha256=digest(Path(__file__)),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "controls", "checkpoint", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
