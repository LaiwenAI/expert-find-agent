"""Test projection-local controls for sparse qualification recheck stability."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import linear

from experiments.retrieval import enterprise_fallback_batch_geometry as base
from experiments.retrieval.enterprise_isolated_parsing import blocks

ARMS = ("full", "compact", "reduction_off", "first_qkv", "all_qkv")
PUBLIC_TAIL = (
    "Administrative context\n"
    "Use the supplied page numbering throughout the document. "
    "The shared register contains historical meeting notes, drawing revisions, summary tables, and records of earlier decisions. "
    "A working index was prepared to help readers distinguish the source document, the review reference, and the archived record. "
    "All cross-references should point to the edition named in the index. "
    "The correspondence register has separate entries for submitted materials, copied attachments, and notes from routine discussion. "
    "Entries in the correspondence register use a fixed sequence and retain the dates printed on the original documents. "
    "Readers can identify a duplicated attachment by comparing its reference number with the index before consulting the record. "
    "The archive is divided into collections for routine correspondence, sample forms, and published guidance. "
    "The archive summary lists the document title, the edition, the date printed on the cover, and the folder used for storage. "
    "The index is intended to make older records easier to find during a review. "
    "The same numbering applies in the printed edition and the electronic copy. "
    "Filenames preserve the edition label and avoid punctuation that could prevent a string match in other systems. "
    "This administrative appendix describes the format of the record and the arrangement of the archive. "
    "Readers can use the document title and edition when reporting a broken cross-reference. "
    "Historical entries retain their original headings, including headings that differ from the current template. "
    "A list of routine corrections appears at the back of the document. "
    "The archive keeps the corrected page beside its earlier version so readers can locate the exact wording under discussion. "
    "Every copy carries a common publication label. "
    "For navigation, use the heading shown in the supplied table of contents."
)


@contextmanager
def without_reduced_precision_reduction():
    matmul = torch.backends.cuda.matmul
    original = (matmul.allow_bf16_reduced_precision_reduction,
                matmul.allow_bf16_reduced_precision_reduction_split_k)
    matmul.allow_bf16_reduced_precision_reduction = (False, original[1])
    try:
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = original


def wrap_projection(old, selected, total):
    def project(value):
        if value.ndim != 3 or value.shape[0] != len(selected):
            raise ValueError("Expected the compact QKV input geometry")
        carrier = value.new_zeros((total, *value.shape[1:]))
        carrier.index_copy_(0, selected, value)
        return old(carrier).index_select(0, selected)
    return project


@contextmanager
def qkv_geometry(reader, selected, total, all_layers):
    layers = reader.backbone.layers
    targets = [getattr(layer.self_attn, name)
               for layer in (layers if all_layers else layers[:1])
               for name in ("q_proj", "k_proj", "v_proj")]
    original = [module.forward for module in targets]
    try:
        for module, forward in zip(targets, original, strict=True):
            module.forward = wrap_projection(forward, selected, total)
        yield
    finally:
        for module, forward in zip(targets, original, strict=True):
            module.forward = forward


class Probe:
    def __init__(self, reverse):
        self.reverse = reverse
        self.values = defaultdict(list)
        self.blocks = 0

    def block(self, reader, front, suffixes, selected, cached=None):
        inputs = base.padded([front + suffix for suffix in suffixes], reader.tokenizer.pad_token_id)
        full_rows, width = inputs[0].shape
        compact = base.take(inputs, selected)
        turn = self.blocks % len(ARMS)
        cycle = ARMS[::-1] if self.reverse else ARMS
        cycle = cycle[turn:] + cycle[:turn]
        self.blocks += 1
        outputs = {}
        for arm in cycle:
            def execute():
                context = nullcontext()
                if arm == "reduction_off":
                    context = without_reduced_precision_reduction()
                elif arm in ("first_qkv", "all_qkv"):
                    context = qkv_geometry(reader, selected, full_rows, arm == "all_qkv")
                with context:
                    last = base.hidden(reader, inputs if arm == "full" else compact)
                    return last, linear(last, reader.head)
            (state, logits), ms = base.timed(execute)
            outputs[arm] = (state, logits, ms)
        reference = outputs["full"][1]
        primary = outputs["full"][0].index_select(0, selected)
        canonical = reference.index_select(0, selected)
        observed = {}
        for arm, (last, logits, ms) in outputs.items():
            if arm == "full":
                last, logits = last.index_select(0, selected), canonical
            difference = base.qualification(logits) != base.qualification(canonical)
            self.values[arm].append(dict(
                rows=len(selected), slots=(full_rows if arm == "full" else len(selected))*width,
                qualification_changes=int(difference.sum()),
                role_changes=int((logits.argmax(1) != canonical.argmax(1)).sum()),
                logit_changes=int((logits != canonical).any(dim=1).sum()),
                hidden_changes=int((last != primary).any(dim=1).sum()),
                max_abs_logit_delta=float((logits.float()-canonical.float()).abs().max()),
                ms=ms,
            ))
            if cached is not None:
                merged = (reference.clone() if arm == "full" else cached.clone())
                merged.index_copy_(0, selected, logits)
                observed[arm] = merged.argmax(1).cpu().numpy()
            else:
                observed[arm] = logits.argmax(1).cpu().numpy()
        return reference, observed, inputs[0].numel()

    def summarize(self):
        out = {}
        for arm in ARMS:
            rows = self.values[arm]
            fields = ("rows", "slots", "qualification_changes", "role_changes", "logit_changes", "hidden_changes")
            out[arm] = {key: sum(row[key] for row in rows) for key in fields}
            out[arm]["max_abs_logit_delta"] = max((row["max_abs_logit_delta"] for row in rows), default=0.)
            out[arm]["recheck_p50_p95_ms"] = np.quantile(
                [row["ms"] for row in rows], [.5,.95]).tolist() if rows else []
        return dict(blocks=self.blocks, arms=out)


def parse(parser, text, clauses, probe):
    reader = parser.readers["bf16"]
    front, suffixes = base.shared_prompt(reader.tokenizer, text, clauses)
    positions = {tuple(suffix): i for i, suffix in enumerate(suffixes)}
    if len(positions) != len(clauses):
        raise ValueError("Expected distinct suffixes")
    compared = []
    for item in parser.readers.values():
        item.broadcast = True
        item.forks = item.prefix_bytes = 0

    def recheck(active, prefix, items, cached, alternate):
        if active is not reader:
            raise ValueError("Wrong recheck dtype")
        choice = (base.qualification(cached) != base.qualification(alternate)) | (
            base.qualification_margin(alternate) <= base.NARROW
        )
        selected = torch.where(choice)[0]
        if not selected.numel():
            raise ValueError("Missing recheck rows")
        canonical, results, slots = probe.block(reader, prefix, items, selected, cached)
        compared.append(([positions[tuple(item)] for item in items], results))
        return canonical, slots

    reference = base.evaluate(text, clauses, parser.readers, "consensus", recheck=recheck)["labels"]
    labels = {arm: reference.copy() for arm in ARMS}
    for positions_, options in compared:
        for arm, values in options.items():
            labels[arm][positions_] = values
    return labels, front, suffixes


def score_labels(aggregate, labels, target=None, gold=None):
    canonical = labels["full"]
    for arm, values in labels.items():
        candidate = values == base.QUALIFICATION
        comparison = canonical == base.QUALIFICATION
        aggregate[arm].update(requests=1, qualification_changes=bool(np.any(candidate != comparison)),
                              role_changes=int(np.count_nonzero(values != canonical)))
        if target is not None:
            aggregate[arm].update(targets=1, correct=bool(values[target] == gold),
                                  missed_qualification=bool(not candidate[target] and gold == base.QUALIFICATION),
                                  false_qualification=bool(candidate[target] and gold != base.QUALIFICATION))


@torch.inference_mode()
def run(args):
    if (args.output.exists() or base.digest(args.queries) != base.DEVELOPMENT
            or base.digest(args.controls) != base.CONTROLS
            or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("Require the pinned cohorts, GPU Slurm and fresh output")
    torch.backends.cuda.matmul.allow_tf32 = False
    parser = base.ComparisonParser(args.checkpoint)
    control = json.loads(args.reference.read_bytes())
    if parser.readers["bf16"].provenance["revision"] != control["models"]["parser"]:
        raise ValueError("Changed parser checkpoint")
    requests = base.rows(args.queries)
    public = [item for item in json.loads(args.controls.read_bytes()) if item["language"] == "en"]
    if len(requests) != 80 or len(public) != 36 or any(row["split"] != "development" for row in requests):
        raise ValueError("Unexpected cohort")
    probes = {}
    for name, source in (("development", requests), ("expanded_public", public)):
        natural = Probe(args.reverse)
        forced = Probe(args.reverse)
        outputs = dict(natural=defaultdict(Counter), forced=defaultdict(Counter))
        for row in (reversed(source) if args.reverse else source):
            text = row["text"] + ("\n\n" + PUBLIC_TAIL if name == "expanded_public" else "")
            segments = base.spans(text)
            decisions, front, suffixes = parse(parser, text, segments, natural)
            index, gold = base.target(text, segments, row) if name == "expanded_public" else (None, None)
            score_labels(outputs["natural"], decisions, index, gold)
            if index is not None:
                for batch in blocks(front, suffixes):
                    positions, selected = zip(*batch, strict=True)
                    if index not in positions:
                        continue
                    local = torch.tensor([positions.index(index)], device="cuda")
                    _, measurements, _ = forced.block(parser.readers["bf16"], front, selected, local)
                    score_labels(outputs["forced"], measurements, 0, gold)
                    break
        probes[name] = dict(natural=natural.summarize() | {"output": outputs["natural"]},
                            forced=forced.summarize() | {"output": outputs["forced"]})
        print(name, natural.blocks, forced.blocks, flush=True)
    base.write(args.output, dict(
        cohorts=probes, model=control["models"]["parser"], public_extension=PUBLIC_TAIL,
        script_sha256=base.digest(Path(__file__)), query_sha256=base.digest(args.queries),
        controls_sha256=base.digest(args.controls), job=os.environ["SLURM_JOB_ID"],
        reverse=args.reverse, gpu=torch.cuda.get_device_name(), pytorch=torch.__version__,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("queries", "controls", "checkpoint", "reference", "output"):
        cli.add_argument(field, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
