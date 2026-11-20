"""Calibrate a no-extra-model guard for packed enterprise qualification rechecks."""

import argparse
from functools import partial
import json
from pathlib import Path

import torch

from experiments.retrieval import enterprise_clause_recheck as clause
from experiments.retrieval.enterprise_packed_clause_recheck import PackedParser
from experiments.retrieval.enterprise_qualification_student import synthetic
from experiments.retrieval.enterprise_routed_review import write


ARMS = ("whole_block", "packed_block", "packed_consensus")


def support(first, second, packed):
    """Smallest, across clauses, median margin supporting the packed decision."""
    margin = clause.qualification_lead(packed)
    direction = torch.where(margin > 0, 1., -1.)
    cached_support = torch.stack((
        margin.abs(), direction * clause.qualification_lead(first),
        direction * clause.qualification_lead(second),
    ))
    return cached_support.median(dim=0).values.amin().item()


class ConsensusParser(PackedParser):
    def __init__(self, checkpoint, calibration_report):
        super().__init__(checkpoint)
        self.calibration = []
        self.mode = "calibrate"
        contrasts = [case for dataset in synthetic().values() for case in dataset]
        panels = [[case["text"] for i, case in enumerate(contrasts) if (i // 2) % 2 == language]
                  for language in (0, 1)]
        composites = ["\n\n".join(panel[(i + 7 * offset) % len(panel)]
                                    for offset in range(4, 4 + (i % 3 + 1) * 4))
                      for panel in panels for i in range(36)]
        for text in [case["text"] for case in contrasts] + composites:
            self.evaluate(text, clause.spans(text), "broadcast_prefix", 2.)
        harmful = [margin for margin, changed in self.calibration if changed]
        self.cutoff = max(0., max(harmful)) if harmful else None
        calibration_report.update(
            synthetic_requests=len(contrasts), composite_requests=len(composites),
            triggered_blocks=len(self.calibration),
            harmful_packed_blocks=len(harmful),
            cutoff=self.cutoff,
            blocks_retained=sum(self.cutoff is not None and margin > self.cutoff
                                for margin, _ in self.calibration),
        )
        print(f"Calibration complete: {calibration_report}", flush=True)
        del self.calibration

    def recheck(self, reader, front, suffixes, first, second):
        if self.mode in ("whole_block", "packed_block"):
            return super().recheck(reader, front, suffixes, first, second)

        if second is None:
            raise ValueError("Expected two cached precision checks")
        # An uninformative calibration leaves every canonical recheck enabled.
        if self.mode == "packed_consensus" and self.cutoff is None:
            self.run_counts.update(canonical_rechecks=1)
            logits, slots = reader.independent(front, suffixes)
            self.run_counts.update(fallback_rows=len(suffixes), original_block_rows=len(suffixes),
                                   fallback_slots=slots)
            return logits, slots

        packed, slots = reader.packed(front, suffixes)
        margin = support(first, second, packed)
        self.run_counts.update(fallback_rows=len(suffixes), original_block_rows=len(suffixes),
                               fallback_slots=slots, packed_rechecks=1)
        if self.mode == "calibrate" or margin <= self.cutoff:
            original, dense_slots = reader.independent(front, suffixes)
            if self.mode == "calibrate":
                membership = packed.argmax(dim=1) == clause.QUALIFICATION
                target = original.argmax(dim=1) == clause.QUALIFICATION
                self.calibration.append((margin, bool((membership != target).any().item())))
            self.run_counts.update(fallback_slots=dense_slots, canonical_rechecks=1)
            return original, slots + dense_slots
        self.run_counts.update(accepted_packed=1)
        return packed, slots


def run(queries, corpus, roles, vectors, metadata, models, control_file, output):
    clause.search.role.require_cuda(output)
    calibration = {}
    intermediate = output.with_suffix(".intermediate.json")
    if intermediate.exists():
        raise FileExistsError(intermediate)
    try:
        clause.run(queries, corpus, roles, vectors, metadata, models, control_file, intermediate,
                   parser_class=partial(ConsensusParser, calibration_report=calibration),
                   arms=ARMS, script_path=Path(__file__))
        report = json.loads(intermediate.read_bytes())
        write(output, dict(report, calibration=calibration))
    finally:
        intermediate.unlink(missing_ok=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "metadata", "models", "control_file", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
