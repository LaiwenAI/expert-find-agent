"""Share the request across all clause rows during independent parser fallback."""

import argparse
from pathlib import Path

from experiments.retrieval import enterprise_clause_recheck as clause
from experiments.retrieval.enterprise_cross_precision import WIDE, disagreements, qualification_gap


ARMS = ("whole_block", "packed_block", "packed_guard")


class PackedParser(clause.ClauseParser):
    def recheck(self, reader, front, suffixes, first, second):
        if self.mode == "whole_block":
            return super().recheck(reader, front, suffixes, first, second)
        if second is None:
            raise ValueError("Expected two cached precision checks before recheck")
        packed, slots = reader.packed(front, suffixes)
        self.run_counts.update(
            fallback_rows=len(suffixes), original_block_rows=len(suffixes), fallback_slots=slots,
            packed_rechecks=1)
        if self.mode == "packed_guard" and (qualification_gap(packed) <= WIDE
                                              or disagreements(packed, first)
                                              or disagreements(packed, second)):
            packed, dense_slots = reader.independent(front, suffixes)
            self.run_counts.update(fallback_slots=dense_slots, canonical_rechecks=1)
            slots += dense_slots
        return packed, slots


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "metadata", "models", "control_file", "output"):
        cli.add_argument(name, type=Path)
    clause.run(**vars(cli.parse_args()), parser_class=PackedParser, arms=ARMS, script_path=Path(__file__))
