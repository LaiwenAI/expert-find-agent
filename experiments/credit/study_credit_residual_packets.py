"""Test exact requirement retention against span and layout counterfactuals."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from expert_search import requirement_packets as packet_builder
from expert_search.requirement_packets import Span, clauses, form_packet
from experiments.credit.study_credit_scope_gate import CASES, SURFACE
from experiments.followir.core17_reranking import digest, write_json


def probe(kind, base, extra):
    role = f"The candidate {base}."
    mandatory = f"The candidate also must meet this condition: they {extra}."
    optional = f"This additional condition is optional: they {extra}."
    if kind == "standalone":
        return role, role
    if kind == "mandate_trailing":
        return f"{role} {mandatory}", role
    if kind == "mandate_leading":
        return f"{mandatory}\n{role}", role
    if kind == "mandate_attached":
        return f"The candidate {base} and additionally {extra}.", None
    if kind == "explicit_preference":
        return f"{role}\n{optional}", role
    if kind == "unrecognized_preference":
        return f"{role} A nice-to-have would be that they {extra}.", role
    if kind == "mixed":
        return f"{optional}\n{role} They must also provide a verifiable reference.", role
    if kind == "mixed_same_clause":
        return f"{role} Optional: they {extra}, but external proof is required.", role
    raise ValueError("Unknown control kind")


KINDS = ("standalone", "mandate_trailing", "mandate_leading", "mandate_attached",
         "explicit_preference", "unrecognized_preference", "mixed", "mixed_same_clause")


def inspect(controls):
    results, time_us = defaultdict(Counter), defaultdict(list)
    for _, base, extra in CASES:
        for kind in KINDS:
            text, role = probe(kind, base, extra)
            offset = text.index(role) if role else None
            anchor = Span(offset, offset + len(role)) if role else None
            start = perf_counter_ns()
            packet = form_packet(text, anchor)
            time_us[kind].append((perf_counter_ns() - start) / 1e3)
            exact_partition(text, packet)
            results[kind].update(
                cases=1, role_only=int(packet.role_only),
                raw_regex_role_only=int(SURFACE.search(text) is None),
                required_text_pending=int(bool(packet.unresolved)),
                optional_spans=len(packet.optional),
            )

    public = Counter()
    spans_with_role = 0
    for row in controls:
        text = row["text"]
        # No verified role-only anchor accompanies these full enterprise requests.
        packet = form_packet(text, None)
        exact_partition(text, packet)
        public.update(documents=1, unresolved_spans=len(packet.unresolved), optional_spans=len(packet.optional))
        spans_with_role += packet.role_only
        for quote, label in row["gold"]:
            needle = text.find(quote)
            if needle < 0:
                raise ValueError("Missing source anchor")
            if any(part.start <= needle and needle + len(quote) <= part.end for part in packet.unresolved):
                public.update(annotated_spans_retained=1, **{f"retained_{label}": 1})
            public.update(annotated_spans=1, **{f"total_{label}": 1})
    public["role_only_documents"] = spans_with_role
    return dict(contrast={k: dict(c, p95_us=float(np.quantile(time_us[k], .95))) for k, c in results.items()},
                public_controls=public)


def exact_partition(text, packet):
    role = () if packet.role is None else (packet.role,)
    all_parts = sorted((*role, *packet.optional, *packet.unresolved), key=lambda part: part.start)
    if tuple(all_parts) != clauses(text):
        raise ValueError("Lost a source clause")
    previous = 0
    for part in all_parts:
        if text[previous:part.start].strip():
            raise ValueError("Lost a non-whitespace source interval")
        previous = part.end
    if text[previous:].strip():
        raise ValueError("Lost the final source interval")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("public_controls", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    controls = json.loads(args.public_controls.read_bytes())
    write_json(args.output, dict(script_sha256=digest(Path(__file__)),
                                 packet_builder_sha256=digest(Path(packet_builder.__file__)), controls_sha256=digest(args.public_controls),
                                 **inspect(controls)))
