"""Recover owner postings from pinned publisher note bytes on demand."""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from functools import partial
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from experiments.credit import legacy_cascade_transfer as capture
from experiments.credit import legacy_contribution_text as source
from experiments.followir.core17_reranking import digest, write_json

TASKS = tuple(sorted(source.PHRASES))
STATES = (("no_cache", 0), ("half_cache", 50), ("resident", 100))
REPEATS = 5


@dataclass(frozen=True)
class Record:
    doi: str
    file: Path
    offsets: dict
    complete: bool
    admissible_owners: frozenset
    reference: dict


def segment(line, admissible):
    heading, sep, tail = line.decode().partition(":")
    label = source.LABELS.get(heading.strip().casefold())
    if not sep or label is None:
        return None
    names = tail.strip().removesuffix(".").split()
    if not names or not all(source.INITIAL.fullmatch(name) for name in names):
        return None
    valid = frozenset(names) & admissible
    return label, valid, len(valid) == len(set(names))


def prepare(directory, scratch):
    files = json.loads((directory / "capture.json").read_bytes())["checksums"]
    skipped = Counter()
    records = []
    for ordinal, (doi, checksum) in enumerate(sorted(files.items())):
        raw = capture.checked_xml(directory, doi, checksum)
        document = source.note(raw, doi)
        if document is None:
            skipped["unsupported_license_or_missing_note"] += 1
            continue
        _, normalized, labels, _ = document
        meta = ET.fromstring(raw).find("./front/article-meta")
        names = meta.findall("./contrib-group/contrib[@contrib-type='author']/name")
        initials = [
            "".join(part[0] for part in source.NAME_PARTS.findall(
                f"{name.findtext('given-names', '')} {name.findtext('surname', '')}"
            )).upper()
            for name in names
        ]
        counts = Counter(initials)
        admissible = frozenset(k for k, n in counts.items() if n == 1 and source.INITIAL.fullmatch(k))
        chunks = [line.encode() for line in source.SENTENCE.split(normalized)]
        joined = b" ".join(chunks)
        offsets, claimed, duplicated = {}, {}, set()
        complete = len(admissible) == len(names) and bool(chunks)
        position = 0
        for line in chunks:
            parsed = segment(line, admissible)
            if parsed is None:
                complete = False
            else:
                role, owners, resolved = parsed
                complete &= resolved
                if role in claimed:
                    complete = False
                    duplicated.add(role)
                else:
                    claimed[role] = owners
                    offsets[role] = (position, len(line))
            position += len(line) + 1
        for role in duplicated:
            claimed.pop(role)
            offsets.pop(role)
        if any(owners != labels.get(role) for role, owners in claimed.items()):
            raise ValueError("Compiled owners disagree with contribution reference")
        if complete and (claimed != labels or not claimed):
            raise ValueError("Complete note disagrees with publisher parser")
        path = scratch / f"{ordinal}.txt"
        path.write_bytes(joined)
        records.append(Record(doi, path, offsets, complete, admissible, claimed))
    if not records:
        raise ValueError("No licensed notes")
    return records, dict(skipped)


def confirmed(postings, pair, complete):
    verified = [postings.get(role) for role in pair]
    if all(value is not None for value in verified):
        overlap = tuple(sorted(verified[0] & verified[1]))
        if overlap:
            return "supported", overlap
    if complete:
        return "no_declared_match", ()
    return "unresolved", ()


def receive_whole(record, pair, warm):
    if record.complete and any(role not in record.offsets for role in pair):
        return ("no_declared_match", ()), 0, 0
    if all(role in warm or role not in record.offsets for role in pair):
        return confirmed(warm, pair, record.complete), 0, 0
    raw = record.file.read_bytes()
    readings = {}
    for line in source.SENTENCE.split(raw.decode()):
        parsed = segment(line.encode(), record.admissible_owners)
        if parsed is not None and parsed[0] in record.offsets:
            readings[parsed[0]] = parsed[1]
    return confirmed(readings, pair, record.complete), len(raw), 1


def receive_spans(record, pair, warm, *, coalesce=False):
    pending = [role for role in pair if role not in warm and role in record.offsets]
    if record.complete and any(role not in record.offsets for role in pair):
        return ("no_declared_match", ()), 0, 0
    if not pending:
        return confirmed(warm, pair, record.complete), 0, 0
    locations = [record.offsets[role] for role in pending]
    postings = dict(warm)
    with record.file.open("rb", buffering=0) as stream:
        if coalesce:
            begin = min(offset for offset, _ in locations)
            end = max(offset + length for offset, length in locations)
            stream.seek(begin)
            block = stream.read(end - begin)
            payloads = [block[offset - begin:offset - begin + length] for offset, length in locations]
            consumed, reads = len(block), 1
        else:
            payloads = []
            for offset, length in locations:
                stream.seek(offset)
                payloads.append(stream.read(length))
            consumed, reads = sum(map(len, payloads)), len(payloads)
    for role, (_, length), chunk in zip(pending, locations, payloads, strict=True):
        parsed = segment(chunk, record.admissible_owners)
        if parsed is None or parsed[0] != role or len(chunk) != length:
            raise ValueError("Missing or changed source span")
        postings[role] = parsed[1]
    return confirmed(postings, pair, record.complete), consumed, reads


def run(folder, output):
    if not os.environ.get("SLURM_JOB_ID") or output.exists():
        raise ValueError("Run on a Slurm compute node with a fresh output")
    rows = {}
    with TemporaryDirectory(dir=output.parent, prefix="note-spans-") as scratch:
        started = perf_counter()
        records, skipped = prepare(folder, Path(scratch))
        build_seconds = perf_counter() - started
        cases = [(record, pair) for record in records for pair in combinations(TASKS, 2)]
        for state, percent in STATES:
            measured = (("whole", receive_whole), ("spans", receive_spans),
                        ("coalesced", partial(receive_spans, coalesce=True)))
            counts = {name: Counter() for name, _ in measured}
            samples = {name: [] for name in counts}
            verdicts = Counter()
            for i, (record, pair) in enumerate(cases):
                warm = {role: owners for role, owners in record.reference.items()
                        if int.from_bytes(source.stable(record.doi, role)[:4], "big") % 100 < percent}
                reference = confirmed(record.reference, pair, record.complete)
                verdicts[reference[0]] += 1
                for repeat in range(REPEATS):
                    order = (i + repeat) % len(measured)
                    for name, method in measured[order:] + measured[:order]:
                        now = perf_counter()
                        verdict, read_bytes, reads = method(record, pair, warm)
                        elapsed = (perf_counter() - now) * 1000
                        if verdict != reference:
                            raise ValueError(f"Mismatch in {state} {name}, {record.doi}, {pair}")
                        samples[name].append(elapsed)
                        if not repeat:
                            counts[name].update(bytes=read_bytes, reads=reads, opened=int(read_bytes > 0))
            rows[state] = dict(
                verdicts=dict(verdicts),
                methods={name: dict(**work, p95_ms=float(np.quantile(samples[name], .95)),
                                    mean_ms=float(np.mean(samples[name]))) for name, work in counts.items()},
            )
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], code_sha256=digest(Path(__file__)),
        capture_sha256=digest(folder / "capture.json"), notes=len(records),
        complete_notes=sum(record.complete for record in records), omitted=skipped,
        queries=len(cases), timing_repeats=REPEATS, build_seconds=build_seconds, states=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
