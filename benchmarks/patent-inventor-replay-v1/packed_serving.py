"""Size-matched grant-role indexes and a loopback HTTP search replay."""

import argparse
from collections import Counter
from dataclasses import dataclass
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import combinations
import json
from pathlib import Path
import random
import socket
import sqlite3
from struct import pack, unpack_from
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter_ns, sleep

from replay_roles import cohorts, exhaustive, stable_number


MODE_SHIFT = 29
OFFSET_MASK = (1 << MODE_SHIFT) - 1
WIDTHS = (0, 8, 12, 16)


@dataclass
class PackedIndex:
    directory: bytes
    payload: bytes

    @property
    def size(self):
        return len(self.directory) + len(self.payload)

    def get(self, grant):
        entry, next_entry = unpack_from("<II", self.directory, grant * 4)
        width = WIDTHS[entry >> MODE_SHIFT]
        start, end = entry & OFFSET_MASK, next_entry & OFFSET_MASK
        if not width:
            return width, ()
        result = tuple(
            (int.from_bytes(self.payload[bit // 8:bit // 8 + 3], "little") >> (bit % 8))
            & ((1 << width) - 1)
            for bit in range(start, end, width)
        )
        return width, result


def materialize(entries):
    words, payload, bit_offset = [], bytearray(), 0
    for width, codes in entries:
        if bit_offset > OFFSET_MASK:
            raise ValueError("Index exceeds directory range")
        words.append(pack("<I", (WIDTHS.index(width) << MODE_SHIFT) | bit_offset))
        for code in codes:
            start = bit_offset // 8
            value = code << (bit_offset % 8)
            bit_offset += width
            while len(payload) < (bit_offset + 7) // 8:
                payload.append(0)
            while value:
                payload[start] |= value & 255
                value >>= 8
                start += 1
    if bit_offset > OFFSET_MASK:
        raise ValueError("Index exceeds directory range")
    words.append(pack("<I", bit_offset))
    return PackedIndex(b"".join(words), bytes(payload))


def make_indexes(rows, priority, hashes):
    if len(hashes) > 65536:
        raise ValueError("The dense-ID control requires at most 65536 inventors")

    def fingerprints(width, roster):
        return tuple(sorted({hashes[actor] & ((1 << width) - 1) for actor in roster}))

    sketch = [(12, fingerprints(12, roster)) for roster in rows]
    small = [(8, fingerprints(8, roster)) for roster in rows]
    full = [(16, roster) for roster in rows]
    names = ("sketch12", "partial_exact", "mixed", "exact_full", "unindexed")
    entries = [sketch, [(0, ()) for _ in rows], small, full, [(0, ()) for _ in rows]]
    payload_cap = sum(width * len(codes) for width, codes in sketch)

    for selected in entries[1:3]:
        remaining = payload_cap - sum(width * len(codes) for width, codes in selected)
        for slot in priority:
            width, codes = selected[slot]
            extra = 16 * len(rows[slot]) - width * len(codes)
            if extra <= remaining:
                selected[slot] = full[slot]
                remaining -= extra

    indexes = dict(zip(names, map(materialize, entries)))
    for name, expected in zip(names, entries):
        assert all(indexes[name].get(slot) == entry for slot, entry in enumerate(expected))
    for name in ("partial_exact", "mixed"):
        assert indexes[name].size <= indexes["sketch12"].size
    return indexes


def query_pairs(groups):
    labels = sorted(label for label, rows in groups.items() if len(rows) >= 12)
    return [(a, b) for a, b in combinations(labels, 2) if a[:3] == b[:3]]


def setup(groups, path, prior_pairs=None):
    pairs = query_pairs(groups)
    reference = {pair: exhaustive((groups[pair[0]], groups[pair[1]])) for pair in pairs}
    workload = pairs if prior_pairs is None else prior_pairs
    counts = Counter(label for pair in workload for label in pair)
    names = sorted({person for rows in groups.values() for row in rows for person in row["inventors"]})
    ranks = {person: rank for rank, person in enumerate(names)}
    hashes = [stable_number("actor-v1", person) for person in names]
    grant_ids, rosters, postings, priority = [], [], {}, []
    for label, rows in sorted(groups.items()):
        group = []
        for rank, row in enumerate(rows, start=1):
            slot = len(grant_ids)
            grant = row["patent_number"]
            roster = tuple(sorted(ranks[person] for person in row["inventors"]))
            group.append(slot)
            grant_ids.append(grant)
            rosters.append(roster)
            priority.append((-counts[label], rank, grant, slot))
        postings[label] = group
    indexes = make_indexes(rosters, [slot for *_, slot in sorted(priority)], hashes)
    with sqlite3.connect(path) as store:
        store.execute("CREATE TABLE roster (slot INTEGER PRIMARY KEY, codes BLOB NOT NULL)")
        store.executemany("INSERT INTO roster VALUES (?, ?)", [
            (slot, pack(f"<{len(roster)}H", *roster)) for slot, roster in enumerate(rosters)
        ])
    common = (grant_ids, rosters, postings, names, hashes, reference)
    return indexes, common


class Search:
    def __init__(self, index, common, path, visible, budget, source_delay=0):
        self.index = index
        self.grants, rosters, self.postings, self.names, self.hashes, _ = common
        self.visible = {
            slot: roster for slot, roster in enumerate(rosters)
            if stable_number("availability-v1", self.grants[slot]) % 100 < visible
        }
        self.budget = budget
        self.source_delay = source_delay
        self.store = sqlite3.connect(path, check_same_thread=False)

    def request(self, labels):
        pools = [self.postings[label] for label in labels]
        decoded = [[self.index.get(slot) for slot in pool] for pool in pools]
        hidden_widths = {width for side in decoded for width, _ in side if width != 16}
        if 0 in hidden_widths or not hidden_widths:
            excluded = (set(), set())
        else:
            width = min(hidden_widths)
            mask = (1 << width) - 1
            fingerprints = [
                [{(self.hashes[code] if encoding == 16 else code) & mask for code in codes}
                 for encoding, codes in side]
                for side in decoded
            ]
            unions = [set().union(*side) for side in fingerprints]
            excluded = tuple(
                {slot for slot, codes in zip(pools[side], fingerprints[side])
                 if codes.isdisjoint(unions[1 - side])}
                for side in range(2)
            )

        known, checked = ({}, {}), ({}, {})
        rank_by_slot = {}
        remaining = [[], []]
        reads = 0

        def observe(lookup, side, slot, codes):
            for code in codes:
                previous = lookup[side].get(code)
                if previous is None or rank_by_slot[slot] < rank_by_slot[previous]:
                    lookup[side][code] = slot

        for side, slots in enumerate(pools):
            for rank, (slot, (width, codes)) in enumerate(zip(slots, decoded[side]), start=1):
                rank_by_slot[slot] = rank
                if slot in excluded[side]:
                    continue
                if slot in self.visible:
                    observe(checked, side, slot, self.visible[slot])
                    observe(known, side, slot, self.visible[slot])
                elif width == 16:
                    observe(known, side, slot, codes)
                else:
                    remaining[side].append(slot)

        def ranked(source):
            keys = source[0].keys() & source[1].keys()
            return sorted(keys, key=lambda code: (
                -sum(1_000_000 // (10 + rank_by_slot[source[side][code]]) for side in range(2)), code,
            ))[:5]

        def open_roster(slot, side):
            nonlocal reads
            if self.source_delay:
                sleep(self.source_delay)
            blob, = self.store.execute("SELECT codes FROM roster WHERE slot = ?", (slot,)).fetchone()
            codes = unpack_from(f"<{len(blob) // 2}H", blob)
            observe(known, side, slot, codes)
            observe(checked, side, slot, codes)
            reads += 1

        opened = set()
        positions = [0, 0]
        while reads < self.budget:
            options = []
            for order, actor in enumerate(ranked(known)):
                missing = [(known[side][actor], side) for side in range(2)
                           if checked[side].get(actor) != known[side][actor]]
                if missing and len(missing) <= self.budget - reads:
                    options.append((len(missing), order, missing))
            if options:
                _, _, missing = min(options)
                for slot, side in missing:
                    if slot not in opened:
                        open_roster(slot, side)
                        opened.add(slot)
                continue

            candidates = [(rank_by_slot[remaining[side][pos]], side, remaining[side][pos])
                          for side, pos in enumerate(positions) if pos < len(remaining[side])]
            if not candidates:
                break
            _, side, slot = min(candidates)
            open_roster(slot, side)
            positions[side] += 1

        answer = [
            [self.names[actor], self.grants[checked[0][actor]], self.grants[checked[1][actor]]]
            for actor in ranked(checked)
        ]
        return {"answer": answer, "reads": reads}


def replay_http(search, reference, rounds):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def do_POST(self):
            pair = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response = json.dumps(search.request(pair), separators=(",", ":")).encode()
            self.send_response_only(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    port = server.server_port

    def replay_client(sequence):
        connection = HTTPConnection("127.0.0.1", port, timeout=60)
        out = []
        try:
            for pair in sequence:
                data = json.dumps(pair).encode()
                start = perf_counter_ns()
                connection.request("POST", "/", data)
                response = connection.getresponse()
                result = json.loads(response.read())
                latency_ms = (perf_counter_ns() - start) / 1e6
                if response.status != 200:
                    raise RuntimeError(response.status)
                out.append((pair, result, latency_ms))
        finally:
            connection.close()
        return out

    def summary(values):
        times = sorted(item[2] for item in values)
        return {
            "requests": len(times),
            "p50_ms": round(times[(len(times) - 1) * 50 // 100], 4),
            "p95_ms": round(times[(len(times) - 1) * 95 // 100], 4),
        }

    try:
        pairs = list(reference)
        replay_client(pairs)
        expected = None
        timings = []
        for repeat in range(rounds):
            sequence = pairs.copy()
            random.Random(3721 + repeat).shuffle(sequence)
            observed = replay_client(sequence)
            current = {pair: item for pair, item, _ in observed}
            if len(current) != len(reference) or (expected is not None and current != expected):
                raise AssertionError("Repeated endpoint output changed")
            expected = current
            timings.append(summary(observed))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        search.store.close()

    exact = hits = total_reads = 0
    for pair, result in expected.items():
        predicted = tuple(map(tuple, result["answer"]))
        gold = reference[pair]
        exact += predicted == gold
        hits += len({row[0] for row in predicted} & {row[0] for row in gold})
        total_reads += result["reads"]
    return {"exact": exact, "hits": hits, "reads": total_reads, "rounds": timings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rosters", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-delay", action="store_true", help="Run the held-out source-cost sweep")
    parser.add_argument("--development-prior", action="store_true", help="Use earlier query templates for held-out admission")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rosters.read_text().splitlines()]
    args.scratch.mkdir(exist_ok=True)
    output = {}
    with TemporaryDirectory(prefix="patent-rosters-", dir=args.scratch) as directory:
        groups_by_split = cohorts(rows)
        earlier = query_pairs(groups_by_split["development"]) if args.development_prior else None
        for split, groups in groups_by_split.items():
            if (args.source_delay or args.development_prior) and split != "held_out":
                continue
            database = Path(directory) / f"{split}.sqlite"
            indexes, common = setup(groups, database, earlier)
            output[split] = {
                "total_queries": len(common[-1]),
                "reference_slots": sum(map(len, common[-1].values())),
                "packed_index_bytes": {name: index.size for name, index in indexes.items()},
                "panels": {},
            }
            if args.source_delay:
                configs = [
                    ("unindexed_12", "unindexed", 12),
                    ("unindexed_exhaustive", "unindexed", sum(map(len, groups.values()))),
                    ("sketch12", "sketch12", 12),
                    ("partial_exact", "partial_exact", 12),
                    ("mixed", "mixed", 12),
                    ("exact_full", "exact_full", 12),
                ]
                for delay_ms in (0, 0.2, 2):
                    output[split]["panels"][f"{delay_ms}ms_added_per_read"] = {
                        name: replay_http(
                            Search(indexes[index], common, database, 70, budget, delay_ms / 1000), common[-1], 3,
                        )
                        for name, index, budget in configs
                    }
            else:
                for visible in (40, 70, 90):
                    for budget in (4, 12):
                        key = f"{visible}pct_{budget}reads"
                        output[split]["panels"][key] = {
                            name: replay_http(Search(index, common, database, visible, budget), common[-1], 3)
                            for name, index in indexes.items()
                        }

    args.output.write_text(json.dumps(output, indent=2) + "\n")
