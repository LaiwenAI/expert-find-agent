"""Stream a time-frozen SE-PEF expert panel from the official Zip archive."""

import argparse
from collections import Counter
from hashlib import sha256
import heapq
import io
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


URL = "https://zenodo.org/records/8332748/files/SE-PEF.zip?download=1"
SIZE = 4_336_948_003
WINDOW = 16 * 1024 * 1024
PREFIX = "SE-PEF/SE-PEF/"
SNAPSHOT = 1577836799  # 2019-12-31 23:59:59 UTC, upstream training boundary.
TEST_START = 1609459199  # 2020-12-31 23:59:59 UTC, upstream test boundary.
SAMPLE = {"val": 256, "test": 512}
ANSWERS_PER_EXPERT = 8
SALT = "sepef-public-temporal-v1"


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.start = -1
        self.buffer = b""
        self.bytes_transferred = 0

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = {0: 0, 1: self.position, 2: SIZE}[whence] + offset
        if self.position < 0:
            raise OSError("Negative file position")
        return self.position

    def read(self, length=-1):
        size = min(length if length >= 0 else SIZE, SIZE - self.position)
        parts = []
        while size > 0:
            if not self.start <= self.position < self.start + len(self.buffer):
                self.start = self.position
                end = min(SIZE, self.start + WINDOW) - 1
                request = Request(URL, headers={"Range": f"bytes={self.start}-{end}",
                                                "Accept-Encoding": "identity"})
                with urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {self.start}-{end}/{SIZE}":
                        raise IOError("Official archive range response differs")
                    self.buffer = response.read(end - self.start + 2)
                if len(self.buffer) != end - self.start + 1:
                    raise IOError("Incomplete official archive range")
                self.bytes_transferred += len(self.buffer)
            offset = self.position - self.start
            take = min(len(self.buffer) - offset, size)
            parts.append(self.buffer[offset:offset + take])
            self.position += take
            size -= take
        return b"".join(parts)


def digest(data):
    return sha256(data).hexdigest()


def load_experts(archive):
    name = PREFIX + "expert_test_data.json"
    raw = archive.read(name)
    data = json.loads(raw)
    profiles = {}
    for author, item in data.items():
        history = list(zip(item["answer_timestamps"], item["answers"], strict=True))
        if any(not isinstance(stamp, int) or stamp >= SNAPSHOT for stamp, _ in history):
            raise ValueError("Author history crosses the official profile cutoff")
        if history:
            profiles[str(author)] = heapq.nlargest(ANSWERS_PER_EXPERT, history)
    all_answers = [answer for history in profiles.values() for _, answer in history]
    if len(set(all_answers)) != len(all_answers):
        raise ValueError("An answer is duplicated across author profiles")
    return profiles, {name: digest(raw)}


def query_candidates(archive, split, authors):
    qname, source = PREFIX + f"{split}/qrels.json", PREFIX + f"{split}/data.jsonl"
    raw_qrels = archive.read(qname)
    qrels = json.loads(raw_qrels)
    seen, eligible, known_authors = set(), 0, Counter()
    raw_digest = sha256()

    def process(file):
        nonlocal eligible
        for raw in file:
            raw_digest.update(raw)
            row = json.loads(raw)
            query, stamp = str(row["id"]), row["timestamp"]
            if query in seen or not isinstance(stamp, int):
                raise ValueError("Invalid or repeated query identity")
            seen.add(query)
            if (split == "val" and not SNAPSHOT <= stamp < TEST_START) or (split == "test" and stamp < TEST_START):
                raise ValueError("Query falls outside its official temporal split")
            labels = sorted(set(map(str, row["expert_ids"])))
            if labels != sorted(qrels[query]):
                raise ValueError("Released question labels differ from its qrels")
            if not labels or not set(labels) <= authors or not row["text"].strip():
                continue
            eligible += 1
            community = query.split("_", 1)[0]
            known_authors[community] += 1
            yield dict(id=query, timestamp=stamp, text=row["text"], expert_ids=labels, community=community)

    with archive.open(source) as file:
        panel = heapq.nsmallest(SAMPLE[split], process(file), key=lambda row: digest(f"{SALT}:{row['id']}".encode()))
    if seen != qrels.keys() or len(panel) != SAMPLE[split]:
        raise ValueError("Released questions and qrels have different inventories or sample shortage")
    return panel, seen, dict(released=len(seen), eligible=eligible, communities=dict(known_authors),
                             selected_communities=dict(Counter(row["community"] for row in panel))), {
        qname: digest(raw_qrels), source: raw_digest.hexdigest()
    }


def read_answers(archive, wanted, output):
    name = PREFIX + "answer_collection.json"
    found = set()
    raw_digest = sha256()
    with archive.open(name) as stream, output.open("x") as writer:
        first = stream.readline()
        raw_digest.update(first)
        if first.strip() != b"{":
            raise ValueError("Unexpected upstream answer collection layout")
        for line in stream:
            raw_digest.update(line)
            line = line.strip()
            if line == b"}":
                continue
            boundary = line.find(b'":')
            if boundary < 2 or not line.startswith(b'"'):
                raise ValueError("Unexpected upstream answer record")
            key = json.loads(line[:boundary+1])
            if key not in wanted:
                continue
            if key in found:
                raise ValueError("Duplicate answer content")
            text = json.loads(line[boundary+2:].rstrip(b","))
            if not isinstance(text, str):
                raise ValueError("Answer must be text")
            found.add(key)
            writer.write(json.dumps(dict(answer_id=key, text=text), ensure_ascii=True) + "\n")
    if found != wanted:
        raise ValueError(f"Missing {len(wanted - found)} indexed answer texts")
    return {name: raw_digest.hexdigest()}


def run(output):
    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the stream build on a Slurm compute node")
    output.mkdir()
    remote = RemoteZip()
    with ZipFile(remote) as archive:
        profiles, upstream = load_experts(archive)
        (output / "profiles.json").write_text(json.dumps(profiles, ensure_ascii=True) + "\n")
        print("Temporal profiles:", len(profiles), flush=True)
        reports = {}
        selected_ids = set()
        for split in ("val", "test"):
            panel, ids, reports[split], checks = query_candidates(archive, split, profiles.keys())
            if ids & selected_ids:
                raise ValueError("Official evaluation splits overlap")
            selected_ids.update(ids)
            upstream.update(checks)
            (output / f"{split}.json").write_text(json.dumps(panel, ensure_ascii=True) + "\n")
            print(split, reports[split]["released"], reports[split]["eligible"], flush=True)
        wanted = {answer for history in profiles.values() for _, answer in history}
        upstream.update(read_answers(archive, wanted, output / "answers.jsonl"))
    report = dict(
        job=os.environ["SLURM_JOB_ID"], archive_url=URL, archive_size=SIZE, bytes_transferred=remote.bytes_transferred,
        script_sha256=digest(Path(__file__).read_bytes()), profile_cutoff_utc=SNAPSHOT, test_start_utc=TEST_START,
        expert_profiles=len(profiles), source_answers=len(wanted), answers_per_expert=ANSWERS_PER_EXPERT,
        split=reports, source_sha256=upstream,
        derived_sha256={path.name: digest(path.read_bytes()) for path in sorted(output.iterdir())},
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Finished temporal panel", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
