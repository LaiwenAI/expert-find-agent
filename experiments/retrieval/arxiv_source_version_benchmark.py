"""Measure source-version drift and bounded identity checks on historical arXiv records."""

import argparse
from calendar import monthrange
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from experiments.retrieval.cof_public_provenance_audit import canonical
from experiments.retrieval.cof_rank_disagreement_review import sha, write_json

ATOM = "{http://www.w3.org/2005/Atom}"
TOTAL = "{http://a9.com/-/spec/opensearch/1.1/}totalResults"
ID = re.compile(r"(?P<work>24\d{2}\.\d{4,5})v(?P<version>[1-9]\d*)$")
MONTHS = (1, 4, 7, 10)
SAMPLE = 64
SEED = "source-version-cs-ir-2024"
API = "https://export.arxiv.org/api/query"


def fingerprint(value):
    return sha256(value.encode()).hexdigest()


class FeedCache:
    def __init__(self, root, offline):
        self.root = root
        self.offline = offline
        self.last = 0.0
        self.feeds = {}

    def fetch(self, params):
        query = urlencode(params)
        target = self.root / (fingerprint(query) + ".xml")
        if not target.exists():
            if self.offline:
                raise ValueError("Missing pinned feed in offline mode")
            for i in range(3):
                sleep(max(0, 3.1 - (monotonic() - self.last)))
                try:
                    request = Request(f"{API}?{query}", headers={"User-Agent": "ExpertSearchResearch/0.1"})
                    self.last = monotonic()
                    with urlopen(request, timeout=40) as response:
                        raw = response.read(8_000_001)
                    if len(raw) > 8_000_000:
                        raise ValueError("Unexpected response size")
                    ET.fromstring(raw)
                    target.write_bytes(raw)
                    break
                except (HTTPError, URLError, TimeoutError):
                    if i == 2:
                        raise
                    sleep(2 ** i)
        self.feeds[query] = sha(target)
        return ET.fromstring(target.read_bytes())


def entries(feed):
    result = {}
    for row in feed.findall(ATOM + "entry"):
        name = row.findtext(ATOM + "id").rsplit("/", 1)[-1]
        identifier = ID.fullmatch(name)
        if identifier is None:
            raise ValueError(f"Unexpected work or version identifier: {name}")
        authors = [canonical(node.findtext(ATOM + "name")) for node in row.findall(ATOM + "author")]
        title = canonical(row.findtext(ATOM + "title"))
        abstract = canonical(row.findtext(ATOM + "summary"))
        if not title or not abstract or not authors or any(not name for name in authors):
            raise ValueError("A feed entry has incomplete content metadata")
        if name in result:
            raise ValueError("Duplicate version in one feed")
        result[name] = dict(
            work=identifier["work"], version=int(identifier["version"]),
            title=fingerprint(title), abstract=fingerprint(abstract),
            byline=fingerprint(json.dumps(authors)), author_set=fingerprint(json.dumps(sorted(set(authors)))),
        )
    return result


def cohort(cache):
    selected = {}
    frame_sizes = {}
    for month in MONTHS:
        limit = monthrange(2024, month)[1]
        search = f"cat:cs.IR AND submittedDate:[2024{month:02d}010000 TO 2024{month:02d}{limit:02d}2359]"
        scope = {"search_query": search, "sortBy": "submittedDate", "sortOrder": "ascending"}
        frame = {}
        boundary = None
        for offset in range(0, 4000, 500):
            feed = cache.fetch({**scope, "start": offset, "max_results": 500})
            count = int(feed.findtext(TOTAL))
            if boundary is not None and boundary != count:
                raise ValueError("The monthly listing changed during paging")
            boundary = count
            batch = entries(feed)
            if frame.keys() & batch.keys():
                raise ValueError("Repeated listing entry")
            frame.update(batch)
            if len(frame) >= count:
                break
        if len(frame) != boundary or len(frame) < SAMPLE or len({x["work"] for x in frame.values()}) != len(frame):
            raise ValueError("Incomplete monthly sampling frame")
        frame_sizes[str(month)] = len(frame)
        ordered = sorted(frame.values(), key=lambda row: fingerprint(f"{SEED}:{row['work']}"))
        for row in ordered[:SAMPLE]:
            if row["work"] in selected:
                raise ValueError("One work was present in two sampled months")
            selected[row["work"]] = row
    return selected, frame_sizes


def historical(cache, latest):
    all_versions = {(entry["work"], entry["version"]): entry for entry in latest.values()}
    for version in (1, 2):
        requested = sorted(work + f"v{version}" for work, row in latest.items() if row["version"] > version)
        for offset in range(0, len(requested), 60):
            batch = requested[offset:offset+60]
            feed = cache.fetch({"id_list": ",".join(batch), "max_results": len(batch)})
            found = entries(feed)
            if found.keys() != set(batch):
                raise ValueError(f"Exact version feed mismatch for batch starting {batch[0]}")
            all_versions.update(((row["work"], row["version"]), row) for row in found.values())
    return all_versions


POLICIES = {
    "latest": (),
    "title": ("title",),
    "title_abstract": ("title", "abstract"),
    "title_author_set": ("title", "author_set"),
}


def select_version(version_list, target, fields):
    if not fields:
        return version_list[0], 0
    for i, row in enumerate(version_list):
        if all(row[field] == target[field] for field in fields):
            return row, i
    return None, len(version_list) - 1


def summarize(rows, version):
    drift, stats = Counter(), {arm: Counter() for arm in POLICIES}
    for versions in rows:
        latest = versions[0]
        if version > latest["version"]:
            continue
        target = next(row for row in versions if row["version"] == version)
        if version == 2 and version == latest["version"]:
            continue
        drift["works"] += 1
        drift["has_later_version"] += latest["version"] > version
        for field in ("title", "abstract", "byline", "author_set"):
            drift[f"changed_{field}"] += latest[field] != target[field]
        drift["changed_author_set_at_same_title"] += latest["title"] == target["title"] and latest["author_set"] != target["author_set"]
        for method, fields in POLICIES.items():
            policy = stats[method]
            result, reads = select_version(versions, target, fields)
            policy["additional_version_reads"] += reads
            policy["abstentions"] += result is None
            if result:
                policy["correct_version"] += result["version"] == version
                policy["correct_author_set"] += result["author_set"] == target["author_set"]
                policy["correct_ordered_byline"] += result["byline"] == target["byline"]
                policy["wrong_author_set"] += result["author_set"] != target["author_set"]
    return dict(drift=drift, policies=stats)


def run(cache, output, offline):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    feeds = FeedCache(cache, offline)
    latest, frame_sizes = cohort(feeds)
    lookup = historical(feeds, latest)
    rows = []
    for work, row in sorted(latest.items()):
        chain = [row]
        chain.extend(lookup[work, version] for version in (1, 2) if version < row["version"])
        rows.append(chain)
    output.mkdir(parents=True)
    samples = output / "cohort.json"
    write_json(samples, rows)
    write_json(output/"results.json", dict(
        monthly_listing_sizes=frame_sizes, cohort_sha256=sha(samples), script_sha256=sha(Path(__file__)),
        api_feed_hashes=feeds.feeds, v1_target=summarize(rows, 1), historical_v2_target=summarize(rows, 2),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--offline", action="store_true")
    run(**vars(parser.parse_args()))
