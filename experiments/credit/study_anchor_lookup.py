"""Investigate literal and lexical study anchors under source withholding."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import log1p
from pathlib import Path
import re
from time import perf_counter
import unicodedata
from xml.etree import ElementTree as ET

import numpy as np

from experiments.followir.core17_reranking import digest, write_json


STYLES = ("full", "fragment", "damaged", "shared_fragment")
ARMS = ("bm25_rank", "bm25_calibrated", "visible_anchor", "frozen_anchor")
MIN_SPAN = 3


def tokens(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized))


def load_catalog(paths):
    catalog = {}
    receipts = []
    for group, root in enumerate(paths):
        receipt = root / "capture.json"
        manifest = json.loads(receipt.read_bytes())
        receipts.append(digest(receipt))
        parts = [(root, manifest)] if "checksums" in manifest else [
            (root / name, manifest[name]) for name in ("calibration", "test")
        ]
        for parent, part in parts:
            for doi, expected in part["checksums"].items():
                raw = (parent / f"{doi.rsplit('.', 1)[-1]}.xml").read_bytes()
                if sha256(raw).hexdigest() != expected:
                    raise ValueError("Changed source capture")
                meta = ET.fromstring(raw).find("./front/article-meta")
                if meta.findtext("article-id[@pub-id-type='doi']") != doi:
                    raise ValueError("Source DOI does not match its capture")
                node = meta.find("title-group/article-title")
                if node is not None:
                    title = tokens(" ".join(node.itertext()))
                    if len(title) >= 6:
                        catalog.setdefault(doi, (title, group))
    return catalog, receipts


class Index:
    def __init__(self, catalog):
        self.spans = defaultdict(set)
        self.terms = defaultdict(dict)
        self.titles = {}
        self.lengths = {}
        for doi, (title, _) in catalog.items():
            self.titles[doi] = title
            self.lengths[doi] = len(title)
            for term, count in Counter(title).items():
                self.terms[term][doi] = count
            for i in range(len(title) - MIN_SPAN + 1):
                self.spans[title[i:i + MIN_SPAN]].add(doi)
        self.average = np.mean(list(self.lengths.values()))

    def rank(self, clue, unavailable=None):
        scores = defaultdict(float)
        documents = len(self.lengths)
        for term in sorted(set(clue)):
            relevant = self.terms.get(term, {})
            idf = log1p((documents - len(relevant) + .5) / (len(relevant) + .5))
            for doi, count in relevant.items():
                if doi != unavailable:
                    norm = 1.2 * (.25 + .75 * self.lengths[doi] / self.average)
                    scores[doi] += idf * (2.2 * count) / (count + norm)
        return min(scores.items(), key=lambda pair: (-pair[1], pair[0])) if scores else (None, 0.)

    def matches(self, clue):
        if len(clue) < MIN_SPAN:
            return set()
        shortlist = min((self.spans.get(clue[i:i + MIN_SPAN], ())
                         for i in range(len(clue) - MIN_SPAN + 1)), key=len)
        return {doi for doi in shortlist if any(
            self.titles[doi][i:i + len(clue)] == clue
            for i in range(self.lengths[doi] - len(clue) + 1)
        )}

    def anchor(self, clue, unavailable=None, frozen=False):
        postings = self.matches(clue)
        eligible = postings if frozen else postings - {unavailable}
        single = next(iter(eligible)) if len(eligible) == 1 else None
        return single if single != unavailable else None


def queries(catalog, index):
    for doi, (title, group) in sorted(catalog.items()):
        fingerprint = int.from_bytes(sha256(doi.encode()).digest()[:8], "big")
        offset = fingerprint % (len(title) - 5)
        fragment = title[offset:offset + 6]
        cut = 1 + fingerprint % 4
        yield doi, group, "full", title
        yield doi, group, "fragment", fragment
        yield doi, group, "damaged", fragment[:cut] + fragment[cut + 1:]
        shared = [span for span in (title[i:i + 3] for i in range(len(title) - 2))
                  if len(index.spans[span]) > 1]
        if shared:
            yield doi, group, "shared_fragment", shared[fingerprint % len(shared)]


def compare(catalog):
    started = perf_counter()
    index = Index(catalog)
    index_ms = (perf_counter() - started) * 1000
    panel = list(queries(catalog, index))
    # Calibrate on earlier source acquisition, then evaluate later acquisitions.
    absent_calibration = defaultdict(list)
    for doi, group, style, clue in panel:
        if group == 0:
            absent_calibration[style].append(index.rank(clue, doi)[1])
    cutoffs = {style: float(np.quantile(scores, .95, method="higher"))
               for style, scores in absent_calibration.items()}
    results = {style: {arm: Counter() for arm in ARMS} for style in STYLES}
    timing = {arm: [] for arm in ARMS if arm != "bm25_calibrated"}

    def timed(arm, callback):
        start = perf_counter()
        value = callback()
        timing[arm].append((perf_counter() - start) * 1000)
        return value

    for doi, group, style, clue in panel:
        if group == 0:
            continue
        for outage in (False, True):
            unavailable = doi if outage else None
            ranked, score = timed("bm25_rank", lambda: index.rank(clue, unavailable))
            selected = {
                "bm25_rank": ranked,
                "bm25_calibrated": ranked if score > cutoffs.get(style, float("inf")) else None,
                "visible_anchor": timed("visible_anchor", lambda: index.anchor(clue, unavailable)),
                "frozen_anchor": timed("frozen_anchor", lambda: index.anchor(clue, unavailable, frozen=True)),
            }
            for arm, target in selected.items():
                row = results[style][arm]
                if outage:
                    row.update(withheld=1, withheld_wrong=int(target is not None))
                else:
                    row.update(available=1, correct=int(target == doi), wrong=int(target is not None and target != doi))
    return dict(
        articles=len(catalog), calibration_articles=sum(group == 0 for _, group in catalog.values()),
        test_articles=sum(group != 0 for _, group in catalog.values()), span_keys=len(index.spans),
        calibrated_cutoffs=cutoffs, p95_ms={arm: float(np.quantile(values, .95)) for arm, values in timing.items()},
        index_ms=index_ms, results=results,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("captures", type=Path, nargs="+")
    args = cli.parse_args()
    catalog, receipts = load_catalog(args.captures)
    write_json(args.output, dict(capture_sha256=receipts, script_sha256=digest(Path(__file__)), **compare(catalog)))
