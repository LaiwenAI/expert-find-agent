"""Build a DOI- and contributor-disjoint publisher source panel."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from expert_search.retrieval import tokens
from experiments.credit import harvest_scoped_credit as capture
from experiments.credit import scoped_credit_search as source
from experiments.credit.scoped_source_confirmation import graph


def prepare(directory, old_projection, output, old_folders, additional):
    if output.exists():
        raise FileExistsError(output)
    before, known_ids, known_names = capture.prior(old_folders)
    dev = set(json.loads(old_projection.read_text())["documents"])
    if not dev <= before:
        raise ValueError("Previous JATS archives must cover the calibration corpus")
    receipts = [root / "capture.json" for root in (directory, *additional)]
    allowed, excluded, encountered = {}, Counter(), {}
    for receipt in receipts:
        manifest = json.loads(receipt.read_text())
        seen = set()
        for item in capture.xml_sources(receipt.parent):
            raw = item.read_bytes()
            checksum = sha256(raw).hexdigest()
            doc = capture.meta(raw)
            doi = doc.findtext("article-id[@pub-id-type='doi']")
            if manifest["hashes"].get(doi) != checksum:
                raise ValueError("Found an unexpected publication capture")
            seen.add(doi)
            if doi in encountered:
                if checksum != encountered[doi]:
                    raise ValueError("The same DOI has changed between captures")
                excluded["duplicate_capture"] += 1
                continue
            encountered[doi] = checksum
            byline = capture.authors(doc)
            if doi in before:
                excluded["prior_doi"] += 1
            elif known_names.intersection(capture.normalized_name(a) for a in byline):
                excluded["prior_name"] += 1
            elif known_ids.intersection(entry[0] for entry in capture.joined(byline)):
                excluded["prior_orcid"] += 1
            else:
                allowed[doi] = item
        if seen != manifest["hashes"].keys():
            raise ValueError("Missing captured records")
    with TemporaryDirectory() as temp:
        for path in allowed.values():
            (Path(temp) / path.name).symlink_to(path.resolve())
        rows = source.source_rows({"pairs": [], "sources": {}}, [Path(temp)])
    corpus = {doi: {key: value for key, value in row.items() if key != "result"} for doi, row in rows.items()}
    pairs = graph(corpus)
    if len({p[2] for p in pairs}) < 3:
        raise ValueError(f"Need three independent linked components, got {len(pairs)} linked pairs")
    frequencies = Counter(word for doc in corpus.values() for word in set(tokens(doc["index"])))
    clues = {}
    for doi in sorted({d for a, b, _ in pairs for d in (a, b)}):
        result = rows[doi]["result"]
        if source.whitespace(result) in source.whitespace(corpus[doi]["index"]):
            raise ValueError("Held-out clue is already indexed")
        terms = {word for word in tokens(result) if len(word) > 3 and not word.isdigit() and word not in source.STOPS}
        selected = sorted(terms, key=lambda word: (frequencies[word] == 0, frequencies[word], source.stable(doi, word)))[:8]
        if len(selected) < 3:
            raise ValueError("Insufficient clue terms")
        clues[doi] = dict(result=result, keywords=" ".join(selected))
    output.mkdir()
    projection = output / "projection.json"
    source.save(projection, dict(documents=corpus))
    source.save(output / "panel.json", dict(frozen_sha256=source.digest(projection), pairs=pairs, clues=clues))
    source.save(output / "manifest.json", dict(
        publication_digests={d: corpus[d]["sha256"] for d in sorted(corpus)},
        projected_count=len(corpus), pairs=len(pairs), components=len({p[2] for p in pairs}),
        excluded_prior_identity=dict(excluded), insufficient_abstracts=len(allowed) - len(corpus),
        capture_sha256=[source.digest(path) for path in receipts], previous_projection_sha256=source.digest(old_projection),
        projection_sha256=source.digest(projection), panel_sha256=source.digest(output / "panel.json"),
    ))
    print(dict(papers=len(corpus), pairs=len(pairs), components=len({p[2] for p in pairs})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "old_projection", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("old_folders", nargs="+", type=Path)
    parser.add_argument("--additional", type=Path, action="append", default=[])
    prepare(**vars(parser.parse_args()))
