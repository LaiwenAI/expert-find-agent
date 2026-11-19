"""Capture publisher articles for a source-gate transfer panel."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from experiments.credit import harvest_scoped_credit as capture
from experiments.credit.plos_role_inventory import read_url


QUERY = (
    'journal:"PLOS ONE" AND title:("microgrid" OR "power load forecasting" OR "photovoltaic power forecasting" '
    'OR "short-term load forecasting" OR "distribution network optimization") '
    'AND publication_date:[2021-01-01T00:00:00Z TO 2026-09-30T23:59:59Z]'
)
SECONDARY = (
    'journal:"PLOS ONE" AND title:("wind power" OR "solar power forecasting" OR "power system operation" '
    'OR "wind farm" OR "electricity demand forecasting" OR "power system stability") '
    'AND publication_date:[2021-01-01T00:00:00Z TO 2026-09-30T23:59:59Z]'
)
ROWS = 80


def collect(directory, old_folders, prior_projection, query):
    directory.mkdir(exist_ok=True, parents=True)
    summary = directory / "capture.json"
    if summary.exists():
        raise FileExistsError(summary)
    before, _, _ = capture.prior(old_folders)
    indexed = set(json.loads(prior_projection.read_text())["documents"])
    if not before or not indexed <= before:
        raise ValueError("Previous JATS archives must cover the development corpus")
    request = capture.search(query, ROWS)
    raw_path = directory / "search.json"
    if not raw_path.exists():
        raw_path.write_bytes(read_url(request))
    rows = json.loads(raw_path.read_text())["response"]["docs"]
    ids = sorted({row["id"] for row in rows if capture.is_article(row["id"])})
    fresh = [doi for doi in ids if doi not in before]
    files, failed = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for doi, (raw, error) in zip(fresh, pool.map(lambda doi: capture.candidate_xml(directory, doi), fresh),
                                     strict=True):
            if raw is not None:
                files[doi] = sha256(raw).hexdigest()
            else:
                failed[doi] = error
    with summary.open("x") as stream:
        json.dump(dict(query=request, listed_dois=ids, matched_prior=len(ids) - len(fresh), count=len(files),
                       hashes=files, rejected=failed, search_sha256=sha256(raw_path.read_bytes()).hexdigest()), stream)
        stream.write("\n")
    print("Published candidates captured:", len(files), "not captured:", len(failed), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("prior_projection", type=Path)
    parser.add_argument("old_folders", nargs="+", type=Path)
    parser.add_argument("--secondary", action="store_true")
    args = parser.parse_args()
    collect(args.directory, args.old_folders, args.prior_projection, SECONDARY if args.secondary else QUERY)
