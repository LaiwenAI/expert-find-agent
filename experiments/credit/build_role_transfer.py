"""Build a DOI-disjoint test set from a second PLOS Search capture."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from experiments.credit.counterfactual_role_edits import (
    CATALOG_SHA256, SOURCE_FIELD, article_url, matched_groups, write_json,
)
from experiments.credit.plos_role_inventory import LICENSE, declared_roles, read_url


TRANSFER_SHA256 = "c3458a0cbe48c790a58139e7264b3afd03b1fb020cafb89604564f1df42f6627"


def build(snapshot, prior_catalog, output, download):
    catalog = (snapshot / "search.json").read_bytes()
    original = prior_catalog.read_bytes()
    if sha256(catalog).hexdigest() != TRANSFER_SHA256 or sha256(original).hexdigest() != CATALOG_SHA256:
        raise ValueError("Unexpected article catalogs")
    original_ids = {r["id"] for r in json.loads(original)["response"]["docs"]}
    catalog_ids = [r["id"] for r in json.loads(catalog)["response"]["docs"]]
    if len(catalog_ids) != len(set(catalog_ids)) or len(catalog_ids) != 47:
        raise ValueError("Unexpected transfer article inventory")
    selected = [doi for doi in catalog_ids if doi not in original_ids]

    def inspect(doi):
        source = snapshot / f"{doi.rsplit('.', 1)[-1]}.xml"
        if not source.exists():
            if not download:
                raise FileNotFoundError(source)
            raw = read_url(article_url(doi))
            declared_roles(raw, doi)
            source.write_bytes(raw)
        raw = source.read_bytes()
        authors = declared_roles(raw, doi)
        if len(authors) < 2 or not all(authors):
            return "incomplete_author_declarations", None, []
        groups = matched_groups(doi, authors, cap=1)
        if not groups:
            return "no_matched_groups", None, []
        for group in groups:
            group["split"] = "test"
        metadata = dict(doi=doi, source_sha256=sha256(raw).hexdigest(), source_field=SOURCE_FIELD,
                        license=LICENSE, attribution=f"https://doi.org/{doi}", xml_url=article_url(doi),
                        split="test", roles_by_author=authors)
        return "included", metadata, groups

    with ThreadPoolExecutor(max_workers=4) as pool:
        captured = list(pool.map(inspect, selected))
    counts = Counter(status for status, _, _ in captured)
    sources = [metadata for status, metadata, _ in captured if status == "included"]
    groups = [group for _, _, cases in captured for group in cases]
    output.mkdir(parents=True, exist_ok=False)
    pairs = output / "pairs.jsonl"
    with pairs.open("w") as file:
        for row in sorted(groups, key=lambda group: (group["doi"], group["id"])):
            file.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    write_json(output / "manifest.json", dict(
        version="plos-credit-transfer-v1", catalog_sha256=TRANSFER_SHA256,
        old_catalog_sha256=CATALOG_SHA256, sources=sources, counts=counts,
        excluded_old_articles=sorted(original_ids.intersection(catalog_ids)),
        split_groups={"test": len(groups)}, split_sources={"test": len(sources)},
        pairs_sha256=sha256(pairs.read_bytes()).hexdigest(),
    ))
    print(dict(sources=len(sources), groups=len(groups), counts=counts, catalog_overlap=len(catalog_ids)-len(selected)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "prior_catalog", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    build(args.snapshot, args.prior_catalog, args.output, args.download)
