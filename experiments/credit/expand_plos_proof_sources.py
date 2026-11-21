"""Extend the cross-publisher proof inventory with bounded public author discovery."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from experiments.credit import harvest_scoped_credit as harvest
from experiments.credit.engineering_cross_source_panel import owner_key
from experiments.credit.freeze_conjunction_calibration import freeze
from experiments.credit.plos_role_inventory import read_url
from experiments.credit.plos_selective_review_holdout import build
from experiments.followir.core17_reranking import write_json


def discover(folders, output, excluded, europe_pmc):
    seed, _, _ = build(folders, excluded)
    frozen, _, occupied = freeze(seed, 40)
    used_papers = {doi for pair in frozen for doi in pair["papers"]}
    visited = set()
    anchors = []
    for folder in folders:
        for path in harvest.xml_sources(folder):
            try:
                meta = harvest.meta(path.read_bytes())
            except (ValueError, harvest.ET.ParseError):
                continue
            doi = meta.findtext("article-id[@pub-id-type='doi']")
            if doi not in seed or doi in visited or doi in used_papers:
                continue
            visited.add(doi)
            candidates = []
            for i, author in enumerate(harvest.authors(meta)):
                key = owner_key(doi, i, author)
                if key not in seed[doi][0] or key in occupied:
                    continue
                for identifier, spelling, _ in harvest.joined([author]):
                    candidates.append((key, identifier, spelling, doi))
            if candidates:
                selected = min(candidates, key=lambda item: sha256((doi+item[0]).encode()).digest())
                anchors.append(selected)
    anchors = sorted(anchors, key=lambda row: sha256(("raw-proof-holdout:"+row[3]).encode()).digest())[:48]
    full = set(seed)

    def lookup(anchor):
        _, identifier, spelling, doi = anchor
        quoted = spelling.replace("\\","\\\\").replace('"','\\"')
        query = f'journal:"PLOS ONE" AND author:"{quoted}" AND publication_date:[2022-01-01T00:00:00Z TO 2026-10-31T23:59:59Z]'
        try:
            if europe_pmc:
                query = f'AUTHORID:{identifier} AND JOURNAL:"PLoS One" AND PUB_YEAR:[2022 TO 2026]'
                params = urlencode(dict(query=query, format="json", pageSize=20))
                with urlopen(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}", timeout=12) as reply:
                    response = reply.read()
                found = [row.get("doi", "").casefold() for row in json.loads(response)["resultList"]["result"]]
            else:
                response = read_url(harvest.search(query, 18))
                found = [row["id"] for row in json.loads(response)["response"]["docs"]]
            candidates = list(dict.fromkeys(item for item in found if harvest.is_article(item)
                                            and item not in full and item != doi))[:3]
            return doi, identifier, candidates, sha256(response).hexdigest()
        except (OSError, ValueError, KeyError):
            return doi, identifier, [], None

    with ThreadPoolExecutor(max_workers=4) as workers:
        candidates = list(workers.map(lookup, anchors))
    unique = sorted({doi for _, _, docs, _ in candidates for doi in docs})
    print(f"Screening {len(unique)} candidate papers from {len(candidates)} author lookups", flush=True)

    def verify(doi):
        try:
            with urlopen(harvest.article_url(doi), timeout=15 if europe_pmc else 35) as reply:
                raw = reply.read()
            harvest.declared_roles(raw, doi)
            ids = {row[0] for row in harvest.joined(harvest.authors(harvest.meta(raw)))}
            return doi, (raw, ids)
        except (OSError, ValueError, harvest.ET.ParseError):
            return doi, None

    with ThreadPoolExecutor(max_workers=4) as workers:
        data = dict(workers.map(verify, unique))
    audits, saved = [], {}
    for parent, identifier, options, search_hash in candidates:
        linked = []
        for doi in options:
            found = data[doi]
            if found and identifier in found[1]:
                linked.append(doi)
                if doi not in saved:
                    (output / (doi.rsplit(".", 1)[-1] + ".xml")).write_bytes(found[0])
                    saved[doi] = sha256(found[0]).hexdigest()
        audits.append(dict(anchor_doi=parent, search_sha256=search_hash, screened=len(options), linked=linked))
    return dict(backend="europe_pmc" if europe_pmc else "plos", queries=len(anchors), searches_succeeded=sum(row["search_sha256"] is not None for row in audits),
                candidate_documents=len(unique), downloads_verified=len(saved), documents=saved, checks=audits)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("elife_notes", "engineering_panel", "output", "report"):
        cli.add_argument(name, type=Path)
    cli.add_argument("folders", nargs="+", type=Path)
    cli.add_argument("--europe-pmc", action="store_true")
    args = cli.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.output.mkdir(exist_ok=False)
    provenance = json.loads(args.elife_notes.read_bytes())
    engineering = json.loads(args.engineering_panel.read_bytes())
    identities = {row["engineering_key"] for notes in provenance.values() for row in notes.values()}
    identities.update(owner for article in engineering["articles"].values() for owner in article["roles"])
    report = discover(args.folders, args.output, identities, args.europe_pmc)
    write_json(args.report, report)
    print({k: v for k,v in report.items() if isinstance(v, int)}, flush=True)


if __name__ == "__main__":
    main()
