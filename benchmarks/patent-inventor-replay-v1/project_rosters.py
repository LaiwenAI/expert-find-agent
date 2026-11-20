"""Project the PatentsView grant roster onto a frozen HUPD grant catalog."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

INVENTOR_MEMBER = "g_inventor_disambiguated.tsv"
INVENTOR_ARCHIVE_MD5 = "ea55c1db6a56e59f54baf34818502a7e"


def project(catalog: Path, archive: Path, output: Path) -> None:
    grants = {
        row["patent_number"]: row
        for row in map(json.loads, catalog.read_text().splitlines())
    }
    checksum = hashlib.md5()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    if digest != INVENTOR_ARCHIVE_MD5:
        raise ValueError(f"Unexpected inventor archive checksum: {digest}")

    rosters: dict[str, dict[int, str]] = {}
    with ZipFile(archive) as bundle, bundle.open(INVENTOR_MEMBER) as stream:
        rows = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t")
        for row in rows:
            patent = row["patent_id"]
            if patent not in grants:
                continue
            sequence = int(row["inventor_sequence"])
            inventor = row["inventor_id"]
            if not inventor:
                raise ValueError(f"Empty inventor identifier in {patent}, slot {sequence}")
            roster = rosters.setdefault(patent, {})
            earlier = roster.setdefault(sequence, inventor)
            if earlier != inventor:
                raise ValueError(f"Conflicting inventor identifiers in {patent}, slot {sequence}")

    eligible = []
    for patent, roster in sorted(rosters.items()):
        if sorted(roster) != list(range(len(roster))):
            raise ValueError(f"Noncontiguous slots in {patent}: {sorted(roster)}")
        eligible.append({
            **grants[patent],
            "inventors": list(dict.fromkeys(roster[slot] for slot in sorted(roster))),
        })
    pending = output.with_suffix(".part")
    pending.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in eligible))
    pending.replace(output)
    print(json.dumps({
        "catalog_grants": len(grants),
        "joined_grants": len(eligible),
        "inventor_appearances": sum(len(row["inventors"]) for row in eligible),
        "distinct_inventors": len({actor for row in eligible for actor in row["inventors"]}),
    }, sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("catalog", type=Path)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    project(args.catalog, args.archive, args.output)
