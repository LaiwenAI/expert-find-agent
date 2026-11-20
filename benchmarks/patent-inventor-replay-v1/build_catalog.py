"""Select granted records from the pinned HUPD January 2016 metadata snapshot."""

import argparse
import hashlib
import json
from pathlib import Path

from pyarrow import feather


FIELDS = [
    "patent_number", "patent_issue_date", "main_cpc_label",
    "invention_title", "application_number",
]
HUPD_SHA256 = "4cc1b41a5233818ba1db85f2b6fd9edde29f8e92645919c27ed8e6a04fc4f20b"


def build(source: Path, output: Path) -> None:
    if hashlib.sha256(source.read_bytes()).hexdigest() != HUPD_SHA256:
        raise ValueError("Unexpected HUPD metadata checksum")

    by_grant = {}
    table = feather.read_table(source, columns=FIELDS)
    for row in table.to_pylist():
        patent_id = row["patent_number"]
        if not (
            patent_id and patent_id.isascii() and patent_id.isdecimal()
            and row["main_cpc_label"] and row["patent_issue_date"] and row["invention_title"]
        ):
            continue
        row["patent_issue_date"] = row["patent_issue_date"].date().isoformat()
        previous = by_grant.setdefault(patent_id, row)
        if previous != row:
            raise ValueError(f"Multiple application records mapped to grant {patent_id}")

    with output.open("w") as handle:
        for patent_id in sorted(by_grant):
            handle.write(json.dumps(by_grant[patent_id], sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
