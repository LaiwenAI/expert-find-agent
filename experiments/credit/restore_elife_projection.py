"""Reconstruct only admitted records from a frozen publisher manifest."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from experiments.credit.elife_joint_postings_holdout import fetch, read_object
from experiments.credit.elife_role_nomination import parse, projection
from experiments.followir.core17_reranking import digest, write_json


def restore(manifest_path, target):
    manifest = json.loads(manifest_path.read_bytes())
    if target.exists():
        if digest(target) != manifest["projection_sha256"]:
            raise ValueError(f"Wrong cached projection: {target}")
        return
    retained = [record for record in manifest["source_inventory"] if record["exclusion"] is None]
    files = [{"path": record["filename"], "sha": record["git_blob"]} for record in retained]
    with TemporaryDirectory(dir=target.parent) as directory:
        xml = Path(directory)
        fetch(xml, files, workers=16)
        rows = []
        for record in retained:
            name = record["filename"]
            raw = read_object(xml / name, record["git_blob"])
            if sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Wrong publisher checksum: {name}")
            row, reason = parse(raw, name)
            if row is None or reason is not None:
                raise ValueError(f"Previously retained article failed parsing: {name}")
            rows.append(row)
        reconstructed = xml / "projection.json"
        write_json(reconstructed, projection(rows))
        if digest(reconstructed) != manifest["projection_sha256"]:
            raise ValueError("Reconstructed projection differs from the archived digest")
        reconstructed.replace(target)


if __name__ == "__main__":
    restore(*map(Path, sys.argv[1:]))
