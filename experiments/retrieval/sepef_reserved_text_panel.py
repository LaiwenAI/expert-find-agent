"""Build an equal-budget text control with one slot per active community."""

import argparse
from hashlib import sha256
from http.client import HTTPException
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import sleep, time_ns
from urllib.request import Request, urlopen
from zipfile import ZipFile

from experiments.retrieval.sepef_budgeted_history import ASSETS, mask_history
from experiments.retrieval.sepef_temporal_panel import (
    ANSWERS_PER_EXPERT, PREFIX, SIZE, URL, RemoteZip, digest, read_answers,
)


class RetryingRemoteZip(RemoteZip):
    @staticmethod
    def fetch(bounds):
        try:
            return RemoteZip.fetch(bounds)
        except (OSError, HTTPException):
            start, end = bounds
            expected_size = end - start + 1
            for delay in (1, 2, 4):
                sleep(delay)
                request = Request(f"{URL}&retry={time_ns()}", headers={
                    "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                })
                try:
                    with urlopen(request, timeout=45) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                            raise OSError("Unexpected official archive range")
                        payload = response.read(expected_size + 1)
                        if len(payload) != expected_size:
                            raise OSError("Incomplete official archive range")
                        return payload
                except (OSError, HTTPException):
                    if delay == 4:
                        raise


def file_sha(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def build(base, output, workers):
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= workers <= 4:
        raise ValueError("Workers must be between one and four")
    previous = json.loads((base / "manifest.json").read_bytes())
    expected = {"profiles.json", "answers.jsonl", "val.json", "test.json"}
    if (set(previous["derived_sha256"]) != expected
            or any(file_sha(base / name) != checksum for name, checksum in previous["derived_sha256"].items())
            or previous["answers_per_expert"] != ANSWERS_PER_EXPERT):
        raise ValueError("Changed anchor panel")
    old_profiles = json.loads((base / "profiles.json").read_bytes())
    old_ids = {answer for history in old_profiles.values() for _, answer in history}
    remote = RetryingRemoteZip(workers)
    with TemporaryDirectory(dir=output.parent, prefix=".sepef-reservation-") as temporary:
        work = Path(temporary)
        final = work / "panel"
        final.mkdir()
        with ZipFile(remote) as archive:
            name = "expert_test_data.json"
            raw = archive.read(PREFIX + name)
            if digest(raw) != previous["source_sha256"][PREFIX + name] or digest(raw) != ASSETS[name]:
                raise ValueError("Changed source history")
            history = json.loads(raw)
            _, count, profiles = mask_history(history, ANSWERS_PER_EXPERT, "reserve_active_sites")
            wanted = {answer for records in profiles.values() for _, answer in records}
            if count != len(wanted) or count != len(old_ids) or count != previous["source_answers"]:
                raise ValueError("Unequal history budgets")
            if profiles.keys() != old_profiles.keys():
                raise ValueError("Different expert population")
            new_ids = wanted - old_ids
            delta = work / "new.jsonl"
            if not new_ids:
                raise ValueError("No history changed under reservation")
            print("Equal-budget reservations:", len(new_ids), "new source answers", flush=True)
            upstream = read_answers(archive, new_ids, delta)
            if any(previous["source_sha256"][name] != checksum for name, checksum in upstream.items()):
                raise ValueError("Changed answer collection")
        (final / "profiles.json").write_text(json.dumps(profiles, ensure_ascii=True) + "\n")
        seen = set()
        with (final / "answers.jsonl").open("x") as target:
            for stream in (base / "answers.jsonl", delta):
                with stream.open() as source:
                    for line in source:
                        answer = json.loads(line)["answer_id"]
                        if answer in wanted:
                            if answer in seen:
                                raise ValueError("Duplicate retained answer text")
                            target.write(line)
                            seen.add(answer)
        if seen != wanted:
            raise ValueError("Missing selected answer text")
        for split in ("val", "test"):
            copyfile(base / f"{split}.json", final / f"{split}.json")
        report = previous | dict(
            script_sha256=file_sha(Path(__file__)),
            anchor_manifest_sha256=file_sha(base / "manifest.json"),
            selection="reserve_active_sites",
            answers_replaced=len(new_ids),
            bytes_transferred=remote.bytes_transferred,
            derived_sha256={name: file_sha(final / name) for name in sorted(expected)},
        )
        (final / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        final.rename(output)
    print("Reserved-community text panel complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    build(**vars(parser.parse_args()))
