"""Restore original cooperation fields and keep both overlap screens binding."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.audit_enterprise_panel_neighbors import write_private
from experiments.enterprise_eligibility_pilot import xian_cooperation
from experiments.freeze_enterprise_panel import digest_file, read_rows, screen
from experiments.retrieval.enterprise_routed_review import write


PILOT_PROTOCOL = (
    Path(__file__).resolve().parents[1] / "benchmarks/enterprise-eligibility-pilot-v1/protocol.json"
)


def pilot_groups(packet, frame_hash, representatives):
    receipt = json.loads((packet / "receipt.json").read_bytes())
    exclusions = packet / "holdout-exclusions.jsonl"
    blank = packet / "blank.jsonl"
    if (digest_file(exclusions) != receipt["exclusions_sha256"]
            or digest_file(blank) != receipt["blank_sha256"]):
        raise ValueError("The original eligibility pilot files changed")
    expected = {item["component_sha256"]: item["text_sha256"] for item in representatives}
    rows = read_rows(exclusions)
    groups = {row["component_sha256"]: row["original_text_sha256"] for row in rows}
    if (len(groups) != len(rows) or len(groups) != receipt["pilot_size"]
            or digest_file(PILOT_PROTOCOL) != receipt["protocol_sha256"]
            or json.loads(PILOT_PROTOCOL.read_bytes())["frame_sha256"] != frame_hash
            or any(expected.get(group) != digest for group, digest in groups.items())):
        raise ValueError("Eligibility exclusions do not belong to the frozen frame")
    return groups


def build(frame, representatives, sources, history, original_stats, archives, pilot, output, mask_contact):
    frame_hash = digest_file(frame)
    excluded = pilot_groups(pilot, frame_hash, representatives)
    xi_ids = {item["id"] for item in sources if item["id"].startswith("xian-")}
    coordination = xian_cooperation(archives, xi_ids)
    extended = [
        row | {"zh": row["zh"] + "\n\n" + coordination[row["id"]]}
        if coordination.get(row["id"]) else row for row in sources
    ]
    restored, restored_stats = screen(extended + history, set(), set())
    extended_by_id = {row["id"]: row for row in extended}
    original_by_id = {row["id"]: row for row in sources}
    restored_keys = {(row["id"], row["component_sha256"]) for row in restored}
    original_keys = {(row["id"], row["component_sha256"]) for row in representatives}
    counts = Counter()
    partitions = {name: [] for name in ("pilot", "provisional-holdout", "quarantined")}
    for item in representatives:
        identifier = item["id"]
        group = item["component_sha256"]
        new = extended_by_id[identifier]
        original = original_by_id[identifier]
        if sha256(original["zh"].encode()).hexdigest() != item["text_sha256"]:
            raise ValueError("Restoration input differs from the frozen source")
        title, _, body = original["zh"].partition("\n")
        additional = coordination.get(identifier)
        unchanged_group = (identifier, group) in restored_keys
        if group in excluded:
            partition = "pilot"
            counts["pilot_groups_changed_on_rescreen"] += int(not unchanged_group)
        else:
            partition = "provisional-holdout" if unchanged_group else "quarantined"
        counts["representatives_with_restored_field"] += bool(additional)
        counts[f"{partition}_restored"] += bool(additional)
        partitions[partition].append(dict(
            source=new["batch"], source_locator=identifier, source_row=new["source_row"],
            component_sha256=group, frozen_text_sha256=item["text_sha256"],
            restored_text_sha256=sha256(new["zh"].encode()).hexdigest(),
            original_group_reproduced=unchanged_group,
            title=mask_contact(title), technical_body=mask_contact(body),
            cooperation_requirements=mask_contact(additional) if additional is not None else None,
        ))
    if len(partitions["pilot"]) != len(excluded) or counts["representatives_with_restored_field"] != 241:
        raise ValueError("Unexpected pilot or restored representative count")

    with TemporaryDirectory(prefix=".cooperation-", dir=output.parent) as directory:
        staging = Path(directory) / "projection"
        staging.mkdir(mode=0o700)
        for name, rows in partitions.items():
            write_private(staging / f"{name}.jsonl", rows)
        write(staging / "receipt.json", dict(
            frame_sha256=frame_hash, protocol_sha256=digest_file(PILOT_PROTOCOL),
            source_rows=len(sources), historical_rows=len(history), original_groups=len(representatives),
            original_screen=original_stats, restored_screen=restored_stats,
            unchanged_groups=len(restored_keys & original_keys),
            added_or_changed_restored_groups=len(restored_keys - original_keys),
            original_groups_not_reproduced=len(original_keys - restored_keys),
            source_rows_with_restored_field=sum(map(bool, coordination.values())),
            field_counts=counts, partition_sizes={k: len(v) for k, v in partitions.items()},
            human_qualification_labels=0, expert_queries_run=0,
        ))
        staging.rename(output)
