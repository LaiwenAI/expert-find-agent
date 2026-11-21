"""Screen new regional enterprise groups against three exposure strata."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from experiments.enterprise_municipal_frame import freeze_snapshot
from experiments.enterprise_regional_frame import COUNTS, table_records, zibo_records
from experiments.external_enterprise_overlap import build_packet, gpu
from experiments.external_enterprise_frame import digest_file, historical, package
from experiments.audit_enterprise_panel_neighbors import windows, write_private


ENCODER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CONTACT = re.compile(
    r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}"
    r"|(?<![\dA-Za-z])(?:\+?86[- ]?)?1[3-9]\d{9}(?![\dA-Za-z])",
    re.IGNORECASE,
)


def checked_rows(args):
    frame = json.loads(args.frame.read_bytes())
    original = json.loads(args.previous_index.read_bytes())
    report = json.loads(args.report.read_bytes())
    earlier, hashes = freeze_snapshot(args.earlier)
    first, first_hashes = package(args.municipal, "first", 30)
    if (hashes | {"first": first_hashes} != original["expanded_source_hashes"]
            or digest_file(args.workbook) != original["historical_workbook_sha256"]):
        raise ValueError("Earlier dataset changed")
    additions = []
    for name in COUNTS:
        source = report["sources"][name]
        notice = args.regional / ("zibo-notice.html" if name == "zibo" else f"{name}.html")
        if source["notice_sha256"] != digest_file(notice):
            raise ValueError(f"Changed notice in {name}")
        if name == "zibo":
            attachment = args.regional / "zibo.xls"
            if source["attachment_sha256"] != digest_file(attachment):
                raise ValueError("Changed Zibo attachment")
            additions.extend(zibo_records(attachment))
        else:
            additions.extend(table_records(notice, name))

    old = original["records"]
    new = frame["disjoint"]
    selected = old + new
    ids = {record["id"] for record in selected}
    if (len(ids) != len(selected) or len(old) != report["reproduced_previous_groups"]
            or len(new) != report["disjoint_new_groups"]
            or len(selected) != report["combined_groups"]
            or {record["id"]: record for record in frame["combined"]} !=
            {record["id"]: record for record in selected}):
        raise ValueError("Combined group metadata differs from the earlier and incremental selections")
    records = {record["id"]: record for record in earlier + first + additions}
    for item in selected:
        record = records[item["id"]]
        if sha256(record["zh"].encode()).hexdigest() != item["text_sha256"]:
            raise ValueError("Selected demand text changed")
    prior = historical(args.workbook)
    if len(prior) != report["historical_records"]:
        raise ValueError("Historical inventory changed")
    return [records[item["id"]] for item in selected], prior, len(old)


def prepare(args):
    external, history, earlier = checked_rows(args)
    projections = []
    masked_contacts = 0
    for row in external + history:
        text, masked = CONTACT.subn("[contact]", row["zh"])
        masked_contacts += masked
        projections.append(windows(text))
    payload = dict(frame_sha256=digest_file(args.frame), external=len(external), earlier=earlier,
                   focus=list(range(earlier, len(external))), views=projections)
    write_private(args.output, [payload])
    print(f"Projected {len(projections)} requests. Masked {masked_contacts} contact-like strings.")


def packet(args):
    external, history, earlier = checked_rows(args)
    semantic = json.loads(args.semantic.read_bytes())
    repeat = json.loads(args.repeat.read_bytes())
    if ({key: value for key, value in semantic.items() if key != "order_offset"} !=
            {key: value for key, value in repeat.items() if key != "order_offset"}
            or {semantic["order_offset"], repeat["order_offset"]} != {0, 1}):
        raise ValueError("Reversed-order GPU screen disagrees")
    if (semantic["frame_sha256"] != digest_file(args.frame)
            or semantic["projection_sha256"] != digest_file(args.projection)
            or semantic["earlier_groups"] != earlier or semantic["encoder_revision"] != ENCODER_REVISION
            or set(semantic["pairs"]) != {"earlier_frame", "historical", "new_groups"}
            or semantic["pairs"].keys() != semantic["counts"].keys()):
        raise ValueError("GPU screen does not match the frozen regional projection")
    build_packet(external, history, list(range(earlier, len(external))), semantic,
                 args.output, earlier=earlier, salt="enterprise-regional-overlap-v1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for command, fields in (
        (prepare, ("output",)),
        (gpu, ("inputs", "checkpoint", "output")),
        (packet, ("projection", "semantic", "repeat", "output")),
    ):
        mode = modes.add_parser(command.__name__)
        if command is not gpu:
            fields = ("earlier", "municipal", "regional", "workbook", "previous_index", "report", "frame") + fields
        for field in fields:
            mode.add_argument(field, type=Path)
        if command is gpu:
            mode.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
        mode.set_defaults(command=command)
    args = parser.parse_args()
    args.command(args)
