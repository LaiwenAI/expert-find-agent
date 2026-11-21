"""Extend the provisional enterprise frame using structured public notices."""

import argparse
from collections import Counter
import json
from pathlib import Path

from bs4 import BeautifulSoup
from xlrd import open_workbook

from experiments.enterprise_municipal_frame import freeze_snapshot, usable
from experiments.enterprise_review_gate import scenario_power
from experiments.external_enterprise_frame import digest_file, historical, package, row
from experiments.freeze_enterprise_panel import grams, normalize, screen
from experiments.retrieval.enterprise_gate_design_audit import paired_pass_probability, passing_boundary


COUNTS = {"zibo": 37, "liaocheng": 53, "jiangyan": 52, "zhangjiagang": 67}
# Column captions from the published attachments.
ISSUER = "\u4f01\u4e1a\u540d\u79f0"
NUMBER = "\u5e8f\u53f7"
TITLE = "\u9700\u6c42\u540d\u79f0"
SCHEMAS = {
    "liaocheng": (NUMBER, ISSUER, "\u6240\u9700\u6280\u672f\u540d\u79f0",
                  "\u9762\u4e34\u7684\u6280\u672f\u96be\u9898(100\u5b57\u4ee5\u5185)"),
    "jiangyan": (NUMBER, ISSUER, TITLE, "\u9700\u6c42\u7b80\u4ecb"),
    "zhangjiagang": ("\u9700\u6c42\u7f16\u53f7", ISSUER, TITLE, "\u9700\u6c42\u5185\u5bb9"),
    "zibo": (NUMBER, ISSUER, "\u6280\u672f\u96be\u9898\u540d\u79f0", "\u6280\u672f\u96be\u9898\u5185\u5bb9"),
}


def cell_text(cell):
    # Keep headings and paragraphs separated without storing presentation whitespace.
    return " ".join(cell.get_text(" ", strip=True).split())


def to_records(name, entries):
    numbers = [number for number, *_ in entries]
    if numbers != list(range(1, COUNTS[name] + 1)):
        raise ValueError(f"Unexpected published row sequence in {name}")
    out = []
    for number, organization, title, body in entries:
        record = row(f"{name}-{number}", name, title, body, organization, number)
        record["complete"] = bool(title.strip() and body.strip())
        out.append(record)
    return out


def zibo_records(path):
    book = open_workbook(path, on_demand=True)
    try:
        sheet, = book.sheets()
        header = sheet.row_values(1)
        columns = [header.index(name) for name in SCHEMAS["zibo"]]
        target = header.index("\u9884\u671f\u8fbe\u5230\u7684\u76ee\u6807\u548c\u6280\u672f\u6307\u6807")
        entries = []
        for values in sheet.get_rows():
            number, issuer, title, body = [values[column].value for column in columns]
            if not isinstance(number, float) or not number.is_integer():
                continue
            body = str(body).strip()
            if body and (goals := str(values[target].value).strip()):
                body = f"{body}\n{goals}"
            entries.append((int(number), str(issuer).strip(), str(title).strip(), body))
        return to_records("zibo", entries)
    finally:
        book.release_resources()


def table_records(path, name):
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    fields = SCHEMAS[name]
    matches = []
    for table in soup.find_all("table"):
        cells_by_row = [
            tr.find_all(["td", "th"], recursive=False)
            for tr in table.find_all("tr") if tr.find_parent("table") is table
        ]
        for index, cells in enumerate(cells_by_row):
            values = [cell_text(cell) for cell in cells]
            if all(field in values for field in fields):
                matches.append((values, cells_by_row[index + 1:]))
                break
    (header, data), = matches
    sequence, issuer, title, body = (header.index(field) for field in fields)
    entries, inherited = [], []
    index = 0
    while index < len(data):
        cells = data[index]
        if not cells:
            raise ValueError(f"Unexpected empty row in {name}")
        value = cell_text(cells[0])
        if len(cells) == len(header) and value.isdecimal():
            values = [cell_text(cells[column]) for column in (issuer, title, body)]
            span = int(cells[sequence].get("rowspan", 1))
            continuation = data[index + 1:index + span]
            if len(continuation) != span - 1:
                raise ValueError(f"Truncated continuation in {name}")
            if name == "zhangjiagang":
                expected = [1 if col == body else span for col in range(len(header))]
                if [int(cell.get("rowspan", 1)) for cell in cells] != expected or any(
                    len(extra) != 1 or extra[0].get("colspan", "1") != "1" for extra in continuation
                ):
                    raise ValueError("Unexpected demand description continuation structure")
                values[2] = "\n".join([values[2], *(cell_text(extra[0]) for extra in continuation)]).strip()
            elif span > 1:
                # Liaocheng merges the sequence and issuer across distinct requests.
                spanned = {col for col, cell in enumerate(cells) if int(cell.get("rowspan", 1)) == span}
                remaining = [col for col in range(len(header)) if col not in spanned]
                if name != "liaocheng" or not {issuer, sequence} <= spanned or any(
                    len(extra) != len(remaining) for extra in continuation
                ):
                    raise ValueError("Unexpected shared issuer continuation structure")
                for offset, extra in enumerate(continuation, 1):
                    inherited.append((int(value), offset, values[0], cell_text(extra[remaining.index(title)]),
                                      cell_text(extra[remaining.index(body)])))
            entries.append((int(value), *values))
            index += span
            continue
        values = [cell_text(cell) for cell in cells]
        is_blank_footer = index == len(data) - 1 and len(values) == len(header) and not any(values)
        if values != header and not is_blank_footer:
            raise ValueError(f"Unaccounted for table row in {name}")
        index += 1

    records = to_records(name, entries)
    for number, offset, organization, headline, description in inherited:
        record = row(f"{name}-{number}-{offset}", name, headline, description, organization, number)
        record["complete"] = bool(headline.strip() and description.strip())
        records.append(record)
    return records


def planning(groups, protocol):
    floor = protocol["coverage_floor"]
    ceiling = protocol["conditional_risk_ceiling"]
    confidence = protocol["confidence"]
    k = protocol["top_k"]
    boundary = passing_boundary(groups, k, protocol["minimum_grounded_p5_gain"], (1 - confidence) / 2)
    return {
        "groups": groups,
        "single_arm_absolute_pass_probability": [
            dict(coverage=.75, adverse=risk, probability=scenario_power(groups, floor, ceiling, confidence, .75, risk))
            for risk in (.02, .03)
        ],
        "paired_quality_pass_probability": [
            dict(gain=.03, discordance=rate, probability=paired_pass_probability(groups, k, boundary, .03, rate))
            for rate in (.20, .40)
        ],
    }


def run(earlier, municipal, new, workbook, protocol_path, previous, previous_index, output, private):
    if output.exists() or private.exists():
        raise FileExistsError("Require fresh output paths")
    old, old_hashes = freeze_snapshot(earlier)
    first, first_hashes = package(municipal, "first", 30)
    baseline = json.loads(previous.read_bytes())
    lock = json.loads(previous_index.read_bytes())
    if (old_hashes != baseline["prior_public_source_hashes"] or first_hashes != baseline["added_xian_source"]
            or digest_file(protocol_path) != baseline["protocol_sha256"]
            or digest_file(workbook) != lock["historical_workbook_sha256"]):
        raise ValueError("Earlier source, historical workbook, or protocol changed")
    history = historical(workbook)
    old_usable = usable(old + first)
    prior_selected, _ = screen(old_usable + history, set(), set())
    if prior_selected != lock["records"] or len(prior_selected) != baseline["expanded_provisional_groups"]:
        raise ValueError("Could not reproduce previously selected groups")
    prior_signatures = {record["component_sha256"] for record in prior_selected}

    input_files = {name: new / f"{name}.html" for name in COUNTS if name != "zibo"}
    intake = {name: table_records(path, name) for name, path in input_files.items()}
    intake["zibo"] = zibo_records(new / "zibo.xls")
    old_as_exposure = [record | {"split": "prior"} for record in old_usable]
    eligible = usable([record for records in intake.values() for record in records])
    combined, summary = screen(old_usable + eligible + history, set(), set())
    disjoint, exposure = screen(eligible + old_as_exposure + history, set(), set())
    source = {record["id"]: record["batch"] for record in old_usable + eligible}
    combined_by_source = Counter(source[r["id"]] for r in combined)
    disjoint_by_source = Counter(source[r["id"]] for r in disjoint)
    short_by_source = Counter(
        record["batch"] for record in eligible if len(grams(normalize(record["zh"]))) < 64
    )
    source_stats = {
        name: dict(numbered=COUNTS[name], parsed_demands=len(records),
                   complete_title_and_body=sum(record["complete"] for record in records),
                   complete_with_explicit_issuer=len(usable(records)),
                   shorter_than_screen_minimum=short_by_source[name],
                   disjoint_candidate_groups=disjoint_by_source[name],
                   notice_sha256=digest_file(new / ("zibo-notice.html" if name == "zibo" else f"{name}.html")))
        for name, records in sorted(intake.items())
    }
    source_stats["zibo"]["attachment_sha256"] = digest_file(new / "zibo.xls")
    protocol = json.loads(protocol_path.read_bytes())
    region = {name: "xian" for name in old_hashes | {"first": first_hashes}}
    region |= {"nantong": "jiangsu", "jiangyan": "jiangsu", "zhangjiagang": "jiangsu",
               "zibo": "shandong", "liaocheng": "shandong"}
    by_region = Counter(region[source[r["id"]]] for r in combined)
    report = dict(
        sources=source_stats, historical_records=len(history), reproduced_previous_groups=len(prior_selected),
        combined_groups=len(combined), net_groups=len(combined) - len(prior_selected),
        prior_group_signatures_preserved=sum(
            record["component_sha256"] in prior_signatures for record in combined
        ),
        disjoint_new_groups=len(disjoint), combined_by_source=dict(sorted(combined_by_source.items())),
        combined_by_region=dict(sorted(by_region.items())),
        combined_screen={k: v for k, v in summary.items() if k != "original_sealed_groups"},
        new_only_screen={k: v for k, v in exposure.items() if k != "original_sealed_groups"},
        planning_scenarios=[planning(n, protocol) for n in (len(prior_selected), len(combined), 500)],
        protocol_sha256=digest_file(protocol_path), previous_report_sha256=digest_file(previous),
        script_sha256=digest_file(Path(__file__)), expert_queries_run=0, human_review_labels_added=0,
    )
    with private.open("x", encoding="utf-8") as stream:
        private.chmod(0o600)
        json.dump(dict(combined=combined, disjoint=disjoint), stream, ensure_ascii=True)
        stream.write("\n")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("earlier", "municipal", "new", "workbook", "protocol_path", "previous",
                     "previous_index", "output", "private"):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
