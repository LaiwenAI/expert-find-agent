"""Expand the enterprise trial frame with two pinned published compilations."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import pdfplumber

from experiments.audit_enterprise_panel_neighbors import windows, write_private
from experiments.enterprise_municipal_frame import freeze_snapshot, usable
from experiments.enterprise_regional_frame import COUNTS, planning, table_records, zibo_records
from experiments.external_enterprise_frame import digest_file, historical, package, row
from experiments.freeze_enterprise_panel import grams, normalize, screen
from experiments.retrieval.enterprise_routed_review import write


UNIT = "\u5355\u4f4d\u540d\u79f0"
TITLE = "\u9700\u6c42\u540d\u79f0"
BODY = "\u9700\u6c42\u5185\u5bb9"
DISCLOSURE = "\u540c\u610f\u516c\u5f00\u9700\u6c42\u4fe1\u606f"
YES, NO, PARTIAL = "\u662f", "\u5426", "\u90e8\u5206\u516c\u5f00"
CHECKED = "\uf052\uf0fe"
CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
FUJIAN_HEADER = ("\u5e8f\u53f7", "\u9700\u6c42\u5355\u4f4d", "\u9879\u76ee\u540d\u79f0",
                 "\u4ea7\u4e1a\u6280\u672f\u96be\u9898", "\u6280\u672f\u521b\u65b0\u9700\u6c42\u4fe1\u606f",
                 "\u4ea7\u5b66\u7814\u5408\u4f5c\u9700\u6c42")
FROZEN_PUBLICATIONS = {
    "yangzhou": "c73d5f2ca3f744fbbfd5e05ad798a07a42feb63b3c0e7b70a0bba11d435bf0b7",
    "fujian": "3f6151aba115a6d1aa39f7ae269c7d70eb7beae093e24c9aaebd381e5a248877",
}
FROZEN_HISTORY = "72cf6506ca1e14b5efca48c7d3ac4f7eae07c663de8fb67cbe366eb633cd2624"
CONTACT = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)(?:\+?86[ -]?)?1[3-9]\d{9}(?!\d)")


def key(cell):
    return "".join((cell or "").split())


def text(cell):
    return CJK_SPACE.sub("", " ".join((cell or "").split()))


def content(parts):
    return CONTACT.sub("[contact]", "\n".join(text(x) for x in parts if text(x)))


def value_after(cells, position):
    return next((cell for cell in cells[position + 1:] if key(cell)), "")


def disclosure(cell):
    selected = set(re.findall(f"[{CHECKED}]({YES}|{NO}|{PARTIAL})", key(cell)))
    return "".join(sorted(selected)) if len(selected) == 1 else "unresolved"


def yangzhou(path):
    output, status = [], Counter()
    form = None
    reading_body = False

    def finish(current):
        if current is None:
            return
        p = current["page"]
        flag = current.get("disclosure", "unresolved")
        status[{YES: "full_disclosure", NO: "excluded_no", PARTIAL: "excluded_partial"}.get(
            flag, "excluded_unresolved"
        )] += 1
        if flag == YES:
            output.append(row(
                f"yangzhou2023-p{p}", "yangzhou2023", current.get("title", ""),
                content(current["body"]), current["company"], source_row=p,
            ))

    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 439:
            raise ValueError("Changed Yangzhou page inventory")
        for page in pdf.pages[10:]:
            for table in page.extract_tables():
                for cells in table:
                    if len(cells) not in (5, 6, 7):
                        continue
                    label, field = map(key, cells[:2])
                    if label == UNIT:
                        finish(form)
                        form = dict(page=page.page_number, company=text(value_after(cells, 0)), body=[])
                        reading_body = False
                    if form is None:
                        continue
                    if label == TITLE:
                        form["title"] = content([value_after(cells, 0)])
                    elif label == DISCLOSURE:
                        form["disclosure"] = disclosure(value_after(cells, 0))
                    elif BODY in (label, field):
                        reading_body = True
                        form["body"].append(value_after(cells, int(field == BODY)))
                    elif reading_body:
                        if not field and label in ("", "\u8bf4\u660e", "\u660e"):
                            form["body"].append(value_after(cells, 1))
                        else:
                            reading_body = False
            page.close()
    finish(form)
    if sum(status.values()) != 206:
        raise ValueError("Expected 206 separately identified Yangzhou forms")
    return output, status


def fujian(path):
    records = []
    section = sequence = 0
    current = None

    def finish(item):
        if item:
            number, start, issuer, title, body = item
            records.append(row(f"fujian2025-s{section}-{number}", "fujian2025",
                               text(title), content(body), text(issuer), source_row=start))

    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 46:
            raise ValueError("Changed Fujian page inventory")
        for page in pdf.pages[4:45]:
            tables = [table for table in page.extract_tables() if len(table[0]) in (7, 8)]
            if len(tables) != 1:
                raise ValueError("Demand table does not match the frozen Fujian layout")
            for cells in tables[0]:
                header = tuple(map(key, cells[:6]))
                if header[:2] == FUJIAN_HEADER[:2] and header[3:] == FUJIAN_HEADER[3:] and (
                    header[2] == FUJIAN_HEADER[2] or page.page_number == 45 and header[2] == "\u9875\u76ee\u540d\u79f0"
                ):
                    finish(current)
                    current = None
                    section, sequence = section + 1, 0
                    continue
                number = key(cells[0])
                if number.isdecimal() and int(number) == sequence + 1:
                    finish(current)
                    sequence += 1
                    current = sequence, page.page_number, cells[1], cells[2], list(cells[3:6])
                elif not number and current and not key(cells[1]):
                    # The source occasionally splits the title and description across pages.
                    num, start, issuer, title, body = current
                    joined = [(a or "") + "\n" + (b or "") for a, b in zip(body, cells[3:6], strict=True)]
                    current = num, start, issuer, (title or "") + "\n" + (cells[2] or ""), joined
                else:
                    raise ValueError(f"Unexpected Fujian sequence on page {page.page_number}")
            page.close()
    finish(current)
    if section != 12 or len(records) != 109:
        raise ValueError(f"Changed Fujian publication inventory: {section} sections, {len(records)} requests")
    return records


def old_rows(earlier, municipal, regional, workbook, protocol, report, index, municipal_report):
    old, hashes = freeze_snapshot(earlier)
    first, first_hashes = package(municipal, "first", 30)
    if (hashes != municipal_report["prior_public_source_hashes"]
            or first_hashes != municipal_report["added_xian_source"]
            or digest_file(workbook) != FROZEN_HISTORY):
        raise ValueError("Changed earlier exposure snapshots")
    if digest_file(protocol) != report["protocol_sha256"]:
        raise ValueError("Changed frozen evaluation protocol")
    for name, checks in report["sources"].items():
        notice = regional / ("zibo-notice.html" if name == "zibo" else f"{name}.html")
        if digest_file(notice) != checks["notice_sha256"]:
            raise ValueError("Changed prior regional notice")
    zibo = regional / "zibo.xls"
    if digest_file(zibo) != report["sources"]["zibo"]["attachment_sha256"]:
        raise ValueError("Changed legacy workbook")
    expanded = zibo_records(zibo) + [r for name in COUNTS if name != "zibo"
                                    for r in table_records(regional / f"{name}.html", name)]
    records = usable(old + first + expanded)
    history = historical(workbook)
    reproduced, _ = screen(records + history, set(), set())
    if reproduced != index["combined"] or len(reproduced) != report["combined_groups"]:
        raise ValueError("Prior 355-group frame did not reproduce")
    return records, history, reproduced, hashes | {"first": first_hashes}


def run(args):
    if args.overlap_review and args.projection is None:
        raise ValueError("Reproduce the semantic projection before building an overlap packet")
    destinations = [path for path in (args.output, args.private, args.projection) if path is not None]
    for option in (args.overlap_review, args.eligibility_pilot, args.cooperation_projection):
        if option:
            destinations.append(option[1])
    if len(set(destinations)) != len(destinations) or any(path.exists() for path in destinations):
        raise FileExistsError("Require fresh output paths")
    if any(digest_file(getattr(args, name)) != fingerprint
           for name, fingerprint in FROZEN_PUBLICATIONS.items()):
        raise ValueError("Changed public PDF snapshot")
    prior, history, selected, old_hashes = old_rows(
        args.earlier, args.municipal, args.regional, args.workbook, args.protocol,
        json.loads(args.previous.read_bytes()), json.loads(args.previous_index.read_bytes()),
        json.loads(args.municipal_report.read_bytes()),
    )
    yang, choices = yangzhou(args.yangzhou)
    fu = fujian(args.fujian)
    new = usable([r for r in yang + fu if r["zh"].partition("\n")[0]])
    combined, summary = screen(prior + new + history, set(), set())
    disjoint, exposure = screen(new + [r | {"split": "prior"} for r in prior] + history, set(), set())
    batches = {r["id"]: r["batch"] for r in prior + new}
    retained = {r["component_sha256"] for r in combined}
    spec = json.loads(args.protocol.read_bytes())
    write(args.private, dict(combined=combined, disjoint=disjoint,
                             prior_index_sha256=digest_file(args.previous_index)))
    if args.cooperation_projection:
        from experiments.enterprise_cooperation_projection import build

        pilot, output = args.cooperation_projection
        build(args.private, combined, prior + new, history, summary, (args.earlier, args.municipal),
              pilot, output, lambda value: CONTACT.sub("[contact]", value))
    if args.eligibility_pilot:
        from experiments.enterprise_eligibility_pilot import freeze

        protocol, output = args.eligibility_pilot
        freeze(args.private, combined, {r["id"]: r for r in prior + new}, protocol,
               output, lambda value: CONTACT.sub("[contact]", value), (args.earlier, args.municipal))
    if args.projection is not None:
        representatives = selected + disjoint
        if ({r["id"]: r for r in representatives} != {r["id"]: r for r in combined}
                or len(representatives) != len(combined)):
            raise ValueError("Combined exposure groups must match the disjoint frame")
        rows = {r["id"]: r for r in prior + new + history}
        external = [rows[r["id"]] for r in representatives]
        texts = [r["zh"] for r in external + history]
        views = [windows(CONTACT.sub("[contact]", value)) for value in texts]
        write_private(args.projection, [dict(
            frame_sha256=digest_file(args.private), external=len(combined), earlier=len(selected),
            focus=list(range(len(selected), len(combined))), views=views,
        )])
        if args.overlap_review:
            from experiments.external_enterprise_overlap import build_packet

            screen_path, review_directory = args.overlap_review
            semantic = json.loads(screen_path.read_bytes())
            if (semantic["frame_sha256"] != digest_file(args.private)
                    or semantic["projection_sha256"] != digest_file(args.projection)
                    or semantic["earlier_groups"] != len(selected)):
                raise ValueError("Semantic screen does not bind the reproduced frame and projection")
            external_masked = [r | {"zh": CONTACT.sub("[contact]", r["zh"])} for r in external]
            historical_masked = [r | {"zh": CONTACT.sub("[contact]", r["zh"])} for r in history]
            build_packet(external_masked, historical_masked, range(len(selected), len(combined)),
                         semantic, review_directory, earlier=len(selected),
                         salt="enterprise-public-compendium-v4")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(dict(
            sources={
                "yangzhou2023": dict(sha256=digest_file(args.yangzhou), disclosure=choices),
                "fujian2025": dict(sha256=digest_file(args.fujian), listed=len(fu)),
            },
            eligible_before_grouping=Counter(r["batch"] for r in new),
            short_by_source=Counter(r["batch"] for r in new if len(grams(normalize(r["zh"]))) < 64),
            historical_records=len(history), prior_groups=len(selected),
            combined_groups=len(combined), net_groups=len(combined) - len(selected),
            prior_signatures_preserved=sum(r["component_sha256"] in retained for r in selected),
            combined_by_source=Counter(batches[r["id"]] for r in combined),
            new_disjoint_by_source=Counter(batches[r["id"]] for r in disjoint),
            combined_screen={k: v for k,v in summary.items() if k != "original_sealed_groups"},
            new_only_screen={k: v for k,v in exposure.items() if k != "original_sealed_groups"},
            previous_publication_hashes=old_hashes,
            historical_sha256=digest_file(args.workbook), protocol_sha256=digest_file(args.protocol),
            script_sha256=digest_file(Path(__file__)), prior_report_sha256=digest_file(args.previous),
            scenarios=[planning(n, spec) for n in (len(selected), len(combined))],
            expert_queries_run=0, human_review_labels_added=0,
        ), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("earlier", "municipal", "regional", "workbook", "protocol", "previous",
                  "previous_index", "municipal_report", "yangzhou", "fujian", "output", "private"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--overlap-review", nargs=2, type=Path,
                        metavar=("SEMANTIC_SCREEN", "NEW_REVIEW_DIRECTORY"))
    parser.add_argument("--eligibility-pilot", nargs=2, type=Path,
                        metavar=("PILOT_PROTOCOL", "NEW_PILOT_DIRECTORY"))
    parser.add_argument("--cooperation-projection", nargs=2, type=Path,
                        metavar=("FROZEN_PILOT_DIRECTORY", "NEW_PRIVATE_PROJECTION_DIRECTORY"))
    run(parser.parse_args())
