"""Freeze an eligibility pilot and validate independent reviewer copies."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory

from experiments.audit_enterprise_panel_neighbors import write_private
from experiments.external_enterprise_frame import COOPERATION, attachments, detail_rows
from experiments.freeze_enterprise_panel import digest_file, issuer_key, read_rows
from experiments.retrieval.enterprise_routed_review import write


OPTIONS = {"pass", "fail", "uncertain"}


def quotas(sizes, total, floor):
    if not sizes or not 0 < floor <= min(sizes.values()):
        raise ValueError("Every source stratum must support the quota floor")
    spare = {source: n - floor for source, n in sizes.items()}
    extras = total - floor * len(sizes)
    capacity = sum(spare.values())
    if not 0 <= extras <= capacity or capacity == 0:
        raise ValueError("Invalid remaining sample quota")
    shares = {source: divmod(extras * n, capacity) for source, n in spare.items()}
    result = {source: floor + whole for source, (whole, _) in shares.items()}
    remainder = total - sum(result.values())
    for source in sorted(shares, key=lambda key: (-shares[key][1], key))[:remainder]:
        result[source] += 1
    return result


def xian_cooperation(archives, selected_ids):
    by_case = {}
    for folder in archives:
        paths = sorted(path for path in folder.glob("xian-*") if path.suffix in {".rar", ".zip"})
        for path in paths:
            for member, content in attachments(path):
                number = re.match(r"[0-9]+", PurePosixPath(member).name)
                key = f"xian-{number[0]}" if number else None
                if key not in selected_ids:
                    continue
                fields = [row[-1] for row in detail_rows(content) if len(row) > 1 and row[0] == COOPERATION]
                if key in by_case or len(fields) != 1:
                    raise ValueError("Original Xi'an cooperation field does not join uniquely")
                by_case[key] = fields[0]
    if by_case.keys() != selected_ids:
        raise ValueError("A frozen Xi'an case lacks its original cooperation field")
    return by_case


def freeze(frame, representatives, source_rows, protocol, output, mask_contact, xian_archives):
    spec = json.loads(protocol.read_bytes())
    if spec["frame_sha256"] != digest_file(frame) or len(representatives) != spec["frame_size"]:
        raise ValueError("Reproduced input does not match the frozen pilot frame")
    by_source = defaultdict(list)
    for item in representatives:
        record = source_rows[item["id"]]
        if (sha256(record["zh"].encode()).hexdigest() != item["text_sha256"]
                or record["source_row"] != item["workbook_row"]):
            raise ValueError("Changed representative source or text")
        title, _, body = record["zh"].partition("\n")
        if not (title.strip() and body.strip() and issuer_key(record["company"])):
            raise ValueError("Incomplete title, technical body, or issuer")
        by_source[record["batch"]].append(item)
    if len(by_source) != spec["source_strata"]:
        raise ValueError("Changed sampling strata")
    population = {source: len(items) for source, items in sorted(by_source.items())}
    selected = quotas(population, spec["pilot_size"], spec["minimum_per_stratum"])
    xi_ids = {item["id"] for item in representatives if item["id"].startswith("xian-")}
    cooperation = xian_cooperation(xian_archives, xi_ids)
    seed = spec["selection_seed"]
    cards, keys = {}, {}
    cooperation_in_pilot = 0
    for source, items in sorted(by_source.items()):
        leaders = sorted(items, key=lambda item: sha256(
            f"{seed}:select:{item['component_sha256']}".encode()).hexdigest())[:selected[source]]
        for item in leaders:
            record = source_rows[item["id"]]
            case = sha256(f"{seed}:blind:{item['component_sha256']}".encode()).hexdigest()[:24]
            title, _, body = record["zh"].partition("\n")
            supplemental = cooperation.get(item["id"])
            cooperation_in_pilot += bool(supplemental)
            cards[case] = dict(
                case_id=case, source=source, locator=item["id"], source_row=record["source_row"],
                issuer=mask_contact(record["company"]), title=mask_contact(title),
                body=mask_contact(body),
                unindexed_original_cooperation=mask_contact(supplemental) if supplemental is not None else None,
                judgments=dict.fromkeys(spec["fields"]),
                reviewer=None, rationale=None,
            )
            keys[case] = dict(case_id=case, component_sha256=item["component_sha256"],
                              original_text_sha256=item["text_sha256"])
    if len(cards) != spec["pilot_size"] or len({v["component_sha256"] for v in keys.values()}) != len(cards):
        raise ValueError("Pilot requires exactly one representative per group")
    ordered = [cards[key] for key in sorted(cards)]
    with TemporaryDirectory(prefix=".eligibility-", dir=output.parent) as directory:
        staging = Path(directory) / "packet"
        staging.mkdir(mode=0o700)
        form = staging / "blank.jsonl"
        for file in (form, staging / "reviewer-a.jsonl", staging / "reviewer-b.jsonl"):
            write_private(file, ordered)
        write_private(staging / "holdout-exclusions.jsonl", [keys[key] for key in sorted(keys)])
        write(staging / "receipt.json", dict(
            protocol_sha256=digest_file(protocol), blank_sha256=digest_file(form),
            exclusions_sha256=digest_file(staging / "holdout-exclusions.jsonl"),
            frame_size=spec["frame_size"], pilot_size=len(cards), remainder_size=spec["frame_size"] - len(cards),
            population_by_source=population, sampled_by_source=selected,
            original_cooperation_fields=dict(
                checked=len(cooperation), nonempty=sum(map(bool, cooperation.values())),
                nonempty_in_pilot=cooperation_in_pilot,
            ),
            independent_reviews=0, expert_queries_run=0,
        ))
        staging.rename(output)


def validate(template, review):
    rows = read_rows(review)
    reviewers = set()
    results = []
    editable = {"reviewer", "rationale", "judgments"}
    for original, row in zip(template, rows, strict=True):
        if row.keys() != original.keys() or any(row[k] != original[k] for k in original.keys() - editable):
            raise ValueError("Review changed a case or its input fields")
        values = row["judgments"]
        if values.keys() != original["judgments"].keys():
            raise ValueError("Review changed the judgment fields")
        if all(choice is None for choice in values.values()):
            if row["reviewer"] is not None or row["rationale"] is not None:
                raise ValueError("Blank review contains attribution")
            results.append(None)
            continue
        if not all(isinstance(choice, str) and choice in OPTIONS for choice in values.values()):
            raise ValueError("Every field needs pass, fail, or uncertain")
        if not all(isinstance(row[key], str) and row[key].strip() for key in ("reviewer", "rationale")):
            raise ValueError("Completed judgments need an attributed rationale")
        reviewers.add(row["reviewer"].strip())
        results.append(values)
    if len(reviewers) > 1:
        raise ValueError("One reviewer per file")
    return results, reviewers


def check(packet, first, second):
    receipt = json.loads((packet / "receipt.json").read_bytes())
    blank = packet / "blank.jsonl"
    exclusions = packet / "holdout-exclusions.jsonl"
    if (receipt["blank_sha256"] != digest_file(blank)
            or receipt["exclusions_sha256"] != digest_file(exclusions)):
        raise ValueError("The baseline review form or group exclusions changed")
    template = read_rows(blank)
    groups = read_rows(exclusions)
    case_ids = [row["case_id"] for row in template]
    if (len(template) != receipt["pilot_size"] or len(set(case_ids)) != len(template)
            or [row["case_id"] for row in groups] != case_ids
            or len({row["component_sha256"] for row in groups}) != len(template)):
        raise ValueError("Changed inventory of unique cases")
    a, ids_a = validate(template, first)
    b, ids_b = validate(template, second)
    if ids_a & ids_b:
        raise ValueError("Independent reviewers must have distinct identifiers")
    statuses = Counter()
    for left, right in zip(a, b, strict=True):
        if left is None or right is None:
            statuses["awaiting_both_reviews"] += 1
        elif left != right or "uncertain" in left.values():
            statuses["adjudication_required"] += 1
        elif "fail" in left.values():
            statuses["agreed_ineligible"] += 1
        else:
            statuses["agreed_eligible"] += 1
    return dict(review_cases=len(template), statuses=statuses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("packet", "first", "second"):
        parser.add_argument(field, type=Path)
    print(json.dumps(check(**vars(parser.parse_args())), indent=2))
