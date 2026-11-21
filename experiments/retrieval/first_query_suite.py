"""Build a reproducible first-query scorecard from frozen paired evaluation runners."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/first-query-suite-v1/protocol.json"


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def ordered_runs(path, pins):
    runs = load(path)["runs"]
    if len(runs) != 2 or {item["reverse"] for item in runs} != {False, True}:
        raise ValueError("Each result needs forward and reverse runs")
    for run in runs:
        if any(run[field] != value for field, value in pins.items()):
            raise ValueError(f"Changed input fingerprint: {path}")
    return sorted(runs, key=lambda run: run["reverse"])


def timing(left, right):
    values = [float(left), float(right)]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Nonpositive or nonfinite timing")
    return sorted(values)


def observed(arm, n):
    if not 0 <= arm["known"] <= arm["issued"] <= n:
        raise ValueError("Invalid observed proof counts")
    return {"known": arm["known"], "issued": arm["issued"]}


def open_topic(first, second, spec):
    records = {}
    for name, panel in spec["panels"].items():
        a, b = (run["panels"][name] for run in (first, second))
        n, quota = panel["queries"], panel["issued_quota"]
        if a["matched_reads_per_arm"] != 3 * n or b["matched_reads_per_arm"] != 3 * n:
            raise ValueError("Open-topic read budget changed")
        if (a["cases"] != b["cases"] or a["decision_signature"] != b["decision_signature"]
                or a["reachable_after_reads"] != b["reachable_after_reads"]):
            raise ValueError("Open-topic outcomes differ across replay orders")
        counts = a["cases"]
        rows = {}
        for arm, natural in counts["natural"]["arms"].items():
            matched = counts["quota"]["arms"][arm]
            if matched["issued"] != quota:
                raise ValueError("Matched issuance changed; use the frozen quota")
            rows[arm] = {
                "natural": observed(natural, n), "matched": observed(matched, n),
                "reachable_after_reads": a["reachable_after_reads"][arm],
                "local_p95_ms": timing(a["p95_ms"][arm], b["p95_ms"][arm]),
            }
        records[name] = {
            "queries": n, "reads_per_query": 3, "arms": rows,
            "paired_known_delta_per_query": counts["quota"]["paired_known_delta_per_query"],
            "missing_label_count_bounds": counts["quota"]["joint_minus_sequential_missing_bounds"],
        }
    return records


def linked_role(first, second, spec):
    records = {}
    for name, n in spec["panels"].items():
        a, b = (run["panels"][name] for run in (first, second))
        if a["query_count"] != n or b["query_count"] != n or a["index"] != b["index"]:
            raise ValueError("Linked-role cohort or index changed between replay orders")
        for depth in spec["depths"]:
            x, y = a["depths"][str(depth)], b["depths"][str(depth)]
            if ({k: v for k, v in x.items() if k != "p95_ms"} !=
                    {k: v for k, v in y.items() if k != "p95_ms"}):
                raise ValueError("Linked-role outputs differ across replay orders")
            if x["verified_exact"] != n or x["reads_per_arm"] != 3 * n:
                raise ValueError("The same-output claim or the three-read budget failed")
            records[f"{name}_{depth}"] = {
                "queries": n, "sources": a["sources"], "known": x["tagged"], "issued": x["issued"],
                "exact_plan_and_response": x["verified_exact"], "reads_per_query": 3,
                "full_role_bytes": a["index"]["full"]["role_bytes"],
                "linked_role_bytes": a["index"]["linked_only"]["role_bytes"],
                "packed_role_bytes": a["packed_projected_payload_bytes"],
                "extra_link_mask_bytes": a["adjacency_payload_bytes"],
                "arms": {
                    name: dict(role_rows_mean=rows, planning_p95_ms=timing(
                        x["p95_ms"][name + "_plan"], y["p95_ms"][name + "_plan"]),
                        local_p95_ms=timing(x["p95_ms"][name], y["p95_ms"][name]))
                    for name, rows in x["planning_rows_mean"].items()
                },
            }
    return records


def trec(first, second, spec):
    n = spec["queries"]
    if (first["topic_count"] != n or second["topic_count"] != n
            or first["nomination_universe"] != second["nomination_universe"]
            or first["comparison_signature"] != second["comparison_signature"]
            or first["results"] != second["results"]):
        raise ValueError("TREC replay cohorts or outcomes differ")
    arms = {}
    for name, data in first["results"].items():
        arms[name] = {k: data[k] for k in ("supported", "issued", "reads", "checks", "oracle_ceiling")}
        arms[name]["supported_p5"] = data["supported"] / (5 * n)
        arms[name]["paired_p5_pp_vs_fitted"] = data["paired_p_at_5_pp"]["fitted_legacy"]
    return {"trec_2006": {"queries": n, "arms": arms}}


def cof(first, second, spec):
    records = {}
    for name, panel in spec["panels"].items():
        a, b = (run["results"][name] for run in (first, second))
        if (a["queries"] != panel["queries"] or b["queries"] != panel["queries"]
                or a["signature"] != b["signature"] or a["outcomes"] != b["outcomes"]):
            raise ValueError("CoF replay cohorts or outcomes differ")
        arms = {}
        for arm, result in a["outcomes"].items():
            matched = result["matched_issuance"]
            if matched["cited"] != panel["cited_quota"]:
                raise ValueError("CoF matched citation count changed")
            arms[arm] = {
                "matched": matched,
                "reads": result["full"]["reads"], "tokens": result["full"]["tokens"],
                "paired_known_p5_pp": result["paired_matched_p5_delta_pp"],
                "missing_label_p5_pp": result["matched_p5_partial_label_interval_pp"],
            }
        records[name] = {"queries": panel["queries"], "arms": arms}
    return records


ADAPTERS = {"open_topic": open_topic, "linked_role": linked_role, "trec": trec, "cof": cof}


def quality_section(spec, optional):
    protocol_path = ROOT / spec["protocol"]
    if digest(protocol_path) != spec["protocol_sha256"]:
        raise ValueError("Changed frozen enterprise scoring protocol")
    if optional is None:
        return {"status": "awaiting_independent_reviews", "scope": spec["scope"], "groups": spec["groups"]}
    result = load(optional)
    protocol = load(protocol_path)
    if ((result["study"], result["groups"], result["primary"], result["comparator"]) !=
            (protocol["study"], spec["groups"], protocol["primary"], protocol["primary_comparator"])
            or set(result["arms"]) != set(protocol["arms"])):
        raise ValueError("Enterprise report differs from the registered blinded study")
    return {"status": "adjudicated_diagnostic", "scope": spec["scope"],
            "sha256": digest(optional), "score": result}


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def render_markdown(path, report):
    def pair(a, b):
        return f"{a} / {b}"

    def span(values):
        return "\u2013".join(f"{value:.3f}" for value in values)

    columns = {
        "open_topic": ("Tag-supported / issued", "At frozen quota", "Local P95 (ms)"),
        "linked_role": ("Exact plans / requests", "Planning P95 (ms)", "Local P95 (ms)"),
        "trec": ("Supported / issued", "Reads", "Checks"),
        "cof": ("Known relevant / matched cited", "Reads", "Tokens"),
    }
    titles = {
        "open_topic": "PLOS and eLife open topic",
        "linked_role": "Independent eLife exact role-index comparison",
        "trec": "TREC Enterprise 2006",
        "cof": "CoF reviewer relevance",
    }
    lines = ["# Frozen first-query comparisons", "",
             "Separate panels use their registered label and timing definitions.", ""]
    for key, track in report["public"].items():
        headers = ("Version", "Panel", "Arm", *columns[key])
        lines.extend([f"## {titles[key]}", "", "| " + " | ".join(headers) + " |",
                      "|" + "|".join(" --- " for _ in headers) + "|"])
        for version in ("reference", "candidate"):
            for panel, record in track.get(version, {}).items():
                for arm, row in record["arms"].items():
                    if key == "open_topic":
                        natural, matched = row["natural"], row["matched"]
                        measures = (pair(natural["known"], natural["issued"]),
                                    pair(matched["known"], matched["issued"]), span(row["local_p95_ms"]))
                    elif key == "linked_role":
                        measures = (pair(record["exact_plan_and_response"], record["queries"]),
                                    span(row["planning_p95_ms"]), span(row["local_p95_ms"]))
                    elif key == "trec":
                        measures = (pair(row["supported"], row["issued"]), row["reads"], row["checks"])
                    else:
                        measures = (pair(row["matched"]["known"], row["matched"]["cited"]),
                                    row["reads"], row["tokens"])
                    lines.append("| " + " | ".join(map(str, (version, panel, arm, *measures))) + " |")
        lines.append("")
    lines.extend(["## Enterprise diagnostic", "",
                  f"{report['enterprise']['groups']} frozen groups. Status is {report['enterprise']['status'].replace('_', ' ')}.",
                  ""])
    with path.open("x", encoding="utf-8") as out:
        out.write("\n".join(lines))


def collect(manifest, output, updates, enterprise, markdown=None):
    if output.exists() or (markdown is not None and (markdown.exists() or output == markdown)):
        raise FileExistsError("Require fresh, distinct report paths")
    settings = load(manifest)
    requested = {}
    for item in updates:
        name, separator, path = item.partition("=")
        if not separator or name not in settings["public_tracks"] or name in requested:
            raise ValueError("Use one TRACK=RESULT.json per registered replacement")
        requested[name] = Path(path)
    tracks = {}
    for key, spec in settings["public_tracks"].items():
        frozen = ROOT / spec["archive"]
        if digest(frozen) != spec["archive_sha256"]:
            raise ValueError(f"Frozen benchmark archive changed: {key}")
        adapter = ADAPTERS[key]
        reference = adapter(*ordered_runs(frozen, spec["pins"]), spec)
        track = {"reference_sha256": digest(frozen), "reference": reference}
        if key in requested:
            updated = requested[key]
            candidate = adapter(*ordered_runs(updated, spec["pins"]), spec)
            for panel, result in reference.items():
                current = candidate[panel]
                if not result["arms"].keys() <= current["arms"].keys():
                    raise ValueError(f"A registered comparison is missing on {key}/{panel}")
                if key == "linked_role" and any(
                    current[field] != result[field] for field in (
                        "queries", "sources", "known", "issued", "exact_plan_and_response",
                    )
                ):
                    raise ValueError(f"Exact proof output changed on {key}/{panel}")
                for arm in spec["frozen_controls"]:
                    left = {k: v for k, v in result["arms"][arm].items() if not k.endswith("p95_ms")}
                    right = {k: v for k, v in current["arms"][arm].items() if not k.endswith("p95_ms")}
                    if left != right:
                        raise ValueError(f"Frozen control changed on {key}/{panel}/{arm}")
            track |= {"candidate_sha256": digest(updated), "candidate": candidate}
        tracks[key] = track
    report = {"suite": settings["suite"], "manifest_sha256": digest(manifest), "public": tracks,
              "enterprise": quality_section(settings["enterprise"], enterprise)}
    write(output, report)
    if markdown is not None:
        render_markdown(markdown, report)


def replay(manifest, track, output, inputs):
    spec = load(manifest)["public_tracks"][track]
    if output.exists() or len(inputs) != spec["input_count"]:
        raise ValueError("Require a fresh output and the registered runner's input count")
    inputs = [str(Path(item).resolve()) for item in inputs]
    with TemporaryDirectory(dir=output.parent) as path:
        first, second = (Path(path) / f"{name}.json" for name in ("forward", "reverse"))
        for target, flags in ((first, []), (second, ["--reverse"])):
            subprocess.run([sys.executable, "-m", spec["runner"], *inputs, str(target), *flags],
                           cwd=ROOT, check=True)
        result = {"runs": [load(first), load(second)]}
        temporary = Path(path) / "paired.json"
        write(temporary, result)
        ADAPTERS[track](*ordered_runs(temporary, spec["pins"]), spec)
        write(output, result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    scores = sub.add_parser("collect")
    scores.add_argument("output", type=Path)
    scores.add_argument("--result", action="append", default=[])
    scores.add_argument("--enterprise-result", type=Path)
    scores.add_argument("--markdown", type=Path)
    run = sub.add_parser("run")
    run.add_argument("track", choices=tuple(ADAPTERS))
    run.add_argument("output", type=Path)
    run.add_argument("inputs", nargs="+")
    args = cli.parse_args()
    if args.command == "collect":
        collect(MANIFEST, args.output, args.result, args.enterprise_result, args.markdown)
    else:
        replay(MANIFEST, args.track, args.output.resolve(), args.inputs)
