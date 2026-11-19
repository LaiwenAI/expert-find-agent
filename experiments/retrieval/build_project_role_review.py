"""Bind provisional role rescues to exact private views and audit equal-check controls."""

import argparse
import json
from pathlib import Path

from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval.enterprise_project_role_rescue import ARMS, profile_subset
from experiments.retrieval.enterprise_routed_review import digest, write


def build(dataset, panel, trace, output):
    if output.exists():
        raise FileExistsError(output)
    lineage = json.loads(trace.read_bytes())
    if lineage["prepared_sha256"] != digest(panel):
        raise ValueError("Changed proposed spans")
    prepared = json.loads(panel.read_bytes())
    if prepared["private_manifest_sha256"] != digest(dataset / "manifest.json"):
        raise ValueError("Changed private manifest")
    manifest = json.loads((dataset/"manifest.json").read_bytes())
    queries, experts, corpus = (profile_subset(dataset, key, manifest) for key in ("queries","experts","corpus"))
    cases = review.index(prepared["cases"], lambda row:(row["query"], row["expert"]))
    observed = review.index(lineage["private_results"], lambda row:(row["query"], row["expert"]))
    if observed.keys() != cases.keys():
        raise ValueError("Incomplete role-rescue trace")
    methods = {arm:{qid:[] for qid in queries} for arm in ARMS}
    paired = dict(eligible_slots=0, whole_source_first_supported=0, role_passage_first_supported=0)
    for key, result in observed.items():
        qid, expert = key
        for arm in ARMS:
            pool = {(v["id"], v["start"], v["end"]):v for v in cases[key]["options"][arm]}
            witnesses = []
            for item in result["decisions"][arm]:
                slot = item["id"], item["start"], item["end"]
                if slot not in pool or item["roles"] != pool[slot]["roles"]:
                    raise ValueError("Result must come from a prepared role-bearing span")
                source = corpus[item["id"]]
                if source["text"][item["start"]:item["end"]] != pool[slot]["text"]:
                    raise ValueError("Source text and prepared role span differ")
                if item["supported"]:
                    witnesses.append(dict(source_id=item["id"],start=item["start"],end=item["end"]))
            if witnesses:
                methods[arm][qid].append((expert, review.sources(corpus, expert, witnesses)))
        if all(result["decisions"][arm] for arm in ARMS):
            paired["eligible_slots"] += 1
            for arm in ARMS:
                paired[f"{arm}_first_supported"] += result["decisions"][arm][0]["supported"]
    forms, mapping, _, counts = review.packet_rows(queries, experts, methods)
    for form, private in zip(forms, mapping, strict=True):
        info = cases[(private["query_id"],private["expert_id"])]
        form["prior_qualifications"] = info["criteria"][1:]
    output.mkdir(mode=0o700)
    blind, private = output/"blind-review.jsonl", output/"private-mapping.jsonl"
    write(blind, forms, lines=True)
    write(private, mapping, lines=True)
    write(output/"manifest.json", dict(
        queries=len(queries), candidate_views=len(forms),
        outputs_sha256={p.name:digest(p) for p in (blind,private)},
        inputs_sha256=dict(private_trace=digest(trace), prepared_panel=digest(panel)),
    ))
    report = review.check(output, blind)
    if any(set(field) != {"missing"} for field in report["counts"].values()):
        raise ValueError("Expected unlabeled blinded forms")
    for reviewer in ("a","b"):
        write(output/f"reviewer-{reviewer}.jsonl", forms, lines=True)
    write(output/"summary.json", dict(
        review_cases=len(forms), human_judgments=0, paired_first_check_conditioned_on_both_eligible=paired,
        packet_sha256=digest(output/"manifest.json"), **counts,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","panel","trace","output"):
        cli.add_argument(name,type=Path)
    build(**vars(cli.parse_args()))
