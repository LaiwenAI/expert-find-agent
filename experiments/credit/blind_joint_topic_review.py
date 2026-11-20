"""Freeze blind evidence cards and score completed independent reviews."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from experiments.credit import cross_source_identity as identities
from experiments.followir.core17_reranking import digest, write_json


DIMENSIONS = ("topic_a", "role_a", "topic_b", "role_b", "same_person")
VERDICTS = {"supported", "contradicted", "unclear"}


def compact(node):
    return " ".join("".join(node.itertext()).split())


def source_paths(folder):
    result = {}
    for path in identities.xml_sources(folder):
        meta = ET.parse(path).getroot().find("./front/article-meta")
        doi = meta.findtext("article-id[@pub-id-type='doi']")
        result[doi] = path
    return result


def citation(catalogue, doi, person, paths):
    authors = [row for row in catalogue.ownership[doi]["authors"] if row["person"] == person]
    if len(authors) != 1:
        raise ValueError("Expected a unique credited author in the original source")
    ordinal = authors[0]["ordinal"]
    document = ET.parse(paths[doi]).getroot()
    meta = document.find("./front/article-meta")
    original = meta.findall("contrib-group/contrib[@contrib-type='author']")[ordinal-1]
    name = original.find("name")
    identifier = original.findtext("contrib-id[@contrib-id-type='orcid']")
    if not identifier or not identities.orcid_id(identifier):
        raise ValueError("Expected a publisher ORCID for the cited contributor")
    notes = {node.get("id"): compact(node) for node in document.findall("./back//fn[@fn-type='con']")}
    footnotes = {item for ref in original.findall("xref[@ref-type='fn']") for item in ref.get("rid", "").split()}
    direct = [compact(role) for role in original.findall("role")]
    credited = direct or [notes[key] for key in sorted(footnotes & notes.keys())]
    if not credited:
        raise ValueError("Cited author has no original contribution excerpt")
    title, abstract = catalogue.documents[doi]
    return dict(
        doi=doi, title=title, abstract=abstract, author_position=ordinal,
        author=" ".join(compact(part) for part in name if compact(part)) if name is not None else None,
        publisher_orcid=identities.orcid_id(identifier), original_credit=credited,
    )


def stable_id(*pieces):
    return sha256(json.dumps(pieces, ensure_ascii=True).encode()).hexdigest()[:24]


def make_cards(rows, traces, reference, metadata, trial):
    all_cards = {}
    key = {}
    for name, panel_rows in rows.items():
        catalogue, paths = metadata[name]
        targets = reference["panels"][name]["arms"]
        desired = targets["full"]["calibrated"]["issued"]
        known_selection = {}
        for mode in trial.MODES:
            cuts = trial.matched_issuance(panel_rows, desired, mode)
            stats = trial.score_panel(panel_rows, mode, cuts)
            if stats != targets[mode]["matched_to_full_issuance"]:
                raise ValueError(f"Frozen issued counts changed for {name}/{mode}")
            known_selection[mode] = cuts
        entries = []
        for j, ((topic_pair, estimates), (atoms, candidates)) in enumerate(zip(panel_rows, traces[name], strict=True)):
            outputs = {}
            for mode in trial.MODES:
                if not trial.qualifies(estimates[mode], known_selection[mode][j]):
                    outputs[mode] = None
                    continue
                person, a, b = candidates[mode]
                uid = stable_id(name, atoms, person, a, b)
                outputs[mode] = uid
                if uid not in all_cards:
                    all_cards[uid] = dict(
                        case_id=uid, requirements=[{"topic": topic, "role": role} for topic, role in atoms],
                        sources=[citation(catalogue, doi, person, paths) for doi in (a, b)],
                    )
            entries.append(dict(topic_pair=topic_pair, outputs=outputs))
        key[name] = dict(queries=len(panel_rows), fixed_issuance=desired, rows=entries)
    ordered = sorted(all_cards.values(), key=lambda item: item["case_id"])
    return ordered, key


def build(args):
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit import selective_joint_topic_check as trial
    from experiments.followir.core17_verifier import Reranker

    base, live = trial.BASE, trial.live
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = live.verify_embedding(args.embedding)
    encoder, reader = QwenTorchEmbedding(args.embedding), Reranker(args.reranker)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, checksum = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (checkpoint, checksum, reader.provenance["revision"]) != (
        expected["embedding_checkpoint"], expected["elife_projection_sha256"], expected["reranker_checkpoint"],
    ):
        raise ValueError("Frozen inputs changed")
    plos_subjects = base.publisher_topics([args.plos_xml])
    life_subjects, _, life_cases = live.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_subjects, base.query_panel(base.annotation_postings(plos, plos_subjects))),
                  elife=(life, life_subjects, life_cases))
    rows, traces = {}, {}
    with TemporaryDirectory() as scratch:
        for name, (catalogue, subjects, queries) in panels.items():
            index = live.LocalIndex(catalogue, encoder, Path(scratch) / name)
            traces[name] = []
            rows[name] = trial.collect(index, reader, subjects, queries, 0, audit_rows=traces[name])
    truth = trial.report(rows["plos"], rows)
    if (truth["fold_cutoffs"], truth["transfer_cutoffs"]) != (expected["fold_cutoffs"], expected["transfer_cutoffs"]):
        raise ValueError("Frozen cutoffs changed")
    metadata = dict(plos=(plos, source_paths(args.plos_xml)), elife=(life, source_paths(args.elife_xml)))
    cards, key = make_cards(rows, traces, expected, metadata, trial)
    packet_id = stable_id(cards)
    write_json(args.packet, dict(packet_id=packet_id, dimensions=DIMENSIONS, cards=cards))
    write_json(args.key, dict(packet_id=packet_id, reference_sha256=digest(args.reference), panels=key))
    print(f"Frozen {len(cards)} distinct evidence cards")


def read_review(path, packet_id, expected):
    contents = json.loads(path.read_bytes())
    if contents.get("packet_id") != packet_id or not str(contents.get("reviewer", "")).strip():
        raise ValueError("Review requires its packet ID and a reviewer identifier")
    labels = contents["labels"]
    if set(labels) - expected:
        raise ValueError("Review contains unfamiliar cases")
    for item in labels.values():
        if set(item) != set(DIMENSIONS) or any(v not in VERDICTS for v in item.values()):
            raise ValueError("Each reviewed case needs all five judgments")
    return contents["reviewer"], labels


def verdict(case, reviews):
    if any(case not in labels for labels in reviews):
        return None
    judgments = [labels[case] for labels in reviews]
    if all(all(value == "supported" for value in row.values()) for row in judgments):
        return True
    if any(all(row[dimension] == "contradicted" for row in judgments) for dimension in DIMENSIONS):
        return False
    return None


def score(args):
    key = json.loads(args.key.read_bytes())
    panels = key["panels"]
    case_ids = {uid for panel in panels.values() for row in panel["rows"]
                for uid in row["outputs"].values() if uid is not None}
    reviews = dict(read_review(path, key["packet_id"], case_ids) for path in args.reviews)
    if len(reviews) < 2 or len(reviews) != len(args.reviews):
        raise ValueError("Use at least two distinct reviewer identifiers")
    judgments = {case: verdict(case, list(reviews.values())) for case in case_ids}
    result = {}
    for name, panel in panels.items():
        arms = {}
        for mode in panel["rows"][0]["outputs"]:
            issued = [row["outputs"][mode] for row in panel["rows"] if row["outputs"][mode] is not None]
            known = sum(judgments[case] is True for case in issued)
            unresolved = sum(judgments[case] is None for case in issued)
            weights = Counter(issued)
            weights.subtract(row["outputs"]["full"] for row in panel["rows"] if row["outputs"]["full"] is not None)
            lower = upper = 0
            for case, weight in weights.items():
                if judgments[case] is True:
                    lower += weight
                    upper += weight
                elif judgments[case] is None:
                    lower += min(weight, 0)
                    upper += max(weight, 0)
            arms[mode] = dict(issued=len(issued), supported=known, unresolved=unresolved,
                             supported_bounds=[known, known+unresolved],
                             supported_delta_vs_full_bounds=[lower, upper])
        result[name] = dict(queries=panel["queries"], arms=arms)
    write_json(args.output, dict(packet_id=key["packet_id"], reviewers=len(reviews),
                                  conclusive_cases=sum(value is not None for value in judgments.values()),
                                  panels=result))


def render(args):
    packet = json.loads(args.packet.read_bytes())
    embedded = json.dumps(packet, ensure_ascii=True).replace("<", chr(92) + "u003c")
    template = Path(__file__).with_suffix(".html").read_text()
    marker = "__REVIEW_PACKET__"
    if template.count(marker) != 1:
        raise ValueError("Expected one packet marker in the reviewer template")
    with args.output.open("x") as stream:
        stream.write(template.replace(marker, embedded))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("build")
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reranker", "reference", "packet", "key"):
        freeze.add_argument(name, type=Path)
    view = commands.add_parser("render")
    view.add_argument("packet", type=Path)
    view.add_argument("output", type=Path)
    adjudicate = commands.add_parser("score")
    adjudicate.add_argument("key", type=Path)
    adjudicate.add_argument("output", type=Path)
    adjudicate.add_argument("reviews", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "build":
        if not os.environ.get("SLURM_JOB_ID") or args.packet.exists() or args.key.exists():
            raise ValueError("Require a Slurm allocation and fresh output paths")
        build(args)
    elif args.command == "render":
        render(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        score(args)


if __name__ == "__main__":
    main()
