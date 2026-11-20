"""Measure joint nomination in a mixed publisher catalog with fixed requests."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import cross_source_identity as ids
from experiments.credit import joint_nomination_blind_review as audit
from experiments.credit import live_joint_role_postings as typed
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("owner_gap", "joint", "personal_roles")


def raw_identifiers(folder):
    return {
        valid
        for path in ids.xml_sources(folder)
        for node in ET.parse(path).getroot().findall(".//contrib[@contrib-type='author']/contrib-id[@contrib-id-type='orcid']")
        if (valid := ids.orcid_id(node.text or ""))
    }


def merge(plos, elife):
    if plos.documents.keys() & elife.documents.keys() or {doc[0].casefold() for doc in plos.documents.values()} & {
        doc[0].casefold() for doc in elife.documents.values()
    }:
        raise ValueError("Publisher catalogs intersect")
    # These snapshots exclude cross-publisher identity matches. Namespace
    # fixture-local IDs so a token collision cannot create a new person.
    ownership = {}
    for publisher, catalog in (("plos", plos), ("elife", elife)):
        for doi, data in catalog.ownership.items():
            authors = [
                {**author, "person": f"{publisher}/{author['person']}" if author["person"] else None}
                for author in data["authors"]
            ]
            ownership[doi] = dict(data, authors=authors)
    return live.BASE.pinned.Catalogue(plos.documents | elife.documents, ownership)


def balanced(ranks, publishers, cap):
    output = []
    for facet in ranks:
        counts = Counter()
        selected = []
        for doi in facet:
            publisher = publishers[doi]
            if counts[publisher] < cap:
                selected.append(doi)
                counts[publisher] += 1
        output.append({doi: i for i, doi in enumerate(selected, 1)})
    return tuple(output)


def source_plan(index, postings, atoms, ranks, arm, depth):
    if arm != "personal_roles":
        return live.inspect(index, atoms, ranks, arm, depth=depth)
    initial, extras = live.simple_order(ranks, depth=depth)
    facets = tuple(postings.facet(ranking, duty, 100, True)
                   for ranking, (_, duty) in zip(ranks, atoms, strict=True))
    plan = typed.source_plan(ranks, facets, [*initial, extras[0]], depth)
    return plan, index.open_sources(plan)


def reference_in_pool(postings, subjects, atoms, ranks):
    eligible = []
    for (topic, duty), facet in zip(atoms, ranks, strict=True):
        records = defaultdict(set)
        for doi in facet:
            if topic not in subjects.get(doi, ()):
                continue
            for person, roles in postings.personal[doi].items():
                if duty in roles:
                    records[person].add(doi)
        eligible.append(records)
    left, right = eligible
    return any(len(left[person] | right[person]) > 1 for person in left.keys() & right.keys())


def count(index, postings, queries, subjects, publishers, mixed, offset):
    views = ("global64", "balanced64", "global128") if mixed else ("original64",)
    results = {view: dict(pool_headroom=0, arms={arm: Counter() for arm in ARMS}) for view in views}
    examples = {view: [] for view in views}
    for number, atoms in enumerate(queries):
        full = index.rank([topic for topic, _ in atoms], depth=len(index.ids) if mixed else 64)
        if mixed:
            small = tuple(dict(list(facet.items())[:64]) for facet in full)
            wide = tuple(dict(list(facet.items())[:128]) for facet in full)
            quotas = balanced(full, publishers, 32)
            if any(len(facet) != 64 for facet in quotas):
                raise ValueError("A balanced view must contain 32 sources per publisher")
            rankings = dict(global64=small, balanced64=quotas, global128=wide)
        else:
            rankings = dict(original64=full)
        for view, ranks in rankings.items():
            depth = 128 if view == "global128" else 64
            sample = {}
            results[view]["pool_headroom"] += reference_in_pool(postings, subjects, atoms, ranks)
            shift = (number + offset) % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                plan, loaded = source_plan(index, postings, atoms, ranks, arm, depth)
                if len(plan) != 3 or set(plan) != loaded.keys():
                    raise AssertionError("Every method must open three distinct declaration files")
                candidates = live.BASE.choose_proofs(
                    live.BASE.read_articles(SimpleNamespace(ownership=loaded), plan, atoms, ranks), ranks,
                )

                def supported(proof):
                    return all(topic in subjects.get(doi, ())
                               for (topic, _), doi in zip(atoms, proof[1:], strict=True))

                first = candidates[0] if candidates else None
                if first:
                    side_ranks = [ranks[i][doi] for i, doi in enumerate(first[1:])]
                    person = f"{publishers[first[1]]}/{first[0].split('/', 1)[-1]}"
                    sample[arm] = dict(match=(person, *first[1:]), known=supported(first),
                                       priority=(max(side_ranks), sum(side_ranks)))
                else:
                    sample[arm] = dict(match=None, known=False, priority=None)
                outcome = results[view]["arms"][arm]
                outcome.update(
                    known=int(bool(first) and supported(first)),
                    observable_proof_after_reads=int(any(map(supported, candidates))),
                    issued=int(bool(first)), reads=len(loaded),
                    unjudged_other_publisher=int(bool(first) and any(doi not in subjects for doi in first[1:])),
                )
            examples[view].append(sample)
    return results, examples


def matched_results(queries, examples):
    contrasts = {
        "global64_vs_original64": ("original64", "global64"),
        "balanced64_vs_global64": ("global64", "balanced64"),
        "global128_vs_global64": ("global64", "global128"),
    }
    results = {}
    for comparison, (control, proposal) in contrasts.items():
        scores = {}
        for arm in ("joint", "personal_roles"):
            rows = [
                dict(atoms=atoms, topic_pair=json.dumps([topic for topic, _ in atoms]),
                     arms=dict(owner_gap=baseline[arm], joint=treatment[arm]))
                for atoms, baseline, treatment in zip(queries, examples[control], examples[proposal], strict=True)
            ]
            scores[arm] = {}
            for policy, masks in audit.select(rows).items():
                if policy == "natural":
                    continue
                outcome = audit.summarize(rows, masks)
                scores[arm][policy] = dict(
                    control=outcome["arms"]["owner_gap"], proposed=outcome["arms"]["joint"],
                    common_requests=outcome["both_issue"], overlap_changes=outcome["overlap_changes"],
                    paired_known_delta_per_request=outcome["paired_known_delta_per_query"],
                )
        results[comparison] = scores
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on a Slurm GPU")
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    base = live.BASE
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    elife, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.manifest) != expected["elife_manifest_sha256"]
    ):
        raise ValueError("Frozen inputs differ from reference")
    if raw_identifiers(args.plos_xml) & raw_identifiers(args.elife_xml):
        raise ValueError("Cross-publisher identity overlaps require a new linked fixture.")
    plos_subjects = base.publisher_topics([args.plos_xml])
    elife_subjects, _, elife_cases = live.elife.build_requests(elife)["topics"]
    panels = dict(plos=(plos, plos_subjects, base.query_panel(base.annotation_postings(plos, plos_subjects))),
                  elife=(elife, elife_subjects, elife_cases))
    merged = merge(plos, elife)
    publishers = {doi: name for name, (catalog, _, _) in panels.items() for doi in catalog.ids}
    encoder = QwenTorchEmbedding(args.embedding)
    results = {}
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        common = live.LocalIndex(merged, encoder, root / "mixed")
        joint_postings = typed.RolePostings(merged)
        for name, (catalog, subjects, cases) in panels.items():
            original = live.LocalIndex(catalog, encoder, root / name)
            prior, original_examples = count(
                original, typed.RolePostings(catalog), cases, subjects, publishers, False, args.order_offset,
            )
            combined, mixed_examples = count(common, joint_postings, cases, subjects, publishers, True, args.order_offset)
            reference = expected["panels"][name]["arms"]["structured"]
            for arm in ("owner_gap", "joint"):
                for metric in ("known", "issued", "reads"):
                    if prior["original64"]["arms"][arm][metric] != reference[arm][metric]:
                        raise ValueError(f"Original catalog reference changed for {name}/{arm}/{metric}")
            results[name] = dict(
                queries=len(cases), original_articles=len(catalog.ids), **prior, **combined,
                matched=matched_results(cases, original_examples | mixed_examples),
            )
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), order_offset=args.order_offset, mixed_articles=len(merged.ids),
        embedding_checkpoint=checkpoint, script_sha256=digest(Path(__file__)),
        live_reference_sha256=digest(args.reference), panels=results,
    ))


if __name__ == "__main__":
    main()
