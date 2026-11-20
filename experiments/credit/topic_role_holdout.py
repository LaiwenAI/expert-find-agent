"""Transfer the fixed topical gate and test coverage-aware role screening."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from experiments.credit import topic_role_postings as pilot
from experiments.credit.cross_source_identity import load, normalized_name, orcid_id
from experiments.credit.plos_role_inventory import declared_roles


ARMS = ("topic_eight", "topic_four", "union_four", "person_four", "compact_eight", "adaptive")
CUTOFF = -2.0625
QUALITY_MARGIN = .01
OFF_LABEL_MARGIN = .01


def metadata(raw):
    return ET.fromstring(raw).find("./front/article-meta")


def title(meta):
    node = meta.find("./title-group/article-title")
    return pilot.compact(" ".join(node.itertext())) if node is not None else ""


def identities(meta):
    authors = meta.findall("./contrib-group/contrib[@contrib-type='author']")
    ids = {valid for author in authors for item in author.findall("contrib-id[@contrib-id-type='orcid']")
           if (valid := orcid_id(item.text or ""))}
    names = {name for author in authors if (name := normalized_name(author))}
    return ids, names


def leaf_topics(meta):
    return sorted({pilot.compact(leaf.findtext("subject", ""))
                   for tree in meta.findall("./article-categories/subj-group[@subj-group-type='Discipline-v3']")
                   for leaf in tree.iter("subj-group")
                   if leaf.find("subj-group") is None and leaf.findtext("subject")})


def prepare(pilot_projection, raw_folders, candidate_folders, output, manifest):
    prior = json.loads(pilot_projection.read_bytes())
    if output.exists() or manifest.exists() or prior["fixture"] != pilot.FIXTURE:
        raise ValueError("Expected fresh outputs and the pilot projection")
    pilot_dois = {row["doi"] for row in prior["sources"]} | set(prior["excluded_missing_text"])
    pilot_topics = Counter(topic for row in prior["sources"] for topic in row["topics"])
    pilot_hashes = {row["source_sha256"] for row in prior["sources"]}
    old_ids, old_names, old_titles, seen = set(), set(), set(), set()
    for folder in raw_folders:
        for path in sorted(folder.glob("*.xml")):
            digest = pilot.digest(path)
            if digest in pilot_hashes and digest not in seen:
                seen.add(digest)
                meta = metadata(path.read_bytes())
                ids, names = identities(meta)
                old_ids.update(ids)
                old_names.update(names)
                old_titles.add(title(meta).casefold())
    if pilot_hashes != seen:
        raise ValueError("Incomplete pilot snapshot inventory")
    files, counts = {}, Counter()
    trials = {}
    for folder in candidate_folders:
        for path in sorted(folder.glob("*.xml")):
            raw = path.read_bytes()
            meta = metadata(raw)
            doi = meta.findtext("article-id[@pub-id-type='doi']")
            if not doi or doi in pilot_dois or doi in trials:
                continue
            article_title = title(meta)
            abstract = meta.find("abstract")
            if not article_title or abstract is None:
                counts["missing_text"] += 1
                continue
            try:
                labels = declared_roles(raw, doi)
            except ValueError:
                counts["incompatible_declaration"] += 1
                continue
            if not labels or not all(labels):
                counts["incomplete_declaration"] += 1
                continue
            ids, names = identities(meta)
            if ids & old_ids or names & old_names or article_title.casefold() in old_titles:
                counts["prior_title_or_author"] += 1
                continue
            if not leaf_topics(meta):
                counts["missing_topics"] += 1
                continue
            files[doi] = path
            trials[doi] = dict(text=f"{article_title}\n{pilot.compact(' '.join(abstract.itertext()))}",
                               topics=leaf_topics(meta))
    # Reuse the canonical anonymous-ID compiler, while restricting it to
    # each folder's pinned records. Duplicate snapshots have been deduplicated.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as staged:
        stage = Path(staged)
        for i, path in enumerate(files.values()):
            (stage / f"{i}.xml").symlink_to(path.resolve())
        annotated = load((stage,))
    sources = [dict(annotated[doi], **trials[doi]) for doi in sorted(trials)]
    pilot.write_json(output, dict(sources=sources, original_topics=sorted(t for t, n in pilot_topics.items() if n >= 4)))
    pilot.write_json(manifest, dict(projection_sha256=pilot.digest(output), original_projection_sha256=pilot.digest(pilot_projection),
                                   excluded=dict(counts), articles=[
        {key: row[key] for key in ("doi", "source_sha256", "license", "url")} for row in sources]))


def select(catalog, ranked, pair, coverage, method):
    if method == "topic_eight":
        return ranked[:8]
    if method == "topic_four":
        return ranked[:4]
    if method in ("union_four", "person_four"):
        return catalog.candidates(ranked, pair, {"union_four": "role_union", "person_four": "person_postings"}[method],
                                  coverage, 4)
    possible = catalog.person[pair]
    complete = catalog.visible[coverage]
    retained = [i for i in ranked[:8] if i in possible or i not in complete]
    if method == "adaptive" and len(retained) < 4:
        return catalog.candidates(ranked, pair, "person_postings", coverage, 4)
    return retained


def interval(values, groups):
    return dict(pilot.clustered_paired_interval(values, groups), sampling_unit="whole topic")


def compare_latency(rows, baseline, groups):
    keys = sorted(set(groups))
    indices = {group: [i for i, g in enumerate(groups) if g == group] for group in keys}
    rng = np.random.default_rng(20261116)
    samples = [np.asarray([i for index in draw for i in indices[keys[index]]])
               for draw in rng.integers(len(keys), size=(4000, len(keys)))]
    value = np.asarray(rows)
    control = np.asarray(baseline)
    differences = [np.quantile(value[item], .95) - np.quantile(control[item], .95) for item in samples]
    return np.quantile(differences, [.025, .975]).tolist()


def analyze(rows, reference):
    pairs = list(zip(rows, reference, strict=True))
    groups = [r["topic"] for r in rows]
    grounded = interval([(a["grounded"]-b["grounded"])/pilot.TOP for a, b in pairs], groups)
    off_label = interval([(a["cited_off_label"]-b["cited_off_label"])/pilot.TOP for a, b in pairs], groups)
    latency = compare_latency([r["ms"] for r in rows], [r["ms"] for r in reference], groups)
    totals = {key: sum(row[key] for row in rows)
              for key in ("grounded", "cited_off_label", "issued", "reads", "characters")}
    return dict(totals, queries=len(rows), p95_ms=float(np.quantile([r["ms"] for r in rows], .95)),
                grounded_delta=grounded, off_label_delta=off_label, paired_p95_delta_ci95_ms=latency,
                outputs_differ=sum(a["witnesses"] != b["witnesses"] for a, b in pairs),
                passes_quality_gate=(grounded["percentile_95"][0] >= -QUALITY_MARGIN
                                     and off_label["percentile_95"][1] <= OFF_LABEL_MARGIN),
                p95_reduction_interval_below_zero=bool(latency[1] < 0))


def read_result(catalog, topic, pair, selected):
    values = catalog.verify(topic, selected)
    witness = {}
    for source, score in sorted(zip(selected, values, strict=True), key=lambda item: -item[1]):
        if score <= CUTOFF:
            continue
        for owner in catalog.bindings.get((source, pair), ()):
            witness.setdefault(owner, source)
            if len(witness) == pilot.TOP:
                break
        if len(witness) == pilot.TOP:
            break
    grounded = sum(i in catalog.topics[topic] for i in witness.values())
    return dict(issued=len(witness), grounded=grounded, cited_off_label=len(witness)-grounded,
                witnesses=tuple(witness.items()))


def run(projection, manifest, embedding, reranking, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or output.exists():
        raise ValueError("Require a new output on Slurm CUDA")
    pinned = json.loads(manifest.read_bytes())
    if pinned["projection_sha256"] != pilot.digest(projection):
        raise ValueError("Input projection does not match the frozen manifest")
    source = json.loads(projection.read_bytes())
    articles = source["sources"]
    old_topics = set(source["original_topics"])
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reranker = Reranker(reranking)
    catalog = pilot.Catalog(articles, encoder, reranker)
    topics = sorted(t for t, ids in catalog.topics.items() if len(ids) >= 4)
    queries = list(pilot.query_panel(catalog, topics))
    panels = {cover: {arm: [] for arm in ARMS} for cover in pilot.CACHE}
    initial_topic, _, _ = queries[0]
    read_result(catalog, initial_topic, queries[0][2], catalog.rank(initial_topic)[:8])
    for i, (topic, _, pair) in enumerate(queries):
        start = perf_counter()
        ranked = catalog.rank(topic)
        rank_ms = (perf_counter() - start) * 1000
        offset = i % len(ARMS)
        for cover in pilot.CACHE:
            for arm in ARMS[offset:] + ARMS[:offset]:
                start = perf_counter()
                selected = select(catalog, ranked, pair, cover, arm)
                stats = read_result(catalog, topic, pair, selected)
                elapsed = rank_ms + (perf_counter() - start) * 1000
                panels[cover][arm].append(dict(stats, topic=topic, ms=elapsed, reads=len(selected),
                                               characters=sum(len(catalog.texts[s]) for s in selected)))
    comparisons = {}
    for cover, arms in panels.items():
        comparisons[cover] = {}
        for name, rows in arms.items():
            control = arms["topic_eight"]
            same_budget = arms["topic_four"]
            comparisons[cover][name] = dict(
                against_eight=analyze(rows, control),
                novel_topics=analyze([r for r in rows if r["topic"] not in old_topics],
                                     [r for r in control if r["topic"] not in old_topics]),
            )
            if name in ("union_four", "person_four", "topic_four"):
                comparisons[cover][name]["against_four"] = analyze(rows, same_budget)
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=pilot.digest(Path(__file__)),
        manifest_sha256=pilot.digest(manifest), projection_sha256=pilot.digest(projection),
        embedding=encoder_info["revision"], reranker=reranker.provenance["revision"], sources=len(articles),
        indexed={entry: len(rows) for entry, rows in catalog.visible.items()}, queries=len(queries),
        topics=len(set(t for t, _, _ in queries)), novel_topics=len({t for t, _, _ in queries if t not in old_topics}),
        frozen_cutoff=CUTOFF, quality_margin=QUALITY_MARGIN, off_label_margin=OFF_LABEL_MARGIN,
        panels=comparisons,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    for name in ("pilot_projection", "output", "manifest"):
        build.add_argument(name, type=Path)
    build.add_argument("--raw-folders", nargs="+", type=Path, required=True)
    build.add_argument("--candidate-folders", nargs="+", type=Path, required=True)
    replay = sub.add_parser("run")
    for name in ("projection", "manifest", "embedding", "reranking", "output"):
        replay.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    command = args.pop("command")
    (prepare if command == "prepare" else run)(**args)
