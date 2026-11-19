"""Share source confirmation across authors with authoritative role declarations."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import compound_contributor_search as base
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("joint", "factor_same_slots", "factor_sources", "factor_qualified_sources", "literal")
ABSTRACT_SENTENCE = re.compile(r"(?<=[.!?])\s+")
SOURCE_SYSTEM = ("Decide whether this publication is the specific study described by the clue. "
                 "A shared broad topic is insufficient. Use only the supplied text. "
                 "Reply 1 for a clearly matching study or 0 otherwise.")


def abstract_clue(article):
    words = [part.split() for part in ABSTRACT_SENTENCE.split(article.abstract)]
    usable = [parts for parts in words if 17 <= len(parts) <= 44]
    if not usable:
        return ""
    chosen = usable[len(usable) // 2]
    return " ".join(word for i, word in enumerate(chosen) if i % 4 != 1)


def prepare_views(fixture, articles):
    expected = {doi: a.digest for doi, a in articles.items()}
    if fixture["sources"] != expected:
        raise ValueError("Publisher XML differs from the fixed comparison")
    for query in fixture["queries"]:
        yield dict(query, style="title_fragment")
        alternative = abstract_clue(articles[query["doi"]])
        if alternative:
            yield dict(query, clue=alternative, style="abstract_fragment")


def measure(query, state, arm, index, joint, source_reader, sections):
    started = perf_counter()
    sources = index.sources(query, state)
    candidates = index.nominate(query, sources, "marginal")
    need = set(query["roles"])
    eligible = {pair for pair in candidates if need <= index.articles[pair[0]].roles[pair[1] - 1]}
    if arm == "literal":
        clue_terms = set(base.tokens(query["clue"]))
        matches = [doi for doi in sources if any(clue_terms <= terms for terms in sections[doi])]
        accepted = {pair for pair in eligible if len(matches) == 1 and pair[0] == matches[0]}
        questions = []
    elif arm == "joint":
        question = " and ".join(query["roles"])
        questions = []
        for doi, ordinal in candidates:
            article = index.articles[doi]
            lead = " ".join(article.abstract.split()[:80])
            credited = ", ".join(sorted(article.roles[ordinal - 1]))
            questions.append((
                f"Does contributor {ordinal} personally have both {question} on the study about "
                f"'{query['clue']}'? The source must match the described study and the same contributor must have both roles.",
                f"Study title: {article.title}\nAbstract excerpt: {lead}\nContributor {ordinal}: {credited}",
            ))
        scores, clipped = joint(questions)
        if clipped or not np.isfinite(scores).all():
            raise ValueError("Invalid joint reader")
        accepted = {pair for pair, score in zip(candidates, scores, strict=True) if score > base.CUTOFF}
    else:
        scoped = [doi for doi, ordinal in candidates if arm != "factor_qualified_sources" or (doi, ordinal) in eligible]
        if arm != "factor_same_slots":
            scoped = list(dict.fromkeys(scoped))
        questions = [(f"Study clue: {query['clue']}",
                      f"Study title: {index.articles[doi].title}\nAbstract excerpt: "
                      + " ".join(index.articles[doi].abstract.split()[:80])) for doi in scoped]
        scores, clipped = source_reader(questions) if questions else ([], 0)
        if clipped or not np.isfinite(scores).all():
            raise ValueError("Invalid source reader")
        confirmed = {doi for doi, score in zip(scoped, scores, strict=True) if score > base.CUTOFF}
        accepted = {pair for pair in eligible if pair[0] in confirmed}
    elapsed = (perf_counter() - started) * 1000
    valid = {(query["doi"], ordinal) for ordinal in query["gold"]} if state == "present" else set()
    false = accepted - valid
    attribution_errors = sum(doi == query["doi"] for doi, _ in false)
    return dict(
        available=bool(set(candidates) & valid), supported=len(accepted & valid), wrong_person=attribution_errors,
        wrong_source=len(false) - attribution_errors, issued=int(bool(accepted)), exact=int(accepted == valid),
        request_candidates=len(candidates), reader_pairs=len(questions), forwards=int(bool(questions)), ms=elapsed,
    ), accepted


def summarize(panels):
    output = {}
    for (style, state, category), rows in panels.items():
        section = output.setdefault(style, {}).setdefault(state, {}).setdefault(category, {})
        for arm in ARMS:
            items = [r[arm][0] for _, r in rows]
            sums = {key: sum(item[key] for item in items) for key in items[0] if key != "ms"}
            for _, result in rows:
                a, b = result["joint"][0], result["factor_same_slots"][0]
                if a["reader_pairs"] != b["reader_pairs"]:
                    raise ValueError("Unequal fixed-budget comparison")
            groups = [doi for doi, _ in rows]
            delta = [(methods[arm][0]["supported"] - methods["joint"][0]["supported"]) / base.BUDGET
                     for _, methods in rows]
            section[arm] = dict(sums, queries=len(items), p95_ms=float(np.quantile([x["ms"] for x in items], .95)),
                paired_supported_p5_delta_vs_joint=clustered_paired_interval(delta, groups),
                identical_to_factor_same_slots=sum(methods[arm][1] == methods["factor_same_slots"][1]
                                                    for _, methods in rows))
    return output


def run(fixture_path, checkpoint, output, folders):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    articles = base.capture(folders)
    fixture = json.loads(fixture_path.read_bytes())
    views = list(prepare_views(fixture, articles))
    index = base.Index(articles)
    sections = {doi: [set(base.tokens(text)) for text in (a.title, *ABSTRACT_SENTENCE.split(a.abstract))]
                for doi, a in articles.items()}
    tokenizer, model, info = base.reader_module.load_reader(checkpoint)
    joint = base.reader_module.reader_scorer(tokenizer, model)
    source = base.reader_module.reader_scorer(tokenizer, model, system=SOURCE_SYSTEM)
    for arm in ARMS:
        measure(views[0], "present", arm, index, joint, source, sections)
    panels = defaultdict(list)
    for i, query in enumerate(views):
        category = "both_by_one_person" if query["gold"] else "split_across_people"
        for state in ("present", "withheld"):
            shift = (i + int(state == "withheld")) % len(ARMS)
            results = {arm: measure(query, state, arm, index, joint, source, sections)
                       for arm in ARMS[shift:] + ARMS[:shift]}
            panels[query["style"], state, category].append((query["doi"], results))
        if (i + 1) % 40 == 0:
            print(f"Completed {i+1}/{len(views)} factored requests", flush=True)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        checkpoint=info["revision"], fixture_sha256=digest(fixture_path), code_sha256=digest(Path(__file__)),
        cutoff=base.CUTOFF, query_views=len(views), panels=summarize(panels)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("fixture_path", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("folders", type=Path, nargs="+")
    run(**vars(parser.parse_args()))
