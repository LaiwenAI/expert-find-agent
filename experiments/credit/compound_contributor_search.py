"""Test owner-bound role intersections before one fixed-size verification batch."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import math
import os
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.credit.plos_role_inventory import declared_roles
from experiments.docred import oracle_role_evidence as reader_module
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("byline", "article_union", "marginal", "replace_one", "owner_intersection")
SOURCE_DEPTH = 6
BUDGET = 5
CUTOFF = .25
STOP = frozenset("a an and as at by for from in of on or the to using with study analysis based new".split())


@dataclass(frozen=True)
class Article:
    doi: str
    title: str
    abstract: str
    roles: tuple
    digest: str


def clean(element):
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def capture(folders):
    articles = {}
    for folder in folders:
        for path in sorted(folder.glob("*.xml")):
            if path.name.startswith("."):
                continue
            raw = path.read_bytes()
            checksum = sha256(raw).hexdigest()
            root = ET.fromstring(raw)
            metadata = root.find("./front/article-meta")
            doi = metadata.findtext("article-id[@pub-id-type='doi']")
            title = clean(metadata.find("./title-group/article-title"))
            abstract = clean(metadata.find("abstract"))
            memberships = tuple(frozenset(roles) for roles in declared_roles(raw, doi))
            if len(memberships) < 2 or not all(memberships) or not title or not abstract:
                continue
            previous = articles.setdefault(doi, Article(doi, title, abstract, memberships, checksum))
            if previous.digest != checksum:
                raise ValueError(f"Conflicting input snapshots for {doi}")
    return dict(sorted(articles.items()))


def stable(*values):
    return sha256(repr(values).encode()).hexdigest()


def clue(article):
    terms = [term for term in tokens(article.title) if term not in STOP and len(term) > 2]
    unique = list(dict.fromkeys(terms))
    if len(unique) < 5:
        return ""
    return " ".join(unique[::2][:7])


def cases(articles, limit):
    capable = sorted((d for d, article in articles.items() if clue(article)), key=stable)
    selected = []
    for doi in capable:
        article = articles[doi]
        declared = sorted(set().union(*article.roles))
        contrasts = []
        for pair in combinations(declared, 2):
            required = frozenset(pair)
            owners = [i for i, roles in enumerate(article.roles, 1) if required <= roles]
            split_authors = [i for i, roles in enumerate(article.roles, 1) if roles & required and i not in owners]
            if split_authors and len(owners) <= 3:
                contrasts.append((pair, owners))
        positives = [(r, g) for r, g in contrasts if g]
        negatives = [(r, g) for r, g in contrasts if not g]
        paired = [(r, g, alt) for r, g in positives for alt, _ in negatives if len(set(r) & set(alt)) == 1]
        if not paired:
            continue
        roles, gold, neg = min(paired, key=lambda row: stable(doi, *row))
        for pair, owners in ((roles, gold), (neg, [])):
            selected.append(dict(doi=doi, clue=clue(article), roles=pair, gold=owners))
        if len(selected) == 2 * limit:
            break
    return selected


class Index:
    def __init__(self, articles):
        self.articles = articles
        self.postings = defaultdict(list)
        self.author_postings = defaultdict(set)
        self.role_union = {}
        term_counts = {d: Counter(tokens(a.title) * 3 + tokens(a.abstract)) for d, a in articles.items()}
        frequency = Counter(term for terms in term_counts.values() for term in terms)
        average = sum(sum(terms.values()) for terms in term_counts.values()) / len(term_counts)
        for doi, terms in term_counts.items():
            norm = 1.2 * (.25 + .75 * sum(terms.values()) / average)
            for term, count in terms.items():
                n = frequency[term]
                weight = math.log1p((len(articles) - n + .5) / (n + .5))
                self.postings[term].append((doi, weight * count * 2.2 / (count + norm)))
            memberships = articles[doi].roles
            self.role_union[doi] = set().union(*memberships)
            for ordinal, roles in enumerate(memberships, 1):
                for role in roles:
                    self.author_postings[role].add((doi, ordinal))

    def sources(self, query, state):
        totals = defaultdict(float)
        for term in sorted(set(tokens(query["clue"]))):
            for doi, value in self.postings[term]:
                if state == "withheld" and doi == query["doi"]:
                    continue
                totals[doi] += value
        return sorted(totals, key=lambda d: (-totals[d], d))[:SOURCE_DEPTH]

    def nominate(self, query, sources, arm):
        first, second = (self.author_postings[r] for r in query["roles"])
        rows = [(doi, person) for doi in sources for person in range(1, len(self.articles[doi].roles) + 1)]
        ranks = {doi: i for i, doi in enumerate(sources)}
        both = set(query["roles"])
        marginal = {pair: int(pair in first) + int(pair in second) for pair in rows}

        def source_order(pair):
            return ranks[pair[0]], pair[1]

        def soft_order(pair):
            rank, ordinal = source_order(pair)
            return -(1 / (1 + rank) + .20 * marginal[pair]), rank, ordinal

        if arm == "byline":
            return sorted(rows, key=source_order)[:BUDGET]
        if arm == "article_union":
            return sorted(rows, key=lambda p: (not both <= self.role_union[p[0]], *source_order(p)))[:BUDGET]
        ordered = sorted(rows, key=soft_order)
        if arm == "marginal":
            return ordered[:BUDGET]
        qualified = [pair for pair in ordered if marginal[pair] == 2]
        if arm == "owner_intersection":
            return sorted(ordered, key=lambda p: marginal[p] != 2)[:BUDGET]
        chosen = ordered[:BUDGET]
        if chosen and marginal[chosen[-1]] < 2:
            alternate = next((pair for pair in qualified if pair not in chosen), None)
            if alternate is not None:
                chosen[-1] = alternate
        return chosen


def observe(query, state, arm, index, read):
    began = perf_counter()
    sources = index.sources(query, state)
    pairs = index.nominate(query, sources, arm)
    required = " and ".join(query["roles"])
    prompts = []
    for doi, ordinal in pairs:
        source = index.articles[doi]
        lead = " ".join(source.abstract.split()[:80])
        credited = ", ".join(sorted(source.roles[ordinal - 1]))
        prompts.append((
            f"Does contributor {ordinal} personally have both {required} on the study about "
            f"'{query['clue']}'? The source must match the described study and the same contributor must have both roles.",
            f"Study title: {source.title}\nAbstract excerpt: {lead}\nContributor {ordinal}: {credited}",
        ))
    margins, clipped = read(prompts) if prompts else ([], 0)
    elapsed = (perf_counter() - began) * 1000
    if clipped or not np.isfinite(margins).all():
        raise ValueError("Invalid reader scores")
    accepted = {pair for pair, margin in zip(pairs, margins, strict=True) if margin > CUTOFF}
    valid = {(query["doi"], i) for i in query["gold"]} if state == "present" else set()
    local_false = {pair for pair in accepted - valid if pair[0] == query["doi"]}
    return dict(source_found=int(query["doi"] in sources), candidates=len(pairs), topk_hit=len(valid.intersection(pairs)),
                supported=len(accepted & valid), wrong_person=len(local_false),
                wrong_source=len(accepted - valid - local_false), issued=int(bool(accepted)),
                correct_output=int(accepted == valid), ms=elapsed)


def summarize(table):
    results = {}
    for (state, case_type), items in table.items():
        if any(any(v["candidates"] != methods["byline"]["candidates"] for v in methods.values())
               for _, methods in items):
            raise ValueError("Unequal number of candidate pairs")
        panel = results.setdefault(state, {}).setdefault(case_type, {})
        for arm in ARMS:
            rows = [methods[arm] for _, methods in items]
            fields = {key: sum(row[key] for row in rows) for key in rows[0] if key != "ms"}
            delta = [(arms[arm]["supported"] - arms["marginal"]["supported"]) / BUDGET for _, arms in items]
            errors = [(arms[arm]["wrong_source"] - arms["marginal"]["wrong_source"]) / BUDGET for _, arms in items]
            groups = [doi for doi, _ in items]
            panel[arm] = dict(queries=len(rows), p95_ms=float(np.quantile([r["ms"] for r in rows], .95)),
                paired_supported_p5_delta_vs_marginal=clustered_paired_interval(delta, groups),
                paired_wrong_source_p5_delta_vs_marginal=clustered_paired_interval(errors, groups), **fields)
    return results


def run(folders, checkpoint, output, limit):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    articles = capture(folders)
    query_rows = cases(articles, limit)
    if len(query_rows) < 30:
        raise ValueError("Insufficient compound contributor contrasts")
    index = Index(articles)
    tokenizer, model, info = reader_module.load_reader(checkpoint)
    read = reader_module.reader_scorer(tokenizer, model)
    for arm in ARMS:
        observe(query_rows[0], "present", arm, index, read)
    results = defaultdict(list)
    for i, query in enumerate(query_rows):
        category = "both_by_one_person" if query["gold"] else "split_across_people"
        for state in ("present", "withheld"):
            shift = (i + int(state == "withheld")) % len(ARMS)
            observations = {arm: observe(query, state, arm, index, read) for arm in ARMS[shift:] + ARMS[:shift]}
            results[state, category].append((query["doi"], observations))
        if (i + 1) % 25 == 0:
            print(f"Completed {i+1}/{len(query_rows)} compound queries in both states", flush=True)
    fixture = dict(source_license="https://creativecommons.org/licenses/by/4.0/",
                   sources={doi: article.digest for doi, article in articles.items()}, queries=query_rows)
    fixture_path = output.with_name("public-queries.json")
    write_json(fixture_path, fixture)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        checkpoint=info["revision"], articles=len(articles), queries=len(query_rows), candidate_budget=BUDGET,
        cutoff=CUTOFF, code_sha256=digest(Path(__file__)), fixture_sha256=digest(fixture_path), panels=summarize(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--limit", type=int, default=60)
    run(**vars(parser.parse_args()))
