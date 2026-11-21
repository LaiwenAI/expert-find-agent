"""Freeze additional author-disjoint source pairs from declarations alone."""

import argparse
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

from experiments.credit import held_open_proof_search as base
from experiments.credit.role_first_proof_decoding import read

MAX_GROUPS = 100


def freeze(paths, prior):
    articles = base.load_catalog(paths)
    recorded = {
        row["doi"]: {author["person"] for author in row["authors"] if author["person"]}
        for path in paths for row in read(path)["sources"]
    }
    used_papers = {paper for panel in read(prior)["panels"].values() for row in panel["pairs"]
                   for paper in row["papers"]}
    used_people = set().union(*(recorded[paper] for paper in used_papers))
    by_author = defaultdict(list)
    for paper, article in sorted(articles.items()):
        if paper not in used_papers and not (recorded[paper] & used_people):
            for person in sorted(article["roles"]):
                by_author[person].append(paper)
    candidates = {pair for papers in by_author.values() for pair in combinations(papers, 2)}
    frozen = []
    for pair in sorted(candidates, key=lambda pair: base.stable("held-source-acceptance-v1", pair)):
        left, right = (articles[paper]["roles"] for paper in pair)
        people = recorded[pair[0]] | recorded[pair[1]]
        if used_people & people or used_papers.intersection(pair):
            continue
        left_roles = set().union(*left.values())
        right_roles = set().union(*right.values())
        bindings = {roles for author in left.keys() & right.keys() for roles in product(left[author], right[author])
                    if roles[0] != roles[1] and roles[1] in left_roles and roles[0] in right_roles
                    and not base.known_owners(articles, pair, roles[::-1])}
        if not bindings:
            continue
        choices = sorted(bindings, key=lambda roles: base.stable("held-source-role-v1", pair, roles))
        frozen.append(dict(papers=pair, bindings=choices[:3]))
        used_people.update(people)
        used_papers.update(pair)
        if len(frozen) == MAX_GROUPS:
            break
    if not frozen:
        raise ValueError("No unused linked source group")
    return dict(pairs=frozen, considered_pairs=len(candidates),
                prior_groups_sha256=base.digest(prior), source_sha256=base.INPUTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "prior", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Require a fresh output")
    base.write_json(args.output, freeze((args.confirmation, args.extended), args.prior))


if __name__ == "__main__":
    main()
