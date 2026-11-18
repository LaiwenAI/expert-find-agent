"""Build and replay two-article contributor checks with publisher-linked identities."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import re
from time import perf_counter_ns
import unicodedata
from xml.etree import ElementTree as ET

from experiments.credit.counterfactual_role_edits import SOURCE_FIELD, article_url, write_json
from experiments.credit.plos_role_inventory import LICENSE, declared_roles


def orcid_id(value):
    raw = value.strip().rsplit("/", 1)[-1].upper()
    if not re.fullmatch(r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]", raw):
        return None
    digits = raw.replace("-", "")
    total = 0
    for digit in digits[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    return raw if digits[-1] == ("X" if check == 10 else str(check)) else None


def normalized_name(author):
    name = " ".join((author.findtext("name/given-names") or "", author.findtext("name/surname") or ""))
    normalized = unicodedata.normalize("NFKD", name.casefold())
    return "".join(char for char in normalized if char.isalnum()) or None


def xml_sources(folder):
    return sorted(path for path in folder.glob("*.xml") if not path.name.startswith("."))


def load(folders):
    raw_sources = {}
    for folder in folders:
        for path in xml_sources(folder):
            raw = path.read_bytes()
            metadata = ET.fromstring(raw).find("./front/article-meta")
            doi = metadata.findtext("article-id[@pub-id-type='doi']")
            if doi in raw_sources:
                if raw_sources[doi]["source_sha256"] != sha256(raw).hexdigest():
                    raise ValueError(f"Conflicting snapshots for {doi}")
                continue
            roles = declared_roles(raw, doi)
            authors = metadata.findall("./contrib-group/contrib[@contrib-type='author']")
            rows = []
            for ordinal, (author, tags) in enumerate(zip(authors, roles, strict=True), 1):
                ids = {key for item in author.findall("contrib-id[@contrib-id-type='orcid']")
                       if (key := orcid_id(item.text or ""))}
                if len(ids) > 1:
                    raise ValueError(f"Conflicting published contributor IDs at {doi}")
                rows.append(dict(ordinal=ordinal, roles=tags, person=next(iter(ids), None),
                                 name=normalized_name(author)))
            raw_sources[doi] = dict(doi=doi, source_sha256=sha256(raw).hexdigest(),
                                    source_field=SOURCE_FIELD, license=LICENSE, url=article_url(doi), authors=rows)
    for field in ("person", "name"):
        tokens = sorted({author[field] for s in raw_sources.values() for author in s["authors"] if author[field]})
        anonymous = {token: f"{field}-{i+1:04d}" for i, token in enumerate(tokens)}
        for source in raw_sources.values():
            for author in source["authors"]:
                author[field] = anonymous.get(author[field])
    return raw_sources


def pairs(left, right, field):
    return sorted((a["ordinal"], b["ordinal"]) for a, b in product(left, right)
                  if a[field] and a[field] == b[field])


def eligible(source, role):
    return [author for author in source["authors"] if role in author["roles"]]


def inventory(sources):
    articles = defaultdict(list)
    for doi, source in sources.items():
        for person in {a["person"] for a in source["authors"] if a["person"]}:
            articles[person].append(doi)
    pairs_of_articles = sorted({pair for docs in articles.values() for pair in combinations(sorted(docs), 2)})
    cases = []
    for first, second in pairs_of_articles:
        a, b = sources[first], sources[second]
        left_roles = sorted({role for person in a["authors"] for role in person["roles"]})
        right_roles = sorted({role for person in b["authors"] for role in person["roles"]})
        bins = defaultdict(list)
        for f, s in product(left_roles, right_roles):
            if f == s:
                continue
            left, right = eligible(a, f), eligible(b, s)
            declared = pairs(left, right, "person")
            if declared:
                if any(s in a["authors"][i - 1]["roles"] or f in b["authors"][j - 1]["roles"] for i, j in declared):
                    continue
                kind = "positive"
            elif all(row["person"] for row in left + right):
                kind = "no_link"
            else:
                continue
            digest = sha256(json.dumps([first, second, f, s]).encode()).hexdigest()
            bins[kind].append(dict(
                id=digest, left=first, right=second, left_role=f, right_role=s,
                gold=declared, kind=kind,
            ))
        for kind in ("positive", "no_link"):
            cases.extend(sorted(bins[kind], key=lambda x: x["id"])[:4])
    return cases


def evaluate(sources, cases):
    report = {}
    for scenario in ("visible", "identity_hidden", "simulated_homonym"):
        by_method = defaultdict(Counter)
        total_ns = Counter()
        for case in cases:
            if scenario != "visible" and not case["gold"]:
                continue
            left = eligible(sources[case["left"]], case["left_role"])
            right = eligible(sources[case["right"]], case["right_role"])
            gold = set(map(tuple, case["gold"]))
            if scenario == "identity_hidden":
                hidden = {j for _, j in gold}
                right = [dict(a, person=None) if a["ordinal"] in hidden else a for a in right]
                gold = set()
            elif scenario == "simulated_homonym":
                first, _ = min(gold)
                holder = next(a for a in left if a["ordinal"] == first)
                distractor = next((a for a in right if a["person"] and a["person"] != holder["person"]), None)
                if holder["name"] is None or distractor is None:
                    continue
                right = [dict(a, name=holder["name"]) if a["ordinal"] == distractor["ordinal"] else a
                         for a in right]
            for method in ("paper_pool", "name_join", "orcid_join"):
                start = perf_counter_ns()
                if method == "paper_pool":
                    answers = {(a["ordinal"], b["ordinal"]) for a, b in product(left, right)}
                else:
                    answers = set(pairs(left, right, "name" if method == "name_join" else "person"))
                pending = not answers and not all(a["person"] for a in left + right)
                total_ns[method] += perf_counter_ns() - start
                count = by_method[method]
                count["queries"] += 1
                count["exact_links_and_status"] += answers == gold and (scenario != "identity_hidden" or pending)
                count["unsupported_links"] += len(answers - gold)
                count["positive_links_found"] += len(answers & gold)
                count["unresolved"] += pending
        report[scenario] = {
            method: dict(count, mean_microseconds=total_ns[method] / count["queries"] / 1000)
            for method, count in by_method.items()
        }
    return report


def main(args):
    if args.replay:
        if args.sources:
            raise ValueError("Choose XML inputs or a frozen fixture")
        manifest = json.loads((args.replay / "manifest.json").read_bytes())
        payload = {}
        for filename, digest in manifest["files"].items():
            data = (args.replay / filename).read_bytes()
            if sha256(data).hexdigest() != digest:
                raise ValueError(f"Changed frozen fixture: {filename}")
            payload[filename] = json.loads(data)
        report = evaluate({row["doi"]: row for row in payload["sources.json"]}, payload["queries.json"])
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps(report, indent=2))
        return
    if not args.sources:
        raise ValueError("Provide at least one XML capture directory")
    sources = load(args.sources)
    cases = inventory(sources)
    if not any(case["gold"] for case in cases):
        raise ValueError("No cases requiring evidence across two distinct articles")
    args.output.mkdir(parents=True, exist_ok=False)
    used = {case[field] for case in cases for field in ("left", "right")}
    write_json(args.output / "sources.json", [sources[doi] for doi in sorted(used)])
    write_json(args.output / "queries.json", cases)
    report = dict(all_sources=len(sources), used_sources=len(used), kinds=dict(Counter(c["kind"] for c in cases)),
                  results=evaluate(sources, cases))
    manifest = dict(all_sources=len(sources), used_sources=len(used), kinds=report["kinds"],
                    input_sources=[{k: sources[doi][k] for k in ("doi", "url", "source_sha256")}
                                   for doi in sorted(sources)],
                    files={name: sha256((args.output / name).read_bytes()).hexdigest()
                           for name in ("queries.json", "sources.json")})
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path, nargs="*")
    parser.add_argument("--replay", type=Path)
    main(parser.parse_args())
