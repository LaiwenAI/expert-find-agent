"""Build and evaluate paired source-specific contributor constraints."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from experiments.credit.harvest_scoped_credit import joined, authors, meta, prior
from experiments.credit.plos_role_inventory import ROLES, declared_roles


ARMS = ("document_union", "owner_union", "attributed_scan", "attributed_postings")
ROLE_BITS = {name: 1 << i for i, name in enumerate(ROLES)}


def save(target, value):
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def records(capture, sources, old_folders):
    old_dois, old_ids, old_names = prior(old_folders)
    metadata = json.loads(capture.read_text())
    expected = metadata["archive"]
    if len(old_dois) != expected["old_articles"] or len(old_ids) != expected["old_orcids"]:
        raise ValueError("Changed prior exposure inventory")
    raw_rows = {}
    for doi, checksum in metadata["source_digests"].items():
        raw = (sources / f"pone.{doi.rsplit('.',1)[-1]}.xml").read_bytes()
        if sha256(raw).hexdigest() != checksum or doi in old_dois:
            raise ValueError("Changed or reused publisher source")
        byline = authors(meta(raw))
        roles = declared_roles(raw, doi)
        rows = []
        for i, (person, labels) in enumerate(zip(byline, roles, strict=True), 1):
            ident = next(iter(joined([person])), None)
            if ident and ident[0] not in old_ids and ident[2] not in old_names and labels:
                rows.append(dict(ordinal=i, owner=ident[0], roles=labels))
        if len({row["owner"] for row in rows}) != len(rows):
            raise ValueError("Repeated owner in one source")
        raw_rows[doi] = dict(source_sha256=checksum, owners=rows)
    ids = sorted({row["owner"] for record in raw_rows.values() for row in record["owners"]})
    aliases = {ident: f"owner-{i+1:04d}" for i, ident in enumerate(ids)}
    for record in raw_rows.values():
        for row in record["owners"]:
            row["owner"] = aliases[row["owner"]]
    return raw_rows


def labels(left, right, roles_left, right_role):
    return sorted(owner for owner in left.keys() & right.keys()
                  if all(role in left[owner] for role in roles_left) and right_role in right[owner])


def build_cases(sources):
    owned = defaultdict(list)
    for doi, record in sources.items():
        for row in record["owners"]:
            owned[row["owner"]].append(doi)
    paired_articles = sorted({pair for dois in owned.values() for pair in combinations(sorted(dois),2)})
    pairs = []
    for left, right in paired_articles:
        a,b = ({r["owner"]:set(r["roles"]) for r in sources[doi]["owners"]} for doi in (left,right))
        categories = defaultdict(list)
        possible_left = sorted(set().union(*(a[owner] for owner in a.keys() & b.keys())))
        possible_right = sorted(set().union(*b.values()))
        for two_left in combinations(possible_left,2):
            for before,after in combinations(possible_right,2):
                original = labels(a,b,two_left,before)
                changed = labels(a,b,two_left,after)
                if not original and not changed:
                    continue
                changed_kind = "label_flip" if original != changed else "stable"
                if not original and changed:
                    original, changed, before, after = changed, original, after, before
                row = dict(
                    id=fingerprint([left,right,two_left,before,after]), left=left, right=right,
                    roles_left=two_left, role_before=before, role_after=after,
                    before=original, after=changed, category=changed_kind,
                )
                categories[changed_kind].append(row)
        for kind, cap in (("label_flip",6),("stable",3)):
            pairs.extend(sorted(categories[kind],key=lambda row:row["id"])[:cap])
        swaps = {}
        for two_left in combinations(possible_left,2):
            for right_role in possible_right:
                original = labels(a,b,two_left,right_role)
                if not original or right_role in two_left:
                    continue
                for moved in two_left:
                    after_left = tuple(sorted((set(two_left)-{moved})|{right_role}))
                    changed = labels(a,b,after_left,moved)
                    if changed == original:
                        continue
                    key = fingerprint([left,right,two_left,right_role,after_left,moved])
                    swaps[key] = dict(
                        id=key,left=left,right=right,roles_left=two_left,roles_left_after=after_left,
                        role_before=right_role,role_after=moved,before=original,after=changed,category="scope_swap",
                    )
        pairs.extend(swaps[key] for key in sorted(swaps)[:4])
    return pairs


def prepare(capture, sources, output, old_folders):
    items = records(capture,sources,old_folders)
    pairs = build_cases(items)
    if not pairs or not any(p["category"]=="label_flip" for p in pairs):
        raise ValueError("No eligible cross-source contrast")
    save(output, dict(capture_sha256=sha256(capture.read_bytes()).hexdigest(), sources=items, pairs=pairs))
    print(dict(sources=len(items),pairs=len(pairs),categories=dict(Counter(p["category"] for p in pairs))),flush=True)


class EvidenceIndex:
    def __init__(self, sources):
        self.roles = {}
        self.postings = defaultdict(set)
        self.unions = {}
        for doi, doc in sources.items():
            masks = {}
            for row in doc["owners"]:
                mask = 0
                for role in row["roles"]:
                    mask |= ROLE_BITS[role]
                    self.postings[doi,role].add(row["owner"])
                masks[row["owner"]] = mask
            self.roles[doi] = masks
            self.unions[doi] = 0
            for mask in masks.values():
                self.unions[doi] |= mask

    def match(self, method, left, right, roles_left, role_right):
        if method == "attributed_postings":
            postings = [self.postings.get((left,role),set()) for role in roles_left]
            postings.append(self.postings.get((right,role_right),set()))
            seed,*others = sorted(postings,key=len)
            return sorted(seed.intersection(*others))
        lhs,rhs = self.roles[left],self.roles[right]
        req_l = sum(ROLE_BITS[role] for role in roles_left)
        req_r = ROLE_BITS[role_right]
        both = lhs.keys() & rhs.keys()
        if method == "attributed_scan":
            return sorted(o for o in both if lhs[o] & req_l == req_l and rhs[o] & req_r == req_r)
        if method == "owner_union":
            need = req_l | req_r
            return sorted(o for o in both if (lhs[o]|rhs[o]) & need == need)
        if method == "document_union":
            return sorted(both) if self.unions[left] & req_l == req_l and self.unions[right] & req_r == req_r else []
        raise ValueError("Unknown attribution strategy")


def evaluate(path, output):
    fixture = json.loads(path.read_text())
    cases = fixture["pairs"]
    if cases != json.loads(json.dumps(build_cases(fixture["sources"]))):
        raise ValueError("Frozen cases differ from publisher rows")
    start = perf_counter_ns()
    idx = EvidenceIndex(fixture["sources"])
    build_us = (perf_counter_ns()-start)/1000
    counts = defaultdict(Counter)
    duration = defaultdict(list)
    for i,row in enumerate(cases):
        offset = i % len(ARMS)
        for arm in ARMS[offset:]+ARMS[:offset]:
            samples = []
            expected = [set(row[field]) for field in ("before","after")]
            for _ in range(7):
                start = perf_counter_ns()
                found = [
                    idx.match(arm,row["left"],row["right"],row["roles_left"],row["role_before"]),
                    idx.match(arm,row["left"],row["right"],row.get("roles_left_after",row["roles_left"]),row["role_after"]),
                ]
                samples.append((perf_counter_ns()-start)/1000)
            actual = list(map(set,found))
            duration[arm].append(float(np.median(samples)))
            count = counts[arm]
            count["matched_pairs"] += actual == expected
            count[f'{row["category"]}_exact'] += actual == expected
            count["unsupported_attributions"] += sum(len(a-b) for a,b in zip(actual,expected,strict=True))
            count["missed_attributions"] += sum(len(b-a) for a,b in zip(actual,expected,strict=True))
    save(output,dict(
        fixture_sha256=sha256(path.read_bytes()).hexdigest(), script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=len(fixture["sources"]),
        pairs=len(cases), article_pairs=len({(row["left"],row["right"]) for row in cases}),
        categories=dict(Counter(row["category"] for row in cases)),
        own_ids=len({a["owner"] for s in fixture["sources"].values() for a in s["owners"]}),
        paired_search_p50_p95_us={name:np.quantile(times,[.5,.95]).tolist() for name,times in duration.items()},
        results={name:dict(count) for name,count in counts.items()}, index_build_us=build_us,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(required=True)
    create = sub.add_parser("prepare")
    for name in ("capture","sources","output"):
        create.add_argument(name,type=Path)
    create.add_argument("old_folders",type=Path,nargs="+")
    create.set_defaults(method=prepare)
    test = sub.add_parser("evaluate")
    for name in ("path","output"):
        test.add_argument(name,type=Path)
    test.set_defaults(method=evaluate)
    options = vars(cli.parse_args())
    options.pop("method")(**options)
