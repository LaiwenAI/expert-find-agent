"""Recover enterprise candidates from role-bearing artifact entries at two reads."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from math import log
import os
from pathlib import Path
import re
from time import perf_counter
import unicodedata

import numpy as np

from experiments.retrieval import enterprise_packed_role_search as packed
from experiments.retrieval import enterprise_project_role_rescue as profile
from experiments.retrieval import enterprise_source_linkability as link
from experiments.retrieval.enterprise_packed_role_load import init_fixture
from experiments.retrieval.enterprise_role_first_query import recover, recovery_options, require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write

DEPTH = 256
ARMS = ("lexical_windows", "dense_windows", "dense_entries", "sparse_entries")
PROJECT_ROLES = {"profile_project_lead", "profile_project_participant"}
ORDINAL = re.compile(r"^(?:[\uff08(]\s*)?\d{1,4}\s*[)\u3001.\uff09\uff0e]\s*")


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def project_columns(role_text):
    columns = [part.strip() for part in profile.COLUMNS.split(ORDINAL.sub("", role_text)) if part.strip()]
    labels = [i for i, part in enumerate(columns) if part.casefold() in profile.LEAD | profile.PARTICIPATE]
    return columns[:labels[0]] if len(labels) == 1 else []


def entry_identity(text, roles, first_role, specificity):
    if roles & PROJECT_ROLES:
        columns = project_columns(first_role)
        if specificity is None:
            return ("project", normalized(columns[0])) if columns and sum(c.isalpha() for c in columns[0]) >= 6 else None
        ids = {normalized(text[a:b]) for a, b in link.locators("project", text)}
        if len(ids) == 1:
            return "project", "locator:" + next(iter(ids))
        titles = {normalized(column) for column in columns if sum(c.isalpha() for c in column) >= 6}
        titles = {title for title in titles if specificity[title] < 16}
        if titles:
            return "project", max(titles, key=lambda title: (len(title), -specificity[title], title))
        return None
    if roles == {"profile_listed_inventor"}:
        numbers = {normalized(text[a:b]) for a, b in link.locators("patent", text)}
        if len(numbers) == 1:
            digits = "".join(c for c in next(iter(numbers)) if c.isdigit())
            return "patent", digits
    return None


def compile_entries(windows, sources, title_key="first_column"):
    specificity = None
    if title_key == "specific":
        specificity = Counter()
        for item in windows:
            view = item["view"]
            if set(view["roles"]) & PROJECT_ROLES:
                specificity.update({normalized(column) for column in project_columns(view["text"])})
    lines = defaultdict(list)
    for item in windows:
        view = item["view"]
        source = sources[view["id"]]
        text = source["text"]
        if source["expert_id"] != item["expert"] or text[view["start"]:view["end"]] != view["text"]:
            raise ValueError("Role span does not match the original inventory")
        start = text.rfind("\n", 0, view["start"]) + 1
        end = text.find("\n", view["end"])
        if end < 0:
            end = len(text)
        lines[view["id"], start, end].append(item)
    entries = []
    identities = set()
    counts = Counter()
    for (source, start, end), group in lines.items():
        text = sources[source]["text"]
        original = text[start:end]
        stripped = original.strip()
        if len(group) != 1 or not 16 <= len(stripped) <= profile.MAX_PASSAGE:
            counts["ambiguous_or_long_lines"] += 1
            continue
        item = group[0]
        role = item["view"]
        begin = start + len(original) - len(original.lstrip())
        stop = begin + len(stripped)
        role_begin = role["start"] + len(role["text"]) - len(role["text"].lstrip())
        role_stop = role["start"] + len(role["text"].rstrip())
        if not begin <= role_begin < role_stop <= stop or "\n" in stripped:
            counts["ambiguous_or_long_lines"] += 1
            continue
        key = entry_identity(stripped, set(role["roles"]), role["text"], specificity)
        if key is None or profile.NEGATION.search(stripped):
            counts["unidentified_lines"] += 1
            continue
        owner = item["expert"]
        identity = owner, key
        if identity in identities:
            counts["duplicate_owner_entries"] += 1
            continue
        identities.add(identity)
        entries.append(dict(expert=owner, artifact=key[0], view=dict(
            id=source, start=begin, end=stop, roles=role["roles"], text=stripped,
        )))
        counts[key[0]] += 1
    counts["owners"] = len({item["expert"] for item in entries})
    counts["owners_with_multiple_entries"] = sum(
        count >= 2 for count in Counter(item["expert"] for item in entries).values())
    counts["input_role_lines"] = len(lines)
    return entries, dict(counts)


class EntryIndex:
    def __init__(self, entries, encoder):
        import torch
        self.entries = entries
        self.encoder = encoder
        self.device = torch
        self.matrix = torch.from_numpy(encoder.encode([row["view"]["text"] for row in entries])).to("cuda")
        self.postings = defaultdict(list)
        for i, row in enumerate(entries):
            for token in sorted(profile.terms(row["view"]["text"])):
                self.postings[token].append(i)
        self.weight = {token: log((len(entries) + 1) / (len(ids) + 1)) + 1
                       for token, ids in self.postings.items()}

    def ranks(self, criteria, dense):
        if dense:
            from experiments.retrieval.enterprise_first_query import EMBED_TASK
            prompts = [f"Instruct: {EMBED_TASK}\nQuery: {criterion}" for criterion in criteria]
            queries = self.device.from_numpy(self.encoder.encode(prompts)).to("cuda")
            similarity = (queries @ self.matrix.T).cpu().numpy()
            return np.argsort(-similarity, axis=1, kind="stable")[:, :DEPTH]
        ranks = []
        for criterion in criteria:
            scores = Counter()
            for token in sorted(profile.terms(criterion) & self.postings.keys()):
                weight = self.weight[token]
                for i in self.postings[token]:
                    scores[i] += weight
            ranks.append(sorted(scores, key=lambda i: (-scores[i], i))[:DEPTH])
        return ranks

    def select(self, criteria, dense):
        per_criterion = [{int(idx): rank for rank, idx in enumerate(row, 1)}
                         for row in self.ranks(criteria, dense)]
        by_owner = defaultdict(list)
        for idx in sorted(set().union(*per_criterion)):
            by_owner[self.entries[idx]["expert"]].append(idx)
        absent = DEPTH + 1
        best = (absent, 0, 0, 0)
        for candidates in by_owner.values():
            for a, b in combinations(candidates, 2):
                ranks = [min(row.get(a, absent), row.get(b, absent)) for row in per_criterion]
                score = max(ranks), sum(ranks), a, b
                if score < best:
                    best = score
        worst, _, a, b = best
        if worst == absent:
            return None
        return self.entries[a], self.entries[b]


def act(text, arm, fixture, windows, entries):
    from experiments.retrieval.enterprise_first_query import CRITERION_CHARS, K, TASK, evaluate
    from experiments.retrieval.enterprise_guarded_evidence_bundle import verify_bundle

    parser, encoder, index, sources, role_index, reader = fixture
    reader.change("guarded_rescue")
    started = perf_counter()
    initial = evaluate(text, "broadcast_prefix", parser, encoder, index, sources, reader)
    primary = [dict(expert=eid, citations=[list(ref) for ref in dict.fromkeys(refs)])
               for eid, passed, refs in initial["decisions"] if passed]
    criteria = [text.strip().splitlines()[0][:CRITERION_CHARS],
                *(text[a:b] for a, b in initial["qualifications"])]
    issued = {item["expert"] for item in primary}
    ranked = [row[0] for row in initial["decisions"]]
    reader.counts.clear()
    added, clipped, attempted, selection_ms, selected = [], 0, False, None, None
    if not issued and all(len(c) <= CRITERION_CHARS for c in criteria):
        if arm != ARMS[0] and len(criteria) >= 2:
            original = recovery_options(criteria, ranked, issued, role_index, "global_on_abstain")
            if len(original) == 2:
                start_select = perf_counter()
                if arm == "dense_windows":
                    proposal = windows.select(criteria)
                    selected = None if proposal is None else proposal[0]
                else:
                    selected = entries.select(criteria, dense=arm == "dense_entries")
                selection_ms = (perf_counter() - start_select) * 1000
        if selected is None:
            added, _, attempted, clipped = recover(criteria, ranked, issued, role_index, reader,
                                                    "global_on_abstain", TASK, K)
        else:
            attempted = True
            added, clipped = verify_bundle(selected, None, criteria, reader, guarded=False)
    return dict(qualifications=initial["qualifications"], ranked=initial["retrieved"], primary=primary,
                added=added, attempted=attempted, allocated=selected is not None,
                selection_ms=selection_ms, ms=(perf_counter()-started)*1000,
                work=dict(reader.counts), clipped=initial["clipped_pairs"] + clipped)


def run(queries, corpus, roles, vectors, vector_meta, models, serial, output, private, reverse, title_key):
    from experiments.retrieval.enterprise_guarded_evidence_bundle import BundleIndex

    require_cuda(output, private)
    texts, fixture, fingerprint = init_fixture(queries, corpus, roles, vectors, vector_meta, models, serial)
    source_windows, _ = fixture[4]
    encoder, sources = fixture[1], fixture[3]
    indexed, audit = compile_entries(source_windows, sources, title_key=title_key)
    if not indexed:
        raise ValueError("The original inventory has no role-bearing entries")
    started = perf_counter()
    windows = BundleIndex(source_windows, encoder, DEPTH, True)
    window_index_ms = (perf_counter()-started)*1000
    started = perf_counter()
    entries = EntryIndex(indexed, encoder)
    entry_index_ms = (perf_counter()-started)*1000
    observations = []
    ids = [row["id"] for row in rows(queries)]
    arms = ARMS[::-1] if reverse else ARMS
    for i, (qid, text) in enumerate(zip(ids, texts, strict=True)):
        frozen = packed.one(text, "guarded_rescue", *fixture)
        order = arms[i % len(arms):] + arms[:i % len(arms)]
        records = {arm: act(text, arm, fixture, windows, entries) for arm in order}
        baseline = records[ARMS[0]]
        for result in records.values():
            if (result["qualifications"], result["ranked"], result["primary"]) != (
                    frozen["qualifications"], frozen["ranked"], frozen["initial"]):
                raise ValueError("The primary output changed")
            if result["work"].get("logical_pairs", 0) != baseline["work"].get("logical_pairs", 0):
                raise ValueError("Recovery work is not matched per request")
        if (baseline["added"], baseline["attempted"], baseline["clipped"]) != (
                frozen["extra"], frozen["invoked"], frozen["clipped"]):
            raise ValueError("Original recovery output changed")
        observations.append(dict(query=qid, arms=records))
        if (i+1) % 20 == 0:
            print(i+1, "original requests replayed", flush=True)

    write(private, dict(observations=observations, **fingerprint))
    totals = {}
    reference = [row["arms"][ARMS[0]] for row in observations]
    for arm in ARMS:
        records = [row["arms"][arm] for row in observations]
        selected_ms = [r["selection_ms"] for r in records if r["selection_ms"] is not None]
        totals[arm] = dict(
            eligible=len(selected_ms), allocated=sum(r["allocated"] for r in records),
            issued=sum(bool(r["primary"] or r["added"]) for r in records),
            slots=sum(len(r["primary"])+len(r["added"]) for r in records),
            multi_window_additions=sum(len(x["citations"])>1 for r in records for x in r["added"]),
            changed_additions=sum(x["added"] != y["added"] for x, y in zip(records, reference, strict=True)),
            recovery_work=dict(sum((Counter(row["work"]) for row in records), Counter())),
            serial_p95_ms=float(np.quantile([r["ms"] for r in records], .95)),
            selection_p95_ms=float(np.quantile(selected_ms,.95)) if selected_ms else None,
        )
    write(output, dict(slurm_job=os.environ["SLURM_JOB_ID"], reverse=reverse, proposal_depth=DEPTH, title_key=title_key,
                       runner_sha256=digest(Path(__file__)), private_sha256=digest(private),
                       compile_audit=audit, window_index_ms=window_index_ms, entry_index_ms=entry_index_ms,
                       arms=totals, **fingerprint))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "vector_meta", "models", "serial", "output", "private"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    cli.add_argument("--title-key", choices=("first_column", "specific"), default="first_column")
    run(**vars(cli.parse_args()))
