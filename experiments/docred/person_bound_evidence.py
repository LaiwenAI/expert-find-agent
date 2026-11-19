"""Contrast people credited with different roles on the same work in Re-DocRED."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


DATA = {
    "dev": "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f",
    "test": "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c",
}
ROLES = {
    "P50": "author", "P57": "director", "P58": "screenwriter", "P86": "composer",
    "P112": "founder", "P161": "cast member", "P162": "producer", "P170": "creator",
    "P175": "performer", "P178": "developer", "P676": "lyricist",
}
ROLE_CUES = {
    "P50": "author wrote written writing novel", "P57": "director directed direction",
    "P58": "screenwriter screenplay script written", "P86": "composer composed music score",
    "P112": "founder founded established created", "P161": "cast stars starring actors role played portrays",
    "P162": "producer produced production", "P170": "creator created designed creation",
    "P175": "performer performed recorded sung sings singer", "P178": "developer developed development",
    "P676": "lyrics lyricist words written",
}
ARMS = ("lexical_two", "local_two", "joint_two", "shared_two", "full")
TASK = ("Determine whether the supplied source explicitly supports the requested relation for the named person "
        "and the named work. A different person or a different contribution does not support the claim. "
        "Use only the supplied text.")
STOP = frozenset("a an and the in on by is of at for with from to film work his her their its was were".split())


def canonical(mentions):
    return max((m["name"].strip() for m in mentions), key=lambda name: (len(tokens(name)), len(name), name))


def stable(*key):
    return sha256(repr(key).encode()).hexdigest()


def flagged_pairs(documents):
    for doc in documents:
        entities = doc["vertexSet"]
        canonical_names = [canonical(mentions) for mentions in entities]
        roles = defaultdict(set)
        labels = {}
        for item in doc["labels"]:
            h, t, r = (item[k] for k in ("h", "t", "r"))
            if r in ROLES and entities[t][0]["type"] == "PER" and entities[h][0]["type"] != "PER":
                roles[h, r].add(t)
                labels[h, t, r] = frozenset(item.get("evidence", ()))
        sentences = [" ".join(s) for s in doc["sents"]]
        if len(sentences) < 2:
            continue
        for (work, relation), people in sorted(roles.items()):
            rivals = set().union(*(person for (h, r), person in roles.items() if h == work and r != relation)) - people
            valid = []
            for target in people:
                positive_name = tuple(tokens(canonical_names[target]))
                if not positive_name:
                    continue
                for rival in rivals:
                    negative_name = tuple(tokens(canonical_names[rival]))
                    if not negative_name or positive_name == negative_name:
                        continue
                    valid.append((target, rival))
            if not valid or not tokens(canonical_names[work]):
                continue
            target, rival = min(valid, key=lambda p: stable(doc["title"], work, relation, p))
            yield dict(
                doc=doc["title"], work=canonical_names[work], role=relation,
                people=(canonical_names[target], canonical_names[rival]),
                annotated=labels[work, target, relation],
                sentences=sentences, id=stable(doc["title"], work, relation),
            )


def profile(sentences):
    words = [tuple(tokens(s)) for s in sentences]
    proper = [{t.casefold() for t in s.split()[1:] if len(t) >= 3 and t[0].isupper() and t.isalpha()}
              for s in sentences]
    frequency = Counter(word for row in words for word in set(row))
    return words, proper, {w: math.log1p((len(words)-n+.5)/(n+.5)) for w, n in frequency.items()}


def coverage(query, words, idf):
    unique = frozenset(query)
    total = sum(idf.get(word, 1.) for word in sorted(unique)) or 1.
    return sum(idf[word] for word in sorted(unique & set(words))) / total


def contains(words, query):
    n = len(query)
    return bool(n and any(words[i:i+n] == query for i in range(len(words)-n+1)))


def choose(case, person, parsed, arm):
    words, proper, idf = parsed
    actor, work = tuple(tokens(person)), tuple(tokens(case["work"]))
    role = tuple(tokens(ROLE_CUES[case["role"]]))
    p = np.array([max(float(contains(row, actor)), .6*coverage(actor, row, idf)) for row in words])
    w = np.array([max(float(contains(row, work)), .6*coverage(work, row, idf)) for row in words])
    r = np.array([coverage(role, row, idf) for row in words])
    single = p + w + r
    order = sorted(range(len(words)), key=lambda i: (-single[i], i))
    if arm == "lexical_two":
        return tuple(sorted(order[:2]))
    if arm == "local_two":
        first = order[0]
        second = min((i for i in range(len(words)) if i != first), key=lambda i: (abs(i-first), -single[i], i))
        return tuple(sorted((first, second)))

    exclude = frozenset(actor + work + role) | STOP

    def pair_score(pair):
        a, b = pair
        shared = (proper[a] & proper[b]) - exclude
        link = min(.25, .1*len(shared)) + .1*(b-a == 1)
        joint = max(p[a], p[b]) + max(w[a], w[b]) + max(r[a], r[b])
        return joint + link, single[a] + single[b], -a, -b

    return max(combinations(range(len(words)), 2), key=pair_score)


def shared_window(case, parsed):
    words, _, idf = parsed
    names = [tuple(tokens(person)) for person in case["people"]]
    work = tuple(tokens(case["work"]))
    role = tuple(tokens(ROLE_CUES[case["role"]]))
    signals = [
        [max(float(contains(row, name)), .6 * coverage(name, row, idf)) for row in words]
        for name in names + [work]
    ]
    role_scores = [coverage(role, row, idf) for row in words]

    def score(pair):
        a, b = pair
        actor_evidence = [max(values[a], values[b]) for values in signals[:2]]
        work_score = max(signals[2][a], signals[2][b])
        role_score = max(role_scores[a], role_scores[b])
        return sum(actor_evidence) + work_score + role_score, min(actor_evidence), -a, -b

    return max(combinations(range(len(words)), 2), key=score)


def observe(model, case, rotation):
    parsed = profile(case["sentences"])
    rows, truncated = {}, 0
    order = ARMS[rotation:] + ARMS[:rotation]
    reverse_person = int(case["id"][0], 16) % 2 == 1
    positions = (1, 0) if reverse_person else (0, 1)
    for arm in order:
        start = perf_counter()
        if arm == "full":
            selected = [tuple(range(len(parsed[0])))] * 2
        elif arm == "shared_two":
            selected = [shared_window(case, parsed)] * 2
        else:
            selected = [choose(case, person, parsed, arm) for person in case["people"]]
        request = [(f"Was {case['people'][i]} credited as a {ROLES[case['role']]} of {case['work']}?",
                    "\n".join(case["sentences"][k] for k in selected[i])) for i in positions]
        margins, cuts = model.score_pairs(request, task=TASK)
        ms = (perf_counter()-start)*1000
        if reverse_person:
            margins.reverse()
        if not np.isfinite(margins).all():
            raise ValueError("Nonfinite support margin")
        truncated += cuts
        evidence = case["annotated"]
        rows[arm] = dict(
            positive=margins[0], rival=margins[1], ms=ms, sentences=len(selected[0]),
            kept_all=int(bool(evidence) and evidence <= set(selected[0])),
        )
    return None if truncated else (case, rows)


def summarize(rows, cuts):
    report = {}
    docs = [case["doc"] for case, _ in rows]
    for arm in ARMS:
        cutoff = cuts[arm]
        results = [values[arm] for _, values in rows]
        win = [int(row["positive"] > row["rival"]) for row in results]
        hits = [int(row["positive"] > cutoff and row["rival"] <= cutoff) for row in results]
        ref_win = [int(values["lexical_two"]["positive"] > values["lexical_two"]["rival"]) for _, values in rows]
        ref_hits = [int(values["lexical_two"]["positive"] > cuts["lexical_two"]
                        and values["lexical_two"]["rival"] <= cuts["lexical_two"]) for _, values in rows]
        report[arm] = dict(
            ordered=sum(win), both_correct=sum(hits),
            accepted_supported=sum(row["positive"] > cutoff for row in results),
            accepted_rival=sum(row["rival"] > cutoff for row in results),
            annotated_evidence_retained=sum(row["kept_all"] for row in results),
            ms_p95=float(np.quantile([row["ms"] for row in results], .95)),
            mean_sentences=float(np.mean([row["sentences"] for row in results])),
            order_delta_vs_lexical=clustered_paired_interval([a-b for a,b in zip(win,ref_win,strict=True)], docs),
            both_delta_vs_lexical=clustered_paired_interval([a-b for a,b in zip(hits,ref_hits,strict=True)], docs),
        )
    return dict(pairs=len(rows), documents=len(set(docs)), with_annotated_evidence=sum(bool(c["annotated"]) for c, _ in rows),
                roles=dict(Counter(c["role"] for c, _ in rows)), arms=report)


def run(snapshot, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    panels, omissions, titles = {}, {}, {}
    encoder = Reranker(reranker)
    for name, digest_expected in DATA.items():
        location = snapshot / f"{name}_revised.json"
        if digest(location) != digest_expected:
            raise ValueError(f"Changed Re-DocRED file: {name}")
        documents = json.loads(location.read_bytes())
        titles[name] = {doc["title"] for doc in documents}
        cases = list(flagged_pairs(documents))
        observe(encoder, cases[0], 0)
        measured = [observe(encoder, case, i % len(ARMS)) for i, case in enumerate(cases)]
        panels[name] = [row for row in measured if row is not None]
        omissions[name] = len(cases) - len(panels[name])
    if titles["dev"] & titles["test"]:
        raise ValueError("Overlapping document titles")
    dev = panels["dev"]
    cap = math.floor(.05 * len(dev))
    cuts = {arm: sorted((row[arm]["rival"] for _, row in dev), reverse=True)[cap] for arm in ARMS}
    write_json(output, dict(
        data_sha256=DATA, checkpoint=encoder.provenance["revision"], job=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)), omissions=omissions,
        dev_rival_cap=cap, thresholds=cuts, panels={name:summarize(rows,cuts) for name,rows in panels.items()},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "reranker", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
