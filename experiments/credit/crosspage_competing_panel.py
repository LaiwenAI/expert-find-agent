"""Freeze new CrossPage queries over an already reconstructed set of PDFs."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.credit.crosspage_pdf_panel import REVISION, hash_order, load, write


CORRECTIONS = Path("audit/admin/final_adjudication/qrel_corrections.jsonl")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def build(root, pilot, audit, output, manifest):
    prior = json.loads(pilot.read_bytes())
    audited = json.loads(audit.read_bytes())
    paths = [Path("data/v1") / f"{name}.jsonl" for name in ("corpus", "queries", "qrels")]
    corpus, queries, qrels = [load(root / path) for path in paths]
    family = {row["doc_id"]: row["family_id"] for row in corpus}
    excluded = {family[doc] for doc in audited["documents"]}
    allowed = {doc for doc in prior["documents"] if family[doc] not in excluded}
    used_queries = {row["query_id"] for row in prior["queries"] + audited["rows"]}
    adjustments = {(row["query_id"], row["doc_id"]): row["new_grade"]
                   for row in load(root / CORRECTIONS) if row["release"] == "v1"}
    grades = defaultdict(dict)
    for row in qrels:
        query_id, doc, grade = row["query_id"], row["doc_id"], row["grade"]
        corrected = adjustments.pop((query_id, doc), grade)
        if corrected != grade:
            raise ValueError("Released qrel does not include the expected correction")
        if doc in allowed and query_id not in used_queries:
            grades[query_id][doc] = grade
    if adjustments:
        raise ValueError("Expected corrected qrels are missing")

    cases = []
    for row in sorted(queries, key=lambda r: r["query_id"]):
        qid, arity = row["query_id"], row["n"]
        if arity not in (2, 3):
            continue
        pool = grades[qid]
        order = sorted(pool, key=lambda doc: hash_order("competing", qid, doc))
        partial = [doc for doc in order if 0 < pool[doc] < arity]
        for gold in (doc for doc in order if pool[doc] == arity):
            negatives = {}
            for doc in partial:
                if family[doc] != family[gold]:
                    negatives.setdefault(family[doc], doc)
                if len(negatives) == 3:
                    break
            if len(negatives) != 3:
                continue
            cases.append(dict(
                query_id=qid, arity=arity, query=row["query_text_eval_full"],
                conditions=[condition["concept"] for condition in row["conditions"]],
                grades={doc: pool[doc] for doc in sorted([gold, *negatives.values()])},
            ))
            break
    sources = sorted({doc for row in cases for doc in row["grades"]})
    if not cases:
        raise ValueError("No eligible queries")
    write(output, dict(queries=cases, documents={doc: prior["documents"][doc] for doc in sources}))
    write(manifest, dict(
        upstream_revision=REVISION,
        upstream_sha256={str(path): digest(root / path) for path in [*paths, CORRECTIONS]},
        pilot_panel_sha256=digest(pilot), audited_panel_sha256=digest(audit),
        panel_sha256=digest(output), excluded_families=sorted(excluded),
        included_queries={row["query_id"]: row["grades"] for row in cases},
        families={doc: family[doc] for doc in sources},
    ))
    print(f"{len(cases)} held queries, {len(sources)} PDFs, "
          f"{len({family[doc] for doc in sources})} PDF families.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "pilot", "audit", "output", "manifest"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()))
