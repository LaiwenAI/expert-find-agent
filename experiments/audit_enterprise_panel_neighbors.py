"""Flag semantic and organizational overlap for local blind review of a sealed panel."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding import QwenEmbedding
from experiments.freeze_enterprise_panel import digest_file, exposure_ids, read_rows, workbook_rows
from experiments.litsearch.dense_cache import verify_checkpoint


WINDOW = 640
THRESHOLDS = (.85, .90, .95)
REVIEW_FLOOR = .90
REVIEW_LIMIT = 24
SALT = "enterprise-panel-neighbor-audit-v1"


def checked_records(workbook, cohort, legacy, registry, panel, exposure):
    summary = json.loads((panel / "summary.json").read_bytes())
    index = panel / "sealed-index.json"
    if digest_file(index) != summary["sealed_index_sha256"]:
        raise ValueError("Sealed index differs from the audited inventory")
    required = dict(workbook=workbook, cohort=cohort, legacy=legacy, registry=registry)
    expected = summary["input_sha256"]
    if any(digest_file(path) != expected[name] for name, path in required.items()):
        raise ValueError("A previously audited input changed")
    if [digest_file(path) for path in exposure] != expected["exposure_manifests"]:
        raise ValueError("Use the original exposure files in audited order")
    selected = json.loads(index.read_bytes())["records"]
    source = read_rows(cohort)
    by_id = {row["id"]: row for row in source}
    if len(by_id) != len(source) or len(selected) != summary["selected_groups"]:
        raise ValueError("Unexpected source or panel inventory")
    for row in selected:
        record = by_id[row["id"]]
        if (record["split"] != "sealed" or record["source_row"] != row["workbook_row"]
                or sha256(record["zh"].encode()).hexdigest() != row["text_sha256"]):
            raise ValueError("A panel representative changed")
    exposure_set = {row["id"] for row in source if row["split"] != "sealed"}
    exposure_set.update(row["id"] for row in read_rows(legacy))
    exposure_set.update(json.loads(registry.read_bytes())["demand_ids"])
    exposure_set.update(set().union(*(exposure_ids(path) for path in exposure)))
    by_id.update(workbook_rows(workbook, exposure_set - by_id.keys()))
    representatives = [row["id"] for row in selected]
    if len(set(representatives)) != len(representatives) or exposure_set.intersection(representatives):
        raise ValueError("A sealed representative was exposed or duplicated")
    return by_id, representatives, sorted(exposure_set), summary["sealed_index_sha256"]


def windows(text):
    title, _, body = text.partition("\n")
    return f"{title}\n{body[:WINDOW]}".strip(), f"{title}\n{body[-WINDOW:]}".strip()


def nearest(vectors, query_count):
    similarity = np.maximum.reduce([
        a[:query_count] @ b.T for a in vectors for b in vectors
    ])
    np.fill_diagonal(similarity[:, :query_count], -np.inf)
    result = {}
    for name, start, end in (("within_panel", 0, query_count),
                             ("prior_use", query_count, similarity.shape[1])):
        scores = similarity[:, start:end]
        best = np.argmax(scores, axis=1)
        result[name] = [(int(i), int(j + start), float(scores[i, j])) for i, j in enumerate(best)]
    return result


def make_packet(rows, pairs):
    scopes = {}
    selected = {}
    for name, edges in pairs.items():
        possible = {}
        for i, j, score in edges:
            key = tuple(sorted((i, j)))
            possible[key] = max(score, possible.get(key, -1.))
        flagged = sorted(((key, score) for key, score in possible.items() if score >= REVIEW_FLOOR),
                         key=lambda item: (-item[1], item[0]))
        for key, score in flagged[:REVIEW_LIMIT]:
            selected.setdefault(key, {})[name] = score
        scopes[name] = {
            "nearest_scores_at_least": {str(level): sum(score >= level for _, _, score in edges)
                                       for level in THRESHOLDS},
            "distinct_pairs_above_review_floor": len(flagged),
            "pairs_in_packet": min(REVIEW_LIMIT, len(flagged)),
        }
    visible, mapping = [], []
    for (i, j), types in selected.items():
        left, right = rows[i], rows[j]
        case = sha256(f"{SALT}:{left['id']}:{right['id']}".encode()).hexdigest()[:24]
        if int(case, 16) % 2:
            left, right = right, left
        visible.append(dict(
            case_id=case,
            left={"issuer": left["company"], "request": left["zh"]},
            right={"issuer": right["company"], "request": right["zh"]},
            intent=None, organizational_link=None, reviewer=None, rationale=None,
        ))
        mapping.append(dict(case_id=case, left_id=left["id"], right_id=right["id"], candidate_reasons=types))
    return sorted(visible, key=lambda row: row["case_id"]), mapping, scopes


def write_private(path, values):
    with path.open("x", encoding="utf-8") as handle:
        path.chmod(0o600)
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=True) + "\n")


def run(workbook, cohort, legacy, registry, panel, checkpoint, output, exposure):
    if output.exists():
        raise FileExistsError(output)
    data, representatives, known, anchor = checked_records(workbook, cohort, legacy, registry, panel, exposure)
    rows = [data[key] for key in representatives + known]
    provenance = verify_checkpoint(checkpoint)
    if provenance["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected frozen document encoder")
    encoder = QwenEmbedding(checkpoint, max_length=384, batch_size=16)
    if encoder.mx.default_device() != encoder.mx.gpu:
        raise RuntimeError("Local GPU is required")
    views = list(zip(*(windows(row["zh"]) for row in rows), strict=True))
    vectors = []
    start = perf_counter()
    for view in views:
        vectors.append(encoder.encode(list(view)))
        print("Encoded local audit view", len(vectors), "of", len(views), flush=True)
    pairs = nearest(vectors, len(representatives))
    elapsed = perf_counter() - start
    packet, mapping, scopes = make_packet(rows, pairs)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_private(output / "blind-review.jsonl", packet)
    write_private(output / "private-mapping.jsonl", mapping)
    summary = dict(
        audited_index_sha256=anchor, script_sha256=digest_file(Path(__file__)), checkpoint=provenance,
        selected_groups=len(representatives), prior_use_requests=len(known), windows_per_request=len(views),
        review_floor=REVIEW_FLOOR, review_limit_per_scope=REVIEW_LIMIT, diagnostic=scopes,
        blind_review_rows=len(packet), human_reviews=0, elapsed_seconds=elapsed,
        blind_form_sha256=digest_file(output / "blind-review.jsonl"),
        private_mapping_sha256=digest_file(output / "private-mapping.jsonl"),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Local blind review pairs:", len(packet), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workbook", "cohort", "legacy", "registry", "panel", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--exposure", type=Path, action="append", required=True)
    run(**vars(parser.parse_args()))
