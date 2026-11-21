"""Screen a public enterprise frame for semantic and organization overlap."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from experiments.followir.core17_reranking import digest, write_json

THRESHOLDS = (.85, .90, .95)
FLOOR = .90
LIMIT = 24
FALLBACK = 8
SALT = "external-enterprise-overlap-v1"


def records(snapshot, workbook, frame, first_batch=None):
    from experiments.external_enterprise_frame import BATCHES, COUNTS, historical, nantong, package

    index = json.loads(frame.read_bytes())
    if digest(workbook) != index["historical_workbook_sha256"]:
        raise ValueError("Historical workbook changed")
    by_id = {}
    attest = {}
    for batch, count in zip(BATCHES, COUNTS, strict=True):
        rows, attest[batch] = package(snapshot, batch, count)
        by_id.update((row["id"], row) for row in rows)
    path = snapshot / "nantong.html"
    by_id.update((row["id"], row) for row in nantong(path))
    attest["nantong"] = {"notice_sha256": digest(path)}
    if first_batch is not None:
        first, attest["first"] = package(first_batch, "first", 30)
        if any(row["id"] in by_id for row in first):
            raise ValueError("First batch overlaps the notice ID inventory")
        by_id.update((row["id"], row) for row in first)
    expected = index["expanded_source_hashes"] if first_batch is not None else index["publication_sources"]
    if attest != expected:
        raise ValueError("External input snapshots changed")
    rows = []
    for record in index["records"]:
        row = by_id[record["id"]]
        if sha256(row["zh"].encode()).hexdigest() != record["text_sha256"]:
            raise ValueError("Selected demand text changed")
        rows.append(row)
    return rows, historical(workbook)


def prepare(args):
    from experiments.audit_enterprise_panel_neighbors import windows, write_private

    external, history = records(args.snapshot, args.workbook, args.frame, args.first_batch)
    payload = dict(frame_sha256=digest(args.frame), external=len(external),
                   views=[windows(row["zh"]) for row in external + history])
    if args.first_batch is not None:
        focus = [i for i, request in enumerate(external) if request["batch"] == "first"]
        if not focus:
            raise ValueError("No first-batch candidates to screen")
        payload["focus"] = focus
    # Positions and technical text windows suffice for GPU screening.
    # Names, record identifiers and contact fields are kept local.
    write_private(args.output, [payload])


def restrict_rows(values, focus):
    keep = np.zeros(len(values), dtype=bool)
    indices = np.asarray(focus)
    if not len(indices) or indices.min() < 0 or indices.max() >= len(values):
        raise ValueError("Invalid marked candidate rows")
    keep[indices] = True
    values[~keep] = -np.inf


def signal(values, n, earlier=None):
    if earlier is None:
        destinations = {"within_frame": (0, values[:, :n])}
    else:
        if not 0 < earlier < n:
            raise ValueError("Expected nonempty earlier and newly added groups")
        destinations = {
            "earlier_frame": (0, values[:, :earlier]),
            "new_groups": (earlier, values[:, earlier:n]),
        }
    destinations["historical"] = (n, values[:, n:])
    pairs, counts = {}, {}
    for scope, (offset, scores) in destinations.items():
        unique = {}
        for i, j in np.argwhere(scores >= FLOOR):
            other = j + offset
            key = (int(min(i, other)), int(max(i, other)))
            unique[key] = max(float(scores[i, j]), unique.get(key, -1.0))
        above_floor = len(unique)
        nearest = scores.max(axis=1)
        if not unique:
            # An empty high-threshold packet cannot tell whether the
            # screen misses overlap. Include the strongest nearest
            # neighbors as a below-threshold diagnostic in this case.
            for i, local in enumerate(scores.argmax(axis=1)):
                other = int(local) + offset
                if np.isfinite(nearest[i]):
                    key = (min(i, other), max(i, other))
                    unique[key] = max(float(nearest[i]), unique.get(key, -1.0))
        ordered = sorted(unique, key=lambda key: (-unique[key], key))
        limit = LIMIT if above_floor else FALLBACK
        counts[scope] = {
            "candidate_groups_at_least": {str(level): int((nearest >= level).sum()) for level in THRESHOLDS},
            "pairs_above_review_floor": above_floor,
            "pairs_selected": min(limit, len(ordered)),
            "exploratory_below_floor": min(limit, len(ordered)) if not above_floor else 0,
        }
        pairs[scope] = [list(key) for key in ordered[:limit]]
    return counts, pairs


def gpu(args):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.litsearch.dense_cache import verify_checkpoint

    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm GPU")
    data = json.loads(args.inputs.read_bytes())
    checkpoint = verify_checkpoint(args.checkpoint)
    encoder = QwenTorchEmbedding(args.checkpoint, max_length=384, batch_size=32)
    views = list(zip(*data["views"], strict=True))
    vectors = {}
    order = (1, 0) if args.order_offset else (0, 1)
    for i in order:
        vectors[i] = encoder.encode(list(views[i]))
    n = data["external"]
    scores = np.maximum.reduce([vectors[i][:n] @ vectors[j].T for i in (0, 1) for j in (0, 1)])
    np.fill_diagonal(scores[:, :n], -np.inf)
    if "focus" in data:
        restrict_rows(scores, data["focus"])
    earlier = data.get("earlier")
    counts, pairs = signal(scores, n, earlier)
    write_json(args.output, dict(counts=counts, pairs=pairs, gpu=torch.cuda.get_device_name(),
                                 frame_sha256=data["frame_sha256"], projection_sha256=digest(args.inputs),
                                 encoder_revision=checkpoint["revision"], order_offset=args.order_offset,
                                 focal_groups=len(data["focus"]) if "focus" in data else n)
               | ({"earlier_groups": earlier} if earlier is not None else {}))
    args.output.chmod(0o600)


def packet(args):
    frame_hash = digest(args.frame)
    semantic = json.loads(args.semantic.read_bytes())
    if semantic["frame_sha256"] != frame_hash:
        raise ValueError("Semantic screen is for a different private frame")
    external, historical_rows = records(args.snapshot, args.workbook, args.frame, args.first_batch)
    focus = ([i for i, row in enumerate(external) if row["batch"] == "first"]
             if args.first_batch else list(range(len(external))))
    build_packet(external, historical_rows, focus, semantic, args.output)


def build_packet(external, historical_rows, focus, semantic, output, earlier=None, salt=SALT):
    from experiments.audit_enterprise_issuer_aliases import close
    from experiments.audit_enterprise_panel_neighbors import write_private
    from experiments.freeze_enterprise_panel import issuer_key

    rows = external + historical_rows
    n = len(external)
    names = [issuer_key(row["company"]) for row in rows]
    if semantic.get("focal_groups", n) != len(focus):
        raise ValueError("Semantic screen used a different focal set")
    org = np.full((n, len(names)), -np.inf, dtype=np.float64)
    for i in focus:
        org[i] = [close(names[i], right) for right in names]
    np.fill_diagonal(org[:, :n], -np.inf)
    org_counts, org_pairs = signal(org, n, earlier)
    mapping = {}
    for method, scopes in (("semantic", semantic["pairs"]), ("organization", org_pairs)):
        for scope, pairs in scopes.items():
            for left, right in pairs:
                if not 0 <= left < n or not left < right < len(rows):
                    raise ValueError("Unexpected private pair position")
                if (earlier is not None and (
                        (scope == "earlier_frame" and not left < earlier <= right < n)
                        or (scope == "new_groups" and not earlier <= left < right < n)
                        or (scope == "historical" and not earlier <= left < n <= right))):
                    raise ValueError("Pair does not match the declared exposure stratum")
                mapping.setdefault((left, right), []).append(f"{method}/{scope}")
    visible, private = [], []
    for (i, j), reasons in mapping.items():
        a, b = rows[i], rows[j]
        identity = sha256(f"{salt}:{a['id']}:{b['id']}".encode()).hexdigest()[:24]
        if int(identity, 16) % 2:
            a, b = b, a
        visible.append(dict(case_id=identity, left={"issuer": a["company"], "request": a["zh"]},
                            right={"issuer": b["company"], "request": b["zh"]},
                            intent=None, organizational_link=None, reviewer=None, rationale=None))
        private.append(dict(case_id=identity, left_id=a["id"], right_id=b["id"], candidate_reasons=reasons))
    visible.sort(key=lambda item: item["case_id"])
    output.mkdir(mode=0o700)
    form = output / "blind-review.jsonl"
    for name in ("blind-review.jsonl", "reviewer-a.jsonl", "reviewer-b.jsonl"):
        write_private(output / name, visible)
    write_private(output / "private-mapping.jsonl", private)
    write_json(output / "summary.json", dict(
        blind_form_sha256=digest(form), blind_review_rows=len(visible),
        selected_external_groups=n, focal_groups=len(focus), historical_requests=len(historical_rows),
        semantic=semantic["counts"], organization=org_counts, human_reviews=0,
        gpu=semantic["gpu"], encoder_revision=semantic["encoder_revision"],
        script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    subparsers = cli.add_subparsers(dest="mode", required=True)
    for command, names in ((prepare, ("snapshot", "workbook", "frame", "output")),
                           (gpu, ("inputs", "checkpoint", "output")),
                           (packet, ("snapshot", "workbook", "frame", "semantic", "output"))):
        group = subparsers.add_parser(command.__name__)
        for name in names:
            group.add_argument(name, type=Path)
        if command is gpu:
            group.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
        else:
            group.add_argument("--first-batch", type=Path)
        group.set_defaults(command=command)
    args = cli.parse_args()
    args.command(args)
