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


def records(snapshot, workbook, frame):
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
    if attest != index["publication_sources"]:
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

    external, history = records(args.snapshot, args.workbook, args.frame)
    payload = dict(frame_sha256=digest(args.frame), external=len(external),
                   views=[windows(row["zh"]) for row in external + history])
    # Positions and technical text windows suffice for GPU screening.
    # Names, record identifiers and contact fields are kept local.
    write_private(args.output, [payload])


def pairs_for(values, n):
    return {"within_frame": (0, values[:, :n]), "historical": (n, values[:, n:])}


def signal(values, n):
    pairs, counts = {}, {}
    for scope, (offset, scores) in pairs_for(values, n).items():
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
    counts, pairs = signal(scores, n)
    write_json(args.output, dict(counts=counts, pairs=pairs, gpu=torch.cuda.get_device_name(),
                                 frame_sha256=data["frame_sha256"], projection_sha256=digest(args.inputs),
                                 encoder_revision=checkpoint["revision"], order_offset=args.order_offset))
    args.output.chmod(0o600)


def packet(args):
    from experiments.audit_enterprise_issuer_aliases import close
    from experiments.audit_enterprise_panel_neighbors import write_private
    from experiments.freeze_enterprise_panel import issuer_key

    frame_hash = digest(args.frame)
    semantic = json.loads(args.semantic.read_bytes())
    if semantic["frame_sha256"] != frame_hash:
        raise ValueError("Semantic screen is for a different private frame")
    external, historical_rows = records(args.snapshot, args.workbook, args.frame)
    rows = external + historical_rows
    n = len(external)
    names = [issuer_key(row["company"]) for row in rows]
    org = np.asarray([[close(left, right) for right in names] for left in names[:n]], dtype=np.float64)
    np.fill_diagonal(org[:, :n], -np.inf)
    org_counts, org_pairs = signal(org, n)
    mapping = {}
    for method, scopes in (("semantic", semantic["pairs"]), ("organization", org_pairs)):
        for scope, pairs in scopes.items():
            for left, right in pairs:
                if not 0 <= left < n or not left < right < len(rows):
                    raise ValueError("Unexpected private pair position")
                mapping.setdefault((left, right), []).append(f"{method}/{scope}")
    visible, private = [], []
    for (i, j), reasons in mapping.items():
        a, b = rows[i], rows[j]
        identity = sha256(f"{SALT}:{a['id']}:{b['id']}".encode()).hexdigest()[:24]
        if int(identity, 16) % 2:
            a, b = b, a
        visible.append(dict(case_id=identity, left={"issuer": a["company"], "request": a["zh"]},
                            right={"issuer": b["company"], "request": b["zh"]},
                            intent=None, organizational_link=None, reviewer=None, rationale=None))
        private.append(dict(case_id=identity, left_id=a["id"], right_id=b["id"], candidate_reasons=reasons))
    visible.sort(key=lambda item: item["case_id"])
    args.output.mkdir(mode=0o700)
    form = args.output / "blind-review.jsonl"
    for name in ("blind-review.jsonl", "reviewer-a.jsonl", "reviewer-b.jsonl"):
        write_private(args.output / name, visible)
    write_private(args.output / "private-mapping.jsonl", private)
    write_json(args.output / "summary.json", dict(
        blind_form_sha256=digest(form), blind_review_rows=len(visible),
        selected_external_groups=n, historical_requests=len(historical_rows),
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
        group.set_defaults(command=command)
    args = cli.parse_args()
    args.command(args)
