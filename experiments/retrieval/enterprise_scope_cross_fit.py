"""Compare synthetic and teacher-labeled qualification training by issuer fold."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.live_joint_source_request import verify_embedding
from experiments.retrieval.enterprise_bidirectional_scope import check_right_context, infer, train
from experiments.retrieval.enterprise_cross_precision import CONTROLS, QUALIFICATION, evaluate
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT
from experiments.retrieval.enterprise_isolated_parsing import spans
from experiments.retrieval.enterprise_mixed_scope_student import binary_stats, mix, public_result, single
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write
from experiments.retrieval.enterprise_scope_folds import FOLDS
from experiments.retrieval.enterprise_scope_transfer import make_dataset
from experiments.retrieval.enterprise_shared_scope_student import p50_p95, timed

SYNTHETIC, TEACHER, MIXED = "synthetic", "teacher", "mixed"
ARMS = (SYNTHETIC, TEACHER, MIXED)


def document(text, labels):
    return None, None, text, list(range(len(labels))), labels


@torch.inference_mode()
def teacher(checkpoint, development, reverse):
    parser = ComparisonParser(checkpoint)
    for reader in parser.readers.values():
        reader.broadcast = True
    documents, latency = {}, {}
    order = sorted(development, reverse=reverse)
    for key in order:
        text = development[key]["text"]
        clauses = spans(text)
        actual, latency[key] = timed(lambda: evaluate(text, clauses, parser.readers, "consensus")["labels"])
        documents[key] = document(text, actual == QUALIFICATION)
    revision = parser.readers["bf16"].provenance["revision"]
    return documents, latency, revision


def calibration_cutoff(predict, documents):
    scores = [predict(doc) for doc in documents]
    failed = [float(np.abs(score).min()) for doc, score in zip(documents, scores, strict=True)
              if np.any((score > 0) != doc[-1])]
    return max(failed, default=0.)


def fit_documents(arm, synthetic, real, fold_index):
    if arm == SYNTHETIC:
        return synthetic
    budget = len(synthetic)
    rng = np.random.default_rng(991 + fold_index)
    if arm == TEACHER:
        return [real[k] for k in np.resize(rng.permutation(len(real)), budget)]
    assert arm == MIXED and budget % 2 == 0
    real_part = [real[k] for k in np.resize(rng.permutation(len(real)), budget // 2)]
    return real_part + [synthetic[k] for k in rng.permutation(budget)[:budget // 2]]


def run(args):
    require_cuda(args.output)
    if digest(args.development) != DEVELOPMENT or digest(args.heading) != CONTROLS:
        raise ValueError("Changed development or published heading cohort")
    scope = make_dataset()
    if scope != json.loads(args.scope.read_bytes()):
        raise ValueError("Changed synthetic source")
    synthetic = mix(scope["train"])
    heading = single(json.loads(args.heading.read_bytes()))
    dev = {row["id"]: row for row in rows(args.development)}
    lock = json.loads(args.folds.read_bytes())
    folds = [[key for group in groups for key in group] for groups in lock["folds"]]
    keys = [key for fold in folds for key in fold]
    if (lock["development_sha256"] != DEVELOPMENT or len(folds) != FOLDS
            or len(keys) != len(dev) or set(keys) != dev.keys() or not all(folds)):
        raise ValueError("Unknown or overlapping evaluation folds")

    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = verify_embedding(args.embedding)
    reference, reference_ms, reader = teacher(args.parser, dev, args.reverse)
    torch.cuda.empty_cache()
    labels_digest = sha256(json.dumps({key: value[-1].tolist() for key, value in reference.items()},
                                     sort_keys=True).encode()).hexdigest()
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.requires_grad_(False)
    check_right_context(encoder)
    counters = {arm: Counter() for arm in ARMS}
    latency = {arm: defaultdict(list) for arm in ARMS}
    fold_reports = {}
    shortfall = Counter()
    order = range(FOLDS - 1, -1, -1) if args.reverse else range(FOLDS)
    for i in order:
        cal = (i + 1) % FOLDS
        training = [reference[key] for j, keys_ in enumerate(folds) if j not in {i, cal} for key in keys_]
        validation = [reference[key] for key in folds[cal]]
        test = folds[i][::-1] if args.reverse else folds[i]
        shortfall["all_negative_exact"] += sum(not reference[key][-1].any() for key in test)
        report = {}
        for arm in (ARMS[::-1] if args.reverse else ARMS):
            docs = fit_documents(arm, synthetic, training, i)
            labels = np.concatenate([doc[-1] for doc in docs])
            if not labels.any() or labels.all():
                raise ValueError("Fit requires both clause labels")
            weight = min(30., float((~labels).sum() / labels.sum()))
            outcomes = Counter()
            with ExitStack() as hooks:
                head, fit = train(encoder, docs, "bidirectional_lora", hooks, positive_weight=weight)

                def predict(doc):
                    return infer(encoder, [doc], head, True, 1)

                cutoff = calibration_cutoff(predict, validation)
                calibration_skips = sum(np.abs(predict(doc)).min() > cutoff for doc in validation)
                heading_result = public_result(infer(encoder, heading, head, True), cutoff, heading)
                for key in test:
                    doc = reference[key]
                    scores, elapsed = timed(lambda: predict(doc))
                    expected = doc[-1]
                    predicted = scores > 0
                    accept = bool(np.abs(scores).min() > cutoff)
                    changed = bool(np.any(predicted != expected))
                    outcomes.update(binary_stats(predicted, expected))
                    outcomes.update(requests=1, exact=not changed, skipped=accept,
                                    changed_skips=accept and changed,
                                    qualification_requests=bool(expected.any()),
                                    qualification_requests_skipped=accept and bool(expected.any()),
                                    skipped_reference_ms_over_500=accept and reference_ms[key] > 500)
                    latency[arm]["student"].append(elapsed)
                    latency[arm]["guard"].append(elapsed + (0 if accept else reference_ms[key]))
            counters[arm].update(outcomes)
            report[arm] = dict(fit=fit, positive_weight=weight, cutoff=cutoff,
                               training_anchors=len(labels), calibration_skips=int(calibration_skips),
                               heading=heading_result, development=outcomes)
        fold_reports[str(i)] = report
        print(f"Fold {i + 1}: " + ", ".join(f"{arm}={r['development']['skipped']}" for arm, r in report.items()),
              flush=True)

    write(args.output, dict(
        arms=counters, folds=fold_reports, labeled_reference_sha256=labels_digest,
        all_negative_exact=shortfall["all_negative_exact"],
        latency={arm: {mode: p50_p95(values) for mode, values in item.items()} for arm, item in latency.items()},
        reference_ms=p50_p95(list(reference_ms.values())),
        inputs={name: digest(getattr(args, name)) for name in ("development", "heading", "scope", "folds")},
        grouped_folds=[len(groups) for groups in lock["folds"]],
        script_sha256=digest(Path(__file__)), trainer_sha256=digest(Path(train.__code__.co_filename)),
        reader=reader, embedding=checkpoint, gpu=torch.cuda.get_device_name(),
        job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("development", "heading", "scope", "folds", "embedding", "parser", "output"):
        cli.add_argument(field, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
