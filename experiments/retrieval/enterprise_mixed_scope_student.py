"""Train shared qualification readouts on requests with several clause roles."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.retrieval.enterprise_cross_precision import CONTROLS, QUALIFICATION, evaluate
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT
from experiments.retrieval.enterprise_isolated_parsing import spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_qualification_student import target_index
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write
from experiments.retrieval.enterprise_scope_transfer import ROLES, classifier, make_dataset
from experiments.retrieval.enterprise_shared_scope_student import ScopeFeatures, calculate, gold, p50_p95, timed

# Maintain zero, one and multiple qualifications. Rotate role patterns
# independently of the topic order and view, within each held split.
PATTERNS = ((0, 1, 2), (3, 1, 2), (1, 2, 0), (1, 2, 3),
            (2, 0, 3), (2, 1, 3), (0, 3, 0), (0, 3, 1))
ARMS = ("single", "mixed_matched", "mixed", "mixed_anchored")


def mix(panel):
    layouts = defaultdict(lambda: defaultdict(dict))
    for item in panel:
        key = (item["family"], item["language"])
        layouts[key][item["topic"]][item["role"]] = item
    documents = []
    for key, topics in sorted(layouts.items()):
        topic_ids = sorted(topics)
        for start in range(len(topic_ids)):
            order = (topic_ids + topic_ids)[start:start + 3]
            for pattern in PATTERNS:
                parts, offsets, expected = [], [], []
                position = 0
                for topic, role_index in zip(order, pattern, strict=True):
                    role = ROLES[role_index]
                    item = topics[topic][role]
                    clauses, index = target_index(item)
                    parts.append(item["text"])
                    offsets.append(tuple(position + clauses[index][edge] for edge in ("start", "end")))
                    expected.append(role_index == 0)
                    position += len(item["text"]) + 2
                text = "\n\n".join(parts)
                locations = [(clause["start"], clause["end"]) for clause in spans(text)]
                indices = [locations.index(offset) for offset in offsets]
                documents.append((key, tuple(order), text, indices, expected))
    return documents


def matched(documents):
    by_layout = defaultdict(list)
    for doc in documents:
        by_layout[doc[0]].append(doc)
    size = len(PATTERNS)
    return [docs[(offset % (len(docs) // size)) * size + offset]
            for docs in by_layout.values() for offset in range(size)]


def single(panel):
    return [(row.get("pair"), (0,), row["text"], [target_index(row)[1]], [gold(row)])
            for row in panel]


def read_features(encoder, doc, arm):
    _, _, text, indices, _ = doc
    clauses = spans(text)
    if arm != "mixed_anchored":
        return encoder.one(text, clauses, indices, "shared_symmetric")[0]
    shared = encoder.one(text, clauses, list(range(len(clauses))), "shared_symmetric")[0]
    anchors = shared[:, :shared.shape[1] // 2]
    # All clause vectors come from the two existing forwards. Include
    # neighbors as separate features, and use zeros at document boundaries.
    padded = np.pad(anchors, ((2, 2), (0, 0)))
    indices = np.array(indices)
    local = np.concatenate([padded[indices + offset + 2] for offset in (-2, -1, 1, 2)], axis=1)
    joined = np.concatenate((shared[indices], local), axis=1)
    return normalize(torch.from_numpy(joined), dim=1).numpy()


def features(encoder, documents, arm):
    return np.concatenate([read_features(encoder, doc, arm) for doc in documents])


def binary_stats(prediction, expected):
    return dict(qualifications=int(expected.sum()), correct=int((prediction == expected).sum()),
                true_positive=int((prediction & expected).sum()),
                false_positive=int((prediction & ~expected).sum()), cases=len(expected))


def public_result(scores, cutoff, documents):
    flat = np.concatenate([doc[-1] for doc in documents])
    chosen = np.abs(scores) > cutoff
    labels = scores > 0
    complete = (labels == flat).reshape(len(documents), -1).all(axis=1)
    changed = defaultdict(set)
    for i, (key, topics, _, indices, expected) in enumerate(documents):
        for j, (topic, target) in enumerate(zip(topics or range(len(indices)), expected, strict=True)):
            changed[key, topics, topic].add((bool(target), bool(labels[i * len(indices) + j])))
    pairs = [value for value in changed.values() if len({key for key, _ in value}) > 1]
    return binary_stats(labels, flat) | dict(
        documents=len(documents), exact_documents=int(complete.sum()),
        accepted=int(chosen.sum()), accepted_errors=int((chosen & (labels != flat)).sum()),
        eligible_counterfactual_anchors=len(pairs),
        all_counterfactual_views_correct=sum(all(a == b for a, b in pair) for pair in pairs),
    )


@torch.inference_mode()
def run(args):
    require_cuda(args.output)
    script_sha256 = digest(Path(__file__))
    if digest(args.development) != DEVELOPMENT or digest(args.heading) != CONTROLS:
        raise ValueError("Expected frozen development projection and heading controls")
    torch.backends.cuda.matmul.allow_tf32 = False
    scope = make_dataset()
    if scope != json.loads(args.scope.read_bytes()):
        raise ValueError("Changed public scope data")
    mixed = {split: mix(panel) for split, panel in scope.items()}
    heading = single(json.loads(args.heading.read_bytes()))
    encoder = ScopeFeatures(QwenTorchEmbedding(args.embedding))
    parser = ComparisonParser(args.parser)
    for reader in parser.readers.values():
        reader.broadcast = True

    training_by_arm = {"single": single(scope["train"]), "mixed_matched": matched(mixed["train"])}
    if sum(len(doc[-1]) for doc in training_by_arm["mixed_matched"]) != len(scope["train"]):
        raise ValueError("Mixed matched fit needs the same number of labeled anchors")
    heads, report, cutoffs = {}, {}, {}
    for arm in ARMS:
        training = training_by_arm.get(arm, mixed["train"])
        y = np.concatenate([doc[-1] for doc in training])
        heads[arm] = classifier(features(encoder, training, arm), y, (), False)
        calibration = mixed["calibration"]
        raw = calculate(features(encoder, calibration, arm), heads[arm])
        truth = np.concatenate([doc[-1] for doc in calibration])
        cutoffs[arm] = float(np.abs(raw)[(raw > 0) != truth].max(initial=0.))
        report[arm] = dict(training_anchors=len(y), cutoff=cutoffs[arm])
        for split, docset in (("mixed_held", mixed["held"]), ("heading", heading)):
            scores = calculate(features(encoder, docset, arm), heads[arm])
            result = public_result(scores, cutoffs[arm], docset)
            if split == "heading" and result["eligible_counterfactual_anchors"] != 36:
                raise ValueError("Expected 36 independent heading pairs")
            report[arm][split] = result

    counts = {arm: Counter() for arm in ARMS}
    latency = defaultdict(list)
    dev = rows(args.development)
    if len(dev) != 80 or any(row["split"] != "development" for row in dev):
        raise ValueError("Expected 80 development rows")
    for row in (reversed(dev) if args.reverse else dev):
        text = row["text"]
        clauses = spans(text)
        truth, latency_ms = timed(lambda: evaluate(text, clauses, parser.readers, "consensus")["labels"])
        expected = truth == QUALIFICATION
        latency["reference"].append(latency_ms)
        document = (None, None, text, list(range(len(clauses))), [])
        for arm in (ARMS[::-1] if args.reverse else ARMS):
            result, ms = timed(lambda: calculate(read_features(encoder, document, arm), heads[arm]))
            labels = result > 0
            accept = bool((np.abs(result) > cutoffs[arm]).all())
            change = bool((labels != expected).any())
            counts[arm].update(binary_stats(labels, expected))
            counts[arm].update(requests=1, changed_requests=int(change), skipped=int(accept),
                               changed_skips=int(accept and change))
            latency[arm].append(ms)
            latency[arm + "_guard"].append(ms + (0. if accept else latency_ms))

    write(args.output, dict(
        public=report, development=counts, latency={key: p50_p95(val) for key, val in latency.items()},
        mixed_digest=sha256(json.dumps(mixed, ensure_ascii=True).encode()).hexdigest(),
        mixed_documents={key: len(value) for key, value in mixed.items()},
        script_sha256=script_sha256, inputs={key: digest(getattr(args, key))
                                                      for key in ("development", "heading", "scope")},
        encoder=digest(args.embedding / "checkpoint-provenance.json"),
        reader=parser.readers["bf16"].provenance["revision"], gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, reverse=args.reverse,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "heading", "scope", "embedding", "parser", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
