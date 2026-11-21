"""Fit a single-pass open-weight clause parser with right-context attention."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.train_role_decision import Adapter
from experiments.credit.live_joint_source_request import verify_embedding
from experiments.retrieval.enterprise_cross_precision import CONTROLS, QUALIFICATION, evaluate
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT
from experiments.retrieval.enterprise_isolated_parsing import spans
from experiments.retrieval.enterprise_mixed_scope_student import binary_stats, mix, public_result, single
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write
from experiments.retrieval.enterprise_scope_transfer import make_dataset
from experiments.retrieval.enterprise_shared_scope_student import p50_p95, timed


ARMS = ("bidirectional_head", "causal_lora", "bidirectional_lora")
SEED = 4281
EPOCHS = 6
BATCH = 8
MAX_TOKENS = 4096


def tokenize(encoder, documents):
    text = [doc[2] for doc in documents]
    encoded = encoder.tokenizer(text, padding=True, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping").tolist()
    if encoded.input_ids.shape[1] > MAX_TOKENS:
        raise ValueError("Request exceeds the clause parser token budget")
    row_ids, positions = [], []
    for i, (doc, mapping) in enumerate(zip(documents, offsets, strict=True)):
        clauses = spans(doc[2])
        for j in doc[3]:
            clause = clauses[j]
            covered = [k for k, (left, right) in enumerate(mapping)
                       if right > left and left < clause["end"] and right > clause["start"]]
            if not covered:
                raise ValueError("Clause has no overlapping tokens")
            row_ids.append(i)
            positions.append(covered[-1])
    return encoded.to("cuda"), row_ids, positions


def logits(encoder, encoded, row_ids, positions, head, bidirectional):
    mask = encoded.attention_mask
    position_ids = (mask.cumsum(1) - 1).clamp_min(0)
    if bidirectional:
        width = mask.shape[1]
        mask = mask[:, None, None, :].expand(-1, 1, width, -1).bool()
    output = encoder.model(input_ids=encoded.input_ids, attention_mask=mask,
                           position_ids=position_ids, use_cache=False).last_hidden_state
    return head(F.normalize(output[row_ids, positions].float(), dim=-1)).squeeze(1)


def check_right_context(encoder):
    prompts = [
        ("early target\n\nRequired: prior field validation.", 0),
        ("early target\n\nOptional: future site visits.", 0),
    ]
    documents = [(None, None, text, [target], []) for text, target in prompts]
    encoded, indices, positions = tokenize(encoder, documents)
    # Project the full normalized representation to test attention, not the classifier.
    with torch.inference_mode():
        causal = logits(encoder, encoded, indices, positions, nn.Identity(), False)
        complete = logits(encoder, encoded, indices, positions, nn.Identity(), True)
    causal_difference = float((causal[0] - causal[1]).abs().max())
    full_difference = float((complete[0] - complete[1]).abs().max())
    if not full_difference > max(.001, 10 * causal_difference):
        raise ValueError("Future scope did not influence the selected anchor as expected")
    return dict(causal_difference=causal_difference, full_difference=full_difference)


def infer(encoder, documents, head, bidirectional, batch_size=BATCH):
    samples = []
    with torch.inference_mode():
        for start in range(0, len(documents), batch_size):
            inputs = tokenize(encoder, documents[start:start + batch_size])
            result = logits(encoder, *inputs, head, bidirectional).cpu().numpy()
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite clause logits")
            samples.append(result)
    return np.concatenate(samples)


def train(encoder, training, arm, hooks):
    torch.manual_seed(SEED)
    head = nn.Linear(encoder.model.config.hidden_size, 1, device="cuda")
    adapters = []
    if arm.endswith("_lora"):
        for layer in encoder.model.layers[-3:]:
            for projection in ("q_proj", "v_proj"):
                adapter = Adapter(getattr(layer.self_attn, projection))
                hooks.enter_context(patch.object(layer.self_attn, projection, adapter))
                adapters.extend((adapter.a, adapter.b))
    params = list(head.parameters()) + adapters
    optimizer = torch.optim.AdamW([dict(params=head.parameters(), lr=.01),
                                   dict(params=adapters, lr=.002)], weight_decay=.01)
    generator = np.random.default_rng(SEED)
    loss_curve = []
    started = perf_counter()
    for _ in range(EPOCHS):
        epoch_loss = []
        order = generator.permutation(len(training))
        for start in range(0, len(order), BATCH):
            documents = [training[i] for i in order[start:start + BATCH]]
            inputs = tokenize(encoder, documents)
            truth = torch.as_tensor(np.concatenate([doc[-1] for doc in documents]),
                                    dtype=torch.float32, device="cuda")
            decision = logits(encoder, *inputs, head, arm.startswith("bidirectional"))
            loss = F.binary_cross_entropy_with_logits(decision, truth, pos_weight=decision.new_tensor(3.))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        loss_curve.append(float(np.mean(epoch_loss)))
    return head, dict(trainable_parameters=sum(x.numel() for x in params),
                      loss=loss_curve, seconds=perf_counter()-started)


def run(args):
    require_cuda(args.output)
    if digest(args.development) != DEVELOPMENT or digest(args.heading) != CONTROLS:
        raise ValueError("Expected the frozen development and heading cohorts")
    scope = make_dataset()
    if scope != json.loads(args.scope.read_bytes()):
        raise ValueError("Changed public scope data")
    mixed = {split: mix(panel) for split, panel in scope.items()}
    heading = single(json.loads(args.heading.read_bytes()))
    dev = rows(args.development)
    if len(dev) != 80 or any(row["split"] != "development" for row in dev):
        raise ValueError("Expected 80 development requests")
    if args.reverse:
        dev.reverse()
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = verify_embedding(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.requires_grad_(False)
    attention_probe = check_right_context(encoder)
    parser = ComparisonParser(args.parser)
    for reader in parser.readers.values():
        reader.broadcast = True
    reference = []
    reference_ms = []
    reader_revision = parser.readers["bf16"].provenance["revision"]
    with torch.inference_mode():
        for row in dev:
            text = row["text"]
            clauses = spans(text)
            truth, elapsed = timed(lambda: evaluate(text, clauses, parser.readers, "consensus")["labels"])
            reference.append(truth == QUALIFICATION)
            reference_ms.append(elapsed)
    del reader, parser
    torch.cuda.empty_cache()
    print("Frozen parser replay complete", flush=True)
    outcomes = {}
    for arm in (ARMS[::-1] if args.reverse else ARMS):
        bidirectional = arm.startswith("bidirectional")
        counts = Counter()
        latency = defaultdict(list)
        with ExitStack() as hooks:
            head, fit = train(encoder, mixed["train"], arm, hooks)
            scores = infer(encoder, mixed["calibration"], head, bidirectional)
            gold = np.concatenate([doc[-1] for doc in mixed["calibration"]])
            cutoff = float(np.abs(scores[(scores > 0) != gold]).max(initial=0.))
            public = {}
            for name, documents in (("mixed_held", mixed["held"]), ("heading", heading)):
                public[name] = public_result(infer(encoder, documents, head, bidirectional), cutoff, documents)
            for row, expected, fallback in zip(dev, reference, reference_ms, strict=True):
                text = row["text"]
                doc = (None, None, text, list(range(len(expected))), [])
                score, elapsed = timed(lambda: infer(encoder, [doc], head, bidirectional, 1))
                labels = score > 0
                accept = bool((np.abs(score) > cutoff).all())
                changed = bool((labels != expected).any())
                counts.update(binary_stats(labels, expected))
                counts.update(requests=1, changed_requests=changed, skipped=accept, changed_skips=accept and changed)
                latency["student"].append(elapsed)
                latency["guard"].append(elapsed + (0 if accept else fallback))
        outcomes[arm] = dict(fit=fit, cutoff=cutoff, public=public, development=counts,
                             latency={key: p50_p95(value) for key, value in latency.items()})
        print(arm, fit["seconds"], flush=True)
    write(args.output, dict(
        arms=outcomes, reference_ms=p50_p95(reference_ms), attention_probe=attention_probe,
        training_documents=len(mixed["train"]), training_anchors=sum(len(doc[-1]) for doc in mixed["train"]),
        inputs={key: digest(getattr(args, key)) for key in ("development", "heading", "scope")},
        script_sha256=digest(Path(__file__)), embedding=checkpoint,
        reader=reader_revision, gpu=torch.cuda.get_device_name(),
        job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "heading", "scope", "embedding", "parser", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
