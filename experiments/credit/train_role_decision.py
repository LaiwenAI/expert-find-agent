"""Train a small local contributor-role decision model on synthetic descriptions."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.compositional_edits import decide, stable
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_language_router import NISO_SHA256, TASK
from experiments.credit.role_prototypes import normalized
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


SEED = 1977
EPOCHS = 16
BATCH = 14
RANK = 8
ADAPTER_LAYERS = 3


class Adapter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(RANK, base.in_features, device="cuda"))
        self.b = nn.Parameter(torch.zeros(base.out_features, RANK, device="cuda"))
        nn.init.xavier_uniform_(self.a)

    def forward(self, x):
        dense = self.base(x)
        residual = F.linear(F.linear(x.float(), self.a), self.b)
        return dense + residual.to(dense.dtype)


def encode(encoder, texts):
    query = [f"Instruct: {TASK}\nQuery: {value}" for value in texts]
    batch = encoder.tokenizer(query, padding=True, truncation=True, max_length=128,
                              return_tensors="pt").to("cuda")
    output = encoder.model(**batch, use_cache=False).last_hidden_state[:, -1, :]
    return F.normalize(output.float(), dim=-1)


def ridge(x, y):
    targets = F.one_hot(y, num_classes=len(ROLES)).float()
    return targets.T @ torch.linalg.solve(x @ x.T + .1 * torch.eye(len(x), device=x.device), x)


def head_from(weights):
    layer = nn.Linear(weights.shape[1], len(ROLES), device=weights.device)
    with torch.no_grad():
        # The positive scale preserves initial rankings and supplies useful gradients.
        layer.weight.copy_(10. * weights)
        layer.bias.zero_()
    return layer


def train(encoder, base, fit, validation, method, seed):
    torch.manual_seed(seed)
    model = encoder.model
    model.requires_grad_(False)
    head = head_from(base)
    adapters = {}
    if method == "lora":
        for index in range(len(model.layers) - ADAPTER_LAYERS, len(model.layers)):
            block = model.layers[index].self_attn
            for name in ("q_proj", "v_proj"):
                wrapper = Adapter(getattr(block, name))
                setattr(block, name, wrapper)
                adapters[f"{index}.{name}"] = wrapper
    params = dict(head_weight=head.weight, head_bias=head.bias)
    for name, adapter in adapters.items():
        params[f"{name}.a"] = adapter.a
        params[f"{name}.b"] = adapter.b
    initial = {name: tensor.detach().cpu().clone() for name, tensor in params.items()}
    head_params = list(head.parameters())
    adapter_params = [param for adapter in adapters.values() for param in (adapter.a, adapter.b)]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": .006},
        {"params": adapter_params, "lr": .001},
    ], weight_decay=.01)
    fit_text, fit_y = fit
    valid_text, valid_y = validation
    generator = np.random.default_rng(seed)
    best = None
    history = []
    start = perf_counter()
    for epoch in range(EPOCHS):
        losses = []
        for offset in range(0, len(fit_text), BATCH):
            if offset == 0:
                order = generator.permutation(len(fit_text))
            selected = order[offset:offset + BATCH]
            vectors = encode(encoder, [fit_text[i] for i in selected])
            loss = F.cross_entropy(head(vectors), fit_y[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            logits = head(encode(encoder, valid_text))
            accuracy = int((logits.argmax(dim=1) == valid_y).sum())
            dev_loss = float(F.cross_entropy(logits, valid_y))
        history.append(dict(epoch=epoch+1, valid_correct=accuracy, valid_loss=dev_loss,
                            train_loss=mean(losses)))
        ranking = (accuracy, -dev_loss)
        if best is None or ranking > best[0]:
            chosen = {name: value.detach().cpu().clone() for name, value in params.items()}
            best = ranking, epoch+1, chosen
    with torch.no_grad():
        for name, tensor in params.items():
            tensor.copy_(best[2][name])
    return head, best[2], dict(
        method=method, selected_epoch=best[1], validation_correct=best[0][0],
        trainable=sum(t.numel() for t in params.values()),
        changed=sum(int((t != initial[name]).sum()) for name, t in best[2].items()),
        adapter_b_nonzero=sum(int(t.count_nonzero()) for name, t in best[2].items() if name.endswith(".b")),
        history=history, training_seconds=perf_counter()-start,
    )


def evaluate(encoder, predictor, glossary, fixture, sources):
    examples = glossary["examples"]
    with torch.inference_mode():
        raw = predictor(encode(encoder, [row["text"] for row in examples]))
        roles = [ROLES[i] for i in raw.argmax(dim=1).tolist()]
    role_correct = sum(a == row["role"] for a, row in zip(roles, examples, strict=True))
    phrases = defaultdict(list)
    for row in examples:
        phrases[row["role"]].append(row["text"])
    out = []
    with torch.inference_mode():
        for i, row in enumerate(fixture):
            tasks = [min(phrases[role], key=lambda x: stable(row["group"], role, x))
                     for _, role in row["requirements"]]
            start = perf_counter()
            mapped = [ROLES[j] for j in predictor(encode(encoder, tasks)).argmax(dim=1).tolist()]
            requirements = [[slot, role] for (slot, _), role in zip(row["requirements"], mapped, strict=True)]
            status, _, _ = decide(sources, dict(row, requirements=requirements), "author_scoped")
            elapsed = (perf_counter()-start)*1000
            out.append(dict(label=status, roles=mapped, ms=elapsed))
    return dict(
        atomic_correct=role_correct, atomic_total=len(examples), atomic_roles=roles,
        correct=sum(a["label"] == row["reference"] for a, row in zip(out, fixture, strict=True)),
        supported_recovered=sum(a["label"] == row["reference"] == "supported"
                                for a, row in zip(out, fixture, strict=True)),
        unsupported=sum(a["label"] == "supported" and row["reference"] != "supported"
                        for a, row in zip(out, fixture, strict=True)),
        p95_ms=float(np.quantile([a["ms"] for a in out], .95)),
        rows=out,
    )


def inputs(args):
    if digest(args.examples) != NISO_SHA256:
        raise ValueError("Changed NISO examples")
    glossary = json.loads(args.examples.read_bytes())
    descriptions = json.loads(args.training.read_bytes())
    if set(descriptions) != set(ROLES) or any(len(value) != 6 for value in descriptions.values()):
        raise ValueError("Expected the pinned 14-role synthetic training inventory")
    values = {normalized(x) for variants in descriptions.values() for x in variants}
    if len(values) != 84 or values.intersection(normalized(row["text"]) for row in glossary["examples"]):
        raise ValueError("Duplicate or held-out example in training")
    fit, valid = [], []
    split_key = ("fine-tune-split",) if args.seed == SEED else ("fine-tune-split", args.seed)
    for index, role in enumerate(ROLES):
        items = sorted(descriptions[role], key=lambda q: stable(*split_key, role, q))
        fit.extend((x, index) for x in items[:-1])
        valid.append((items[-1], index))
    manifest = json.loads((args.fixture / "manifest.json").read_bytes())
    if any(digest(args.fixture / n) != s for n, s in manifest["files"].items()):
        raise ValueError("Changed compositional fixture")
    fixture = [row for row in json.loads((args.fixture / "queries.json").read_bytes()) if row["split"] == "test"]
    sources = json.loads((args.fixture / "sources.json").read_bytes())
    return fit, valid, glossary, fixture, sources


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(args.embedding)
    fit, valid, glossary, fixture, sources = inputs(args)
    encoder = QwenTorchEmbedding(args.embedding)
    args.output.mkdir()
    def tensors(rows):
        text, labels = zip(*rows, strict=True)
        return list(text), torch.tensor(labels, device="cuda")
    fit, valid = tensors(fit), tensors(valid)
    with torch.no_grad():
        train_x = encode(encoder, fit[0])
        val_x = encode(encoder, valid[0])
        base = ridge(train_x, fit[1])
        full = ridge(torch.cat((train_x, val_x)), torch.cat((fit[1], valid[1])))
    results = {
        "ridge_train70": evaluate(encoder, lambda x: x @ base.T, glossary, fixture, sources),
        "ridge_all84": evaluate(encoder, lambda x: x @ full.T, glossary, fixture, sources),
    }
    metadata = {}
    for method in ("head_only", "lora"):
        head, weights, meta = train(encoder, base, fit, valid, method, args.seed)
        target = args.output / f"{method}.safetensors"
        save_file(weights, str(target))
        reloaded = load_file(str(target))
        if any(not torch.equal(weights[name], reloaded[name]) for name in weights):
            raise ValueError("Checkpoint round-trip changed parameter values")
        meta["checkpoint_sha256"] = digest(target)
        metadata[method] = meta
        results[method] = evaluate(encoder, head, glossary, fixture, sources)
        print(method, "selected epoch", meta["selected_epoch"], flush=True)
    write_json(args.output / "results.json", dict(
        job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        seed=args.seed, script_sha256=digest(Path(__file__)), training_sha256=digest(args.training),
        fixture_sha256=digest(args.fixture / "manifest.json"), examples_sha256=NISO_SHA256,
        embedding=provenance, metadata=metadata, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "examples", "training", "embedding", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    run(parser.parse_args())
