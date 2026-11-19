"""Train and evaluate a compact support verifier on claim-matched examples."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.vitaminc_label_transfer import LABELS, panel, stable
from experiments.scifact.vitaminc_selective_support import DEV_SHA256, DATA_SHA256, cohorts

TRAIN_SHA256 = "7461c6fd1a13459590317c5ccdc8651dd2daf7c1ad8ae4b10ccd88d164fccd5a"
SEED = 20261119
TRAIN_PAIRS = 1200
TRAIN_BATCH = 32
EPOCHS = 2
ARMS = ("frozen", "classification", "contrastive")


def fingerprint(pairs):
    return stable(*(r["unique_id"] for pair in pairs for r in pair)).hex()


def prepare(train_path, dev_path, test_path, output):
    hashes = [digest(path) for path in (train_path, dev_path, test_path)]
    if hashes != [TRAIN_SHA256, DEV_SHA256, DATA_SHA256]:
        raise ValueError("Changed official source files")
    development, test, old_pages, old_claims = cohorts(dev_path, test_path)
    held_pages = old_pages | {a["page"] for a, _ in development + test}
    held_claims = old_claims | {a["claim"].casefold() for a, _ in development + test}
    train = panel(train_path, seed="vitaminc-contrastive-minilm-train-v1", per_control=TRAIN_PAIRS//2,
                  excluded_pages=held_pages, excluded_claims=held_claims)
    data = {name: {"hash": fingerprint(pairs), "pairs": [
        [{k: r[k] for k in ("page", "claim", "evidence", "label", "unique_id")} for r in pair]
        for pair in pairs]} for name, pairs in (("train", train), ("development", development), ("test", test))}
    write_json(output, dict(source_hashes=hashes, data=data))


def inputs(tokenizer, pairs):
    steps = [row for pair in pairs for row in pair]
    tokens = tokenizer([r["evidence"] for r in steps], [r["claim"] for r in steps],
                       add_special_tokens=True, padding=False, truncation=False, verbose=False)
    if max(map(len, tokens["input_ids"])) > 512:
        raise ValueError("A source pair exceeds the shared context limit")
    labels = np.asarray([LABELS.index(row["label"]) for row in steps])
    return {name: data for name, data in tokens.items()}, labels


def support_margin(logits):
    return logits[:, 0] - logits[:, 1:].max(dim=1).values


def upper_tail_loss(logits, labels):
    from torch.nn import functional as F

    margins = support_margin(logits)
    supports = margins[labels == 0]
    comparisons = []
    for label in (1, 2):
        negatives = margins[labels == label]
        if negatives.numel():
            # Pool the riskiest quarter within each failure class. The loss
            # equally weights refutation and insufficient-evidence comparisons.
            hard_count = max(1, (negatives.numel() + 3) // 4)
            hard = negatives.topk(hard_count).values
            comparisons.append(F.softplus(1. + hard[:, None] - supports[None, :]).mean())
    return sum(comparisons) / len(comparisons)


def fit(tokenizer, model, positions, pairs, arm, weights, *, seed=SEED, artifact_name=None):
    import torch
    from safetensors.torch import save_file
    from torch.nn import functional as F

    torch.manual_seed(seed)
    last = str(model.config.num_hidden_layers-1)
    prefixes = ("classifier.", f"roberta.encoder.layer.{last}.")
    selected = {}
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(prefixes))
        if param.requires_grad:
            selected[name] = param
    if not selected or not any(name.startswith(prefixes[1]) for name in selected):
        raise ValueError("Expected classifier and final transformer block")
    initial = {name: value.detach().cpu().clone() for name, value in selected.items()}
    tokens, labels = inputs(tokenizer, pairs)
    model.float().train()
    optimizer = torch.optim.AdamW(selected.values(), lr=3e-5, weight_decay=.01)
    start = perf_counter()
    random = np.random.default_rng(seed)
    loss_by_epoch = []
    updates = 0
    for epoch in range(EPOCHS):
        order = random.permutation(len(pairs))
        losses = []
        for i in range(0, len(order), TRAIN_BATCH//2):
            pair_ids = order[i:i+TRAIN_BATCH//2]
            idx = np.column_stack((2*pair_ids, 2*pair_ids+1)).ravel().tolist()
            batch = tokenizer.pad([{k: v[j] for k,v in tokens.items()} for j in idx], return_tensors="pt").to("cuda")
            target = torch.tensor(labels[idx], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits[:, positions].float()
                loss = F.cross_entropy(logits, target)
                if arm == "contrastive":
                    margin = support_margin(logits).reshape(-1,2)
                    loss = loss + .4*F.softplus(margin[:, 1] - margin[:, 0]).mean()
                elif arm == "risk_tail":
                    loss = loss + .4*upper_tail_loss(logits, target)
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selected.values(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            updates += 1
        loss_by_epoch.append(float(np.mean(losses)))
        print(arm, f"epoch {epoch+1}", loss_by_epoch[-1], flush=True)
    seconds = perf_counter() - start
    model.bfloat16().eval()
    changed = sum(int(torch.count_nonzero(param.detach().cpu() != initial[name])) for name,param in selected.items())
    if not changed:
        raise ValueError("Training produced no changed parameters")
    path = weights/f"{artifact_name or arm}.safetensors"
    if path.exists():
        raise FileExistsError(path)
    save_file({name:param.detach().cpu().contiguous() for name,param in selected.items()}, str(path))
    return dict(trainable=sum(p.numel() for p in selected.values()), changed=changed, updates=updates,
                seconds=seconds, losses=loss_by_epoch, seed=seed, weight_sha256=digest(path))


def evaluate(tokenizer, model, positions, pairs, batch_size):
    import torch

    rows = [r for pair in pairs for r in pair]
    margins, predictions, times = [], [], []
    for begin in range(0,len(rows),batch_size):
        start = perf_counter()
        chosen = rows[begin:begin+batch_size]
        encoded = tokenizer([r["evidence"] for r in chosen], [r["claim"] for r in chosen], padding=True,
                            truncation=False, return_tensors="pt").to("cuda")
        if encoded["input_ids"].shape[1] > 512:
            raise ValueError("A source pair exceeds the shared context limit")
        with torch.inference_mode():
            logits = model(**encoded).logits[:, positions].float()
        values = support_margin(logits).cpu().tolist()
        prediction = logits.argmax(dim=1).cpu().tolist()
        times.append((perf_counter()-start)*1000)
        margins.extend(values)
        predictions.extend(prediction)
    return np.asarray(margins), np.asarray(predictions), times


def cutoff(margins, targets, caps):
    return max(sorted(margins[targets==label], reverse=True)[cap] for label,cap in zip((1,2),caps,strict=True))


def acceptance(mask, target):
    return [int(np.sum(mask & (target == label))) for label in range(3)]


def summarize(observations, pairs, caps):
    labels = {name:np.asarray([LABELS.index(r["label"]) for pair in pairs[name] for r in pair])
              for name in ("development","test")}
    groups = [a["page"] for a,_ in pairs["test"]]
    report = {}
    for arm, value in observations.items():
        dev, test, batched = value["development"], value["test"], value["batched"]
        report[arm] = dict(training=value["training"],
                           item_accuracy=float(np.mean(test[1]==labels["test"])),
                           single_p95_ms=float(np.quantile(test[2],.95)),
                           four_item_batch_p95_ms=float(np.quantile(batched[2],.95)),
                           raw_accepted=acceptance(test[0]>=0,labels["test"]), operating_points={})
        for name, cap in caps.items():
            threshold = cutoff(dev[0], labels["development"], cap)
            dev_mask, test_mask = dev[0]>threshold, test[0]>threshold
            control_threshold = cutoff(observations["frozen"]["development"][0], labels["development"], cap)
            control = observations["frozen"]["test"][0] > control_threshold
            differences = (test_mask.astype(int)-control.astype(int)).reshape(-1,2)
            report[arm]["operating_points"][name] = dict(
                threshold=threshold, development_accepted=acceptance(dev_mask,labels["development"]),
                test_accepted=acceptance(test_mask,labels["test"]),
                batched_threshold_changes=int(np.sum(test_mask != (batched[0]>threshold))),
                support_delta_vs_frozen=clustered_paired_interval(differences[:,0],groups)
                    | {"sampling_unit":"one page with a matched evidence pair"},
                refute_delta_vs_frozen=clustered_paired_interval(differences[labels["test"][1::2]==1,1],groups[:len(groups)//2])
                    | {"sampling_unit":"one page with a matched evidence pair"},
                insufficient_delta_vs_frozen=clustered_paired_interval(differences[labels["test"][1::2]==2,1],groups[len(groups)//2:])
                    | {"sampling_unit":"one page with a matched evidence pair"},
            )
    return report


def run(projection, control, checkpoint, output, weights, *, compare_tail=False):
    import torch
    from experiments.scifact.small_nli_verifier import load

    if output.exists() or weights.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh outputs and Slurm CUDA")
    data = json.loads(projection.read_text())
    if data["source_hashes"] != [TRAIN_SHA256, DEV_SHA256, DATA_SHA256]:
        raise ValueError("Changed public inputs")
    pairs = {name: payload["pairs"] for name,payload in data["data"].items()}
    for name, q in pairs.items():
        if fingerprint(q) != data["data"][name]["hash"]:
            raise ValueError("Changed contrastive pair order")
    control_data = json.loads(control.read_text())
    for name in ("development","test"):
        prior_name = "dev" if name == "development" else "test"
        if control_data[f"{prior_name}_panel_sha256"] != fingerprint(pairs[name]):
            raise ValueError("Different calibration or test cohort")
    pages = [a["page"] for q in pairs.values() for a,_ in q]
    claims = [a["claim"].casefold() for q in pairs.values() for a,_ in q]
    if len(set(pages)) != len(pages) or len(set(claims)) != len(claims):
        raise ValueError("Cohort overlap")
    weights.mkdir()
    observations = {}
    arms = [(arm, arm, SEED) for arm in ARMS]
    if compare_tail:
        arms = [("frozen", "frozen", SEED)] + [
            (f"{arm}_{seed}", arm, seed)
            for seed in (SEED, SEED + 1) for arm in (*ARMS[1:], "risk_tail")
        ]
    for name, arm, seed in arms:
        tokenizer, model, positions, provenance = load(checkpoint,"minilm")
        training = {} if arm == "frozen" else fit(
            tokenizer, model, positions, pairs["train"], arm, weights, seed=seed, artifact_name=name)
        evaluate(tokenizer,model,positions,pairs["development"][:2],1)
        observations[name] = dict(training=training,
                                 development=evaluate(tokenizer,model,positions,pairs["development"],1),
                                 test=evaluate(tokenizer,model,positions,pairs["test"],1),
                                 batched=evaluate(tokenizer,model,positions,pairs["test"],4))
        del model
    caps = dict(matched=control_data["dev_false_support_caps"], strict=[10,10])
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), dataset_hashes=data["source_hashes"],
        script_sha256=digest(Path(__file__)), projection_sha256=digest(projection), control_sha256=digest(control),
        model_revision=provenance["revision"], caps=caps,
        panel_hashes={name:fingerprint(q) for name,q in pairs.items()}, pair_counts={name:len(q) for name,q in pairs.items()},
        report=summarize(observations,pairs,caps),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    for name, action, fields in (
        ("prepare",prepare,("train_path","dev_path","test_path","output")),
        ("run",run,("projection","control","checkpoint","output","weights")),
    ):
        command=sub.add_parser(name)
        for field in fields:
            command.add_argument(field,type=Path)
        command.set_defaults(action=action)
        if name == "run":
            command.add_argument("--compare-tail", action="store_true")
    args=vars(parser.parse_args())
    args.pop("action")(**args)
