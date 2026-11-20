"""Replay the published MERIT retriever on its matched LR-Bench subsets."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.lrbench_frozen_baselines import ASSET_HASHES, text as raw_text

RELEASE = "5fad0ea06148c3c8817aac841de46c98a75eb40c"
BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
ADAPTER_REVISION = "6ddce7e8fdfed417d9fd022124529a421560a3f0"
WEIGHTS = {
    "model-00001-of-00004.safetensors": "99b343597fe840706146144699a8b9188dd3387e43eb61faf0231b70b249d451",
    "model-00002-of-00004.safetensors": "dff635b0f6dbbaad2a2d633ef037ec0a39bc165cc1806c712fbd6fcbcb4526c0",
    "model-00003-of-00004.safetensors": "30b1d4c53d84eb018f642cad7b373f0aabf79699872d8702c1f38577c0a59a2f",
    "model-00004-of-00004.safetensors": "36cbc9c60375693629f25743c1e77ebb1724af58e671b2376463193c7fd21ef6",
}
ADAPTER_SHA = "9d14bf0e74609d1b3d257e7eb2b2b0627c2efc8d96bd4b9c0d72a41318451aca"
DATA_SHA = {
    "pc": "3eee77d4f387fbadd7bb54dc3792a9bbad16808d13f755ef9e6840459a5f90ab",
    "rc": "a478b1eed63734b9d1862fbee41bd4d328af4e72b943296508fea490f045d70e",
}
RENDER_SHA = {
    "prepare_topk_LR.py": "afeb4b05ea6bd055e5328484891d374cace0af4d930e5e6ed949202f31dc0cf1",
    "eval_LR.py": "34097ccb517e6b0a49218e713bfa9b335a9a7ffa972f2cdf6c446be354ef000f",
}
KEYS = ("anchor_id", "positive_id", "negative_id", "type", "positive_ref_score", "negative_ref_score")
COMPACT_TASK = "Retrieve an academic publication relevant to the target paper."


def formatting(root):
    exports = []
    for name, functions, constants in (
        ("prepare_topk_LR.py", {"get_paper_text", "build_prefilter_paper_text"}, {"PAPER_INSTRUCTION"}),
        ("eval_LR.py", {"get_paper_text", "get_qwen3_query_prompt", "build_profile_text",
                        "build_profile_doc_text"}, {"DEFAULT_TASK_PROMPT", "DEFAULT_TASK_PROMPT_RC"}),
    ):
        path = root / name
        if digest(path) != RENDER_SHA[name]:
            raise ValueError("Unexpected author text-rendering file")
        scope = {}
        body = []
        for node in ast.parse(path.read_bytes()).body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                body.append(node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                tag = node.targets[0]
                if isinstance(tag, ast.Name) and tag.id in constants:
                    scope[tag.id] = ast.literal_eval(node.value)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
        exports.append(scope)
    return exports


def load_pairs(root, rate, split, cap):
    source = root / f"evaluations_{split}.json"
    predictions = rate / f"RATE_{split}.json"
    if digest(source) != DATA_SHA[split] or digest(predictions) != ASSET_HASHES[predictions.name]:
        raise ValueError("Changed MERIT or RATE input")
    original = json.loads(predictions.read_bytes())
    replay = {tuple(row[key] for key in KEYS): row for row in original}
    if len(replay) != len(original):
        raise ValueError("RATE comparisons are not unique")
    cases = []
    for row in json.loads(source.read_bytes())[:cap]:
        primary, other = ("paper_id", "author_id") if split == "pc" else ("author_id", "paper_id")
        anchor, positive, negative = (row[name] for name in ("anchor", "positive", "negative"))
        key = (anchor[primary], positive[other], negative[other], row["type"], positive["score"], negative["score"])
        if positive["score"] <= negative["score"] or key not in replay:
            raise ValueError("The published comparison does not align with the RATE release")
        items = ((anchor, positive), (anchor, negative)) if split == "pc" else ((positive, anchor), (negative, anchor))
        cases.append((key[0], positive["score"] - negative["score"], items, replay[key]))
    return cases


@torch.inference_mode()
def embed_8b(model, tokenizer, texts, batch_size, max_length, *, original_pool):
    encoded = {}
    started = perf_counter()
    ordered = sorted(dict.fromkeys(texts), key=lambda t: (len(t), sha256(t.encode()).digest()))
    padded = 0
    truncated = 0
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        lengths = tokenizer(batch, truncation=False, return_length=True)["length"]
        truncated += sum(n > max_length for n in lengths)
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        positions = inputs.attention_mask.sum(dim=1) - 1
        hidden = model(**inputs, use_cache=False).last_hidden_state[torch.arange(len(batch), device="cuda"), positions]
        values = torch.nn.functional.normalize(hidden if original_pool else hidden.float(), dim=1).float().cpu().numpy()
        for t, value in zip(batch, values, strict=True):
            encoded[t] = value
        padded += inputs.input_ids.numel()
    return encoded, dict(texts=len(encoded), batch_size=batch_size, seconds=perf_counter()-started,
                         padded_tokens=padded, truncated_texts=truncated)


def render_final(paper, profile, split, template):
    query = template["get_qwen3_query_prompt"]
    paper_text = template["get_paper_text"](paper)
    if split == "pc":
        return query(template["DEFAULT_TASK_PROMPT"], paper_text), template["build_profile_doc_text"](profile)
    return query(template["DEFAULT_TASK_PROMPT_RC"], template["build_profile_text"](profile)), paper_text


def select_profile(paper, papers, embeddings, render):
    if len(papers) <= 3:
        return papers
    target = embeddings[render(paper)]
    return sorted(papers, key=lambda p: -float(target @ embeddings[render(p)]))[:3]


def paired_metrics(cases, margins):
    groups = [case[0] for case in cases]
    gaps = np.asarray([case[1] for case in cases])
    ref = margins["rate"] > 0
    out = {}
    for arm, predicted in margins.items():
        good = predicted > 0
        errors = (predicted < 0).astype(float) + .5 * (predicted == 0)
        out[arm] = dict(accuracy=float(good.mean()), weighted_kendall_loss=float(np.dot(errors, gaps) / gaps.sum()),
                        correct=int(good.sum()), ties=int((predicted == 0).sum()),
                        paired_accuracy_vs_rate=clustered_paired_interval(good.astype(int) - ref.astype(int), groups))
    return out


def run(args):
    from peft import PeftModel

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Use a fresh output on Slurm CUDA")
    if args.batch_size < 1 or (args.pilot is not None and args.pilot < 1):
        raise ValueError("Invalid batch size or pilot limit")
    for name, known in WEIGHTS.items():
        if digest(args.backbone / name) != known:
            raise ValueError("Unexpected backbone weights")
    if digest(args.adapter / "adapter_model.safetensors") != ADAPTER_SHA:
        raise ValueError("Unexpected MERIT adapter")
    prefilter, template = formatting(args.upstream)
    cases = {split: load_pairs(args.data, args.rate, split, args.pilot) for split in DATA_SHA}
    comparisons = [pair for rows in cases.values() for row in rows for pair in row[2]]
    torch.backends.cuda.matmul.allow_tf32 = False
    compact = QwenTorchEmbedding(args.compact)
    compact_revision = verify_checkpoint(args.compact)["revision"]
    targets = [raw_text(paper) for paper, _ in comparisons]
    sources = [raw_text(p) for _, person in comparisons for p in person["papers"]]
    started = perf_counter()
    compact_inputs = list(dict.fromkeys(sources + [f"Instruct: {COMPACT_TASK}\nQuery: {t}" for t in targets]))
    compact_bank = dict(zip(compact_inputs, compact.encode(compact_inputs), strict=True))
    compact_time = perf_counter() - started
    del compact
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone, padding_side="right", local_files_only=True, trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        args.backbone, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
    ).to("cuda").eval()
    render = prefilter["build_prefilter_paper_text"]
    pretexts = [render(p) for paper, person in comparisons for p in (paper, *person["papers"])
                if len(person["papers"]) > 3]
    bank, pre_cost = embed_8b(model, tokenizer, pretexts, args.batch_size, 2048, original_pool=True)
    print(f"Prepared author prefilter for {pre_cost['texts']} unique texts.", flush=True)

    alternatives = {}
    cheap_pair_scores = {}
    for split, rows in cases.items():
        alternatives[split] = {"author": [], "compact_prefilter": [], "first_three": []}
        cheap_pair_scores[split] = []
        for _, _, pair, _ in rows:
            scores = []
            pairs = {name: [] for name in alternatives[split]}
            for paper, person in pair:
                profile = person["papers"]
                query = compact_bank[f"Instruct: {COMPACT_TASK}\nQuery: {raw_text(paper)}"]
                scored = [(float(query @ compact_bank[raw_text(p)]), p) for p in profile]
                scored.sort(key=lambda entry: -entry[0])
                if not scored:
                    raise ValueError("Released reviewer profile is empty")
                scores.append(float(np.mean([score for score, _ in scored[:3]])))
                views = {
                    "author": select_profile(paper, profile, bank, render),
                    "compact_prefilter": [p for _, p in scored[:3]] if len(profile) > 3 else profile,
                    "first_three": profile[:3],
                }
                for name, selected in views.items():
                    pairs[name].append(render_final(paper, selected, split, template))
            cheap_pair_scores[split].append(scores[0] - scores[1])
            for name, views in pairs.items():
                alternatives[split][name].append(views)

    def final_scores(encode_model, names):
        scores, costs = {}, {}
        for name in names:
            texts = [text for by_arm in alternatives.values() for pair in by_arm[name]
                     for comparison in pair for text in comparison]
            final, costs[name] = embed_8b(encode_model, tokenizer, texts, args.batch_size, 4096, original_pool=False)
            scores[name] = {
                split: np.asarray([float(final[q1] @ final[d1] - final[q2] @ final[d2])
                                   for (q1, d1), (q2, d2) in by_arm[name]])
                for split, by_arm in alternatives.items()
            }
            print(f"Finished {name}: {costs[name]['texts']} distinct final texts.", flush=True)
        return scores, costs

    base, base_times = final_scores(model, ("author",))
    adapted = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    adapter_scores, adapter_times = final_scores(adapted, ("author", "compact_prefilter", "first_three"))
    results = {}
    for split, rows in cases.items():
        margins = {
            "rate": np.asarray([row[3]["positive_score"] - row[3]["negative_score"] for row in rows]),
            "compact_top3": np.asarray(cheap_pair_scores[split]),
            "base_8b_same_profile": base["author"][split],
            **{name: by_split[split] for name, by_split in adapter_scores.items()},
        }
        results[split] = dict(pairs=len(rows), anchors=len({row[0] for row in rows}),
                              metrics=paired_metrics(rows, margins))
    write_json(args.output, dict(
        model_revisions={"backbone": BASE_REVISION, "adapter": ADAPTER_REVISION, "compact": compact_revision},
        release=RELEASE, data_sha256=DATA_SHA, rendering_sha256=RENDER_SHA, results=results,
        component_timing={"compact": dict(texts=len(compact_inputs), seconds=compact_time),
                          "author_prefilter": pre_cost, "backbone_final": base_times, "adapter": adapter_times},
        gpu=torch.cuda.get_device_name(), pilot=args.pilot, script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "upstream", "rate", "backbone", "adapter", "compact", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pilot", type=int)
    run(parser.parse_args())
