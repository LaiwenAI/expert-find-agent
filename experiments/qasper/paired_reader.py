"""Measure the effect of shorter routed context on a fixed local QASPER reader."""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from expert_search.statistics import clustered_paired_interval
from experiments.qasper.adaptive_evidence import EVALUATOR_SHA256, digest, paper_inputs, read_data


SAMPLE_SIZE = 72
SAMPLE_SALT = "qasper-context-reader-v1"
MAX_NEW_TOKENS = 48
ARMS = ("fixed5", "learned")
INSTRUCTION = (
    "Answer the question using only the passages. Give a short answer without explanation. "
    "For a yes/no question, use Yes or No. If the passages cannot answer it, reply Unanswerable."
)


def sample(dataset):
    def order(identity):
        return sha256(f"{SAMPLE_SALT}:{identity}".encode()).digest()

    pids = sorted(dataset, key=order)[:SAMPLE_SIZE]
    return [(pid, min(dataset[pid]["qas"], key=lambda qa: order(qa["question_id"]))) for pid in pids]


def measure(tokenizer, reader, question, passages):
    started = perf_counter()
    contents = "\n\n".join(f"Passage {i + 1}:\n{text}" for i, text in enumerate(passages))
    messages = [{"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": f"{contents}\n\nQuestion: {question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]
    if prompt_tokens + MAX_NEW_TOKENS > reader.config.max_position_embeddings:
        raise ValueError("Reader context overflow")
    with torch.inference_mode():
        ids = reader.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                              pad_token_id=tokenizer.pad_token_id)[0, prompt_tokens:]
    output_tokens = int(ids.numel())
    reply = tokenizer.decode(ids.cpu(), skip_special_tokens=True).strip()
    return dict(answer=reply, seconds=perf_counter() - started,
                input_tokens=prompt_tokens, generation_tokens=output_tokens)


def load_reader(path):
    provenance = json.loads((path / "local-provenance.json").read_bytes())
    if provenance["repo"] != "Qwen/Qwen3-4B-Instruct-2507" or provenance["revision"] != "cdbee75f17c01a7cc42f958dc650907174af0554":
        raise ValueError("Unexpected reader checkpoint")
    for name, checksum in provenance["files"].items():
        if digest(path / name) != checksum:
            raise ValueError(f"Changed checkpoint file: {name}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    reader = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    reader.generation_config.temperature = None
    reader.generation_config.top_p = None
    reader.generation_config.top_k = None
    return tokenizer, reader, provenance


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.retrieval / "test_rows.json"
    retrieval = json.loads((args.retrieval / "summary.json").read_bytes())
    if digest(source) != retrieval["test_rows_sha256"]:
        raise ValueError("Changed retrieval manifest")
    by_question = {row["question_id"]: row for row in json.loads(source.read_bytes())}
    dataset = read_data(args.data, "test")
    evaluator_file = args.data / "qasper_evaluator.py"
    if digest(evaluator_file) != EVALUATOR_SHA256:
        raise ValueError("Changed evaluator")
    spec = importlib.util.spec_from_file_location("qasper_evaluator", evaluator_file)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    gold = evaluator.get_answers_and_evidence(dataset, True)
    tokenizer, reader, provenance = load_reader(args.model)
    measure(tokenizer, reader, "What word appears in the passage?", ["Example."])

    rows = []
    for index, (pid, qa) in enumerate(sample(dataset)):
        qid = qa["question_id"]
        selection = by_question[qid]
        if selection["paper_id"] != pid:
            raise ValueError("Mismatched source document")
        paragraphs, _ = paper_inputs(dataset[pid])
        row = dict(paper_id=pid, question_id=qid, arms={})
        ordered = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
        for arm in ordered:
            output = measure(tokenizer, reader, qa["question"],
                             [paragraphs[i] for i in selection["arms"][arm]["paragraph_indices"]])
            output["answer_f1"] = max(evaluator.token_f1_score(output["answer"], ref["answer"])
                                      for ref in gold[qid])
            row["arms"][arm] = output
        rows.append(row)
        if len(rows) % 12 == 0:
            print("Completed pairs", len(rows), flush=True)
    metrics = {}
    for arm in ARMS:
        outputs = [row["arms"][arm] for row in rows]
        metrics[arm] = dict(
            mean_answer_f1=mean(out["answer_f1"] for out in outputs),
            mean_input_tokens=mean(out["input_tokens"] for out in outputs),
            mean_generation_tokens=mean(out["generation_tokens"] for out in outputs),
            generation_cap_hits=sum(out["generation_tokens"] == MAX_NEW_TOKENS for out in outputs),
            mean_seconds=mean(out["seconds"] for out in outputs),
            p50_p95_seconds=list(map(float, np.quantile([o["seconds"] for o in outputs], [.5, .95]))),
        )
    groups = [row["paper_id"] for row in rows]
    paired = {field: clustered_paired_interval([
        row["arms"]["learned"][field] - row["arms"]["fixed5"][field] for row in rows
    ], groups) for field in ("answer_f1", "seconds")}
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "rows.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    summary = dict(provenance=provenance, questions=len(rows), sample_salt=SAMPLE_SALT,
                   max_new_tokens=MAX_NEW_TOKENS, gpu=torch.cuda.get_device_name(), metrics=metrics,
                   learned_minus_fixed5=paired, retrieval_rows_sha256=digest(source), rows_sha256=digest(output))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-4b-instruct-2507"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
