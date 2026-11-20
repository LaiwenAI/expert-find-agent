"""Calibrate a frozen-embedding qualification student on synthetic scope contrasts."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import enterprise_shared_qualification as search
from experiments.retrieval.enterprise_cross_precision import CONTROLS, QUALIFICATION
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
from experiments.retrieval.enterprise_heading_roles import scoped_context
from experiments.retrieval.enterprise_isolated_parsing import ROLES, spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_routed_review import digest, rows, write


TOPICS = (
    ("Acoustic inspection on welded joints.", "\u710a\u63a5\u63a5\u5934\u58f0\u5b66\u68c0\u6d4b\u3002"),
    ("Verification of polymer seal fatigue.", "\u805a\u5408\u7269\u5bc6\u5c01\u4ef6\u75b2\u52b3\u9a8c\u8bc1\u3002"),
    ("Calibration of thermal cameras at low temperatures.", "\u4f4e\u6e29\u7ea2\u5916\u76f8\u673a\u6821\u51c6\u3002"),
    ("Testing wastewater filtration membranes.", "\u5e9f\u6c34\u8fc7\u6ee4\u819c\u6d4b\u8bd5\u3002"),
    ("Commissioning a robotic dispensing station.", "\u673a\u5668\u4eba\u70b9\u80f6\u7ad9\u8c03\u8bd5\u3002"),
    ("Reliability testing of electrical connectors.", "\u7535\u6c14\u8fde\u63a5\u5668\u53ef\u9760\u6027\u6d4b\u8bd5\u3002"),
    ("Monitoring vibration in rotating assemblies.", "\u65cb\u8f6c\u7ec4\u4ef6\u632f\u52a8\u76d1\u6d4b\u3002"),
    ("Parameter estimation for flow meters.", "\u6d41\u91cf\u8ba1\u53c2\u6570\u4f30\u8ba1\u3002"),
    ("Validation of corrosion monitoring methods.", "\u8150\u8680\u76d1\u6d4b\u65b9\u6cd5\u9a8c\u8bc1\u3002"),
    ("Evaluation of optical coating durability.", "\u5149\u5b66\u6d82\u5c42\u8010\u4e45\u6027\u8bc4\u4f30\u3002"),
    ("Inspection of ceramic components.", "\u9676\u74f7\u96f6\u4ef6\u68c0\u9a8c\u3002"),
    ("Noise characterization of pump assemblies.", "\u6cf5\u7ec4\u4ef6\u566a\u58f0\u8868\u5f81\u3002"),
)
SCOPES = (
    (True, "Completed work required of the applicant:", "\u7533\u8bf7\u4eba\u987b\u5177\u5907\u7684\u65e2\u5f80\u5de5\u4f5c\uff1a"),
    (False, "Deliverables the project team will complete:", "\u9879\u76ee\u56e2\u961f\u9700\u5b8c\u6210\u7684\u4ea4\u4ed8\u5de5\u4f5c\uff1a"),
    (True, "Submit evidence of work the applicant has already completed.", "\u7533\u8bf7\u4eba\u987b\u63d0\u4ea4\u5df2\u5b8c\u6210\u5de5\u4f5c\u7684\u8bc1\u660e\u3002"),
    (False, "The winning contractor will perform the work listed below.", "\u4e2d\u9009\u56e2\u961f\u5c06\u5728\u9879\u76ee\u671f\u95f4\u5b8c\u6210\u4ee5\u4e0b\u5de5\u4f5c\u3002"),
    (False, "Previous work reported by another supplier:", "\u5176\u4ed6\u4f9b\u5e94\u5546\u65e2\u5f80\u5b8c\u6210\u7684\u5de5\u4f5c\uff1a"),
    (False, "Available plant capabilities before engagement:", "\u672c\u5355\u4f4d\u73b0\u6709\u7684\u8bbe\u65bd\u4e0e\u5de5\u4f5c\u57fa\u7840\uff1a"),
)
MODELS = ("target_only", "scope")
ARMS = ("full_parser", "student_guard")


def synthetic():
    generated = {"training": [], "calibration": []}
    for i, topic in enumerate(TOPICS):
        partition = "training" if i < 8 else "calibration"
        for positive, english, chinese in SCOPES:
            for language, (target, heading) in enumerate(zip(topic, (english, chinese), strict=True)):
                fillers = ("Administrative details follow.\n", "\u4ee5\u4e0b\u5217\u51fa\u6709\u5173\u8d44\u6599\u3002\n")[language]
                for spacing in ("", fillers):
                    generated[partition].append(dict(text=f"{heading}\n{spacing}{target}", target=target,
                                                     positive=positive))
    return generated


def target_index(row):
    clauses = spans(row["text"])
    candidates = [i for i, c in enumerate(clauses)
                  if row["text"][c["start"]:c["end"]] == row["target"]]
    if len(candidates) != 1:
        raise ValueError("Missing or duplicate target span")
    return clauses, candidates[0]


def render(text, clauses, selected, arm):
    context, _ = scoped_context(text, clauses)
    output = []
    for i in selected:
        payload = context(i)
        if arm == "target_only":
            payload = dict(target=payload["target"])
        output.append(json.dumps(payload, ensure_ascii=False))
    return output


def fit(encoder, train, arm):
    texts = []
    gold = []
    for row in train:
        clauses, i = target_index(row)
        texts.extend(render(row["text"], clauses, [i], arm))
        gold.append(float(row["positive"]) * 2 - 1)
    x = torch.from_numpy(encoder.encode(texts)).cuda()
    center = x.mean(dim=0)
    x = x - center
    y = torch.tensor(gold, device="cuda")
    intercept = float(y.mean())
    dual = torch.linalg.solve(x @ x.T + .10 * torch.eye(len(x), device="cuda"), y - intercept)
    weight = (x.T @ dual).cpu().numpy()
    return weight, center.cpu().numpy(), intercept


def scores(encoder, head, texts):
    weight, center, intercept = head
    return (encoder.encode(texts) - center) @ weight + intercept


def labeled_scores(encoder, head, data, arm):
    texts = []
    labels = []
    for row in data:
        clauses, i = target_index(row)
        texts.extend(render(row["text"], clauses, [i], arm))
        labels.append(row["positive"] if "positive" in row else row["role"] == ROLES[QUALIFICATION])
    return scores(encoder, head, texts), np.asarray(labels)


def diagnostic(predicted, gold, cutoff):
    correct = (predicted > 0) == gold
    covered = np.abs(predicted) > cutoff
    return dict(cases=len(gold), correct=int(correct.sum()), accepted=int(covered.sum()),
                accepted_errors=int((covered & ~correct).sum()),
                accepted_positive=int((covered & (predicted > 0)).sum()))


class Guard(ComparisonParser):
    def __init__(self, checkpoint, encoder, head, confidence):
        super().__init__(checkpoint)
        self.encoder, self.head, self.confidence = encoder, head, confidence

    def evaluate(self, text, clauses, method, cutoff):
        self.cheap_qualifications = None
        self.skipped = False
        if self.arm == "student_guard" and clauses:
            views = render(text, clauses, range(len(clauses)), "scope")
            ids = self.encoder.tokenizer(views, add_special_tokens=True, truncation=False)["input_ids"]
            if all(len(seq) <= self.encoder.max_length for seq in ids):
                value = scores(self.encoder, self.head, views)
                self.cheap_qualifications = tuple(bool(v > 0) for v in value)
                if all(abs(v) > self.confidence for v in value):
                    self.skipped = True
                    return dict(labels=np.where(value > 0, QUALIFICATION, 0), rechecked=0)
        return super().evaluate(text, clauses, method, cutoff)


def run(queries, corpus, roles, vectors, metadata, models, control_file, output):
    search.role.require_cuda(output)
    if (digest(queries), digest(corpus), digest(control_file)) != (DEVELOPMENT, SOURCE_INVENTORY, CONTROLS):
        raise ValueError("Unexpected development or public controls")
    torch.backends.cuda.matmul.allow_tf32 = False
    synthetic_data = synthetic()
    external = json.loads(control_file.read_bytes())
    enc = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(enc)
    encoder = QwenTorchEmbedding(enc)
    encoder.model.config.use_cache = False
    heads, quality, thresholds = {}, {}, {}
    started = perf_counter()
    for method in MODELS:
        heads[method] = fit(encoder, synthetic_data["training"], method)
        margins, gold = labeled_scores(encoder, heads[method], synthetic_data["calibration"], method)
        errors = np.abs(margins[(margins > 0) != gold])
        thresholds[method] = float(errors.max()) if len(errors) else 0.
        frozen = thresholds[method]
        quality[method] = {
            "calibration": diagnostic(margins, gold, frozen),
            "external": diagnostic(*labeled_scores(encoder, heads[method], external, method), frozen),
        }
    fitting_ms = (perf_counter() - started) * 1000

    sources = {doc["id"]: doc for doc in rows(corpus)}
    stored = json.loads(metadata.read_bytes())
    if (SOURCE_INVENTORY, provenance["revision"], digest(vectors), len(sources)) != tuple(
            stored[k] for k in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Vector cache provenance differs")
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[r["expert_id"]] for r in sources.values()]))
    parser = Guard(models / "qwen3-4b-instruct-2507", encoder, heads["scope"], thresholds["scope"])
    verifier = search.EncodedReader(Reranker(models / "qwen3-reranker-0.6b"))
    fixture = (parser, encoder, index, sources,
               search.role.load_role_index(roles, SOURCE_INVENTORY, sources), verifier)

    def request(text, arm):
        parser.arm = arm
        result = search.one(text, "automatic", *fixture)
        return dict(result, skipped=parser.skipped, cheap=parser.cheap_qualifications)

    for arm in ARMS:
        request("Find an expert in optical sensors. The applicant needs past calibration experience.", arm)
    observations = []
    raw = Counter()
    for i, record in enumerate(rows(queries)):
        text = record["text"]
        pair = {arm: request(text, arm) for arm in (ARMS if i % 2 == 0 else ARMS[::-1])}
        guess = pair["student_guard"]["cheap"]
        if guess is not None:
            required = set(pair["full_parser"]["qualifications"])
            clauses = spans(text)
            expected = [(c["start"], c["end"]) in required for c in clauses]
            raw.update(requests=1, clauses=len(expected),
                       reference_qualifications=sum(expected),
                       qualification_false_negatives=sum(a and not b for a, b in zip(expected, guess, strict=True)),
                       qualification_false_positives=sum(not a and b for a, b in zip(expected, guess, strict=True)))
        observations.append(pair)
        if (i + 1) % 20 == 0:
            print(f"Completed {i+1} private development requests", flush=True)
    control, candidate = ([row[arm] for row in observations] for arm in ARMS)
    samples = np.random.default_rng(20261126).integers(len(control), size=(6000, len(control)))
    a, b = (np.asarray([r["elapsed_ms"] for r in stream]) for stream in (control, candidate))
    delta = np.quantile(b[samples], .95, axis=1) - np.quantile(a[samples], .95, axis=1)
    write(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
        encoder=provenance["revision"], synthetic_sha256=sha256(json.dumps(synthetic_data).encode()).hexdigest(),
        teaching_cases={k: len(v) for k,v in synthetic_data.items()}, fitting_ms=fitting_ms,
        confidence=thresholds, anchor_controls=quality, private_raw_comparison=dict(raw),
        live=dict(requests=len(control), bypassed=sum(r["skipped"] for r in candidate),
                  issued_queries={name: sum(bool(r["initial"] or r["extra"]) for r in stream)
                                  for name, stream in zip(ARMS, (control, candidate), strict=True)},
                  p50_p95_ms={k: np.quantile(v, [.50,.95]).tolist() for k,v in zip(ARMS, (a,b), strict=True)},
                  delta_p95_ci95=np.quantile(delta, [.025,.975]).tolist(),
                  changes={field: sum(x[field] != y[field] for x,y in zip(control,candidate,strict=True))
                           for field in ("qualifications","initial","extra","invoked","decisions","retrieved")})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "metadata", "models", "control_file", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
