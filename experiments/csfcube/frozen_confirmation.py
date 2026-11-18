"""Test frozen LitSearch policies on expert-judged CSFCube candidate pools."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.source_confirmation import accepts
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
from experiments.litsearch.learned_confirmation import Endpoint, METHODS, VIEWS, evaluate
from experiments.litsearch.source_withholding import POOL, TASK
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


FACETS = {"background": "research objective or problem", "method": "research method",
          "result": "main result"}
STRONG_RELEVANCE = 2


def load(root):
    papers_path = root / "abstracts-csfcube-preds.jsonl"
    with papers_path.open() as stream:
        source = {
            int(row["paper_id"]): dict(corpusid=int(row["paper_id"]), title=row["title"],
                                      abstract=" ".join(row["abstract"]))
            for row in map(json.loads, stream)
        }
    papers = [source[pid] for pid in sorted(source)]
    questions, hashes = [], {papers_path.name: digest(papers_path)}
    for facet, meaning in FACETS.items():
        path = root / f"test-pid2anns-csfcube-{facet}.json"
        hashes[path.name] = digest(path)
        for query_id, row in sorted(json.loads(path.read_text()).items()):
            query_id = int(query_id)
            grades = dict(zip(map(int, row["cands"]), row["relevance_adju"], strict=True))
            if (len(grades) != len(row["cands"])
                    or not set(grades) <= source.keys() or not set(grades.values()) <= {0, 1, 2, 3}):
                raise ValueError(f"Unexpected candidate grades for {query_id}")
            grades.pop(query_id, None)
            query = f"Find a paper with a similar {meaning}.\n{paper_text(source[query_id])}"
            questions.append(dict(query=query, query_id=query_id, facet=facet, grades=grades,
                                  corpusids=[pid for pid, grade in grades.items() if grade >= STRONG_RELEVANCE]))
    if len(papers) != 4207 or len(questions) != 50:
        raise ValueError("Unexpected release size")
    return papers, questions, hashes


class JudgedPoolEndpoint(Endpoint):
    @torch.inference_mode()
    def retrieve(self, question, view):
        query = self.encoder.encode([f"Instruct: {TASK}\nQuery: {question['query']}"])[0]
        candidates = question["grades"].keys()
        if view == "withheld":
            candidates = candidates - set(question["corpusids"])
        positions = [self.positions[doc] for doc in sorted(candidates)]
        scores = self.vectors[positions] @ torch.from_numpy(query).cuda()
        chosen = torch.argsort(scores, descending=True, stable=True)[:POOL]
        indices = [positions[i] for i in chosen.tolist()]
        return [self.ids[i] for i in indices], indices, scores[chosen].cpu().numpy()


def summarize(rows, questions, method, threshold):
    report = {}
    for view in VIEWS:
        counts = Counter()
        for row, question in zip(rows, questions, strict=True):
            arm = row["views"][view]
            grades = question["grades"]
            observed = grades[arm["scored"][0]]
            returned = (method == "cascade" and not arm["reader"]) or accepts(arm["scored"], threshold)
            counts["queries"] += 1
            counts["available_strong"] += view == "present" and bool(question["corpusids"])
            counts["pool_strong"] += any(grades[pid] >= STRONG_RELEVANCE for pid in arm["pool"])
            counts["top_strong"] += observed >= STRONG_RELEVANCE
            counts["returned"] += returned
            for grade in range(4):
                counts[f"returned_grade_{grade}"] += returned and observed == grade
        report[view] = dict(counts)
    return report


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    papers, questions, source_hashes = load(args.data)
    reference = json.loads(args.litsearch_result.read_bytes())
    provenance = verify_checkpoint(args.embedding)
    reader = Reranker(args.reranker)
    if provenance != reference["embedding"] or reader.provenance != reference["reranker"]:
        raise ValueError("Different checkpoint provenance")
    texts = [paper_text(row) for row in papers]
    encoder = QwenTorchEmbedding(args.embedding, batch_size=32)
    encoder.model.config.use_cache = False
    vectors = torch.from_numpy(encoder.encode(texts)).cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    lexical_texts = {}
    for i, text in enumerate(texts):
        while text in lexical_texts:
            text += "\n"
        lexical_texts[text] = i
    with TemporaryDirectory() as temporary:
        lexical = build_lexical(lexical_texts, Path(temporary))
        endpoint = JudgedPoolEndpoint(papers, vectors, encoder, lexical, reader)
    fit = reference["fitted"]
    endpoint.head = (*(np.asarray(fit[name]) for name in ("center", "scale", "coefficients")), fit["bias"])
    test = evaluate(endpoint, questions, METHODS, reference["cascade_first"])
    thresholds = reference["thresholds"]
    strict = reference["calibration_sweep"]["0.02"]
    strict_thresholds = {method: row["calibration"]["policy"] for method, row in strict.items()}
    report = {
        level: {method: summarize(test[method], questions, method, threshold)
                for method, threshold in policies.items()}
        for level, policies in (("0.02", strict_thresholds), ("0.10", thresholds))
    }
    write_json(args.output, dict(
        source_hashes=source_hashes, script_sha256=digest(Path(__file__)),
        litsearch_sha256=digest(args.litsearch_result), embedding=provenance, reranker=reader.provenance,
        test=test, thresholds={"0.02": strict_thresholds, "0.10": thresholds}, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "litsearch_result", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
