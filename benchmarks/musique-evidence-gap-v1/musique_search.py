"""Diagnose multi-hop retrieval and verifier misses with paired MuSiQue contexts."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.qwen_reranker import pack_tokens
from experiments.followir.core17_verifier import Reranker
from experiments.followir.core17_reranking import digest

ARCHIVE_SHA = "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
RETRIEVAL_TASK = "Retrieve passages that support answering the question."
TASK = ("Decide whether the passages jointly provide sufficient evidence for a complete "
        "answer to the question. Require evidence for every step of a multi-hop question.")
TOKEN = re.compile(r"[^\W_]+")
METHODS = ("lexical", "dense", "hybrid", "diverse", "linked")
DEPTHS = (4, 6, 12)


class InstructionReader:
    SYSTEM = (
        "Judge whether the supplied passages jointly contain sufficient evidence for a complete answer "
        "to the question. Use only these passages. Require support for every step. "
        "Output exactly 1 for sufficient or 0 otherwise."
    )

    def __init__(self, checkpoint):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        provenance = json.loads((checkpoint / "local-provenance.json").read_bytes())
        if any(digest(checkpoint / name) != value for name, value in provenance["files"].items()):
            raise ValueError("Changed instruction reader checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.bfloat16,
        ).cuda().eval()
        labels = [self.tokenizer.encode(label, add_special_tokens=False) for label in ("1", "0")]
        if any(len(label) != 1 for label in labels) or labels[0] == labels[1]:
            raise ValueError("Expected distinct single-token labels")
        self.labels = [label[0] for label in labels]

    @torch.inference_mode()
    def score_pairs(self, pairs, *, task=None):
        if len(pairs) != 1:
            raise ValueError("Expected single-request timing")
        query, context = pairs[0]
        encode = self.tokenizer.encode
        prefix = encode(f"<|im_start|>system\n{self.SYSTEM}<|im_end|>\n<|im_start|>user\nQuestion: {query}\nPassages:\n",
                        add_special_tokens=False)
        suffix = encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        body = encode(context, add_special_tokens=False)
        ids, raw = pack_tokens(prefix, body, suffix, 1024)
        x = torch.tensor([ids], device="cuda")
        logits = self.model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1, self.labels].float()
        return [(logits[0] - logits[1]).item()], raw > len(ids)


def tokens(text):
    return tuple(TOKEN.findall(text.casefold()))


def document(paragraph):
    return paragraph["title"], paragraph["paragraph_text"]


def key(text):
    return sha256(text.encode()).digest()


def load_pairs(path):
    with path.open("rb") as stream:
        if sha256_file(stream) != ARCHIVE_SHA:
            raise ValueError("Changed MuSiQue archive")
    by_id = defaultdict(dict)
    with ZipFile(path) as archive, archive.open("data/musique_full_v1.0_dev.jsonl") as stream:
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            row = json.loads(line)
            bucket = by_id[row["id"]]
            if row["answerable"] in bucket:
                raise ValueError("Duplicate paired-context side")
            bucket[row["answerable"]] = row
    def checked(pair):
        pos, neg = pair[True], pair[False]
        if pos["question"] != neg["question"]:
            raise ValueError("Different paired questions")
        gold = {document(p) for p in pos["paragraphs"] if p["is_supporting"]}
        assert 2 <= len(gold) <= 4 and gold - {document(p) for p in neg["paragraphs"]}
        return pos, neg, gold
    return [checked(by_id[i]) for i in sorted(by_id, key=lambda i: key("musique-first-v1:" + i))]


def sha256_file(stream):
    digest = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


class Lexical:
    def __init__(self, bank):
        grouped = defaultdict(list)
        doc_lengths = []
        for i, (title, text) in enumerate(bank):
            terms = tokens(title) + tokens(title) + tokens(text)
            doc_lengths.append(len(terms))
            for term, count in Counter(terms).items():
                grouped[term].append((i, count))
        lengths = np.array(doc_lengths, dtype=np.float32)
        self.postings = {}
        self.size = len(bank)
        for term, rows in grouped.items():
            ids, counts = np.array(rows, dtype=np.int32).T
            idf = math.log(1 + (self.size - len(rows) + 0.5) / (len(rows) + 0.5))
            ranks = idf * counts * 2.2 / (counts + 1.2 * (0.25 + 0.75 * lengths[ids] / lengths.mean()))
            self.postings[term] = ids, ranks

    def score(self, text):
        result = np.zeros(self.size, dtype=np.float32)
        for term in dict.fromkeys(tokens(text)):
            if term in self.postings:
                ids, values = self.postings[term]
                result[ids] += values
        return result


def link_index(bank):
    title_docs = defaultdict(list)
    for i, (title, _) in enumerate(bank):
        words = tokens(title)
        if 2 <= len(words) <= 8:
            title_docs[words].append(i)
    outgoing = []
    for title, text in bank:
        words = tokens(text)
        own = tokens(title)
        seen = set()
        links = set()
        for i in range(len(words)):
            for size in range(2, min(len(words) - i, 8) + 1):
                phrase = words[i:i + size]
                if phrase != own and phrase not in seen and phrase in title_docs:
                    seen.add(phrase)
                    links.update(title_docs[phrase])
        outgoing.append(links)
    # Neighbor candidates include either direction of an exact title mention.
    links = [set(group) for group in outgoing]
    for source, group in enumerate(outgoing):
        for target in group:
            links[target].add(source)
    return links


def choices(method, ranks, titles, links, n, depth):
    if method in ("lexical", "dense", "hybrid"):
        return ranks[method][:depth].tolist()
    order = ranks["hybrid"]
    position = ranks["position"]
    chosen = order[:2].tolist()
    seen = set(chosen)
    for slot in range(2, depth):
        if method == "linked" and slot % 2 == 0:
            options = (links[chosen[-1]] | links[chosen[-2]]) - seen
            # Keep links near the base retrieval frontier and require another article.
            options = [doc for doc in options if position[doc] < min(256, n) and titles[doc] != titles[chosen[-1]]]
        elif method == "diverse":
            options = [int(doc) for doc in order[:24] if titles[doc] not in {titles[x] for x in chosen}]
        else:
            options = []
        if not options:
            options = [int(doc) for doc in order[:depth] if doc not in seen]
        next_doc = min(options, key=lambda doc: (position[doc], doc))
        chosen.append(next_doc)
        seen.add(next_doc)
    return chosen


def pct(series):
    values = np.asarray(series, dtype=float)
    return {f"p{level}": round(float(np.percentile(values, level)), 3) for level in (50, 95)}


def joined(rows):
    return "\n\n".join(f"Source: {title}\n{text}" for title, text in rows)


def replace_missing(selected, gold, depth):
    chosen = list(selected)
    missing = sorted(gold - set(chosen))
    slots = [idx for idx in reversed(range(len(chosen))) if chosen[idx] not in gold]
    if len(chosen) != depth or len(slots) < len(missing):
        raise ValueError("Cannot insert annotated evidence at the same paragraph count")
    for idx, row in zip(slots, missing):
        chosen[idx] = row
    assert gold <= set(chosen)
    return chosen


def candidate_ranks(q, encoder, matrix, lexical):
    v = torch.as_tensor(encoder.encode([f"Instruct: {RETRIEVAL_TASK}\nQuery: {q}"])[0], device="cuda")
    dense = (matrix @ v).cpu().numpy()
    sparse = lexical.score(q)
    ordered = {
        "dense": np.argsort(-dense, kind="stable"),
        "lexical": np.argsort(-sparse, kind="stable"),
    }
    rrf = np.empty_like(dense)
    rrf[ordered["dense"]] = 1 / (60 + np.arange(1, len(dense) + 1))
    rrf[ordered["lexical"]] += 1 / (60 + np.arange(1, len(dense) + 1))
    ordered["hybrid"] = np.argsort(-rrf, kind="stable")
    ordered["position"] = np.empty(len(dense), dtype=np.int32)
    ordered["position"][ordered["hybrid"]] = np.arange(len(dense))
    return ordered


def make_negative(neg, gold, lexical, bank_index, depth):
    items = list(dict.fromkeys(map(document, neg["paragraphs"])))
    known = [item for item in items if item in gold]
    scores = lexical.score(neg["question"])
    others = [item for item in items if item not in gold]
    others.sort(key=lambda item: (-float(scores[bank_index[item]]), item) if item in bank_index else (0, item))
    return known + others[:depth - len(known)]


def evaluate(pairs, encoder, reranker, held, calibration, read_depth):
    bank = sorted({document(p) for pos, _, _ in pairs for p in pos["paragraphs"]})
    bank_index = {doc: i for i, doc in enumerate(bank)}
    titles = [doc[0] for doc in bank]
    prepare_begin = perf_counter()
    model_vectors = encoder.encode([title + "\n" + text for title, text in bank])
    matrix = torch.from_numpy(model_vectors).cuda()
    lexical = Lexical(bank)
    links = link_index(bank)
    offline_seconds = perf_counter() - prepare_begin

    q = pairs[0][0]["question"]
    candidate_ranks(q, encoder, matrix, lexical)
    reranker.score_pairs([(q, joined(bank[:6]))], task=TASK)
    calib_scores = []
    for _, neg, gold in pairs[:calibration]:
        view = make_negative(neg, gold, lexical, bank_index, read_depth)
        scores, clipped = reranker.score_pairs([(neg["question"], joined(view))], task=TASK)
        if not clipped:
            calib_scores.append(scores[0])
    if len(calib_scores) < calibration // 2:
        raise RuntimeError("Too few unclipped calibration contexts")
    # Strict upper bound on the observed calibration false-accept fraction.
    cutoff = sorted(calib_scores, reverse=True)[len(calib_scores) // 10]

    outputs = []
    for i, (pos, neg, gold) in enumerate(pairs[calibration:calibration + held]):
        q = pos["question"]
        gold_ids = {bank_index[item] for item in gold}
        gold_titles = {title for title, _ in gold}
        begin = perf_counter()
        ranks = candidate_ranks(q, encoder, matrix, lexical)
        shared_ms = (perf_counter() - begin) * 1000
        selection = {}
        ordered_methods = METHODS[i % len(METHODS):] + METHODS[:i % len(METHODS)]
        selected = {}
        for arm in ordered_methods:
            begin = perf_counter()
            ids = choices(arm, ranks, titles, links, len(bank), max(DEPTHS))
            elapsed = (perf_counter() - begin) * 1000
            chosen = {}
            for depth in DEPTHS:
                picked = ids[:depth]
                chosen[str(depth)] = {
                    "complete": gold_ids <= set(picked),
                    "all_titles": gold_titles <= {titles[j] for j in picked},
                    "gold_hits": len(gold_ids.intersection(picked)),
                }
            selection[arm] = dict(depths=chosen, shared_ms=shared_ms, policy_ms=elapsed)
            selected[arm] = [bank[j] for j in ids[:read_depth]]
        views = {
            "hybrid": selected["hybrid"],
            "linked": selected["linked"],
            "oracle_replacement": replace_missing(selected["hybrid"], gold, read_depth),
            "paired_missing": make_negative(neg, gold, lexical, bank_index, read_depth),
        }
        if read_depth == 4:
            views["gold_only"] = sorted(gold)
            views["dense"] = selected["dense"]
        # Match the reader to each request and report clipping separately per view.
        readout = {}
        names = tuple(views)
        for j in range(len(names)):
            arm = names[(i + j) % len(names)]
            begin = perf_counter()
            values, clipped = reranker.score_pairs([(q, joined(views[arm]))], task=TASK)
            elapsed = (perf_counter() - begin) * 1000
            readout[arm] = {
                "accepted": values[0] > cutoff,
                "clipped": bool(clipped),
                "complete": gold <= set(views[arm]),
                "ms": elapsed,
            }
        outputs.append(dict(hops=len(gold), selection=selection, readout=readout))

    measurements = {}
    for arm in METHODS:
        source = [row["selection"][arm] for row in outputs]
        measurements[arm] = {
            "depths": {str(depth): {
                field: sum(row["depths"][str(depth)][field] for row in source)
                for field in ("complete", "all_titles", "gold_hits")
            } for depth in DEPTHS},
            "shared_query_ms": pct([row["shared_ms"] for row in source]),
            "selection_ms": pct([row["policy_ms"] for row in source]),
        }

    link_delta = Counter()
    for row in outputs:
        for depth in DEPTHS:
            baseline = row["selection"]["hybrid"]["depths"][str(depth)]["complete"]
            candidate = row["selection"]["linked"]["depths"][str(depth)]["complete"]
            link_delta[f"{depth}_gained"] += candidate and not baseline
            link_delta[f"{depth}_lost"] += baseline and not candidate

    reader = {}
    for name in views:
        items = [row["readout"][name] for row in outputs]
        reader[name] = {
            "clipped": sum(row["clipped"] for row in items),
            "reader_ms": pct([row["ms"] for row in items]),
            "unclipped_complete": sum(row["complete"] and not row["clipped"] for row in items),
            "accepted_unclipped_complete": sum(row["complete"] and row["accepted"] and not row["clipped"] for row in items),
            "unclipped_incomplete": sum(not row["complete"] and not row["clipped"] for row in items),
            "accepted_unclipped_incomplete": sum(not row["complete"] and row["accepted"] and not row["clipped"] for row in items),
        }

    paired = Counter()
    for row in outputs:
        natural = row["readout"]["hybrid"]
        restored = row["readout"]["oracle_replacement"]
        negative = row["readout"]["paired_missing"]
        if not natural["clipped"] and not restored["clipped"] and not natural["complete"]:
            paired["incomplete_and_restored_both_unclipped"] += 1
            paired["restore_changes_reject_to_accept"] += not natural["accepted"] and restored["accepted"]
            paired["restore_changes_accept_to_reject"] += natural["accepted"] and not restored["accepted"]
        if not restored["clipped"] and not negative["clipped"]:
            paired["oracle_and_missing_both_unclipped"] += 1
            paired["accept_complete_reject_missing"] += restored["accepted"] and not negative["accepted"]

    return {
        "archive_sha256": ARCHIVE_SHA, "reader_paragraph_cap": read_depth,
        "pool_paragraphs": len(bank), "pool_titles": len(set(titles)),
        "held_pairs": len(outputs), "hop_counts": dict(Counter(row["hops"] for row in outputs)),
        "input_calibration_pairs": calibration, "unclipped_calibration_pairs": len(calib_scores),
        "calibration_cutoff": cutoff, "calibration_accepted": sum(s > cutoff for s in calib_scores),
        "offline_build_seconds": round(offline_seconds, 2), "retrieval": measurements,
        "linked_vs_hybrid_full_evidence_changes": dict(link_delta), "reader": reader, "paired_reader": dict(paired),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("archive", "embedding", "reranker", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--instruction-reader", action="store_true")
    parser.add_argument("--reader-paragraphs", type=int, choices=(4, 6), default=6)
    parser.add_argument("--held", type=int, default=384)
    parser.add_argument("--calibration", type=int, default=80)
    args = parser.parse_args()
    pairs = load_pairs(args.archive)[:args.held + args.calibration]
    encoder = QwenTorchEmbedding(args.embedding, max_length=256, batch_size=64)
    reader = (InstructionReader if args.instruction_reader else Reranker)(args.reranker)
    result = evaluate(pairs, encoder, reader, args.held, args.calibration, args.reader_paragraphs)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
