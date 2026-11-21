"""Complete source and role confirmation in four rows of one 4B forward."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from experiments.credit.engineering_joint_calibration import open_calibration, topic_prompts
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.paired_role_mixed_verifier import availability
from experiments.credit.paired_role_stage_audit import signatures
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_role_factorial_diagnosis import payload
from experiments.credit.source_confirmation_transfer import verify
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("cascade", "dense_joint", "fused_4b", "relative_4b")
CONFIRM = "The document reports the specific study described by the clue."
REJECT = "The document does not establish that it reports the specific study."


def source_row(clue, article):
    title, _, abstract = article.partition("\n")
    excerpt = " ".join(abstract.split()[:224])
    ordered = [CONFIRM, REJECT]
    if sha256(clue.encode()).digest()[0] % 2:
        ordered.reverse()
    options = "\n".join(f"{symbol}. {answer}" for symbol, answer in zip("AB", ordered, strict=True))
    return (
        ("Identify the specific scientific study. Choose exactly one option letter.",
         f"Clue: {clue}\nDocument: {title}\n{excerpt}\n\n{options}"),
        ordered.index(CONFIRM),
    )


def source_margins(scores, positive):
    return [float(row[choice] - row[1 - choice]) for row, choice in zip(scores, positive, strict=True)]


def calibrate(service, plos):
    papers, clues, _ = open_calibration(plos)
    catalog = PostingCatalog(papers, service.encoder)
    report = {}
    for view in ("keywords", "excerpt"):
        texts = [catalog.keywords(entry["result"]) if view == "keywords" else entry["result"]
                 for _, entry in sorted(clues.items())]
        vectors = service.encoder.encode(topic_prompts(texts))
        examples, target = [], []
        for document, clue, vector in zip(sorted(clues), texts, vectors, strict=True):
            if not clue:
                continue
            rivals = catalog.full_rank(vector[None, :], [clue], document)[0]["fused"][:2]
            for doc in (document, *rivals):
                examples.append(source_row(clue, papers[doc]["indexed"]))
                target.append(doc == document)
        margins = []
        for offset in range(0, len(examples), 12):
            prompts, identities = zip(*examples[offset:offset + 12], strict=True)
            scores, _ = service.role.option_logits(prompts)
            margins.extend(source_margins(scores, identities))
        absolute = list(zip(target, margins, strict=True))
        relative = []
        for true, first, second in zip(margins[::3], margins[1::3], margins[2::3], strict=True):
            relative.extend(((True, true - max(first, second)), (False, first - second), (False, second - first)))
        report[view] = {}
        for mode, samples in (("absolute", absolute), ("relative", relative)):
            wrong = [margin for expected, margin in samples if not expected]
            threshold = max(wrong)
            report[view][mode] = dict(threshold=threshold, positives=sum(target), negatives=len(wrong),
                                      positives_passed=sum(expected and score > threshold for expected, score in samples),
                                      negatives_passed=sum(margin > threshold for margin in wrong))
    return report


def score_fused(service, pair, clues, phrases, pools, cutoff, source_cutoff, rivals=()):
    examples = [
        source_row(clue, service.index.articles[doc]["indexed"])
        for doc, clue in zip(pair, clues, strict=True)
    ]
    if rivals:
        examples.extend(source_row(clue, service.index.articles[doc]["indexed"])
                        for doc, clue in zip(rivals, clues, strict=True))
    source_prompts, target = zip(*examples, strict=True)
    role_prompts, choices = service.role.format_batched(phrases, pools)
    scores, tokens = service.role.option_logits([*source_prompts, *role_prompts])
    margins = source_margins(scores[:len(target)], target)
    if rivals:
        margins = [left - right for left, right in zip(margins[:2], margins[2:], strict=True)]
    labels = []
    gaps = []
    for row, candidates in zip(scores[len(target):], choices, strict=True):
        best, next_best = np.argsort(-row[:len(candidates)], kind="stable")[:2]
        labels.append(candidates[best])
        gaps.append(float(row[best] - row[next_best]))
    source_ok = min(margins) > source_cutoff
    role_ok = cutoff is not None and None not in labels and min(gaps) > cutoff
    response = payload(service.index, pair if source_ok and role_ok else None,
                       [{role} for role in labels], source_calls=1)
    return response, dict(tokens=tokens, source_pass=source_ok, role_pass=role_ok,
                          both_pass=source_ok and role_ok, rows=len(scores))


def run_fusion(service, labels, cutoff, lightweight, plos, reference, output, reverse):
    calibration = calibrate(service, plos)
    requests = list(labels)
    counts = defaultdict(lambda: defaultdict(Counter))
    observed = defaultdict(dict)
    fallback = defaultdict(lambda: defaultdict(Counter))
    for req in (requests[::-1] if reverse else requests):
        label = labels[req]
        view = label["view"]
        cascade, plan = availability(service, req, cutoff, lightweight)
        response = dict.fromkeys(ARMS, cascade)
        if plan is not None:
            pair, pools, required, clues, phrases, _, ranking = plan
            dense_ok = verify(service.source, service.index, clues, pair, (0, 1)) > service.cutoffs["source"]
            response["dense_joint"] = payload(service.index, pair if dense_ok else None, required, source_calls=1)
            rivals = tuple(next(doc for doc in side["fused"] if doc != selected)
                           for side, selected in zip(ranking, pair, strict=True))
            for arm, mode, distractors in (("fused_4b", "absolute", ()), ("relative_4b", "relative", rivals)):
                fused, diagnostics = score_fused(service, pair, clues, phrases, pools, cutoff,
                                                  calibration[view][mode]["threshold"], distractors)
                response[arm] = fused
                fused_result = evidence(fused, label)
                diagnostics.update(
                    triggered=1, controls_with_true_pair=pair == tuple(label["papers"]),
                    strict=fused_result["strict"], adverse=fused_result["adverse"],
                )
                fallback[view][arm].update(diagnostics)
        for arm, out in response.items():
            if out["calls"] > 1 or out["calls"] != out["source_calls"] + out["role_calls"]:
                raise ValueError("Exceeded the one-forward cap")
            counts[view][arm].update(evidence(out, label))
            observed[arm][req] = out
    proofs = signatures(observed, requests)
    expected = json.loads(reference.read_bytes())["response_signatures"]
    if proofs["cascade"] != expected["student_then_teacher"] or proofs["dense_joint"] != expected["dense_joint"]:
        raise ValueError("The dense or cascade control did not reproduce prior service proofs")
    write_json(output, dict(calibration=calibration, counts=counts, fallback=fallback,
                            signatures=proofs, requests=len(requests), reverse=reverse,
                            reference_sha256=digest(reference), script_sha256=digest(Path(__file__))))
