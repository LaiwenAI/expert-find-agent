"""Calibrate fixed resident scores on independent same-author source pairs."""

from collections import defaultdict
import json

from experiments.credit.elife_prompt_holdout import build, sample_families
from experiments.credit.engineering_cross_source_search import TOPIC_TASK
from experiments.credit.engineering_joint_calibration import features
from experiments.credit.engineering_role_banks import TASK
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.paired_role_prospective_service import workload
from experiments.credit.paired_role_resident_source import score
from experiments.credit.paired_role_source_postings import pairs_for
from experiments.credit.paired_role_verification import alternative_roles
from experiments.followir.core17_reranking import digest


PRIOR_PROJECTION = "38bc5e6bd42e3b2f427a42a552cae11bc0813e50f2e601ee9e10a298fe47925f"
PRIOR_MANIFEST = "064b9d0a579e58911c1986a17e4eb169dcfd3ab57e4e4640d31a1a62143a753a"
SUPPLEMENT_PROJECTION = "4ce67b1ff52d869f133bd5318bad2573502d12aae065c75c9407c14754171c89"
SUPPLEMENT_MANIFEST = "77b44ff60a0c335be3d7a6233b18af17c55a453ec0e40f799561203241c838d4"
HARD_DEPTH = 4


def calibrate_pairs(service, original, projection, manifest_path, supplement, supplement_manifest, wording):
    if (digest(projection) != PRIOR_PROJECTION or digest(manifest_path) != PRIOR_MANIFEST
            or digest(supplement) != SUPPLEMENT_PROJECTION or digest(supplement_manifest) != SUPPLEMENT_MANIFEST):
        raise ValueError("Expected the frozen and author-screened earlier eLife calibration panel")
    papers, primary, census = build(projection, json.loads(manifest_path.read_bytes()))
    additional, secondary, added_census = build(supplement, json.loads(supplement_manifest.read_bytes()))
    if papers.keys() & additional.keys():
        raise ValueError("Calibration catalogs share a source DOI")
    papers.update(additional)
    held = service.index.articles
    authors = lambda corpus: {a for record in corpus.values() for a in record["roles"] if "#" not in a}
    if set(papers) & set(held) or authors(papers) & authors(held):
        raise ValueError("Calibration shares a DOI or recorded author with the evaluation")
    catalog = PostingCatalog(papers, service.encoder)
    requests = workload(catalog, sample_families(primary) + sample_families(secondary), wording)
    scores_by_view = defaultdict(dict)
    for req, label in requests.items():
        requirements = json.loads(req.payload)["requirements"]
        clues = [item["topic"] for item in requirements]
        phrases = [item["role"] for item in requirements]
        vectors = service.encoder.encode(
            [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
            + [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in phrases]
        )
        required = [{pool[0]} for pool in alternative_roles(
            {name: vectors[2:] @ bank.T for name, bank in service.banks.items()}
        )]
        view = label["view"]
        target = tuple(label["papers"])
        for absent in (None, *target):
            ranking = catalog.full_rank(vectors[:2], clues, absent)
            by_author = pairs_for(catalog, ranking, [set(), set()])
            by_role = pairs_for(catalog, ranking, required)
            near = (set(by_author[:HARD_DEPTH]) | set(by_role[:HARD_DEPTH])) - {target}
            if absent is None and all(doc in side["rank"] for side, doc in zip(ranking, target, strict=True)):
                near.add(target)
            for pair in sorted(near):
                key = target, absent, pair
                if key in scores_by_view[view]:
                    continue
                values = [features(catalog, side, clue, doc)
                          for side, clue, doc in zip(ranking, clues, pair, strict=True)]
                margins = {name: min(float(score(model, row)) for row in values)
                           for name, (model, _) in original[view].items()}
                scores_by_view[view][key] = (pair == target, absent is not None, margins)

    gates, report = {}, {}
    for view, records in scores_by_view.items():
        candidates = list(records.values())
        correct = [margins for gold, _, margins in candidates if gold]
        wrong = [margins for gold, _, margins in candidates if not gold]
        missing_count = sum(missing for _, missing, _ in candidates)
        sufficient = bool(correct and wrong and missing_count)
        gates[view], report[view] = {}, {}
        for name, (model, old_cutoff) in original[view].items():
            cutoff = max(margin[name] for margin in wrong) if sufficient else old_cutoff
            matched_name = name.replace("_features", "_matched")
            gates[view][matched_name] = model, cutoff
            report[view][matched_name] = dict(
                cutoff=cutoff, old_cutoff=old_cutoff, correct_pairs=len(correct), wrong_pairs=len(wrong),
                missing_target_pairs=missing_count, recalibrated=sufficient,
                correct_pass=sum(m[name] > cutoff for m in correct),
                old_correct_pass=sum(m[name] > old_cutoff for m in correct),
                wrong_pass=sum(m[name] > cutoff for m in wrong),
            )
    return gates, dict(views=report, panels=[census, added_census], requests=len(requests),
                       projection_sha256=PRIOR_PROJECTION, manifest_sha256=PRIOR_MANIFEST,
                       supplement_sha256=SUPPLEMENT_PROJECTION, supplement_manifest_sha256=SUPPLEMENT_MANIFEST)
