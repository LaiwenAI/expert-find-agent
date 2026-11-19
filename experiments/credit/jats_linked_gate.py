"""Route contributor phrases into validated publisher JATS role postings."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from experiments.credit.cross_source_identity import orcid_id
from experiments.credit.plos_role_inventory import ROLES, declared_roles
from experiments.followir.core17_reranking import digest, write_json
from xml.etree import ElementTree as ET


ARMS = ("live_text", "dense_point", "ridge_point", "ridge_consensus90", "ridge_consensus_all")
SALT = "jats-linked-gate-v1"
ROLE_INDEX = {label:i for i,label in enumerate(ROLES)}
TASK = "Classify a research contribution into its precise CRediT role."


def words(text):
    return " ".join(re.findall(r"\w+",text.casefold()))


def stable(*parts):
    return sha256(json.dumps(parts).encode()).digest()


def compile_jats(documents, selected, raw_directory):
    registry = {}
    orcids = {}
    raw_to_owner = {}
    for doi in sorted(selected):
        ref = documents[doi]
        path = raw_directory/f"{doi.rsplit('.',1)[-1]}.xml"
        raw = path.read_bytes()
        if sha256(raw).hexdigest()!=ref["sha256"]:
            raise ValueError(f"Changed publisher XML: {doi}")
        field_roles = declared_roles(raw,doi)
        nodes = ET.fromstring(raw).findall("./front/article-meta/contrib-group/contrib[@contrib-type='author']")
        if len(nodes)!=len(field_roles):
            raise ValueError("Misaligned contributor role fields")
        compiled = defaultdict(set)
        authors = set()
        for entry in ref["owners"]:
            ordinal, owner = entry["ordinal"],entry["owner"]
            node = nodes[ordinal-1]
            keys = {key for value in node.findall("contrib-id[@contrib-id-type='orcid']")
                    if (key:=orcid_id(value.text or ""))}
            if len(keys)!=1 or owner in authors:
                raise ValueError("Missing or conflicting publisher contributor identifier")
            raw_key = next(iter(keys))
            if orcids.setdefault(owner,raw_key)!=raw_key or raw_to_owner.setdefault(raw_key,owner)!=owner:
                raise ValueError("An anonymized key disagrees with the original contributor identifier")
            actual = set(field_roles[ordinal-1])
            if actual!=set(entry["roles"]):
                raise ValueError("Verified JATS roles and projection disagree")
            authors.add(owner)
            for role in actual:
                compiled[ROLE_INDEX[role]].add(owner)
        registry[doi] = authors,dict(compiled)
    return registry


def phrase_pools(glossary, training):
    def collect(rows):
        return sorted({row["text"].strip() for row in rows})

    calibrated = []
    held = {}
    for role in ROLES:
        items = collect(row for row in glossary["examples"] if row["role"]==role)
        ordered = sorted(items,key=lambda text:stable(SALT,role,text))
        if len(ordered)<2 or not training[role]:
            raise ValueError("Need independent calibration and evaluation examples")
        calibrated.append((role,ordered[0]))
        held[role] = ordered[1:]
    train_keys = {words(text) for rows in training.values() for text in rows}
    calibration_keys = {words(text) for _,text in calibrated}
    held_keys = {words(text) for rows in held.values() for text in rows}
    if train_keys & (calibration_keys|held_keys) or calibration_keys & held_keys:
        raise ValueError("Overlapping phrase partitions")
    return calibrated,held


def predict(registry, dois, scopes, scores, cutoff):
    ordered = np.argsort(-scores,axis=1,kind="stable")
    if cutoff is None:
        pools = [rank[:1].tolist() for rank in ordered]
    else:
        gaps = scores.max(axis=1,keepdims=True)-scores
        pools = [rank[gap[rank]<=cutoff].tolist() for rank,gap in zip(ordered,gaps,strict=True)]
    plans = {tuple(roles) for roles in product(*pools) if len(set(roles))==len(roles)}
    possible = registry[dois[0]][0] & registry[dois[1]][0]
    matches = []
    for plan in plans:
        eligible = possible.copy()
        for side,role in zip(scopes,plan,strict=True):
            eligible &= registry[dois[side]][1].get(role,set())
        matches.append(eligible)
    common = set.intersection(*matches) if matches else set()
    exact_decision = bool(matches) and all(result==common for result in matches)
    return common,exact_decision,plans


def run(projection, glossary_path, training_path, raw_directory, checkpoint, reranker, output):
    import torch
    from experiments.credit import linked_witness_dispatch as replay
    from experiments.credit.live_linked_witness_verification import verify
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    if digest(projection)!=replay.PROJECTION or digest(glossary_path)!=replay.EXAMPLES:
        raise ValueError("Changed projection or glossary")
    docs = json.loads(projection.read_bytes())["documents"]
    glossary = json.loads(glossary_path.read_bytes())
    training = json.loads(training_path.read_bytes())
    if set(training)!=set(ROLES) or {len(v) for v in training.values()}!={6}:
        raise ValueError("Changed role training inventory")
    calibrated,held = phrase_pools(glossary,training)
    items,links = replay.panel(docs)
    dois = {doi for index, *_ in items for doi in index}
    start = perf_counter()
    registry = compile_jats(docs,dois,raw_directory)
    xml_ms = (perf_counter()-start)*1000
    model_info = replay.verify_checkpoint(checkpoint)
    encoder = replay.QwenTorchEmbedding(checkpoint,batch_size=24)
    encoder.model.config.use_cache = False

    def encode(phrases):
        return encoder.encode([f"Instruct: {TASK}\nQuery: {p}" for p in phrases])

    reader = Reranker(reranker)
    views = {}
    for pair in sorted({item[0] for item in items}):
        records = [(side,row["owner"],set(row["roles"])) for side,doi in enumerate(pair) for row in docs[doi]["owners"]]
        texts = [" ".join(glossary["definitions"][role] for role in sorted(tags)) for _,_,tags in records]
        views[pair] = records,texts,encoder.encode(texts)
    definitions = encoder.encode([glossary["definitions"][role] for role in ROLES])
    train = encode([phrase for role in ROLES for phrase in training[role]])
    target = np.repeat(np.eye(len(ROLES),dtype=np.float32),6,axis=0)
    ridge = target.T @ np.linalg.solve(train@train.T+.1*np.eye(len(train)),train)
    cal_vectors = encode([phrase for _,phrase in calibrated])
    cal_scores = cal_vectors@ridge.T
    residuals = np.sort(cal_scores.max(axis=1)-cal_scores[np.arange(len(calibrated)),
                                                        [ROLE_INDEX[role] for role,_ in calibrated]])
    cutoff = {"ridge_consensus90":float(residuals[int(np.ceil(.90*len(residuals)))-1]),
              "ridge_consensus_all":float(residuals[-1])}
    stats,timing = defaultdict(Counter),defaultdict(list)
    outcomes = defaultdict(lambda:defaultdict(list))
    for i,(pair,family,variant,requested,truth) in enumerate([items[0],*items]):
        scopes,roles = tuple(zip(*requested,strict=True))
        phrases = [min(held[role],key=lambda text:stable(family,role,text)) for role in roles]
        start = perf_counter()
        matrix = encode(phrases)
        prefix = (perf_counter()-start)*1000
        offset = i%len(ARMS)
        for arm in ARMS[offset:]+ARMS[:offset]:
            start = perf_counter()
            if arm == "live_text":
                records,texts,stored = views[pair]
                match_scores = stored @ encoder.encode(phrases,query=True).T
                found, _, _ = verify(records,match_scores,texts,phrases,requested,reader,"owner_mean",2,0.)
                decided, plans = True, None
                elapsed = (perf_counter()-start)*1000
            else:
                logits = matrix@(definitions if arm=="dense_point" else ridge).T
                found,decided,plans = predict(registry,pair,scopes,logits,cutoff.get(arm))
                elapsed = prefix+(perf_counter()-start)*1000
            if not i:
                continue
            result = stats[arm]
            if plans is not None:
                contains = tuple(ROLE_INDEX[role] for role in roles) in plans
                if contains and found-truth:
                    raise AssertionError("Consensus misattributed an included interpretation")
                result["includes_intent"] += contains
                result["plans_total"] += len(plans)
            result["queries"] += 1
            result["exact_decided"] += decided and found==truth
            result["unresolved"] += not decided
            result["positive_correct"] += bool(found&truth)
            result["unsupported_queries"] += bool(found-truth)
            result["unsupported_people"] += len(found-truth)
            result[f"{variant}_unsupported"] += bool(found-truth)
            result[f"{variant}_queries"] += 1
            timing[arm].append(elapsed)
            outcomes[arm][family].append(decided and found==truth)
    reports = {arm:dict(stats[arm],both_variants_exact=sum(len(x)==2 and all(x) for x in outcomes[arm].values()),
                        p95_ms=float(np.quantile(timing[arm],.95))) for arm in ARMS}
    write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),script_sha256=digest(Path(__file__)),
        projection_sha256=replay.PROJECTION,glossary_sha256=replay.EXAMPLES,training_sha256=digest(training_path),
        embedding_revision=model_info["revision"],reranker_revision=reader.provenance["revision"],queries=len(items),article_links=links,checked_articles=len(dois),
        xml_compile_ms=xml_ms,calibration_phrases=len(calibrated),held_phrases=sum(map(len,held.values())),
        thresholds=cutoff,reports=reports,
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("projection","glossary_path","training_path","raw_directory","checkpoint","reranker","output"):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
