"""Replay source discovery and structured expert verification with missing sources."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.retrieval import tokens
from experiments.credit import jats_linked_gate as roles
from experiments.credit import scoped_source_confirmation as source
from experiments.credit.scoped_credit_search import DENSE_TASK
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("top4", "expanded", "absolute_text", "cheap_gate", "oracle_source")


def prepare(projection, raw_directory, output):
    data = json.loads(projection.read_bytes())
    declarations = roles.compile_jats(data["documents"], data["documents"], raw_directory)
    write_json(output, dict(
        projection_sha256=digest(projection),
        rows={doi:[sorted(owners), {role:sorted(ids) for role,ids in sorted(postings.items())}]
              for doi,(owners,postings) in sorted(declarations.items())},
    ))


class ClueIndex(source.Search):
    def __init__(self, payload, encoder, directory):
        super().__init__(payload,encoder,directory)
        self.terms = [set(tokens(r["index"])) for r in payload["documents"].values()]
        frequency = Counter(t for doc in self.terms for t in doc)
        self.idf = {word:math.log1p(len(self.terms)/(1+count)) for word,count in frequency.items()}

    def observe(self, clues, removed, documents, reader):
        begin = perf_counter()
        vectors = self.encoder.encode([f"Instruct: {DENSE_TASK}\nQuery: {clue}" for clue in clues])
        cosines = [self.vectors @ query for query in vectors]
        absent = {self.ids[doi] for doi in removed}
        dense, fused, features = [], [], []
        for text, cosine in zip(clues,cosines,strict=True):
            order = [int(i) for i in np.argsort(-cosine,kind="stable") if i not in absent]
            lexical = {int(i):score for i,score in self.lexical.document_scores(text).items() if int(i) not in absent}
            sparse = sorted(lexical,key=lambda i:(-lexical[i],i))
            combined = {i:1/(60+j) for j,i in enumerate(order,1)}
            for j,i in enumerate(sparse,1):
                combined[i] += 1/(60+j)
            reranked = sorted(combined,key=lambda i:(-combined[i],i))
            dense.append(order)
            fused.append(reranked)
            query = sorted(set(tokens(text)))
            weights = {word:self.idf.get(word,math.log1p(len(self.terms))) for word in query}
            cover = np.array([sum(weights[word] for word in query if word in terms)/max(sum(weights.values()),1e-12)
                              for terms in self.terms])
            features.append((cosine,cover))
        narrow = self.rank(dense,fused,"linked_top4")
        broad = self.rank(dense,fused,"linked_expand")
        vector = None
        if broad:
            per_side = []
            for i, doi in enumerate(broad):
                candidate = self.ids[doi]
                cosine,cover = features[i]
                alternative = next(j for j in dense[i] if j != candidate)
                lexical_max = max(cover[j] for j in dense[i] if j!=candidate)
                per_side.append([float(cosine[candidate]), float(cosine[candidate]-cosine[alternative]),
                                 float(cover[candidate]), float(cover[candidate]-lexical_max),
                                 -math.log1p(dense[i].index(candidate))])
            vector = np.column_stack((np.min(per_side,axis=0),np.max(per_side,axis=0))).ravel()
        base_ms = (perf_counter()-begin)*1000
        text_score, text_ms = None,0.
        if broad:
            begin = perf_counter()
            margins, clipped = reader.score_pairs(
                [(clue,documents[doi]["index"]) for clue,doi in zip(clues,broad,strict=True)],task=source.TASK)
            if clipped or not np.isfinite(margins).all():
                raise ValueError("Invalid source confirmation score")
            text_score = min(margins)
            text_ms = (perf_counter()-begin)*1000
        return dict(top4=narrow,expanded=broad,features=vector,text_score=text_score,
                    source_ms=base_ms,text_ms=text_ms)


def frame(data, inventory, index, reader):
    result = {}
    for left,right,component in inventory["pairs"]:
        pair = left,right
        for view in source.VIEWS:
            clue = [inventory["clues"][doi][view] for doi in pair]
            for status in source.MISSING:
                removed = () if status=="present" else ((left,) if status=="missing_left" else (right,))
                observed = index.observe(clue,removed,data,reader)
                result[pair,view,status] = observed | dict(component=component,truth=pair if not removed else None)
    return result


def fit_gate(training):
    observed = [r for r in training if r["expanded"] is not None]
    x = np.stack([r["features"] for r in observed])
    y = np.array([r["truth"]==r["expanded"] for r in observed])
    bad = ~y
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0),1e-8)
    xs = (x-mean)/scale
    weights = np.linalg.solve(xs.T@xs+8*np.eye(x.shape[1]),xs.T@(y-y.mean()))
    cheap_cutoff = max(xs[bad]@weights,default=float("inf"))
    text_cutoff = max((r["text_score"] for r in observed if r["truth"]!=r["expanded"]),default=float("inf"))
    return mean,scale,weights,float(cheap_cutoff),float(text_cutoff)


def proposal(observed, model, arm):
    if arm in ("top4","expanded"):
        return observed[arm]
    if arm=="oracle_source":
        return observed["truth"]
    if not observed["expanded"]:
        return None
    mean,scale,weights,cheap_cutoff,text_cutoff = model
    if arm=="absolute_text":
        return observed["expanded"] if observed["text_score"]>text_cutoff else None
    confidence = ((observed["features"]-mean)/scale)@weights
    return observed["expanded"] if confidence>cheap_cutoff else None


def register(projection_path, postings_path):
    payload = json.loads(postings_path.read_bytes())
    if payload["projection_sha256"]!=digest(projection_path):
        raise ValueError("Postings and source inventory disagree")
    return {doi:(set(ids),{int(role):set(people) for role,people in posting.items()})
            for doi,(ids,posting) in payload["rows"].items()}


def run(projection_path, cases_path, postings_path, glossary_path, training_path, embedding, reranker, output):
    import torch
    from experiments.credit import linked_witness_dispatch as witness
    from experiments.followir.core17_verifier import Reranker
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    if digest(projection_path)!=witness.PROJECTION or digest(glossary_path)!=witness.EXAMPLES:
        raise ValueError("Changed pinned inputs")
    payload = json.loads(projection_path.read_bytes())
    docs = payload["documents"]
    cases = json.loads(cases_path.read_bytes())
    if cases["frozen_sha256"]!=digest(projection_path) or cases["pairs"]!=[list(x) for x in source.graph(docs)]:
        raise ValueError("Changed source controls")
    registry = register(projection_path,postings_path)
    if set(registry)!=set(docs):
        raise ValueError("Incomplete JATS registry")
    glossary = json.loads(glossary_path.read_bytes())
    training = json.loads(training_path.read_bytes())
    calibration,held = roles.phrase_pools(glossary,training)
    queries,_ = witness.panel(docs)
    encoder = witness.QwenTorchEmbedding(embedding,batch_size=24)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    info = witness.verify_checkpoint(embedding)

    def encode(phrases):
        return encoder.encode([f"Instruct: {roles.TASK}\nQuery: {phrase}" for phrase in phrases])

    bank = encode([phrase for name in roles.ROLES for phrase in training[name]])
    labels = np.repeat(np.eye(len(roles.ROLES),dtype=np.float32),6,axis=0)
    head = labels.T @ np.linalg.solve(bank@bank.T+.1*np.eye(len(bank)),bank)
    val = encode([phrase for _,phrase in calibration])@head.T
    quantiles = np.sort(val.max(axis=1)-val[np.arange(len(calibration)),[roles.ROLE_INDEX[name] for name,_ in calibration]])
    role_cutoff = float(quantiles[int(np.ceil(.90*len(calibration)))-1])
    with TemporaryDirectory() as work:
        index = ClueIndex(payload,encoder,Path(work))
        records = frame(docs,cases,index,reader)
    grouping = {r["component"] for r in records.values()}
    models = {group:fit_gate([r for r in records.values() if r["component"]!=group]) for group in grouping}
    result = defaultdict(Counter)
    timings = defaultdict(list)
    for pair,family,variant,requested,truth in queries:
        scopes,names = tuple(zip(*requested,strict=True))
        texts = [min(held[name],key=lambda text:roles.stable(family,name,text)) for name in names]
        begin = perf_counter()
        scores = encode(texts)@head.T
        role_ms = (perf_counter()-begin)*1000
        for view in source.VIEWS:
            for status in source.MISSING:
                observed = records[pair,view,status]
                for arm in ARMS:
                    begin = perf_counter()
                    selected = proposal(observed,models[observed["component"]],arm)
                    found,decided,plans = roles.predict(registry,selected,scopes,scores,role_cutoff) if selected else (set(),False,set())
                    exact_source = status=="present" and selected==pair
                    expected = truth if status=="present" else set()
                    report = result[view,status,arm]
                    report["requests"] += 1
                    report["positive_requests"] += bool(expected)
                    report["exact_source"] += exact_source
                    report["wrong_source_accept"] += selected is not None and not exact_source
                    report["source_unresolved"] += selected is None
                    report["role_unresolved_given_exact_source"] += exact_source and not decided
                    report["missing_intent_given_exact_source"] += exact_source and tuple(roles.ROLE_INDEX[name] for name in names) not in plans
                    report["correct_positive"] += exact_source and bool(found & expected)
                    report["wrong_source_recommendation"] += bool(found) and not exact_source
                    report["unsupported_recommendation"] += bool(found and (not exact_source or found-expected))
                    report["exact_decision"] += exact_source and decided and found==expected
                    route_ms = observed["source_ms"] if arm!="oracle_source" else 0.
                    if arm=="absolute_text":
                        route_ms += observed["text_ms"]
                    # Skip role processing after a source abstention.
                    timings[view,status,arm].append(route_ms + ((perf_counter()-begin)*1000 + role_ms if selected else 0.))
    reports = {f"{view}/{status}":{arm:dict(result[view,status,arm],
                    factored_p95_ms=float(np.quantile(timings[view,status,arm],.95))) for arm in ARMS}
               for view in source.VIEWS for status in source.MISSING}
    write_json(output,dict(job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)),projection_sha256=digest(projection_path),registry_sha256=digest(postings_path),
        cases_sha256=digest(cases_path),training_sha256=digest(training_path),embedding_revision=info["revision"],
        reranker_revision=reader.provenance["revision"],role_cutoff=role_cutoff,
        source_calibration_rows=len(records),source_components=len(grouping),requests=len(queries),reports=reports))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    prep = sub.add_parser("prepare")
    for name in ("projection","raw_directory","output"):
        prep.add_argument(name,type=Path)
    prep.set_defaults(action=prepare)
    replay = sub.add_parser("run")
    for name in ("projection_path","cases_path","postings_path","glossary_path","training_path","embedding","reranker","output"):
        replay.add_argument(name,type=Path)
    replay.set_defaults(action=run)
    args = vars(parser.parse_args())
    args.pop("action")(**args)
