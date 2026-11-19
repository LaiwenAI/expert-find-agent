"""Replace annotated works and people with an automatic entity index."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import global_role_search as base
from experiments.docred import noisy_role_source_routing as noisy


MODEL = "en_core_web_sm"
MODEL_VERSION = "3.8.0"
ENGINE_VERSION = "3.8.7"
WORK_LABELS = {"WORK_OF_ART", "ORG", "PRODUCT", "EVENT", "FAC", "LAW", "GPE"}
ARMS = ("gold", "auto_sources", "auto_people", "auto_bm25", "title_only", "auto_full")
CUTOFF = .25


def load_documents(snapshot):
    documents = []
    for split, expected in base.HASHES.items():
        path = snapshot / f"{split}_revised.json"
        if base.digest(path) != expected:
            raise ValueError("Changed Re-DocRED input")
        documents.extend(json.loads(path.read_bytes()))
    if len(documents) != 1000:
        raise ValueError("Expected 1,000 source documents")
    return documents


def extract(snapshot, output):
    import spacy
    from spacy.tokens import Doc

    if output.exists() or spacy.__version__ != ENGINE_VERSION:
        raise ValueError("Require fresh output and pinned spaCy")
    model = spacy.load(MODEL, enable=["tok2vec", "ner"])
    if model.meta["version"] != MODEL_VERSION:
        raise ValueError("Unexpected NER checkpoint")
    documents = load_documents(snapshot)
    sentences = [(i, sentence_id, tokens) for i, doc in enumerate(documents)
                 for sentence_id, tokens in enumerate(doc["sents"])]
    stream = (Doc(model.vocab, words=tokens) for _, _, tokens in sentences)
    annotations = [[] for _ in documents]
    started = perf_counter()
    for (doc_id, sentence_id, tokens), result in zip(sentences, model.pipe(stream, batch_size=96), strict=True):
        for ent in result.ents:
            if ent.label_=="PERSON" or ent.label_ in WORK_LABELS:
                if not 0 <= ent.start < ent.end <= len(tokens):
                    raise ValueError("Invalid entity bounds")
                annotations[doc_id].append((sentence_id, ent.start, ent.end, ent.label_))
    base.write_json(output,dict(
        inputs=base.HASHES, extraction_code_sha256=base.digest(Path(__file__)),
        model=MODEL, model_version=MODEL_VERSION, engine_version=ENGINE_VERSION,
        seconds=perf_counter()-started, annotations=annotations,
    ))


def add_occurrences(doc, annotations):
    names, aliases = defaultdict(set), defaultdict(set)
    for sentence_id,start,end,label in annotations:
        key=base.words(" ".join(doc["sents"][sentence_id][start:end]))
        if not key:
            continue
        if label=="PERSON":
            names[key].add((sentence_id,start,end))
        elif label in WORK_LABELS:
            aliases[key].add(sentence_id)
    for key in list(names):
        if len(key)!=1:
            continue
        expanded=[other for other in names if len(other)>1 and other[-1:]==key]
        if len(expanded)==1:
            names[expanded[0]].update(names.pop(key))
    title=base.words(doc["title"])
    if title:
        for i, sentence in enumerate(doc["sents"]):
            words=base.words(" ".join(sentence))
            if any(words[j:j+len(title)]==title for j in range(len(words)-len(title)+1)):
                aliases[title].add(i)
        aliases.setdefault(title,set())
    return sorted((name,tuple(sorted(values))) for name,values in names.items()),aliases


def auto_index(documents, extracted, oracle_profiles):
    aliases=defaultdict(set)
    profiles=[]
    for i,doc in enumerate(documents):
        people, subjects=add_occurrences(doc,extracted[i])
        for key in subjects:
            aliases[key].add(i)
        profiles.append((people,subjects,oracle_profiles[i][2]))
    return aliases, profiles


def person_alias_judgments(documents):
    identities=[]
    for doc in documents:
        aliases=defaultdict(set)
        for mentions in doc["vertexSet"]:
            if mentions[0]["type"]!="PER":
                continue
            person=base.words(base.canonical(mentions))
            for mention in mentions:
                aliases[base.words(mention["name"])].add(person)
        identities.append({alias:next(iter(people)) for alias,people in aliases.items() if alias and len(people)==1})
    return identities


def prepare_views(documents):
    original=base.study_queries(documents,500)
    return [dict(query,style="exact") for query in original]+[
        query for query in noisy.query_styles(original,documents) if query["style"]=="transposition"
    ]


def reader_session(model_path):
    import torch
    from experiments.docred import oracle_role_evidence as oracle
    from experiments.docred import person_bound_evidence as evidence

    if not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ:
        raise ValueError("Require Slurm CUDA")
    tokenizer, model, info=oracle.load_reader(model_path)
    verify=oracle.reader_scorer(tokenizer, model)

    def read(query, pairs, sentences):
        work=" ".join(query["work"])
        role=query["role"]
        context=[]
        for doc,person in pairs:
            text, parsed=sentences[doc]
            name=" ".join(person)
            case=dict(work=work,role=role)
            selected=(evidence.choose(case,name,parsed,"lexical_two") if len(text)>=2 else range(len(text)))
            context.append((f"Was {name} credited in the role of {evidence.ROLES[role]} for {work}?",
                            "\n".join(text[i] for i in selected)))
        if not context:
            return []
        scores, clipped=verify(context)
        if clipped or not np.isfinite(scores).all():
            raise ValueError("Unexpected clipped model result")
        return scores

    return read,evidence.profile,dict(job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
                                     checkpoint=info["revision"], cutoff=CUTOFF)


def judge(query, method, postings, lookups, profiles, aliases, reader, sentences):
    auto_source=method in ("auto_sources","auto_full","auto_bm25","title_only")
    auto_people=method in ("auto_people","auto_full","auto_bm25","title_only")
    route="flat_four" if method=="auto_bm25" else "swap_then_guard"
    lookup=lookups["titles" if method=="title_only" else auto_source]
    # The work index and the person inventory vary independently. Their
    # cue positions come from the same tokenized document text.
    profile=profiles[(auto_source,auto_people)]
    started=perf_counter()
    docs,_,fallback=noisy.route(query,postings,lookup,route)
    predictions, examined=base.rank_people(query,docs,profile)
    scores=reader(query,predictions,sentences) if reader else None
    elapsed=(perf_counter()-started)*1000
    support=query["positive"]
    rivals=query["rivals"]

    def resolve(pair):
        d,name=pair
        return d,aliases[d].get(name,name)

    resolved=[resolve(pair) for pair in predictions]
    available={resolve((d,name)) for d in docs for name,_ in profile[d][0]}
    item=dict(
        opened=len(docs), fallback=fallback, exact_key=query["work"] in lookup.aliases,
        source=any(d in docs for d,_ in support), available=bool(available & support),
        candidates=len(predictions), person_mentions_scored=examined,
        strict_positive=len(set(predictions) & support),
        proposed_positive=len(set(resolved) & support), proposed_rival=len(set(resolved) & rivals),
        ms=elapsed,
    )
    if scores is not None:
        acceptance={pair for pair,score in zip(resolved,scores,strict=True) if score>CUTOFF}
        item.update(accepted_positive=len(acceptance & support), accepted_rival=len(acceptance & rivals),
                    accepted_unjudged=len(acceptance-support-rivals))
    return item


def run(snapshot,extracts,output,gpu_model=None):
    if output.exists():
        raise FileExistsError(output)
    documents=load_documents(snapshot)
    extracted=json.loads(extracts.read_bytes())
    if (extracted["inputs"]!=base.HASHES or extracted["model"]!=MODEL
            or extracted["model_version"]!=MODEL_VERSION or len(extracted["annotations"])!=len(documents)):
        raise ValueError("Incorrect NER projection")
    oracle_aliases, oracle_profiles=base.mentions_index(documents)
    automatic_aliases, automatic_profiles=auto_index(documents,extracted["annotations"],oracle_profiles)
    titles=defaultdict(set)
    for i,doc in enumerate(documents):
        title=base.words(doc["title"])
        if title:
            titles[title].add(i)
    lookups={False:noisy.AliasLookup(oracle_aliases),True:noisy.AliasLookup(automatic_aliases),
             "titles":noisy.AliasLookup(titles)}
    profiles={}
    for source_choice in (False,True):
        for person_choice in (False,True):
            source=automatic_profiles if source_choice else oracle_profiles
            person=automatic_profiles if person_choice else oracle_profiles
            profiles[source_choice,person_choice]=[(p[0],s[1],s[2]) for s,p in zip(source,person,strict=True)]
    identities=person_alias_judgments(documents)
    postings=base.bm25_index(documents)
    queries=prepare_views(documents)
    reader,sentences,info=None,None,{}
    if gpu_model:
        reader, preprocess, info=reader_session(gpu_model)
        sentences=[]
        for doc in documents:
            text=[" ".join(row) for row in doc["sents"]]
            sentences.append((text,preprocess(text)))
    scores=defaultdict(lambda:defaultdict(list))
    groups=defaultdict(list)
    for method in ARMS:
        judge(queries[0],method,postings,lookups,profiles,identities,reader,sentences)
    for i,query in enumerate(queries):
        style=query["style"]
        groups[style].append(query["group"])
        offset=i%len(ARMS)
        for method in ARMS[offset:]+ARMS[:offset]:
            scores[style][method].append(judge(query,method,postings,lookups,profiles,identities,reader,sentences))
    reports={}
    fields=("opened","fallback","exact_key","source","available","candidates","person_mentions_scored",
            "strict_positive","proposed_positive","proposed_rival")
    if reader:
        fields+=("accepted_positive","accepted_rival","accepted_unjudged")
    outcome="accepted_positive" if reader else "proposed_positive"
    for style,methods in scores.items():
        reports[style]={}
        reference=methods["auto_bm25"]
        for method, rows in methods.items():
            report={field:sum(r[field] for r in rows) for field in fields}
            report.update(queries=len(rows), p50_p95_ms=np.quantile([r["ms"] for r in rows],[.5,.95]).tolist(),
                          positive_p5_delta_vs_auto_bm25=clustered_paired_interval(
                              [(x[outcome]-y[outcome])/base.TOP for x,y in zip(rows,reference,strict=True)],groups[style]))
            reports[style][method]=report
    base.write_json(output,dict(
        inputs=base.HASHES, code_sha256=base.digest(Path(__file__)), ner_projection_sha256=base.digest(extracts),
        ner_model=MODEL, ner_version=MODEL_VERSION, engine_version=extracted["engine_version"],
        extraction_seconds=extracted["seconds"], documents=len(documents),
        aliases=dict(annotated=len(oracle_aliases),automatic=len(automatic_aliases)),
        people=dict(annotated=sum(len(p[0]) for p in oracle_profiles),automatic=sum(len(p[0]) for p in automatic_profiles)),
        panels=reports, **info,
    ))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    commands=cli.add_subparsers(dest="command",required=True)
    for name,args in (("extract",("snapshot","output")),("run",("snapshot","extracts","output"))):
        cmd=commands.add_parser(name)
        for arg in args:
            cmd.add_argument(arg,type=Path)
        if name=="run":
            cmd.add_argument("--gpu-model",type=Path)
    args=vars(cli.parse_args())
    {"extract":extract,"run":run}[args.pop("command")](**args)
