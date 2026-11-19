"""Promote raw-text name candidates near role cues inside selected sources."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import global_role_search as search
from experiments.docred import noisy_role_source_routing as noisy


ARMS = ("automatic", "unified", "guarded_reserve", "boundary_reserve")
CONNECTORS = frozenset(("de", "da", "van", "von", "del", "di", "la", "le", "bin", "al", "of"))
NAME = re.compile(r"^[A-Z][A-Za-z]+(?:[-’'][A-Za-z]+)*$")
LEADS = frozenset(("the", "a", "an", "in", "on", "for", "by", "this", "that", "it", "he", "she",
                   "they", "after", "before", "when", "while", "from", "during", "and", "also"))
CUTOFF = .25


def plausible_names(sentence):
    i=0
    while i<len(sentence):
        if not NAME.fullmatch(sentence[i]) or sentence[i].casefold() in LEADS:
            i+=1
            continue
        end=i+1
        while end<len(sentence):
            token=sentence[end]
            if NAME.fullmatch(token):
                end+=1
            elif token in CONNECTORS and end+1<len(sentence) and NAME.fullmatch(sentence[end+1]):
                end+=2
            else:
                break
        name=search.words(" ".join(sentence[i:end]))
        if 2<=sum(token not in CONNECTORS for token in name)<=4 and len(name)<=6:
            yield name,i,end
        i=end


def competing_names(doc, query, automatic):
    candidates=defaultdict(set)
    cue_terms=set(search.words(search.ROLES[query["role"]]))
    cue_locations, existing=automatic[2],{name for name,_ in automatic[0]}
    sentences=defaultdict(list)
    for cue in cue_terms:
        for s,j in cue_locations.get(cue,()):
            sentences[s].append(j)
    for sent, positions in sentences.items():
        for name,start,end in plausible_names(doc["sents"][sent]):
            if name in existing or name==query["work"]:
                continue
            distance=min(min(abs(start-j),abs(end-1-j)) for j in positions)
            if distance<=5:
                candidates[name].add((sent,start,end,distance))
    return candidates


def boundary_mentions(doc, annotations, work, existing):
    proposed=defaultdict(set)
    for sent, start, end, label in annotations:
        if label not in ("PERSON","ORG","PRODUCT","FAC","WORK_OF_ART"):
            continue
        sentence=doc["sents"][sent]
        for a in range(start,end):
            if not NAME.fullmatch(sentence[a]) or sentence[a].casefold() in LEADS:
                continue
            for b in range(a+1,min(end,a+4)+1):
                span=sentence[a:b]
                if not NAME.fullmatch(span[-1]) or any(
                        not NAME.fullmatch(token) and token not in CONNECTORS for token in span):
                    continue
                if len(span)==1 and (len(span[0])<4 or label!="PERSON"):
                    continue
                name=search.words(" ".join(span))
                if name and name not in existing and name!=work:
                    proposed[name].add((sent,a,b))
    return sorted((name,tuple(sorted(positions))) for name,positions in proposed.items())


def propose(query, indices, documents, automatic, extracts, policy):
    if policy=="automatic":
        return search.rank_people(query,indices,automatic)[0]
    changed={}
    strict={}
    for d in indices:
        people, sources, cues=automatic[d]
        if policy=="boundary_reserve":
            existing={name for name,_ in people}
            strict[d]=(boundary_mentions(documents[d],extracts[d],query["work"],existing),sources,cues)
            continue
        extra=competing_names(documents[d],query,automatic[d])
        changed[d]=(people+[(name,tuple(sorted((i,a,b) for i,a,b,_ in occurrences)))
                            for name,occurrences in sorted(extra.items())],sources,cues)
        strict[d]=([ (name,tuple(sorted((i,a,b) for i,a,b,dist in occurrences if dist<=3)))
                     for name,occurrences in sorted(extra.items()) if any(x[3]<=3 for x in occurrences)],
                   sources,cues)
    if policy=="unified":
        return search.rank_people(query,indices,changed)[0]
    baseline=search.rank_people(query,indices,automatic)[0]
    supplemental=search.rank_people(query,indices,strict)[0]
    if not supplemental:
        return baseline
    return baseline[:search.TOP-1]+[supplemental[0]]


def measure(queries, documents, postings, lookup, automatic, extracts, score, context, identity):
    rows=defaultdict(lambda:defaultdict(list))
    for i, query in enumerate(queries):
        offset=i%len(ARMS)
        for arm in ARMS[offset:]+ARMS[:offset]:
            start=perf_counter()
            docs,_,_=noisy.route(query,postings,lookup,"swap_then_guard")
            pairs=propose(query,docs,documents,automatic,extracts,arm)
            margins=score(query,pairs,context) if score else None
            elapsed=(perf_counter()-start)*1000
            refs=[(d,identity[d].get(name,name)) for d,name in pairs]
            positives=query["positive"]
            rivals=query["rivals"]
            # Score a person-source identity once, even if multiple emitted
            # strings are judged aliases of the same individual.
            proposed=set(refs)
            data=dict(proposed_pairs=len(pairs), ranked_positive=len(proposed & positives),
                      ranked_rivals=len(proposed & rivals), ms=elapsed, group=query["group"])
            if margins is not None:
                accepted={p for p,margin in zip(refs,margins,strict=True) if margin>CUTOFF}
                data.update(accepted_positive=len(accepted & positives), accepted_rival=len(accepted & rivals),
                            accepted_unjudged=len(accepted-positives-rivals))
            rows[query["style"]][arm].append(data)
    return rows


def summarize(data, gpu):
    output={}
    outcomes=("proposed_pairs","ranked_positive","ranked_rivals")
    target="ranked_positive"
    if gpu:
        outcomes+=("accepted_positive","accepted_rival","accepted_unjudged")
        target="accepted_positive"
    for style,arms in data.items():
        baseline=arms["automatic"]
        groups=[row["group"] for row in baseline]
        output[style]=dict()
        for name,rows in arms.items():
            assert [row["group"] for row in rows]==groups
            metrics={field:sum(row[field] for row in rows) for field in outcomes}
            deltas=[(a[target]-b[target])/search.TOP for a,b in zip(rows,baseline,strict=True)]
            metrics.update(queries=len(rows),p50_p95_ms=np.quantile([r["ms"] for r in rows],[.5,.95]).tolist(),
                           labeled_p5_delta_vs_automatic=clustered_paired_interval(deltas,groups))
            output[style][name]=metrics
    return output


def missing_diagnostic(views, documents, postings, lookup, automatic, extracts, identities):
    counts=Counter()
    for q in views:
        if q["style"]!="exact":
            continue
        docs=noisy.route(q,postings,lookup,"swap_then_guard")[0]
        gold={(d,name) for d,name in q["positive"] if d in docs}
        present={(d,identities[d].get(name,name)) for d in docs for name,_ in automatic[d][0]}
        if not gold or gold & present:
            continue
        counts["found_source_but_missing_person"]+=1
        found=set()
        for d, person in gold:
            aliases={name for name,canonical in identities[d].items() if canonical==person and len("".join(name))>=3}
            for s,a,b,label in extracts[d]:
                n=search.words(" ".join(documents[d]["sents"][s][a:b]))
                if n in aliases:
                    found.add("exact_span_typed_"+label)
                if any(any(n[i:i+len(alias)]==alias for i in range(len(n)-len(alias)+1)) for alias in aliases):
                    found.add("inside_a_predicted_span")
            existing={name for name,_ in automatic[d][0]}
            if any(n in aliases for n,_ in boundary_mentions(documents[d],extracts[d],q["work"],existing)):
                found.add("boundary_proposal_available")
            if competing_names(documents[d],q,automatic[d]).keys() & aliases:
                found.add("role_local_proposal_available")
        counts.update(found)
    return dict(counts)


def run(snapshot,extracts,output,gpu_model=None):
    if output.exists():
        raise FileExistsError(output)
    documents=auto.load_documents(snapshot)
    extracted=json.loads(extracts.read_bytes())
    if extracted["inputs"]!=search.HASHES or extracted["model_version"]!=auto.MODEL_VERSION:
        raise ValueError("Changed automatic NER projection")
    _, annotation_profiles=search.mentions_index(documents)
    aliases,automatic=auto.auto_index(documents,extracted["annotations"],annotation_profiles)
    lookup=noisy.AliasLookup(aliases)
    postings=search.bm25_index(documents)
    views=auto.prepare_views(documents)
    identities=auto.person_alias_judgments(documents)
    score,context,info=None,None,{}
    if gpu_model:
        score, preprocess, info=auto.reader_session(gpu_model)
        context=[]
        for doc in documents:
            text=[" ".join(sentence) for sentence in doc["sents"]]
            context.append((text,preprocess(text)))
        if info["cutoff"]!=CUTOFF:
            raise ValueError("Changed frozen decision threshold")
    spans=extracted["annotations"]
    measure(views[:8],documents,postings,lookup,automatic,spans,score,context,identities)
    measured=measure(views,documents,postings,lookup,automatic,spans,score,context,identities)
    search.write_json(output,dict(inputs=search.HASHES, code_sha256=search.digest(Path(__file__)),
                                   ner_projection_sha256=search.digest(extracts),source_code_sha256=search.digest(Path(auto.__file__)),
                                   panels=summarize(measured,bool(gpu_model)),
                                   missing_exact_diagnostic=missing_diagnostic(views,documents,postings,lookup,automatic,spans,identities),
                                   **info))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot","extracts","output"):
        cli.add_argument(name,type=Path)
    cli.add_argument("--gpu-model",type=Path)
    run(**vars(cli.parse_args()))
