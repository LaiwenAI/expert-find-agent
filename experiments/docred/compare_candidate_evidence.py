"""Evaluate a joint evidence decision for the last expert candidate slot."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as proposal
from experiments.docred import noisy_role_source_routing as routing
from experiments.docred import oracle_role_evidence as oracle
from experiments.docred import person_bound_evidence as evidence


ARMS = ("keep","replace","independent","joint","joint_reversed")
JOINT = (
    "Check both candidates independently against the work and requested role. Use each candidate's own "
    "excerpt only. A different role or someone else's contribution is insufficient. Choose the person "
    "with the stronger explicit support. If neither is explicitly supported, answer 0. "
    "Answer only 1 for candidate 1, 2 for candidate 2, or 0 for neither."
)


def alternative(query,docs,documents,profiles,extracts,old):
    if not old:
        return None
    used=set(old)
    expanded={}
    for d in docs:
        _, work, cues=profiles[d]
        names=[(name,tuple(sorted(spans))) for name,spans,_ in proposal.pool(documents[d],query,profiles[d],extracts[d])
               if (d,name) not in used]
        expanded[d]=(names,work,cues)
    ranked,_=index.rank_people(query,docs,expanded)
    return ranked[0] if ranked else None


def prompt_data(query,pair,content):
    doc,name=pair
    text,profile=content[doc]
    person=" ".join(name)
    work=" ".join(query["work"])
    requested=evidence.ROLES[query["role"]]
    case=dict(work=work,role=query["role"])
    positions=evidence.choose(case,person,profile,"lexical_two") if len(text)>=2 else range(len(text))
    context="\n".join(text[i] for i in positions)
    question=f"Was {person} credited in the role of {requested} for {work}?"
    return person,question,context


def reader_factory(checkpoint):
    import torch

    tokenizer, model, provenance=oracle.load_reader(checkpoint)
    ids=[tokenizer.encode(i,add_special_tokens=False) for i in ("0","1","2")]
    if any(len(t)!=1 for t in ids):
        raise ValueError("Require single-token decision labels")
    answer_ids=[t[0] for t in ids]
    if len(set(answer_ids))!=3:
        raise ValueError("Require distinct decision tokens")

    def make(system,user):
        return tokenizer.apply_chat_template(
            [{"role":"system","content":system},{"role":"user","content":user}],
            tokenize=True,add_generation_prompt=True,return_dict=False,
        )

    @torch.inference_mode()
    def read(query, values, pair=None):
        work=" ".join(query["work"])
        name=evidence.ROLES[query["role"]]
        encoded=[make(oracle.SYSTEM,f"Source:\n{doc}\n\nQuestion: {text}") for _,text,doc in values]
        if pair is not None:
            (a,_,left),(b,_,right)=pair
            encoded.append(make(JOINT,
                f"Work: {work}\nRole: {name}\nCandidate 1: {a}\nExcerpt 1:\n{left}\n\n"
                f"Candidate 2: {b}\nExcerpt 2:\n{right}"))
        if not encoded:
            return [], None, (0,0,0)
        largest=max(map(len,encoded))
        if largest>model.config.max_position_embeddings:
            raise ValueError("Reader context overflow")
        inputs=tokenizer.pad([{"input_ids":x} for x in encoded],padding=True,
                             padding_side="left",return_tensors="pt").to("cuda")
        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,answer_ids].float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite role comparison outputs")
        binary=(logits[:len(values),1]-logits[:len(values),0]).tolist()
        joint=logits[-1].tolist() if pair is not None else None
        return binary,joint,(len(encoded),sum(map(len,encoded)),len(encoded)*largest)

    return read,provenance["revision"]


def action(flags,last,option,arm):
    if not flags or option is None:
        return None
    if arm=="independent":
        incumbent, challenger=flags[-2:]
        if max(incumbent,challenger)<=auto.CUTOFF:
            return None
        return option if challenger>incumbent else last
    if arm=="replace":
        return option if flags[-1]>auto.CUTOFF else None
    return last if flags[-1]>auto.CUTOFF else None


def evaluate(query,docs,documents,profiles,extracts,context,reader,arm):
    baseline=index.rank_people(query,docs,profiles)[0]
    if not baseline:
        return [],(0,0,0),(),None
    most=baseline[:-1]
    incumbent=baseline[-1]
    challenger=alternative(query,docs,documents,profiles,extracts,baseline) if arm!="keep" else None
    primary=[prompt_data(query,pair,context) for pair in most]
    a=prompt_data(query,incumbent,context)
    b=prompt_data(query,challenger,context) if challenger is not None else None
    if b is None:
        values, _, sizes=reader(query,primary+[a])
        last=incumbent if values[-1]>auto.CUTOFF else None
    elif arm in ("joint","joint_reversed"):
        reverse=bool(routing.stable(query["group"],query["work"],query["role"])[0]%2)
        reverse ^= arm=="joint_reversed"
        variants=(b,a) if reverse else (a,b)
        output=(challenger,incumbent) if reverse else (incumbent,challenger)
        values, logits, sizes=reader(query,primary,pair=variants)
        # Abstain if neither is strictly lower than the better candidate.
        choice=max((1,2),key=lambda i:(logits[i],output[i-1]==incumbent))
        last=output[choice-1] if logits[choice]-logits[0]>auto.CUTOFF else None
    else:
        tail=[a,b] if arm=="independent" else [b]
        values, _, sizes=reader(query,primary+tail)
        last=action(values,incumbent,challenger,arm)
    selected=[pair for pair,margin in zip(most,values[:len(most)],strict=True) if margin>auto.CUTOFF]
    common=tuple(selected)
    if last:
        selected.append(last)
    return selected,sizes,common,last


def observe(views,documents,profiles,extracts,aliases,postings,context,reader):
    data=defaultdict(lambda:defaultdict(list))
    for i,q in enumerate(views):
        offset=i%len(ARMS)
        indexed={}
        for arm in ARMS[offset:]+ARMS[:offset]:
            start=perf_counter()
            docs=proposal.noisy_sources(q,postings,aliases)
            final,sizes,common,last=evaluate(q,docs,documents,profiles,extracts,context,reader,arm)
            elapsed=(perf_counter()-start)*1000
            indexed[arm]=(final,sizes,elapsed,common,last)
        for arm,(final,sizes,elapsed,common,last) in indexed.items():
            ordered=set(final)
            resolved={(doc,aliases_for_names.get(name,name)) for doc,name in ordered
                      for aliases_for_names in [context["identity"][doc]]}
            pos=q["positive"]
            rival=q["rivals"]
            data[q["style"]][arm].append(dict(
                group=q["group"], accepted_positive=len(resolved&pos), accepted_rival=len(resolved&rival),
                accepted_unjudged=len(resolved-pos-rival), ms=elapsed,
                prompts=sizes[0], tokens=sizes[1], padded_tokens=sizes[2],
                differs_when_swapped=int(arm=="joint" and indexed["joint_reversed"][4]!=last),
                earlier_slots_changed=int(common!=indexed["keep"][3]),
            ))
    return data


def report(data):
    panels={}
    fields=("accepted_positive","accepted_rival","accepted_unjudged","prompts","tokens",
            "padded_tokens","differs_when_swapped","earlier_slots_changed")
    for style,arms in data.items():
        baseline=arms["keep"]
        groups=[r["group"] for r in baseline]
        panels[style]={}
        for arm,rows in arms.items():
            assert [r["group"] for r in rows]==groups
            totals={key:sum(r[key] for r in rows) for key in fields}
            differences=[(a["accepted_positive"]-b["accepted_positive"])/index.TOP
                         for a,b in zip(rows,baseline,strict=True)]
            panels[style][arm]=totals|dict(queries=len(rows),
                p95_ms=float(np.quantile([r["ms"] for r in rows],.95)),
                accepted_p5_delta_vs_keep=clustered_paired_interval(differences,groups))
    return panels


def run(snapshot,extracts,checkpoint,output):
    import os
    import torch
    if output.exists() or not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ:
        raise ValueError("Require a new Slurm CUDA experiment")
    documents=auto.load_documents(snapshot)
    projection=json.loads(extracts.read_bytes())
    if projection["inputs"]!=index.HASHES or len(projection["annotations"])!=len(documents):
        raise ValueError("Incorrect NER projection")
    spans=projection["annotations"]
    _, annotated=index.mentions_index(documents)
    aliases,profiles=auto.auto_index(documents,spans,annotated)
    aliased=routing.AliasLookup(aliases)
    postings=index.bm25_index(documents)
    context={}
    for i,doc in enumerate(documents):
        text=[" ".join(line) for line in doc["sents"]]
        context[i]=(text,evidence.profile(text))
    context["identity"]=auto.person_alias_judgments(documents)
    reader,revision=reader_factory(checkpoint)
    views=auto.prepare_views(documents)
    observe(views[:5],documents,profiles,spans,aliased,postings,context,reader)
    measured=observe(views,documents,profiles,spans,aliased,postings,context,reader)
    index.write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),checkpoint=revision,
        code_sha256=index.digest(Path(__file__)),projection_sha256=index.digest(extracts),inputs=index.HASHES,
        cutoff=auto.CUTOFF,panels=report(measured),
    ))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot","extracts","checkpoint","output"):
        cli.add_argument(name,type=Path)
    run(**vars(cli.parse_args()))
