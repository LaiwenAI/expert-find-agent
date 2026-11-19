"""Compare joint and source-factored first-query contributor parsing."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.source_scoped_credit import EvidenceIndex
from experiments.followir.core17_reranking import digest, write_json


FIXTURE = "f04b25afbefd303e61a191506b9a28164a4a9b36d6fc8d210f9c269194ac9c0d"
NISO = "c107b4e780218e58aec4e2a13f2d7072533962a6a73b82d23e4273b43aafd528"
ARMS = ("joint_full", "model_scope_and_role", "factored_full", "factored_local", "oracle")
FORMS = ("explicit", "anaphoric")
SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZab"
ALIASES = ("Cobalt", "Amber")


def stable(*parts):
    return sha256(json.dumps(parts).encode()).digest()


def interval_rows(case, version):
    original = [*case["roles_left"],case["role_before"]]
    if version == "before":
        return [(side, role) for side, role in zip((0,0,1),original,strict=True)]
    if case["category"] == "scope_swap":
        return [(int(role==case["role_after"]),role) for role in original]
    return list(zip((0,0,1),[*case["roles_left"],case["role_after"]],strict=True))


def examples(fixture, glossary):
    phrases = defaultdict(set)
    for row in glossary["examples"]:
        phrases[row["role"]].add(" ".join(row["text"].split()).rstrip(". "))
    result = []
    for case in fixture["pairs"]:
        order = sorted(range(3),key=lambda i:stable(case["id"],i))
        aliases = ALIASES if stable(case["id"],"alias")[0]%2==0 else ALIASES[::-1]
        for form in FORMS:
            for version in ("before","after"):
                atoms = interval_rows(case,version)
                atoms = [atoms[i] for i in order]
                lines = []
                for i,(side,role) in enumerate(atoms,1):
                    phrase = min(phrases[role],key=lambda p:stable(case["id"],role,p))
                    if form == "explicit":
                        sentence = f"For {aliases[side]}, the contribution must be: {phrase}."
                    else:
                        reference = ("former","latter")[side]
                        sentence = f"{phrase}. This contribution must be documented in the {reference} report."
                    lines.append(f"[{i}] {sentence}")
                header = (f"The two reports are {aliases[0]} and {aliases[1]}, in that order. "
                          "Find one person who satisfies all three contribution requirements.")
                result.append(dict(family=case["id"],category=case["category"],version=version,
                                   form=form,dois=(case["left"],case["right"]),aliases=aliases,
                                   text="\n".join([header,*lines]), atoms=atoms, owners=case[version]))
    return result


def scope_atoms(text, aliases):
    header, *lines = text.splitlines()
    if not header.startswith(f"The two reports are {aliases[0]} and {aliases[1]}, in that order. "):
        raise ValueError("Unresolved source order")
    result = []
    for i,line in enumerate(lines,1):
        pattern = rf"\[{i}\] For ({'|'.join(aliases)}), the contribution must be: (.+)\."
        declared = re.fullmatch(pattern,line)
        relative = re.fullmatch(
            rf"\[{i}\] (.+)\. This contribution must be documented in the (former|latter) report\.",line)
        if declared:
            alias,phrase = declared.groups()
            result.append((aliases.index(alias),phrase))
        elif relative:
            phrase,reference = relative.groups()
            result.append((int(reference=="latter"),phrase))
        else:
            raise ValueError("Unresolved source anchor")
    if len(result)!=3:
        raise ValueError("Expected three requirements")
    return result


def make_prompt(tokenizer, glossary, codebook, text, number=None, joint=False):
    if number is None:
        target = f"Classify the research contribution described here:\n{text}"
    else:
        task = ("Select the source report and the contributor role together" if joint
                else "Classify just the contribution")
        target = f"{task} at position [{number}] in this request:\n{text}"
    messages = [
        dict(role="system",content="Use the glossary and the answer codes. Return only the best answer code."),
        dict(role="user",content=f"Glossary:\n{glossary}\n\nAnswer codes:\n{codebook}\n\n{target}"),
    ]
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)


def route(tokenizer, model, digits, definitions, item, mode, roles, balanced):
    import torch

    text = item["text"]
    if mode=="joint_full":
        sides = (1,0) if balanced and stable(item["family"],"joint-side")[0]%2 else (0,1)
        options = [(side,role) for side in sides for role in roles]
        labels = [f"{item['aliases'][side]} / {role}" for side,role in options]
        prompts_text = [(text,n) for n in (1,2,3)]
    else:
        if mode != "model_scope_and_role":
            anchored = scope_atoms(text,item["aliases"])
        options,labels = roles,roles
        prompts_text = ([(text,n) for n in (1,2,3)] if mode in ("factored_full", "model_scope_and_role")
                        else [(phrase,None) for _,phrase in anchored])
    codebook = "\n".join(f"{tag}: {label}" for tag,label in zip(SYMBOLS,labels))
    prompts = [make_prompt(tokenizer,definitions,codebook,t,n,balanced and mode=="joint_full")
               for t,n in prompts_text]
    if mode == "model_scope_and_role":
        aliases = item["aliases"]
        for number in (1,2,3):
            messages = [
                dict(role="system", content="Identify the report required by just the numbered clause. Return A or B."),
                dict(role="user", content=f"A: {aliases[0]}\nB: {aliases[1]}\n"
                                          f"Target position [{number}]. Request:\n{text}"),
            ]
            prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    encoded = tokenizer(prompts,padding=True,return_tensors="pt",add_special_tokens=False).to("cuda")
    positions = (encoded["attention_mask"].cumsum(1)-1).clamp(min=0)
    with torch.inference_mode():
        scores = model(**encoded,position_ids=positions,logits_to_keep=1,use_cache=False).logits[:, -1,digits[:len(options)]]
        scores = scores.float().cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite parser scores")
    prediction = [options[i] for i in scores[:3].argmax(axis=1)]
    if mode == "model_scope_and_role":
        sides = scores[3:,:2].argmax(axis=1).tolist()
        prediction = list(zip(sides,prediction,strict=True))
    elif mode != "joint_full":
        prediction = [(anchor[0],role) for anchor,role in zip(anchored,prediction,strict=True)]
    return prediction,int(encoded["attention_mask"].sum().item())


def attributed(idx, dois, predicted):
    left,right = dois
    possible = idx.roles[left].keys() & idx.roles[right].keys()
    for side,role in predicted:
        possible &= idx.postings.get((dois[side],role),set())
    return sorted(possible)


def summarize(rows):
    reports = {}
    groups = {"all":rows}
    groups.update({name:[row for row in rows if row["form"]==name] for name in FORMS})
    groups.update({name:[row for row in rows if row["category"]==name]
                   for name in ("label_flip","stable","scope_swap")})
    for name,cohort in groups.items():
        arms = {}
        for arm in ARMS:
            counts = Counter()
            paired = defaultdict(list)
            for row in cohort:
                observed = row["arms"][arm]
                counts.update({name:int(observed[name]) for name in (
                    "scope_atoms_correct","role_atoms_correct","scoped_atoms_correct","plan_correct",
                    "expert_set_correct","unsupported_ids","missed_ids","joint_correct")})
                paired[row["family"],row["form"]].append(observed["joint_correct"])
            arms[arm] = dict(counts, pairs_exact=sum(all(pair) and len(pair)==2 for pair in paired.values()),
                             p50_p95_ms=np.quantile([r["arms"][arm]["ms"] for r in cohort],[.5,.95]).tolist(),
                             mean_tokens=float(np.mean([r["arms"][arm]["tokens"] for r in cohort])))
        reports[name] = dict(queries=len(cohort),pairs=len(cohort)//2,arms=arms)
    return reports


def run(fixture_path, examples_path, checkpoint, output, balanced_joint=False):
    import torch

    from experiments.qasper.paired_reader import load_reader

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a new Slurm CUDA run")
    if digest(fixture_path)!=FIXTURE or digest(examples_path)!=NISO:
        raise ValueError("Changed fixtures")
    fixture, glossary = (json.loads(path.read_text()) for path in (fixture_path,examples_path))
    idx = EvidenceIndex(fixture["sources"])
    cohort = examples(fixture,glossary)
    if len(cohort)!=240:
        raise ValueError("Changed query cohort")
    tokenizer, model, provenance = load_reader(checkpoint)
    tokenizer.padding_side = "left"
    encoded = [tokenizer.encode(s,add_special_tokens=False) for s in SYMBOLS]
    if any(len(option)!=1 for option in encoded) or len({option[0] for option in encoded})!=len(SYMBOLS):
        raise ValueError("Require 28 distinct decision tokens")
    digits = [option[0] for option in encoded]
    definitions = "\n".join(f"{role}: {glossary['definitions'][role]}" for role in ROLES)
    complete = []
    for i, item in enumerate([cohort[0],*cohort]):
        offset = stable(item["family"],"roles")[0]%len(ROLES)
        roles = ROLES[offset:]+ROLES[:offset]
        order = ARMS[i%len(ARMS):]+ARMS[:i%len(ARMS)]
        observations = {}
        for arm in order:
            started = perf_counter()
            if arm=="oracle":
                predicted,n = item["atoms"],0
            else:
                predicted,n = route(tokenizer,model,digits,definitions,item,arm,roles,balanced_joint)
            found = attributed(idx,item["dois"],predicted)
            ms = (perf_counter()-started)*1000
            desired,actual = set(item["owners"]),set(found)
            plan_correct = sorted(predicted)==sorted(item["atoms"])
            observations[arm] = dict(
                scope_atoms_correct=sum(a[0]==b[0] for a,b in zip(predicted,item["atoms"],strict=True)),
                role_atoms_correct=sum(a[1]==b[1] for a,b in zip(predicted,item["atoms"],strict=True)),
                scoped_atoms_correct=sum(a==b for a,b in zip(predicted,item["atoms"],strict=True)),
                plan_correct=plan_correct, expert_set_correct=actual==desired,
                unsupported_ids=len(actual-desired),missed_ids=len(desired-actual),
                joint_correct=plan_correct and actual==desired, ms=ms,tokens=n,
            )
        if not observations["oracle"]["joint_correct"]:
            raise ValueError("Known scoped roles disagree with fixture")
        if i:
            complete.append({k:item[k] for k in ("family","category","form")} | {"arms":observations})
        if i and i%60==0:
            print(i, "queries complete", flush=True)
    write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),revision=provenance["revision"],
        fixture_sha256=FIXTURE,examples_sha256=NISO,script_sha256=digest(Path(__file__)),
        balanced_joint=balanced_joint,
        results=summarize(complete),
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fixture_path","examples_path","checkpoint","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--balanced-joint",action="store_true")
    run(**vars(parser.parse_args()))
