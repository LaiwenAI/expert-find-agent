"""Learn which additional person name can replace a fixed-budget candidate."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import global_role_search as index
from experiments.docred import noisy_role_source_routing as routing
from experiments.docred import role_local_person_rescue as rescue


ARMS = ("automatic","fixed_guard","linear_reserve","mlp_reserve","guarded_linear","guarded_mlp")
ROLE_ORDER = tuple(sorted(index.ROLES))
SEED = 381027
STEPS = 120


def pool(doc, query, profile, extracts):
    original=dict(profile[0])
    nearby=rescue.competing_names(doc,query,profile)
    boundary=dict(rescue.boundary_mentions(doc,extracts,query["work"],set(original)))
    names=sorted(original.keys()|nearby.keys()|boundary.keys())
    return [(name,
             set(original.get(name,()))|{(s,a,b) for s,a,b,_ in nearby.get(name,())}|set(boundary.get(name,())),
             (name in original,name in nearby,name in boundary))
            for name in names]


def candidate_features(doc, query, profile, name, spans, origins, rank):
    work=profile[1].get(query["work"],set())
    cues=[p for word in index.words(index.ROLES[query["role"]]) for p in profile[2].get(word,())]
    minimum=20
    before=after=linked=0
    project_name_in_sentence=False
    for s,start,end in spans:
        raw=[token.casefold() for token in doc["sents"][s]]
        project_name_in_sentence|=s in work
        for line,position in cues:
            if line!=s:
                continue
            distance=min(abs(start-position),abs(end-1-position))
            minimum=min(minimum,distance)
            if position < start:
                before=max(before,1/(1+distance))
                bridge=raw[position+1:start]
                linked=max(linked, int(len(bridge)<=3 and ("by" in bridge or "as" in bridge)))
            elif position>=end:
                after=max(after,1/(1+distance))
    return np.array([
        *map(float,origins), min(4,len(name))/4, min(len(spans),4)/4, 1/(1+rank), 1/(1+minimum),
        before,after,linked,float(project_name_in_sentence),
        float(any(s in work and any(cs==s for cs,_ in cues) for s,_,_ in spans)),
        *[float(query["role"]==role) for role in ROLE_ORDER],
    ], dtype=np.float32)


def candidates(query, sources, documents, profiles, extracts):
    features={}
    for rank,d in enumerate(sources):
        for name,spans,origins in pool(documents[d],query,profiles[d],extracts[d]):
            features[d,name]=candidate_features(documents[d],query,profiles[d],name,spans,origins,rank)
    return features


def contrast_set(queries, documents, profiles, extracts, identities, postings, lookup):
    examples=[]
    for q in queries:
        docs=noisy_sources(q,postings,lookup)
        vectors=candidates(q,docs,documents,profiles,extracts)
        positives,rivals=[],[]
        for (d,name), features in vectors.items():
            pair=d,identities[d].get(name,name)
            if pair in q["positive"]:
                positives.append((d,features))
            elif pair in q["rivals"]:
                rivals.append((d,features))
        for d,good in positives:
            for other,bad in rivals:
                if d==other:
                    examples.append((q["group"],good,bad))
    return examples


def noisy_sources(query, postings, lookup):
    return routing.route(query,postings,lookup,"swap_then_guard")[0]


def fit_pairs(examples):
    import torch
    from torch import nn

    groups=sorted({source for source,_,_ in examples},key=lambda i:routing.stable("validation",i))
    n_validation=max(1,len(groups)//5)
    validation=set(groups[:n_validation])
    train=[row for row in examples if row[0] not in validation]
    valid=[row for row in examples if row[0] in validation]
    if len(train)<50 or not valid:
        raise ValueError("Insufficient independently labeled contrasts")
    size=len(train[0][1])
    train_good=torch.as_tensor(np.stack([r[1] for r in train]),device="cuda")
    train_bad=torch.as_tensor(np.stack([r[2] for r in train]),device="cuda")
    val_good=torch.as_tensor(np.stack([r[1] for r in valid]),device="cuda")
    val_bad=torch.as_tensor(np.stack([r[2] for r in valid]),device="cuda")
    counts={}
    scorers={}
    for name in ("linear_reserve","mlp_reserve"):
        torch.manual_seed(SEED)
        m=nn.Linear(size,1) if name=="linear_reserve" else nn.Sequential(nn.Linear(size,32),nn.Tanh(),nn.Linear(32,1))
        m=m.cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=.015,weight_decay=.01)
        best=None
        start=perf_counter()
        for step in range(1,STEPS+1):
            opt.zero_grad(set_to_none=True)
            margin=m(train_good)-m(train_bad)
            loss=torch.nn.functional.softplus(-margin).mean()
            loss.backward()
            opt.step()
            if step%10==0:
                with torch.inference_mode():
                    dif=m(val_good)-m(val_bad)
                    criterion=float(torch.nn.functional.softplus(-dif).mean())
                    order=float((dif>0).float().mean())
                if best is None or criterion<best[0]:
                    best=(criterion,step,order,{k:v.detach().clone() for k,v in m.state_dict().items()})
        torch.cuda.synchronize()
        elapsed=perf_counter()-start
        m.load_state_dict(best[3])
        counts[name]=dict(parameters=sum(p.numel() for p in m.parameters()), validation_loss=best[0],
                          selected_step=best[1], validation_pair_order=best[2], gpu_training_seconds=elapsed)
        m=m.cpu().eval()
        weights=[p.detach().numpy().copy() for p in m.parameters()]
        if name=="linear_reserve":
            def score(x,w=weights):
                return (x@w[0].T+w[1]).ravel()
        else:
            def score(x,w=weights):
                return (np.tanh(x@w[0].T+w[1])@w[2].T+w[3]).ravel()
        scorers[name]=score
    return scorers,validation,dict(contrasts=len(examples),train_pairs=len(train),validation_pairs=len(valid),
                                   train_reference_docs=len(groups)-n_validation,validation_reference_docs=n_validation,
                                   models=counts)


def rank_replacement(features, incumbent, base, score, guarded):
    others=sorted(features.keys()-set(base))
    if guarded:
        others=[pair for pair in others if (features[pair][0] or features[pair][1]) and features[pair][6]>=.25]
    if not others:
        return incumbent, 0.
    choices=[incumbent,*others]
    scores=score(np.stack([features[pair] for pair in choices]))
    selected=int(np.argmax(scores))
    return choices[selected],float(scores[selected]-scores[0])


def calibrate_gate(queries, documents, profiles, extracts, identities, postings, lookup, score):
    observations=[]
    for query in queries:
        docs=noisy_sources(query,postings,lookup)
        prior=index.rank_people(query,docs,profiles)[0]
        if not prior:
            continue
        features=candidates(query,docs,documents,profiles,extracts)
        alternative, margin=rank_replacement(features,prior[-1],prior,score,True)
        if margin<=0:
            continue
        existing={(d,identities[d].get(name,name)) for d,name in prior[:-1]}
        before_d,before_name=prior[-1]
        after_d,after_name=alternative
        before=existing|{(before_d,identities[before_d].get(before_name,before_name))}
        after=existing|{(after_d,identities[after_d].get(after_name,after_name))}
        support,rivals=query["positive"],query["rivals"]
        observations.append((margin,len(after&support)-len(before&support),len(after&rivals)-len(before&rivals)))
    cap=int(.30*len(queries))
    choices=[(0,0,0,max((m for m,_,_ in observations),default=0.))]
    for threshold in sorted({0.,*(m for m,_,_ in observations)}):
        moves=[(gain,rivals) for score,gain,rivals in observations if score>threshold]
        gain=sum(value for value,_ in moves)
        rivals=sum(value for _,value in moves)
        if len(moves)<=cap and rivals<=0:
            choices.append((gain,-rivals,-len(moves),threshold))
    gain,rival_reduction,negative_count,threshold=max(choices)
    return dict(threshold=threshold,validation_queries=len(queries),replacement_cap=cap,
                actual_validation_replacements=-negative_count,validation_known_gain=gain,
                validation_rival_change=-rival_reduction)


def select(query,docs,documents,profiles,extracts,method,base,score,threshold):
    if method=="automatic" or not base:
        return base
    if method=="fixed_guard":
        proposed=rescue.propose(query,docs,documents,profiles,extracts,"guarded_reserve")
        return base[:-1]+[proposed[-1]] if proposed and proposed[-1] not in base else base
    features=candidates(query,docs,documents,profiles,extracts)
    if base[-1] not in features:
        raise ValueError("Baseline candidate lost from training feature space")
    candidate, margin=rank_replacement(features,base[-1],base,score,threshold is not None)
    return base if threshold is not None and margin<=threshold else base[:-1]+[candidate]


def evaluate(views,documents,profiles,extracts,identities,postings,lookup,scorers,gates,reader,context):
    result=defaultdict(lambda:defaultdict(list))
    for i,q in enumerate(views):
        offset=i%len(ARMS)
        for method in ARMS[offset:]+ARMS[:offset]:
            begin=perf_counter()
            docs=noisy_sources(q,postings,lookup)
            prior=index.rank_people(q,docs,profiles)[0]
            cutoff=gates.get(method,{}).get("threshold")
            pairs=select(q,docs,documents,profiles,extracts,method,prior,scorers.get(method),cutoff)
            if len(pairs)!=len(prior):
                raise ValueError("Verification budget changed")
            decided=perf_counter()
            margins=reader(q,pairs,context)
            elapsed=perf_counter()-begin
            grounded=[(d,identities[d].get(name,name)) for d,name in pairs]
            known=q["positive"]
            rivals=q["rivals"]
            accepted={pair for pair,margin in zip(grounded,margins,strict=True) if margin>auto.CUTOFF}
            result[q["style"]][method].append(dict(
                group=q["group"], pairs=len(pairs), replaced=pairs!=prior,
                proposed_positive=len(set(grounded)&known), accepted_positive=len(accepted&known),
                accepted_rival=len(accepted&rivals), accepted_unjudged=len(accepted-known-rivals),
                decision_ms=(decided-begin)*1000, total_ms=elapsed*1000,
            ))
    return result


def aggregate(measured):
    panels={}
    for style,arms in measured.items():
        baseline=arms["automatic"]
        groups=[row["group"] for row in baseline]
        panels[style]={}
        for method,rows in arms.items():
            totals={field:sum(row[field] for row in rows) for field in
                    ("pairs","replaced","proposed_positive","accepted_positive","accepted_rival","accepted_unjudged")}
            assert totals["pairs"]==sum(row["pairs"] for row in baseline)
            deltas=[(a["accepted_positive"]-b["accepted_positive"])/index.TOP for a,b in zip(rows,baseline,strict=True)]
            panels[style][method]=totals|dict(
                queries=len(rows),decision_p95_ms=float(np.quantile([r["decision_ms"] for r in rows],.95)),
                full_p95_ms=float(np.quantile([r["total_ms"] for r in rows],.95)),
                accepted_p5_delta_vs_automatic=clustered_paired_interval(deltas,groups),
            )
    return panels


def run(snapshot,extracts,checkpoint,output):
    if output.exists():
        raise FileExistsError(output)
    documents=auto.load_documents(snapshot)
    projection=json.loads(extracts.read_bytes())
    if projection["inputs"]!=index.HASHES or len(projection["annotations"])!=len(documents):
        raise ValueError("Changed extraction projection")
    spans=projection["annotations"]
    _,annotated=index.mentions_index(documents)
    aliases,profiles=auto.auto_index(documents,spans,annotated)
    lookup=routing.AliasLookup(aliases)
    postings=index.bm25_index(documents)
    identities=auto.person_alias_judgments(documents)
    test=auto.prepare_views(documents)
    held_names={q.get("original",q["work"]) for q in test}
    dev=[q for q in index.study_queries(documents[:500],0) if q["work"] not in held_names]
    training=contrast_set(dev,documents,profiles,spans,identities,postings,lookup)
    reader,preprocess,info=auto.reader_session(checkpoint)
    scorers,validation_groups,train_summary=fit_pairs(training)
    validation=[q for q in dev if q["group"] in validation_groups]
    gates={}
    for name,original in (("guarded_linear","linear_reserve"),("guarded_mlp","mlp_reserve")):
        scorer=scorers[original]
        gates[name]=calibrate_gate(validation,documents,profiles,spans,identities,postings,lookup,scorer)
        scorers[name]=scorer
    context=[]
    for doc in documents:
        text=[" ".join(row) for row in doc["sents"]]
        context.append((text,preprocess(text)))
    evaluate(test[:8],documents,profiles,spans,identities,postings,lookup,scorers,gates,reader,context)
    measured=evaluate(test,documents,profiles,spans,identities,postings,lookup,scorers,gates,reader,context)
    index.write_json(output,dict(
        code_sha256=index.digest(Path(__file__)), projection_sha256=index.digest(extracts),
        input_sha256=index.HASHES, seed=SEED, development_queries=len(dev), training=train_summary,
        gates=gates, panels=aggregate(measured), **info,
    ))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot","extracts","checkpoint","output"):
        cli.add_argument(name,type=Path)
    run(**vars(cli.parse_args()))
