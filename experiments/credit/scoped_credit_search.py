"""Retrieve source descriptions and measure identity-linked expert evidence selection."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.retrieval import tokens
from experiments.credit.conjunctive_source_retrieval import sentences, whitespace
from experiments.credit.harvest_scoped_credit import authors, joined, meta, xml_sources
from experiments.credit.scoped_credit_language import (
    FIXTURE, NISO, SYMBOLS, attributed, examples, route, stable,
)
from experiments.credit.plos_role_inventory import ROLES, declared_roles
from experiments.credit.source_scoped_credit import EvidenceIndex, save
from experiments.followir.core17_reranking import digest, write_json


PROJECTION = "1b9d9b1fae5397b790be93df66b42d8295105f4a57ade1ca3618e6ee40d2c322"
DENSE_TASK = "Retrieve the specific research paper described by this clue."
ARMS = ("dense", "hybrid", "linked_top4", "linked_expand")
VIEWS = ("result", "keywords")
TAGS = ("present", "missing_left", "missing_right")
STOPS = set("and that with their from using results approach model system data this these were have method has can for was the".split())


def source_rows(fixture, roots):
    rows = {}
    for root in roots:
        for path in xml_sources(root):
            raw = path.read_bytes()
            node = meta(raw)
            doi = node.findtext("article-id[@pub-id-type='doi']")
            checksum = sha256(raw).hexdigest()
            if not doi:
                continue
            if doi in rows:
                if rows[doi]["sha256"] != checksum:
                    raise ValueError("Conflicting snapshots for one DOI")
                continue
            title = node.find("./title-group/article-title")
            if title is None:
                continue
            abstract = node.find("abstract")
            if abstract is None:
                continue
            parts = sentences(whitespace(" ".join(abstract.itertext())))
            if len(parts)<3:
                continue
            lead = " ".join(parts[:max(1,len(parts)//2)])
            ending = " ".join(parts[-2:])
            title = whitespace("".join(title.itertext()))
            byline = authors(node)
            complete_roles = declared_roles(raw,doi)
            owners = []
            known = {}
            for ordinal,(author,roles) in enumerate(zip(byline,complete_roles,strict=True),1):
                who = next(iter(joined([author])),None)
                if who is not None:
                    known[ordinal] = who[0]
                    if roles:
                        owners.append(dict(ordinal=ordinal,owner=who[0],roles=roles))
            if len(set(known.values())) != len(known):
                raise ValueError("Repeated ORCID in a byline")
            rows[doi] = dict(
                sha256=checksum, index=title+"\n"+lead, result=ending,
                identities=sorted(known.values()), owners=owners,
            )
    target = {case[field] for case in fixture["pairs"] for field in ("left","right")}
    if not target <= rows.keys():
        raise ValueError("Source clues require abstracts with separated sections")
    ids = sorted({person for row in rows.values() for person in row["identities"]})
    aliases = {}
    for doi,record in fixture["sources"].items():
        if doi not in rows:
            continue
        if rows[doi]["sha256"] != record["source_sha256"]:
            raise ValueError("Expected frozen publisher source")
        known = {item["ordinal"]:item for item in rows[doi]["owners"]}
        for gold in record["owners"]:
            source = known[gold["ordinal"]]
            raw = source["owner"]
            if set(source["roles"])!=set(gold["roles"]) or aliases.setdefault(raw,gold["owner"])!=gold["owner"]:
                raise ValueError("Inconsistent checked roles or identities")
    ids_for_graph = {person:i for i,person in enumerate(ids)}
    for doi,row in rows.items():
        row["identities"] = [ids_for_graph[p] for p in row["identities"]]
        for owner in row["owners"]:
            owner["owner"] = aliases.get(owner["owner"],f"other-{ids_for_graph[owner['owner']]:04d}")
    return rows


def prepare(fixture_path, output, sources):
    if digest(fixture_path)!=FIXTURE:
        raise ValueError("Changed scoped fixture")
    fixture=json.loads(fixture_path.read_text())
    rows=source_rows(fixture,sources)
    corpus_terms=Counter(word for r in rows.values() for word in set(tokens(r["index"])))
    targets={case[field] for case in fixture["pairs"] for field in ("left","right")}
    previous=set(rows)-fixture["sources"].keys()
    probes={}
    for doi in sorted((set(rows)&fixture["sources"].keys())|targets):
        row=rows[doi]
        result=row["result"]
        options=sorted({t for t in tokens(result) if len(t)>3 and not t.isdigit() and t not in STOPS})
        words=sorted(options,key=lambda t:(corpus_terms[t]==0,corpus_terms[t],stable(doi,t)))[:8]
        if len(words)<3 or whitespace(result) in whitespace(row["index"]):
            raise ValueError("Clue is not separate from indexed contents")
        probes[doi]=dict(result=result,keywords=" ".join(words))
    payload=dict(fixture_sha256=FIXTURE,previous_records=len(previous),clues=probes,
                 documents={doi:{k:v for k,v in row.items() if k!="result"} for doi,row in sorted(rows.items())})
    save(output,payload)
    print(dict(docs=len(rows),clues=len(probes),paired_targets=len(targets)),flush=True)


class Search:
    def __init__(self, payload, encoder, directory):
        from experiments.retrieval.lrbench_frozen_baselines import build_lexical

        self.dois=list(payload["documents"])
        self.ids={doi:i for i,doi in enumerate(self.dois)}
        rows=list(payload["documents"].values())
        # The shared builder deduplicates input strings, so add a tied DOI key only at row identity level.
        texts=[row["index"] for row in rows]
        if len(set(texts))!=len(texts):
            raise ValueError("Duplicate indexed documents")
        self.lexical=build_lexical({t:i for i,t in enumerate(texts)},directory)
        self.vectors=encoder.encode(texts)
        self.encoder=encoder
        owners=defaultdict(set)
        for i,row in enumerate(rows):
            for person in row["identities"]:
                owners[person].add(i)
        self.links=[set().union(*(owners[p] for p in row["identities"]))-{i} for i,row in enumerate(rows)]

    def views(self, clues, removed):
        vectors=self.encoder.encode([f"Instruct: {DENSE_TASK}\nQuery: {clue}" for clue in clues])
        dense=[np.argsort(-(self.vectors@q),kind="stable").tolist() for q in vectors]
        both, ranked = [], []
        banned={self.ids[doi] for doi in removed}
        for query,order in zip(clues,dense,strict=True):
            order=[i for i in order if i not in banned]
            lexical={int(key):value for key,value in self.lexical.document_scores(query).items()}
            lexical=sorted((int(i) for i in lexical if int(i) not in banned),
                           key=lambda i:(-lexical[i],i))
            scores={i:1/(60+j) for j,i in enumerate(order,1)}
            for j,i in enumerate(lexical,1):
                scores[i]+=1/(60+j)
            both.append(order)
            ranked.append(sorted(scores,key=lambda i:(-scores[i],i)))
        return both,ranked

    def rank(self, dense, fused, arm):
        a,b = dense if arm=="dense" else fused
        left,right=a[0],b[0]
        if arm in ("dense","hybrid"):
            return None if left==right else (self.dois[left],self.dois[right])
        depth=4 if arm=="linked_top4" else 12
        ranks={target:j for j,target in enumerate(b[:depth],1)}
        pairs=[]
        for i,x in enumerate(a[:depth],1):
            for y in self.links[x]&ranks.keys():
                j=ranks[y]
                if min(i,j)<=4:
                    pairs.append((i+j,max(i,j),x,y))
        if not pairs:
            return None
        _,_,x,y=min(pairs)
        return self.dois[x],self.dois[y]


def summarize(payload, cases, groups):
    evidence=EvidenceIndex(payload["documents"])
    out={}
    for setting, outcomes in groups.items():
        counted={a:Counter() for a in ARMS}
        for item in cases:
            pair=item["dois"]
            desired=set(item["owners"])
            requested=item["predicted"]
            for arm in ARMS:
                found=outcomes[pair][arm]
                score=counted[arm]
                score["questions"]+=1
                score["exact_source_pair"]+=found==pair
                score["unresolved_pair"]+=found is None
                supported=set() if found is None else set(attributed(evidence,found,requested))
                if setting[1]!="present":
                    score["wrong_pair_accepted"]+=found is not None
                    score["unsupported_identity_predictions"]+=len(supported)
                    continue
                oracle=set() if found is None else set(attributed(evidence,found,item["atoms"]))
                score["scoped_query_correct"]+=found==pair and sorted(requested)==sorted(item["atoms"]) and supported==desired
                score["contributor_and_sources_correct"]+=found==pair and supported==desired
                score["known_roles_contributor_and_sources_correct"]+=found==pair and oracle==desired
                score["unsupported_identity_predictions"]+=len(supported-desired)
        out["/".join(setting)] = {
            "source_pairs":len(outcomes),
            "unique_source_pair_answers":{a:dict(
                exact=sum(found[a]==target for target,found in outcomes.items()),
                unresolved=sum(found[a] is None for found in outcomes.values()),
            ) for a in ARMS},
            "repeated_requirement_queries":{a:dict(counts) for a,counts in counted.items()},
        }
    return out


def run(fixture_path, examples_path, projection, embed_path, reader_path, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.qasper.paired_reader import load_reader

    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or digest(fixture_path)!=FIXTURE or digest(examples_path)!=NISO or digest(projection)!=PROJECTION):
        raise ValueError("Require pinned inputs and fresh Slurm CUDA run")
    fixture,glossary,payload=(json.loads(p.read_text()) for p in (fixture_path,examples_path,projection))
    if payload["fixture_sha256"]!=FIXTURE:
        raise ValueError("Changed source projection")
    sample=[q for q in examples(fixture,glossary) if q["form"]=="explicit"]
    pairs=sorted({q["dois"] for q in sample})
    embed_rev=verify_checkpoint(embed_path)["revision"]
    encoder=QwenTorchEmbedding(embed_path,batch_size=24)
    encoder.model.config.use_cache=False
    with TemporaryDirectory() as work:
        index=Search(payload,encoder,Path(work))
        print(len(index.dois),"sources indexed",flush=True)
        groups={}
        times=[]
        single={}
        for view in VIEWS:
            single[view]=Counter()
            for doi,clues in payload["clues"].items():
                dense, fused = index.views([clues[view]],())
                n=index.ids[doi]
                for name,order in (("dense",dense[0]),("hybrid",fused[0])):
                    single[view][f"{name}_at_1"]+=order[0]==n
                    single[view][f"{name}_at_4"]+=n in order[:4]
            for tag in TAGS:
                outcomes={}
                for left,right in pairs:
                    clues=[payload["clues"][d][view] for d in (left,right)]
                    removed=() if tag=="present" else ((left,) if tag=="missing_left" else (right,))
                    start=perf_counter()
                    orders=index.views(clues,removed)
                    row={arm:index.rank(*orders,arm) for arm in ARMS}
                    times.append((perf_counter()-start)*1000)
                    outcomes[left,right]=row
                groups[view,tag]=outcomes
        del encoder
        torch.cuda.empty_cache()
        tokenizer,model,reader_meta=load_reader(reader_path)
        tokenizer.padding_side="left"
        encoded=[tokenizer.encode(item,add_special_tokens=False) for item in SYMBOLS]
        if any(len(x)!=1 for x in encoded):
            raise ValueError("Expected role labels to be single-token")
        digits=[x[0] for x in encoded]
        glossary_text="\n".join(f"{name}: {glossary['definitions'][name]}" for name in ROLES)
        for i,item in enumerate(sample):
            offset=stable(item["family"],"roles")[0]%len(ROLES)
            order=ROLES[offset:]+ROLES[:offset]
            item["predicted"],_=route(tokenizer,model,digits,glossary_text,item,"factored_local",order,True)
            if (i+1)%30==0:
                print(i+1,"queries parsed",flush=True)
        results=summarize(payload,sample,groups)
    write_json(output,dict(
        job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),fixture_sha256=FIXTURE,
        projection_sha256=digest(projection),script_sha256=digest(Path(__file__)),
        corpus=len(payload["documents"]),single_clues=len(payload["clues"]),families=len(pairs),
        embedding_revision=embed_rev,reader_revision=reader_meta["revision"],
        single_source=dict(single),results=results,shared_pair_retrieval_p50_p95_ms=np.quantile(times,[.5,.95]).tolist(),
    ))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(required=True)
    create=subs.add_parser("prepare")
    create.add_argument("fixture_path",type=Path)
    create.add_argument("output",type=Path)
    create.add_argument("sources",type=Path,nargs="+")
    create.set_defaults(action=prepare)
    live=subs.add_parser("run")
    for name in ("fixture_path","examples_path","projection","embed_path","reader_path","output"):
        live.add_argument(name,type=Path)
    live.set_defaults(action=run)
    args=vars(parser.parse_args())
    args.pop("action")(**args)
