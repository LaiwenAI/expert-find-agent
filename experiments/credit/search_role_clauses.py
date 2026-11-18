"""Replay source retrieval, credited-clause selection and author attribution together."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from experiments.credit.compositional_edits import stable
from experiments.credit.conjunctive_source_retrieval import sentences
from experiments.credit.cross_source_identity import load
from experiments.credit.cross_source_search import searchable_text
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.source_confirmation import accepts, measure, notice_kind
from experiments.followir.core17_reranking import digest, write_json


METHODS = ("bm25_ridge", "confirmed_ridge", "confirmed_clause", "confirmed_full_reader")
PROSE_METHODS = ("first_cut", "embedding_cut", "reader_cut", "oracle_cut")
SOURCE_POLICY = {"score": .75, "margin": .05}
STYLES = ("literal_excerpt", "abridged_excerpt")
PARAPHRASE_METHODS = ("bm25", "dense", "one_reader", "reranked", "supplied_source")


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    desired = {}
    for path, field in ((args.prior, "input_sources"), (args.external, "inventory")):
        for row in json.loads(path.read_bytes())[field]:
            expected = desired.setdefault(row["doi"], row["source_sha256"])
            if row["source_sha256"] != expected:
                raise ValueError("Conflicting publisher source")
    loaded = load(args.xml)
    corpus = searchable_text(args.xml)
    if not desired.keys() <= loaded.keys() or not desired.keys() <= corpus.keys():
        raise ValueError("Incomplete publisher inputs")
    rows = {}
    for doi, expected in sorted(desired.items()):
        source = loaded[doi]
        if source["source_sha256"] != expected:
            raise ValueError("Changed publisher record")
        title, abstract = corpus[doi]
        rows[doi] = dict(title=title, abstract=abstract, source_sha256=expected,
                         roles=[r["roles"] for r in source["authors"]])
    write_json(args.output, dict(prior_sha256=digest(args.prior), external_sha256=digest(args.external), records=rows))


def split(text):
    fields = text.splitlines()
    if len(fields) != 3 or any(not line.startswith(tag) for line, tag in zip(
        fields, ("Study: ", "Required action 1: ", "Required action 2: "), strict=True,
    )):
        raise ValueError("Unrecognized request format")
    return [line.split(": ", maxsplit=1)[1] for line in fields]


def render(lead, phrases):
    return f"Study: {lead}\nRequired action 1: {phrases[0]}\nRequired action 2: {phrases[1]}"


def views(record):
    parts = [s for s in sentences(record["abstract"]) if 12 <= len(s.split()) <= 62]
    if not parts:
        raise ValueError("No eligible abstract span")
    full = parts[len(parts)//2]
    shortened = " ".join(word for i, word in enumerate(full.split()) if i % 5 != 2)
    return dict(literal_excerpt=full, abridged_excerpt=shortened)


def summary(rows, methods=METHODS, styles=STYLES):
    results = {}
    for style in styles:
        results[style] = {}
        for state in ("present", "withheld"):
            values = [r for r in rows if r["style"] == style and r["state"] == state]
            results[style][state] = {}
            for method in methods:
                arms = [r["arms"][method] for r in values]
                counts = Counter()
                for row, arm in zip(values, arms, strict=True):
                    target, expected = row["doi"], row["gold"]
                    counts["requests"] += 1
                    counts["source_exact"] += arm["doi"] == target
                    counts["source_selected"] += arm["doi"] is not None
                    counts["unresolved"] += arm["authors"] is None
                    if arm.get("cut_exact") is not None:
                        counts["cut_evaluated"] += 1
                        counts["cut_exact"] += arm["cut_exact"]
                        counts["multiple_cut_options"] += arm["candidate_cuts"] > 1
                    if state == "present":
                        counts["exact"] += arm["doi"] == target and arm["authors"] == expected
                    if arm["authors"] is not None:
                        counts["unsupported_citations"] += len(arm["authors"]) if state == "withheld" or arm["doi"] != target else len(set(arm["authors"]) - set(expected))
                        counts["wrong_source_empty"] += not arm["authors"] and (state == "withheld" or arm["doi"] != target)
                results[style][state][method] = dict(counts, p95_ms=float(np.quantile([x["ms"] for x in arms], .95)))
    return results


def run(args):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.attribution_controls import Reader
    from experiments.credit.factored_role_edits import choose_batched
    from experiments.credit.retrieve_and_attribute import FORMATS, spans, split_request
    from experiments.credit.role_clause_selection import reader_quotes
    from experiments.credit.role_language_router import NISO_SHA256, Router
    from experiments.credit.role_prototypes import normalized
    from experiments.credit.train_role_decision import encode, ridge
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.qasper.paired_reader import load_reader
    from experiments.retrieval.lrbench_frozen_baselines import build_lexical

    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    fixture_hash = digest(args.fixture / "manifest.json")
    catalog = json.loads(args.catalog.read_bytes())
    if catalog["external_sha256"] != fixture_hash:
        raise ValueError("Changed external fixture")
    records = catalog["records"]
    manifest = json.loads((args.fixture / "manifest.json").read_bytes())
    pair_path = args.fixture / "pairs.jsonl"
    if manifest["pairs_sha256"] != digest(pair_path):
        raise ValueError("Changed public group labels")
    for row in manifest["sources"]:
        paper = records[row["doi"]]
        if paper["source_sha256"] != row["source_sha256"] or paper["roles"] != row["roles_by_author"]:
            raise ValueError("Changed contributor declarations")
    groups = [json.loads(line) for line in pair_path.read_text().splitlines()]
    if args.paraphrases is not None:
        from experiments.credit.source_paraphrases import Search, validate

        if args.prose or args.source_gates is None or args.source_reranker is None:
            raise ValueError("Paraphrases require frozen source gates and a reranker with structured role fields")
        gates = json.loads(args.source_gates.read_bytes())
        expected = {"catalog": digest(args.catalog), "fixture": fixture_hash, "queries": digest(args.paraphrases)}
        if gates["inputs"] != expected:
            raise ValueError("Source gate inputs differ from the original calibration run")
        clues = validate(args.paraphrases, args.fixture / "manifest.json", catalog)
        target_views = {row["doi"]: {"paraphrase": row["query"]} for row in clues if row["split"] == "test"}
        source = Search(records, args.embedding, args.source_reranker)
        source_policies = gates["policies"]
        if not all(source_policies[method] for method in ("dense", "one_reader", "reranked")):
            raise ValueError("Expected nonempty frozen source policies")
        methods, styles = PARAPHRASE_METHODS, ("paraphrase",)
    else:
        if args.source_gates is not None or args.source_reranker is not None:
            raise ValueError("Source gates and reranker require paraphrases")
        target_views = {row["doi"]: views(records[row["doi"]]) for row in groups}
        methods, styles = PROSE_METHODS if args.prose else METHODS, STYLES
    phrases = json.loads((args.fixture / "boundary-phrases.json").read_bytes())
    if digest(args.examples) != NISO_SHA256:
        raise ValueError("Changed glossary")
    glossary = json.loads(args.examples.read_bytes())
    provenance = verify_checkpoint(args.embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = source.encoder if args.paraphrases is not None else QwenTorchEmbedding(args.embedding)
    tokenizer, model, reader_provenance = load_reader(args.reader)
    reader = Reader(tokenizer, model)
    definitions = [f"{role}: {glossary['definitions'][role]}" for role in ROLES]
    router = Router(definitions, None, None, encoder, model, tokenizer, 3)
    original = json.loads(args.training.read_bytes())
    ordered = [[(text, ROLES.index(role)) for text in sorted(original[role],
                key=lambda s: stable("fine-tune-split", role, s))] for role in ROLES]
    fit = [r for variants in ordered for r in variants[:-1]]
    held = [v[-1] for v in ordered]
    if args.paraphrases is None:
        ids = sorted(records)
        docs = {}
        segments = {}
        notices = {}
        for i, doi in enumerate(ids):
            paper = records[doi]
            text = paper["title"] + "\n" + paper["abstract"]
            while text in docs:
                text += "\n"
            docs[text] = i
            segments[doi] = [normalized(s).split() for s in [paper["title"], *sentences(paper["abstract"])]]
            notices[doi] = notice_kind(paper["title"])
        with TemporaryDirectory() as folder:
            lexical = build_lexical(docs, Path(folder))
    with torch.inference_mode():
        x = torch.cat([encode(encoder, [t for t, _ in part]) for part in (fit, held)])
        y = torch.tensor([role for _, role in (*fit, *held)], device="cuda")
        bank = ridge(x, y)

        def choose_cut(activity, method, expected):
            candidates = spans(activity)
            if method == "oracle_cut":
                return expected, len(candidates)
            if not candidates:
                return None, 0
            if len(candidates) == 1 or method == "first_cut":
                return candidates[0], len(candidates)
            if method == "embedding_cut":
                atoms = list(dict.fromkeys(part for pair in candidates for part in pair))
                vectors = encode(encoder, atoms) @ bank.T
                values = dict(zip(atoms, vectors, strict=True))

                def proposal_score(pair):
                    scores = values[pair[0]][:, None] + values[pair[1]][None, :]
                    scores.fill_diagonal_(-torch.inf)
                    return scores.max().item()

                return max(candidates, key=proposal_score), len(candidates)
            if len(candidates) > len(router.ids):
                return None, len(candidates)
            options = candidates
            rotation = int(stable(activity, "candidate-order"), 16) % len(options)
            options = options[rotation:] + options[:rotation]
            menu = "\n".join(f"{chr(65+i)}: [{left}] / [{right}]" for i, (left, right) in enumerate(options))
            prompt = (
                "Select the boundary between the two top-level contributor requirements. "
                "Each requirement may contain subordinate clauses about other contributors. "
                "Keep these subordinate clauses within their own requirement. Choose a letter.\n"
                f"Requirement text: {activity}\n{menu}\nAnswer:"
            )
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            scores = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1, router.ids[:len(options)]]
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite boundary scores")
            return options[int(scores.argmax().item())], len(candidates)

        def execute(text, hidden, method, expected, target):
            start = perf_counter()
            if args.prose:
                lead, activity = split_request(text)
                tasks = None
            else:
                lead, *tasks = split(text)
            if args.paraphrases is not None:
                if method == "supplied_source":
                    doi = target if hidden is None else None
                else:
                    observation, _, _, _ = source.run(lead, hidden, method)
                    doi = observation[0] if method == "bm25" or accepts(observation, source_policies[method]) else None
            else:
                scores = lexical.document_scores(lead)
                candidates = [ids[int(i)] for i in sorted(scores, key=lambda i: (-scores[i], int(i)))
                              if ids[int(i)] != hidden]
                if method == "bm25_ridge":
                    doi = candidates[0] if candidates else None
                else:
                    typed = [d for d in candidates if notices[d] == notice_kind(lead)][:4]
                    observation = measure(normalized(lead).split(), typed, segments, "ordered")
                    doi = observation[0] if accepts(observation, SOURCE_POLICY) else None
            source_ms = (perf_counter() - start) * 1000
            authors = None
            cut_exact = None
            candidate_cuts = 0
            roles = records[doi]["roles"] if doi else None
            if roles and all(roles):
                if args.prose:
                    tasks, candidate_cuts = choose_cut(activity, method, expected)
                    cut_exact = tasks == expected
                if tasks is None:
                    return dict(doi=doi, authors=None, cut_exact=cut_exact, candidate_cuts=candidate_cuts,
                                ms=(perf_counter()-start)*1000, source_ms=source_ms)
                if method == "confirmed_full_reader":
                    rankings = choose_batched(router, tasks, [list(range(len(ROLES)))] * len(tasks), 0)
                else:
                    if args.prose or method == "confirmed_clause" or args.paraphrases is not None:
                        tasks = reader_quotes(reader, tasks)
                    rankings = (encode(encoder, tasks) @ bank.T).argmax(1).tolist()
                required = {ROLES[i] for i in rankings}
                if len(required) == len(tasks):
                    authors = [i for i, own in enumerate(roles, 1) if required.issubset(own)]
            return dict(doi=doi, authors=authors, cut_exact=cut_exact, candidate_cuts=candidate_cuts,
                        ms=(perf_counter()-start)*1000, source_ms=source_ms)

        rows = []
        for i, group in enumerate(groups):
            doi = group["doi"]
            for style, lead in target_views[doi].items():
                for side in ("before", "after"):
                    required = group[side]
                    tasks = tuple(min(phrases[r], key=lambda phrase: stable(group["id"], r, phrase)) for r in required)
                    if args.prose:
                        form, _ = FORMATS[i % len(FORMATS)]
                        text = form.format(lead=lead.replace('"', "'"), activity=" and ".join(tasks))
                    else:
                        text = render(lead, tasks)
                    gold = [ordinal for ordinal, own in enumerate(records[doi]["roles"], 1)
                            if set(required).issubset(own)]
                    for state, hidden in (("present", None), ("withheld", doi)):
                        if not i and (args.paraphrases is None or not rows):
                            for method in methods:
                                execute(text, hidden, method, tasks, doi)
                        order = (i + len(rows)) % len(methods)
                        arms = {m: execute(text, hidden, m, tasks, doi) for m in methods[order:] + methods[:order]}
                        if hidden and any(a["doi"] == hidden for a in arms.values()):
                            raise AssertionError("Returned an unavailable document")
                        rows.append(dict(id=group["id"], doi=doi, style=style, state=state,
                                         side=side, gold=gold, arms=arms))
    write_json(args.output, dict(
        job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        catalog_sha256=digest(args.catalog), script_sha256=digest(Path(__file__)),
        training_sha256=digest(args.training), phrase_sha256=digest(args.fixture / "boundary-phrases.json"),
        embedding=provenance, reader=reader_provenance, sources=len(records),
        policy=SOURCE_POLICY if args.paraphrases is None else source_policies,
        prose=args.prose, report=summary(rows, methods, styles), rows=rows,
        source_gate_sha256=digest(args.source_gates) if args.paraphrases is not None else None,
        source_reranker=source.reranker.provenance if args.paraphrases is not None else None,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser("prepare")
    for name in ("prior", "external", "output"):
        build.add_argument(name, type=Path)
    build.add_argument("xml", type=Path, nargs="+")
    replay = subs.add_parser("run")
    for name in ("fixture", "catalog", "examples", "training", "embedding", "reader", "output"):
        replay.add_argument(name, type=Path)
    replay.add_argument("--prose", action="store_true")
    for option in ("--paraphrases", "--source-gates", "--source-reranker"):
        replay.add_argument(option, type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)
