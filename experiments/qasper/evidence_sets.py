"""Select complete evidence sets from identical frozen QASPER reranker scores."""

import argparse
from hashlib import sha256
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.qasper import evidence_shortlist as shortlist


base = shortlist.base
depth = shortlist.depth
ARMS = ("pointwise", "mmr", "coverage", "learned")
DEPTHS = (16, 32)
ANCHORS = 2
WEIGHTS = (0.0, .05, .15, .30, .60)
SALT = "qasper-final-evidence-sets-v1"
RIDGE = 10.0


class SetSelector:
    def __init__(self, pool, fused, probabilities, count):
        self.ids = pool.order[:count]
        self.rank = np.lexsort((self.ids, -fused))
        self.relevance = (fused - fused.min()) / max(fused.max() - fused.min(), 1e-12)
        self.probabilities = probabilities
        vectors = pool.vectors[:count]
        self.similarity = np.maximum(vectors @ vectors.T, 0)
        sections = pool.sections[:count]
        self.same_section = np.equal.outer(sections, sections)
        self.contribution, self.idf = pool.coverage()
        self.contribution = self.contribution[:count]
        self.distances = np.log1p(np.abs(self.ids[:, None] - self.ids[None, :]))

    def state(self, selected):
        remaining = self.idf * np.prod(1 - self.contribution[selected], axis=0)
        lexical_gain = self.contribution @ remaining
        lexical_gain /= max(lexical_gain.max(), 1e-12)
        redundancy = self.similarity[:, selected].max(axis=1)
        adjacent = 1 / (1 + self.distances[:, selected].min(axis=1))
        same_section = self.same_section[:, selected].any(axis=1)
        return np.column_stack([
            self.relevance, self.probabilities, self.relevance ** 2, lexical_gain, redundancy,
            adjacent, same_section, self.relevance * lexical_gain,
            redundancy * lexical_gain, np.full(len(self.ids), len(selected) / base.TOP),
        ])

    def choose(self, arm, weight=0.0, model=None):
        if arm == "pointwise" or weight == 0 or len(self.ids) <= base.TOP:
            return self.ids[self.rank[:base.TOP]].tolist()
        selected = self.rank[:ANCHORS].tolist()
        while len(selected) < base.TOP:
            state = self.state(selected)
            if arm == "mmr":
                adjusted = self.relevance - weight * state[:, 4]
            elif arm == "coverage":
                adjusted = self.relevance + weight * state[:, 3]
            else:
                predicted = model["intercept"] + (state - model["center"]) / model["scale"] @ model["coefficients"]
                adjusted = self.relevance + weight * predicted
            adjusted[selected] = -np.inf
            chosen = int(np.lexsort((self.ids, -adjusted))[0])
            selected.append(chosen)
        return self.ids[selected].tolist()

    def training_states(self, gold_positions, rng):
        for size in range(ANCHORS, min(base.TOP, len(self.ids))):
            default = self.rank[:size].tolist()
            extras = [i for i in self.rank[ANCHORS:] if i not in default]
            samples = [default]
            if size > ANCHORS and extras:
                samples.append(default[:-1] + [int(rng.choice(extras))])
            for selected in samples:
                coverage = (gold_positions[:, selected]).sum(axis=1) / gold_positions.sum(axis=1)
                gain = gold_positions / gold_positions.sum(axis=1, keepdims=True)
                target = np.maximum((coverage[:, None] + gain).max(axis=0) - coverage.max(), 0)
                candidates = [i for i in range(len(self.ids)) if i not in selected]
                yield self.state(selected)[candidates], target[candidates]


def fit(training):
    rng = np.random.default_rng(44)
    state_rows = []
    for row in training:
        for selector in row["selectors"].values():
            index = {text: i for i, text in enumerate(row["paragraphs"])}
            present = set(selector.ids.tolist())
            gold = [{index[text] for text in ref} for ref in row["gold"]]
            feasible = [ref for ref in gold if len(ref) <= base.TOP and ref <= present]
            if feasible:
                membership = np.array([[int(i in ref) for i in selector.ids] for ref in feasible])
                state_rows.extend(selector.training_states(membership, rng))
    x = np.concatenate([features for features, _ in state_rows])
    y = np.concatenate([target for _, target in state_rows])
    center = x.mean(0)
    scale = np.maximum(x.std(0), 1e-9)
    standardized = (x - center) / scale
    coefficients = np.linalg.solve(standardized.T @ standardized + RIDGE * np.eye(x.shape[1]),
                                   standardized.T @ y)
    return dict(center=center, scale=scale, coefficients=coefficients, intercept=np.asarray(y.mean())), len(y)


def select_split(dataset, panel, encoder, reader):
    indexes, offline = shortlist.paper_indexes(dataset, panel, encoder)
    rows = []
    for pid, qa in panel:
        paragraphs, sections, bank, stats = indexes[pid]
        question = qa["question"]
        context = [f"{section}\n{text}" for section, text in zip(sections, paragraphs, strict=True)]
        started = perf_counter()
        dense = shortlist.online_scores(encoder, question, bank)
        pool = shortlist.CandidatePool(question, dense, bank, stats, sections)
        probability = []
        shared = {}
        for start in range(0, len(pool.order), base.BLOCK):
            end = start + base.BLOCK
            batch = pool.order[start:end].tolist()
            values, _ = reader.score(question, context, batch, task=base.TASK)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite reranker margin")
            probability.extend(1 / (1 + math.exp(-max(-40, min(40, value)))) for value in values)
            shared[min(end, len(pool.order))] = (perf_counter() - started) * 1000
        selectors = {}
        for count in DEPTHS:
            take = min(count, len(pool.order))
            probabilities = np.array(probability[:take])
            fused = (1 - base.WEIGHT) * dense[pool.order[:take]] + base.WEIGHT * probabilities
            selectors[count] = SetSelector(pool, fused, probabilities, take)
        rows.append(dict(paper_id=pid, question_id=qa["question_id"], paragraphs=paragraphs, selectors=selectors,
                         shared_ms={n: shared[min(n, len(pool.order))] for n in DEPTHS},
                         gold=base.evidence_sets(qa, paragraphs)))
        if len(rows) % 32 == 0:
            print("Scored shared requests:", len(rows), flush=True)
    return rows, offline


def calibrate(rows, model):
    labeled = [row for row in rows if row["gold"]]
    settings = {}
    for count in DEPTHS:
        selected = {}
        for arm in ARMS[1:]:
            scores = []
            for weight in WEIGHTS:
                correct = []
                for row in labeled:
                    indices = row["selectors"][count].choose(arm, weight, model)
                    result = base.quality([row["paragraphs"][i] for i in indices], row["gold"])
                    correct.append(result["complete_evidence"])
                scores.append(mean(correct))
            selected[arm] = dict(chosen=WEIGHTS[int(np.argmax(scores))],
                                 scores=dict(zip(WEIGHTS, scores, strict=True)))
        settings[count] = selected
    return settings


def test_panels(dataset):
    replicate = shortlist.replication_panel(dataset)
    exclude = {pid for pid, _ in replicate + base.panel(dataset)}

    def order(value):
        return sha256(f"{SALT}:{value}".encode()).digest()

    further = [(pid, min(dataset[pid]["qas"], key=lambda q: order(q["question_id"])))
               for pid in sorted(dataset.keys() - exclude, key=order)[:120]]
    return {"replication": replicate, "additional": further}


def evaluate(rows, model, settings, forced_positive):
    results = []
    for i, row in enumerate(rows):
        groups = {}
        for count, selector in row["selectors"].items():
            offset = i % len(ARMS)
            returned = {}
            for arm in ARMS[offset:] + ARMS[:offset]:
                state = settings[count].get(arm)
                weight = 0
                if state is not None:
                    if forced_positive:
                        weight = max(WEIGHTS[1:], key=lambda value: (state["scores"][value], value))
                    else:
                        weight = state["chosen"]
                start = perf_counter()
                indices = selector.choose(arm, weight, model)
                elapsed = (perf_counter() - start) * 1e6
                if len(set(indices)) != min(len(selector.ids), base.TOP):
                    raise ValueError("Invalid selected evidence set size")
                quality = base.quality([row["paragraphs"][i] for i in indices], row["gold"])
                returned[arm] = dict(order=indices, quality=quality, selection_us=elapsed, weight=weight)
            candidate_text = {row["paragraphs"][i] for i in selector.ids}
            anchors = {row["paragraphs"][i] for i in returned["pointwise"]["order"][:ANCHORS]}
            gold = row["gold"]
            oracle = None if not gold else any(len(ref) <= base.TOP and ref <= candidate_text for ref in gold)
            anchored = None if not gold else any(len(ref | anchors) <= base.TOP and ref <= candidate_text
                                                  for ref in gold)
            groups[count] = dict(arms=returned, shared_ms=row["shared_ms"][count], max_complete=oracle,
                                 max_complete_with_anchors=anchored)
        results.append(dict(paper_id=row["paper_id"], question_id=row["question_id"], depths=groups))
    return results


def summarize(results):
    report = {}
    for count in DEPTHS:
        groups = [row["depths"][count] for row in results]
        scored = [group for group in groups if group["max_complete"] is not None]
        papers = [row["paper_id"] for row in results if row["depths"][count]["max_complete"] is not None]
        arms = {}
        for arm in ARMS:
            pairs = [group["arms"][arm]["quality"]["complete_evidence"]
                     - group["arms"]["pointwise"]["quality"]["complete_evidence"] for group in scored]
            arms[arm] = dict(
                complete=mean(group["arms"][arm]["quality"]["complete_evidence"] for group in scored),
                f1=mean(group["arms"][arm]["quality"]["f1"] for group in scored),
                selection_p95_us=float(np.quantile([group["arms"][arm]["selection_us"] for group in groups], .95)),
                gains=pairs.count(1), losses=pairs.count(-1),
                weight=groups[0]["arms"][arm]["weight"],
                sets_changed=sum(set(group["arms"][arm]["order"]) != set(group["arms"]["pointwise"]["order"])
                                 for group in groups),
                complete_delta=clustered_paired_interval(pairs, papers),
            )
        report[count] = dict(evaluable=len(scored), maximum=sum(group["max_complete"] for group in scored),
                             maximum_with_anchors=sum(group["max_complete_with_anchors"] for group in scored),
                             shared_p95_ms=float(np.quantile([group["shared_ms"] for group in groups], .95)),
                             arms=arms)
    return report


def run(data, embedding, reranker, output, forced_positive=False):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and a Slurm allocation")
    dev = base.read_data(data, "dev")
    test = base.read_data(data, "test")
    if dev.keys() & test.keys():
        raise ValueError("Development and test paper IDs overlap")
    embedding_info = base.verify_checkpoint(embedding)
    if embedding_info["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected embedding checkpoint")
    encoder = base.QwenTorchEmbedding(embedding, max_length=512, batch_size=32)
    if not encoder.torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    encoder.model.config.use_cache = False
    reader = base.Reranker(reranker)
    if reader.provenance["revision"] != "e61197ed45024b0ed8a2d74b80b4d909f1255473":
        raise ValueError("Unexpected reranker checkpoint")
    encoder.encode([base.query_input("What method is used?")])
    reader.score("What method is used?", ["Linear regression."] * base.BLOCK, range(base.BLOCK), task=base.TASK)
    dev_rows, dev_offline = select_split(dev, depth.selected_dev(dev), encoder, reader)
    model, examples = fit([row for row in dev_rows if depth.partition(row["paper_id"]) == "fit"])
    settings = calibrate([row for row in dev_rows if depth.partition(row["paper_id"]) == "calibration"], model)
    offline, results = {"development": dev_offline}, {}
    for name, panel in test_panels(test).items():
        print(name, flush=True)
        states, offline[name] = select_split(test, panel, encoder, reader)
        results[name] = evaluate(states, model, settings, forced_positive)
    base.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=encoder.torch.cuda.get_device_name(), data_assets=base.ASSETS,
        script_sha256=base.digest(Path(__file__)), shortlist_sha256=base.digest(Path(shortlist.__file__)),
        embedding=embedding_info, reranker=reader.provenance, offline=offline, fit_examples=examples,
        model={key: value.tolist() for key, value in model.items()}, settings=settings, forced_positive=forced_positive,
        report={name: summarize(rows) for name, rows in results.items()}, rows=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--forced-positive", action="store_true")
    run(**vars(parser.parse_args()))
