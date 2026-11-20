"""Compare one-forward shared-prefix verification with padded and packed controls."""

import argparse
from collections import Counter
from functools import partial
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import elife_mesh as mesh
from experiments.credit import elife_mesh_scope as scope
from experiments.credit import role_field_ranking as field
from experiments.credit import role_gap_selective_recheck as scoring
from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


ARMS = ("padded", "packed", "prefix_fork", "prefix_tree")


def shared_prefix(sequences):
    count = 0
    for column in zip(*(sequence[:-1] for sequence in sequences)):
        if any(token != column[0] for token in column[1:]):
            break
        count += 1
    return sequences[0][:count], [sequence[count:] for sequence in sequences]


class PrefixReader:
    def __init__(self, reader, packed):
        import torch

        from experiments.retrieval.enterprise_isolated_parsing import padded
        from experiments.retrieval.enterprise_prefix_fork import fork_prefix

        self.torch = torch
        self.reader = reader
        self.backbone = reader.model.model
        self.head = reader.model.lm_head.weight[reader.labels].contiguous()
        self.fork_cache = fork_prefix
        self.pad = padded
        self.packed = packed

    def rank(self, hidden):
        logits = self.torch.nn.functional.linear(hidden.contiguous(), self.head).float()
        return (logits[:, 0] - logits[:, 1]).cpu().tolist()

    def fork(self, front, tails):
        torch = self.torch
        source = self.backbone(input_ids=torch.tensor([front], device="cuda"), use_cache=True).past_key_values
        cache = self.fork_cache(source, len(tails))
        ids, mask, positions = self.pad(tails, self.reader.tokenizer.pad_token_id, len(front))
        prefix_mask = torch.ones((len(tails), len(front)), dtype=torch.bool, device="cuda")
        output = self.backbone(input_ids=ids, attention_mask=torch.cat((prefix_mask, mask), dim=1),
                               position_ids=positions, past_key_values=cache, use_cache=True)
        if source.get_seq_length() != len(front):
            raise ValueError("Shared cache was mutated by a suffix branch")
        return self.rank(output.last_hidden_state[:, -1]), len(front) + ids.numel()

    def tree(self, front, tails):
        torch = self.torch
        tokens = front.copy()
        positions = list(range(len(front)))
        groups = [0] * len(front)
        ends = []
        for branch, tail in enumerate(tails, 1):
            tokens.extend(tail)
            positions.extend(range(len(front), len(front) + len(tail)))
            groups.extend([branch] * len(tail))
            ends.append(len(tokens) - 1)
        indices = torch.arange(len(tokens), device="cuda")
        groups = torch.tensor(groups, device="cuda")
        allowed = (indices[:, None] >= indices[None, :]) & (
            (groups[:, None] == groups[None, :]) | (groups[None, :] == 0))
        mask = torch.zeros((len(tokens), len(tokens)), dtype=self.head.dtype, device="cuda")
        mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
        output = self.backbone(input_ids=torch.tensor([tokens], device="cuda"),
                               attention_mask={"full_attention": mask[None, None]},
                               position_ids=torch.tensor([positions], device="cuda"), use_cache=False)
        return self.rank(output.last_hidden_state[0, ends]), len(tokens)

    def score(self, encoded, arm):
        if arm == "packed":
            scores, tokens = self.packed(encoded)
            return scores, tokens, 0, 1
        with self.torch.inference_mode():
            front, tails = shared_prefix([row["input_ids"] for row in encoded])
            if not front or len(encoded) == 1:
                scores, tokens = self.packed(encoded)
                return scores, tokens, 0, 1
            scores, tokens = self.fork(front, tails) if arm == "prefix_fork" else self.tree(front, tails)
        return scores, tokens, len(front), 2 if arm == "prefix_fork" else 1


def panel(catalog, titles, min_support, reader, forest, reference):
    from experiments.credit.role_gap_title_refill import refill

    queries = list(pilot.query_panel(
        catalog, sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= min_support),
    ))
    rows = {mask: {arm: [] for arm in ARMS} for mask in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        shift = index % len(ARMS)
        for mask in (pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))):
            start = perf_counter()
            vector = catalog.encoder.encode([f"Research about {topic}"], query=True)[0]
            ranked = catalog.rank_for_vector(topic, vector)
            excluded = catalog.visible[mask] - catalog.person[pair]
            sources = refill(ranked, lambda: np.argsort(-(titles @ vector), kind="stable").tolist(), excluded)
            encoded = reader.encode(f"Research about {topic}", [catalog.texts[i] for i in sources])
            common = (perf_counter() - start) * 1000
            results = {}
            for arm in ARMS[shift:] + ARMS[:shift]:
                start = perf_counter()
                if arm == "padded":
                    values, tokens = reader.score(encoded, range(len(encoded)))
                    prefix, forwards = 0, 1
                else:
                    values, tokens, prefix, forwards = forest.score(encoded, arm)
                witnesses, response = scoring.serialize(catalog, topic, pair, sources, dict(enumerate(values)))
                if not np.isfinite(values).all():
                    raise ValueError("Unexpected nonfinite verifier score")
                results[arm] = dict(topic=topic, values=values, witnesses=witnesses, response=response,
                                    tokens=tokens, prefix=prefix, forwards=forwards,
                                    ms=common + (perf_counter() - start) * 1000)
            for arm in ARMS:
                base, row = results["padded"], results[arm]
                before, after = base["witnesses"]["witnesses"], row["witnesses"]["witnesses"]
                row["counts"] = Counter(
                    witness_sets_different=int(set(before) != set(after)),
                    expert_sets_different=int({expert for expert, _ in before} != {expert for expert, _ in after}),
                    first_witness_different=int(before[:1] != after[:1]),
                    tagged_citations=row["witnesses"]["grounded"],
                    off_label_citations=row["witnesses"]["cited_off_label"],
                    tagged_citations_gained=max(0, row["witnesses"]["grounded"] - base["witnesses"]["grounded"]),
                    tagged_citations_lost=max(0, base["witnesses"]["grounded"] - row["witnesses"]["grounded"]),
                    exact_responses=int(row["response"] == base["response"]),
                    scores_different=sum(a != b for a, b in zip(row["values"], base["values"], strict=True)),
                    threshold_flips=sum((a > holdout.CUTOFF) != (b > holdout.CUTOFF)
                                        for a, b in zip(row["values"], base["values"], strict=True)),
                    token_positions=row["tokens"], prefix_tokens_shareable=row["prefix"], forwards=row["forwards"],
                )
                rows[mask][arm].append(row)
    summaries = {}
    for mask, arms in rows.items():
        base = arms["padded"]
        prior = reference[str(mask)]["gap_title_four"]["compared"]["topic_four"]
        for k in ("grounded", "issued", "cited_off_label"):
            if sum(row["witnesses"][k] for row in base) != prior[k]:
                raise ValueError(f"Padded baseline did not reproduce the frozen {k} count")
        summaries[mask] = {
            arm: dict(
                **sum((row["counts"] for row in observed), Counter()),
                requests=len(observed),
                p50_p95_p99_ms=np.quantile([row["ms"] for row in observed], [.50, .95, .99]).tolist(),
                paired_p95_delta_ms=holdout.compare_latency(
                    [row["ms"] for row in observed], [row["ms"] for row in base],
                    [row["topic"] for row in observed],
                ),
            )
            for arm, observed in arms.items()
        }
    return summaries


def run(plos, elife, mesh_snapshot, gap_reference, embedding, reranker, output):
    import torch
    from transformers import AttentionInterface

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention, packed_forward
    from experiments.retrieval.cof_ragged_shared_weights import measure_storage, packed_view

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm CUDA")
    targets = json.loads(gap_reference.read_bytes())["run"]
    frozen = json.loads(mesh_snapshot.read_bytes())
    if pilot.digest(elife) != frozen["projection_sha256"]:
        raise ValueError("Expected the frozen eLife projection")
    torch.backends.cuda.matmul.allow_tf32 = False
    embed = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    rerank = Reranker(reranker)
    if (embed["revision"], rerank.provenance["revision"]) != (targets["embedding"], targets["reranker"]):
        raise ValueError("Expected the frozen checkpoints")
    backbone = rerank.model.model
    if rerank.model.lm_head.bias is not None or set(backbone.config.layer_types) != {"full_attention"}:
        raise ValueError("Expected a bias-free full-attention model")
    AttentionInterface.register(PACKED_ATTENTION, packed_attention)
    with torch.inference_mode():
        packed_backbone = packed_view(backbone, share_weights=True)
        head = rerank.model.lm_head.weight[rerank.labels].contiguous()
    if measure_storage(backbone, packed_backbone)["extra_parameter_bytes"]:
        raise ValueError("Packed backbone must share model weights")
    forest = PrefixReader(rerank, partial(packed_forward, packed_backbone, head))
    reader = batches.BatchReader(rerank)
    reports = {}
    for name, source, support in (("plos", plos, 4), ("elife", elife, scope.MINIMUM_SUPPORT)):
        catalog, titles, _ = field.data_model(source, encoder, rerank)
        if name == "elife":
            complete = {i for i, article in enumerate(catalog.articles) if article["declaration_complete"]}
            catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
            mesh.assign_topics(catalog, frozen["records"], mesh.ROUTINE_DESCRIPTORS, major_only=False)
            for term in tuple(catalog.topics):
                if not scope.MINIMUM_SUPPORT <= len(catalog.topics[term]) <= scope.MAXIMUM_SUPPORT:
                    del catalog.topics[term]
        warm_rows = reader.encode("Research about materials", catalog.texts[:4])
        reader.score(warm_rows, range(4))
        front, tails = shared_prefix([row["input_ids"] for row in warm_rows])
        modified = [tail.copy() for tail in tails]
        modified[0][0] = (modified[0][0] + 1) % rerank.model.config.vocab_size
        with torch.inference_mode():
            for verify in (forest.fork, forest.tree):
                original = verify(front, tails)[0]
                other = verify(front, modified)[0]
                if original[1:] != other[1:]:
                    raise ValueError("Changing a branch affected its siblings")
        forest.score(warm_rows, "packed")
        reports[name] = panel(catalog, titles, support, reader, forest, targets[name]["metrics"])
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=pilot.digest(Path(__file__)),
        gap_reference_sha256=pilot.digest(gap_reference), plos_projection_sha256=pilot.digest(plos),
        elife_projection_sha256=pilot.digest(elife), branch_isolation_check=True, reports=reports,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "elife", "mesh_snapshot", "gap_reference", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
