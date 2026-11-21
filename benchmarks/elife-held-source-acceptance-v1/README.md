# Source acceptance before issuing a same-contributor proof

The [role-wording replay](../role-wording-stress-v1) repeatedly chose
a substitute paper for the same held keyword group after one source
was removed. This comparison separates accepting the selected papers
from validating the requested roles and contributor on those papers.

## Holdout and protocol

Freeze 35 new paper pairs from the 2,011-paper eLife catalog, using
the same two pinned publisher projections. Exclude all papers and
all recorded, resolved identities in the 60 previous groups.
Greedily select author-disjoint pairs in a stable hash order, with
no search scores, text similarity, or policy outputs in the order.
Each pair needs a resolved common author with two distinct roles.
The swapped roles must be declared somewhere on their new
respective papers, with no linked author meeting both swapped
requirements. Keep up to three bindings per pair, yielding 102
positive and 102 swapped requests per entrance. The same groups
are replayed with the left and right designated sources separately
removed, yielding 408 missing-source requests per entrance.
Paper selection and the resulting [manifest](panel.json) were frozen
in job 47115 before any GPU ranking of these groups. Authors without
resolved identities cannot be checked for overlap.

Use the same excerpt/keyword clues, depth-six BM25/dense fusion,
and fixed query-first Ridge role cutoff as the previous work.
A source pair must consist of distinct papers with at least one
resolved shared author. Measure both source-pair substitution and
downstream wrong proofs under the exact designated-source contract.
The test does not judge other papers that may provide substantively
valid alternative evidence.

## Source policies

| Policy | Selection and acceptance |
| --- | --- |
| Linked | Select the admissible pair with the smallest sum of hybrid ranks |
| Anchored | Require a hybrid leader on at least one side and a cosine drop of at most 0.04 on each side |
| Paired rank | Rerank the linked candidates with the previous paired source features and apply a score cutoff |
| Fixed pair score | Score the original linked proposal using the same paired features and apply a cutoff |
| Fixed fingerprint | Score the original linked proposal using the paired features and independent lexical footprint features |

The fingerprint records each clue's distinctive non-stopword tokens
and ordered adjacent token pairs after stopword filtering. It adds
weighted unigram, four-rarest-token and bigram coverage, the difference
between coverage of the assigned and other clue, and unigram coverage
relative to the best source in the same candidate pool. Use the
minimum and maximum across sides as pair features. All signals use
the indexed text and residue from the same retrieval. No additional
query encoding or source read is used in the comparison.

Fit regularized logistic scores on the original 18 development
paper groups. Use the other 12 original development groups and all
30 previously scored held groups for calibration. For each
entrance and policy, set the cutoff from the largest adverse
selected-pair score per calibration group. Evaluate two operating
points admitting at most zero or two adverse groups in calibration.
The calibration set is historical and has been previously explored.
The new panel is held out from both steps. The cutoffs do not
constitute a deployment risk guarantee.

## Results

| Entrance and policy | Exact sources / 35 present trials | Strict proofs / 102 positives | Wrong source proposals / 70 missing trials | Wrong proofs / 408 missing requests |
| --- | ---: | ---: | ---: | ---: |
| Excerpt, linked | 34 | 38 | 4 | 1 |
| Excerpt, anchored | 34 | 38 | 1 | 1 |
| Excerpt, paired or fixed calibrated score | 34 | 38 | 4 | 1 |
| Excerpt, fixed fingerprint | 34 | 38 | 4 | 1 |
| Keyword, linked | 31 | 35 | 0 | 0 |
| Keyword, anchored | 21 | 25 | 0 | 0 |
| Keyword, paired rank, zero-adverse cutoff | 22 | 26 | 0 | 0 |
| Keyword, fixed pair score or fingerprint | 31 | 35 | 0 | 0 |

The 35 new groups contain 34 reachable correct excerpt pairs and
31 reachable correct keyword pairs at depth six. Linked selection
recovers all reachable pairs in both views. No route emits a wrong
proof when both designated papers exist. On excerpts, the four
substitutions with one paper withheld involve three groups. The
anchored route suppresses three substitutes and leaves the single
one that also passes contributor verification. On keywords, the
linked route already emits zero substitutes, while anchoring or
strictly filtering the reranked pair reduces correct proof coverage.

**The excerpt cutoff has no adverse calibration examples.** None of
the three learned policies selects a wrong excerpt pair in the 42
historical calibration groups. The fallback threshold therefore
accepts every selected pair, including the new substitutes. The
more permissive keyword threshold also accepts all selected pairs:
there are fewer than three adverse groups. This replay gives no
evidence that the fingerprint improves source acceptance, and no
out-of-catalog guarantee. Develop the source verifier on calibration
cases that actually expose substitutes, using related alternatives
and separate absent-source workloads, before another frozen test.

Both RTX 4090 runs in Slurm array 47119 on ws1 return identical
cutoffs and counts. The benchmark keeps the same retrieval and
encoding workload across methods and does not time the added CPU
gate. The group count and publisher scope limit transfer claims.
[Aggregate results](results.json).

## Reproduce

Restore the projections, held groups, embedding checkpoint and NISO
snapshot as in the [role-first study](../role-first-proof-decoding-v1).

```sh
export PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.credit.freeze_source_acceptance_panel \
  CONFIRMATION_PROJECTION EXTENDED_PROJECTION \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json NEW_PANEL_JSON

python -m experiments.credit.held_source_acceptance \
  CONFIRMATION_PROJECTION EXTENDED_PROJECTION \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json NEW_PANEL_JSON \
  benchmarks/role-first-proof-decoding-v1/results.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json QWEN_EMBEDDING NEW_RESULTS
```

Use a Slurm GPU allocation for ranking. Repeat with `--reverse` and
a new results path.
