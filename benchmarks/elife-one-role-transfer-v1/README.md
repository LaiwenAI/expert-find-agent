# Transfer one-role proof repair to eLife source search

Restore the [pinned eLife projection](../elife-role-nomination-v1).
The original intake screened its 322 articles for title, byline,
and valid ORCID overlap with the earlier PLOS inventory. Admit
only articles with complete normalized contributor declarations,
at least three abstract sentences, and unique author keys.
Link articles by the publisher ORCID. Authors without a validated
identifier remain local to the source.

Reuse the deterministic engineering scope-swap construction.
Each positive requires the same person to hold two different
roles on designated articles. In the paired negative, both roles
occur on the opposite articles, but no linked author satisfies
the swapped conjunction. Use held NISO wording for the roles.
Index the title and first half of each abstract. Derive source
clues from the held-out half, separately as excerpts and up to
eight high-IDF keywords. Query clause boundaries are supplied.

The [engineering policy](../engineering-one-role-recheck-v1)
is frozen, including the author-join source anchor, both role
bands, the role reader, and the reader margin selected on PLOS.
No threshold is fitted on eLife. The top-six fused source anchor
chooses a pair before checking roles. For attribution, compare
ridge and prototype bands, left-priority recheck, proof-aware
recheck, and proof-aware recheck with no call if neither clause
can complete a same-author proof. Evaluate with given sources
and with the retrieved pair. Hide each target article in turn
to count any output attributed to a substitute source. The
oracle source arm returns no pair when one target is hidden.

## Results

The filter retained 280 articles. The constructed test has 51
positive requests and 51 swaps over ten article pairs. With
both targets visible, anchored search finds the designated
pair on all 102 excerpt requests and 72 keyword requests.

| Clue and role route | Strict proofs / 51 | Adverse / 102 | Substitutions on missing source / 204 | Reader calls / 102 | Mean local ms | Local P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Excerpt, ridge band | 13 | 0 | 0 | 0 | 11.9 | 12.9 |
| Excerpt, prototype band | 23 | 0 | 0 | 0 | 11.9 | 12.9 |
| Excerpt, left priority | 31 | 3 | 2 | 77 | 23.1 | 37.9–38.1 |
| Excerpt, proof-aware | 32 | 3 | 2 | 77 | 23.5 | 38.4–38.6 |
| Excerpt, proof-aware with skip | 32 | 3 | 2 | 33 | 16.8 | 34.5–34.7 |
| Keywords, ridge band | 9 | 0 | 0 | 0 | 10.3–10.4 | 10.6–10.8 |
| Keywords, prototype band | 16 | 0 | 0 | 0 | 10.3–10.4 | 10.6–10.8 |
| Keywords, left priority | 24 | 3 | 0 | 55 | 18.3 | 36.0 |
| Keywords, proof-aware | 25 | 3 | 0 | 55 | 18.8 | 36.1 |
| Keywords, proof-aware with skip | 25 | 3 | 0 | 25 | 14.0–14.1 | 30.2–30.5 |

With given sources, the skip returns the same 32 strict proofs and
three adverse outcomes for both clue views. One adverse outcome is
an output on a role swap. The gap between 25 keyword proofs with
retrieved sources and 32 with given sources quantifies source
availability loss for that route. Hiding an excerpt source causes
two incorrect substitutions after recheck, while both static role
bands emit no substitutes on the same views.

The structural skip saves 44 of 77 calls on excerpt queries and 30
of 55 on keyword queries, with the same outputs under the same
role bands and source decisions. Compared with left priority,
the adaptive choice adds just one proof in each clue view.
Compared with the fastest static route, the skip finds nine
additional strict proofs in each view, adds three adverse
outcomes to each, and increases mean and tail latency. The gate
certifies lack of opportunity within the fixed bands and source
pair. It does not certify source sufficiency or absence of a
valid contributor. The evidence supports pruning futile reads.
Quality and risk improvements from adaptive selection require
further evaluation.

Both tasks in Slurm array 45931 on ws3 RTX 5090 reproduce all
non-timing fields under reversed query order. Times include a
live batch with both clues and both phrases, ranking, source
selection, declaration matching and the selected reader call if
any. Each factored timing charges a reader call even if the
harness reuses its measured output in a paired arm. Initialization,
index construction, raw request parsing and service transport are
outside the timer. These are ten linked article pairs with repeated
wording and roles, within the scope of publisher declarations.
[Full results](results.json)

## Reproduce

Rebuild the eLife projection with the pinned source manifest.
Use the prior engineering search output for the role bands and
the previous single-role recheck output for the frozen cutoff.
Run inside a Slurm CUDA allocation with a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.elife_one_role_transfer \
  ELIFE_PROJECTION NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```

Repeat with `--reverse`.
