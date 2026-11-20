# Select three publications by topical specificity

Test whether a reviewer is better represented by publications
whose match is specific to the target, under a fixed three-source
budget. An article can score well for many similar queries.
Estimate that background affinity offline and subtract half of
it from each cosine score. This is an adaptation of
[Cross-Domain Similarity Local Scaling](https://openreview.net/pdf?id=H196sainb).
The experiment uses [LR-Bench](https://aclanthology.org/2026.acl-long.2035/)
self-rated familiarity comparisons, both directions, and the
published RATE prediction file.

The frozen Qwen3-Embedding-0.6B encoder maps publications into
their source and query representations. The background score
of each source is its mean similarity to its 10 nearest
publication-as-query vectors. Exclude sources with the same
normalized title from their own neighborhood. Discard
publications that share a normalized title with any evaluated
target before compiling the background and candidate histories.
The same filtered histories serve every local route. There is
no learning from ratings or evaluation query embeddings during
index construction. The source corpus is derived from the
released profiles and does not constitute an unseen corpus.

Each method chooses up to three distinct owned sources per
candidate and takes their mean score. The **dense** baseline
uses cosine for both stages. **Specificity selection** uses
the adjusted score for choosing sources and raw cosine for
aggregation. **Specificity** uses the adjusted scores for
both stages. The **two-sided** diagnostic additionally
subtracts half the mean similarity of the query to its 10
nearest source vectors. The query subtraction cancels across
two reviewers for the same paper by construction, but can
change comparisons in the reverse direction. Its extra scan
is included in timing.

Use the official strict pairwise accuracy and rating-gap
weighted Kendall loss. Bootstrap by paper or reviewer anchor.
Treat the paper-centric comparisons as primary. One inference
pass per pair supplies the same real-time query embeddings to
each arm. Report the measured encoding time plus each arm's
scoring time, with scoring order rotated. Timing excludes
offline compilation, serving overhead and source-level
verification. Familiarity labels evaluate the whole reviewer
profile. They do not establish that an individual publication
proves the reviewer's qualifications or contribution.

## Results

Jobs `43429` and `43432` reproduced every route's strict
accuracy on an RTX 5090. The times below are from the second
run. The filtered corpus contains 1,482 distinct publications;
11 title-overlapping publications were excluded. Computing
background penalties adds 6.80 s offline after document
encoding.

| Route | Paper selects reviewer / 212 | P95 | Reviewer selects paper / 1,184 | P95 |
| --- | ---: | ---: | ---: | ---: |
| Dense top-three mean | 143 | 7.65 ms | 900 | 11.86 ms |
| Specificity selection only | 141 | 7.64 ms | 900 | 11.86 ms |
| Specificity selection and scoring | 146 | 7.64 ms | 897 | 11.88 ms |
| Two-sided correction | 146 | 7.71 ms | 884 | 11.99 ms |
| RATE released predictions | 163 | — | 891 | — |

The source correction changed at least one ordered shortlist
in 125 of the 212 reviewer comparisons. Using it only for
selection lost two correct decisions. Using it for both
selection and scoring improved observed accuracy by 1.42
percentage points over dense, with a descriptive 95% paired
interval of −1.90 to +5.02 points. In the reverse direction,
the full two-sided correction lost 16 decisions relative to
dense. The comparison supports keeping cosine top-three mean
as the inexpensive scoring reference and RATE as the primary
published ranking comparator. It leaves open whether a
field-specific background can help, and does not provide
grounds to substitute this source correction for role-specific
candidate evidence. [Aggregate results](results.json)

## Reproduce

Use the pinned inputs listed in the
[LR-Bench release comparison](../../docs/merit-suitability-card.md#lr-bench-pairwise-retrieval).
Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.lrbench_specificity \
  LRBENCH_RELEASE QWEN3_EMBEDDING_CHECKPOINT NEW_RESULTS
```
