# Nominate a challenger from evidence the incumbent leaves unexplained

Use the same three CoF query panels, owned publication vectors and
frozen embedding/reader checkpoints as
[the fixed-issuance transfer](../cof-contrastive-swap-v1/README.md).
The strong initial ranker averages profile-centroid similarity and
best owned-source similarity. Every route protects its first four
reviewers. Only the fifth slot can change. Search ranks six through
ten for one challenger with at least as many readable sources as the
incumbent. If none qualify, keep the original five.

Let `q` be the existing query vector and `s` the unit vector of the
incumbent's best source. Form a residual by removing the positive
projection of `q` onto `s` and normalizing the result. The nomination
score is the maximum, over each challenger's owned publications, of
the equal mean of raw source similarity and residual similarity.
This gives credit to evidence aligned to aspects of the query that
the incumbent's leading publication does not cover in the embedding
space. It is a representation hypothesis, without labels linking
individual abstract passages to target requirements.

| Arm | Challenger | Challenger's three possible reads |
| --- | --- | --- |
| Dense five | None | None |
| Sixth | First eligible challenger in the original order | Top dense sources |
| Raw best | Highest single owned-source cosine | Top dense sources |
| Unshared best | Highest owned-source cosine with the existing global source-sharing penalty | Top dense sources |
| Residual nomination | Highest joint raw and residual source score | Top dense sources |
| Residual evidence | Same nomination as the preceding arm | First dense source, best remaining joint source, next unused dense source |

Each replacement route reads the same number of reviewer–source
pairs on every request. Perform one 0.6B scoring batch for the four
protected reviewers, the incumbent and the one nominated challenger,
reading at most three sources per reviewer. Compare the candidate
maxima over the first two sources. Swap only if the challenger has a
strictly higher score. The selected reviewer gets the first two
positive-margin citations if available. The dense-five reference
skips the extra challenger rows. None of the parameters uses CoF
ratings. The comparison reports separately whether the nominee could
improve known relevance, whether the verifier actually chooses it,
and whether the selected reviewer also receives citations.

Use paired request intervals for known-positive changes and report
bounds when the labels of swapped reviewers are missing. A positive
reviewer with model-selected citations is a ranking and model-based
coverage statistic. It does not adjudicate citation adequacy.
Timings cover a warm query embedding, full-inventory profile and
publication ranking, optional residual selection, verifier inference,
and assembling the ranked shortlist with citations. The shared
retrieval timing is charged to all arms. Serving queues, network and
index build are outside the measurement.

## Results

Slurm array `44280` ran all three panels on RTX 5090 in ws5. The
rotated SIGIR repeat reproduces every non-timing result. The
panels, hard labels, and vector protocols match the prior
fixed-issuance study, and the sixth and dense-five arms reproduce
its pointwise and dense counts. Bold entries identify the best
observed known-positive count within a panel.

| Panel | Route | Possible positive-for-negative swaps | Known positives at five | Cited positives | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| SIGIR (73) | Dense five | — | 63 | 18 | 114.90 |
| SIGIR | Sixth | 9 | 66 | 22 | 146.47 |
| SIGIR | Raw best | 11 | 64 | 20 | 138.03 |
| SIGIR | Unshared best | 11 | 64 | 20 | 138.27 |
| SIGIR | Residual nomination | 13 | **68** | 21 | 138.21 |
| SIGIR | Residual evidence | 13 | 67 | 20 | 138.30 |
| KDD (122) | Dense five | — | **24** | 13 | 165.00 |
| KDD | Sixth | 0 | 20 | 12 | 200.69 |
| KDD | Raw best | 0 | 17 | 10 | 200.15 |
| KDD | Residual nomination | 0 | 17 | 10 | 200.64 |
| KDD | Residual evidence | 0 | 16 | 9 | 200.45 |
| SciRepEval (59) | Dense five | — | 52 | 25 | 140.30 |
| SciRepEval | Sixth | 0 | 51 | 24 | 180.13 |
| SciRepEval | Raw best | 3 | **54** | 27 | 166.10 |
| SciRepEval | Residual nomination | 4 | 52 | 26 | 166.15 |
| SciRepEval | Residual evidence | 4 | 52 | 26 | 165.34 |

All SIGIR replacements read 1,314 pairs; dense five reads 1,095.
Compared with sixth, residual nomination finds four more confirmed
swap opportunities, and the verifier makes three more positive
replacements at the cost of one additional loss. Its observed
P@5 change against dense five is +1.37 percentage points, with
a paired request interval of −0.55 to +3.29 points. It provides
one fewer cited positive than the sixth strategy. Changing the
second source changes 21 SIGIR evidence sets and loses one more
positive relative to residual nomination with dense sources.
The comparable known-positive gains do not transfer to KDD or
SciRepEval. Both panels have partial reviewer judgments and the
missing-label bounds include both improvement and deterioration.

## Relation to MMR

Write `c = max(0, q·s)` and `t = sqrt(1 − c²)` for unit vectors
and `c < 1`. The joint raw and residual objective has the same
source order as `q·v − c/(1+t) · s·v`, for a publication vector
`v`. The residual method therefore implements a query-dependent
redundancy penalty against one incumbent publication. This is a
form of the relevance–redundancy score in
[Carbonell and Goldstein's MMR](https://doi.org/10.1145/290941.291025).
Slurm job `44288` tests two fixed penalty ratios, 0.25 and
0.50, on the fully judged SIGIR panel. Each arm reads 1,314
pairs with the same verification and swap rule.

| Coefficient | Known positives at five | Cited positives | P95 (ms) |
| --- | ---: | ---: | ---: |
| 0.25 | 67 | 20 | 137.81 |
| 0.50 | 67 | 21 | 146.78 |
| Query dependent | 68 | 21 | 138.58 |

Adaptive coefficients range from 0.26 to 0.48, with a median of
0.35. The paired intervals against either fixed rule include zero.
The observed gains are consistent with simple incumbent-specific
redundancy reduction. The adaptive coefficient has no demonstrated
advantage over this MMR control on the panel.

[All GPU aggregates](results.json) · [MMR control](mmr.json)

## Reproduce

Use the pinned CoF snapshot and checkpoints from the linked transfer
study inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_residual_challenger \
  SIGIR EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON --order-offset 0
```

Repeat for KDD and SciRepEval. Run the fixed-penalty control on
SIGIR with the same three checkpoint and output arguments.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_incumbent_penalty \
  EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON
```
