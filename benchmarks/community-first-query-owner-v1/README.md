# Search by account profile or attributed answer on the first query

Compare author-first and source-first search using live open-weight models
and fixed five-answer scrutiny. Reuse the frozen
[Cross Validated snapshots](../crossvalidated-witness-retrieval-v1) and
[MathOverflow snapshot](../community-witness-matched-issuance-v1). Each
question links to an older canonical question whose accepted answer is the
known witness. The same account may own multiple answers. Only answer
bodies enter the index and verifier, and no gold labels enter either.
Apply the prior source and title screens to CV 2025 and cross-site text
screens to MathOverflow. Catalogs are sampled and contain the known
witnesses. [Answer acceptance](https://stackoverflow.com/help/accepted-answer)
records the asker's decision. Other citations remain unjudged.

## Comparison

| Arm | Nomination and quotation |
| --- | --- |
| Profile first | Rank normalized author means and retrieve each of the top five authors' highest-scoring answer. |
| Source dense | Rank all answers by dot product. Use each account's best result and return the first five distinct accounts. |
| Source hybrid | Apply BM25 and dense reciprocal-rank fusion, then return the first five distinct accounts and their best answers. |
| Source bounded | Score the 512-coordinate prefix of every answer. Use a residual norm upper bound to restrict full-length scoring while retaining the dense output. |

The bounded path uses Cauchy–Schwarz on the tail. Its comparison adds
conservative float32 slack and requires the fifth distinct owner's score
to exceed all unevaluated upper bounds before stopping. Ties in exact
score are settled by ascending answer ID. Verify every result against
exhaustive ranking on these panels after selection. The certificate
preserves the dense score's decisions. It does not establish relevance.

Every arm re-encodes the live request and verifies exactly one answer per
account in one five-pair batch. Use the same frozen Qwen3 0.6B encoder
and reranker. The reader sees up to 950 request characters and 850 answer
characters and emits a citation only when its margin is above zero.
Rotate arm order by request and by invocation. Run a warm-up request
outside the aggregate. Measure one-client warm local P95 from request
encoding through verification, plus the separate search region. There is
no service transport, request queue, or approximate index in this test.

Count known accounts, exact witnesses selected, and exact witnesses
issued. The supported P@5 contrast uses fixed five-slot denominators and
paired resampling by clusters of shared answers. Also report issuance
counts and token clipping. Work counts in the result refer to full-length
**answer** scores. Profile scores one full vector per author in addition,
and the bounded arm scores one half-length vector per answer in addition.
The offline build time includes construction of all indexes.

## Results

Both tasks in Slurm array `45666` completed on the ws3 RTX 5090.
Rotating execution order reproduced all counts and paired intervals.
Each cell shows known citations issued / all citations issued.

| Panel | Requests | Profile first | Source dense | Source hybrid | Source bounded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross Validated 2024 | 68 | 30 / 56 | 36 / 67 | 24 / 43 | 36 / 67 |
| Cross Validated 2025 | 27 | 15 / 23 | 16 / 26 | 14 / 23 | 16 / 26 |
| MathOverflow | 20 | 13 / 13 | 13 / 13 | 9 / 9 | 13 / 13 |

The CV 2024 dense-vs-profile advantage appears before verification,
with 59 versus 51 known witnesses in the five-answer sets. The final
supported P@5 difference is +1.76 percentage points (paired 95% interval
+0.30 to +3.53). The 2025 difference is +0.74 points (0 to +2.31).
Dense source-first uses more issuance on both CV cohorts. Without
judging unlisted answers, the gain supports known-witness recovery
at equal verification work, without establishing higher precision
at equal issuance. Each run verifies 2,300 pairs across the four arms
for measured requests. No texts hit the reader's token cap.

The bounded planner preserves every selected and issued answer in
each run. It still computes 98.1% to 99.9% of the full-length answer
scores, in addition to a prefix scan. CPU-search P95 is 0.385–0.449 ms
versus 0.076–0.082 ms for exhaustive dense retrieval. Cross-arm warm
request P95 is 47–49 ms, dominated by encoder and verifier execution.
Use exhaustive dense source-first as a strong first-query comparator.
The prefix bound gives no efficiency benefit at this catalog size and
embedding split. The compressed owner profile remains an applicable
control when per-author storage is constrained. A practical scale
comparison would also need a tuned approximate source index.

[Aggregate results for both runs](results.json).

## Reproduce

Restore the same three snapshot folders and frozen checkpoints. Run on an
allocated GPU, with a fresh output path for each order rotation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.community.first_query_owner_routing \
  CV_2024 CV_2025 MATHOVERFLOW QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS \
  --order-offset 0
```
