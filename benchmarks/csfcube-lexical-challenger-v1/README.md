# Allocate two verification slots to complementary source nominations

Test whether a lexical challenger can recover a strongly relevant study
missed by the first two dense nominations, and whether requiring lexical
agreement after verification suppresses legitimate recommendations.
[CSFCube](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/20f07591c6fcb220ffe637cda29bb3f6-Abstract-round2.html)
provides expert-judged relevance for background, method and result facets.
Use the 50 public query-facet pairs from 34 query papers in the
[earlier replay](../csfcube-frozen-confirmation), excluding the query
paper. Preserve the released candidate pool per request. In a separate
view, remove every grade-2 or grade-3 candidate before retrieval.
Relevance grade is used for calibration and evaluation only, apart from
the withholding intervention.

Keep the same dense six candidates and frozen Qwen3 0.6B embedding and
cross-encoder checkpoints. The lexical challenger ranks the dense six by
BM25 of the query paper title and abstract, omitting the literal facet
instruction. Keep the first dense candidate. Give the second slot to the
lexical first choice if it differs from the first dense candidate, or to
the second dense candidate otherwise. Dense rank resolves lexical ties.
The reranker gets the full query with facet instruction. Use one batch
per arm.

| Policy | Verification slots and acceptance |
| --- | --- |
| Dense two | First two dense candidates, calibrated score and margin |
| Lexical challenger | Dense first and lexical challenger, calibrated score and margin |
| Challenger with agreement | Same batch, withhold any reranker winner that is not the lexical first choice |
| Dense six | All six dense candidates, calibrated score and margin |

Fit the same score-and-margin grid in four query-paper folds and apply it
on the fifth. Preserve the existing 5%, 10% and 20% calibration ceilings
for both rejected relevance grades in the candidate-present view and any
return in the withholding view. These are calibration constraints.
Report realized outcomes separately, including grades 0 and 1.
Pair uncertainty by query paper. Route timing adds separately measured
shared online dense search and each route's verification, plus BM25
when used. It excludes offline encoding, index build, transport and
queueing. The agreement route reuses the challenger batch and timing.

## Results

Slurm job 43323 ran on one RTX 5090 on ws5. Lexical nomination
changes the second slot on 20 of 50 present requests. It exposes a
strong candidate on one additional request and loses all strong
candidates on four. Total strong exposure drops from 35 to 32.
The six-candidate pool contains a strong candidate on 45 requests.
The challenger reranker puts a strong candidate first on 26
requests before applying a threshold, and the hard agreement rule
would reject 11 of those 26 before thresholding.

### Held-out folds at the 10% calibration ceiling

| Policy | Strong returns / 50 | Present grade 1 / 0 | Withheld grade 1 / 0 | Present P95 | Model pairs / request |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense two | 13 | 3 / 1 | 4 / 2 | 26.04 ms | 2 |
| Lexical challenger | 13 | 3 / 1 | 3 / 2 | 26.68 ms | 2 |
| Challenger with agreement | 10 | 1 / 1 | 2 / 3 | 26.68 ms | 2 |
| Dense six | 17 | 2 / 1 | 2 / 3 | 54.42 ms | 6 |

The lexical challenger recovers the same number of strong returns
as dense two at the middle calibration ceiling, with a descriptive
paired interval of −8.2 to +8.0 percentage points. It verifies the
same number of candidate pairs. At the 20% ceiling it returns 15
strong results versus 16 for dense two. Hard agreement also lowers
strong returns at all three ceilings, and at the middle ceiling it
adds one withheld grade-0 return while removing one withheld grade-1
return relative to the challenger alone. Each policy is calibrated
separately, so the additional grade-0 return reflects the fitted
operating point rather than a direct consequence of vetoing a single
candidate.

Retain dense two for the fixed small budget and dense six as the
broader control. These measurements distinguish the value of a
second nomination from agreement as an acceptance requirement.
They do not validate either a lexical replacement rule or a hard
agreement gate for naturally graded facet relevance. The dense-six
advantage remains uncertain on 34 query papers. The task concerns
paper-level relevance inside judged pools, not personal attribution
or full-corpus expert matching. [Results and calibration](results.json)

## Reproduce

Use the pinned CSFCube dataset and model files from the
[provenance manifest](provenance.json). Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.csfcube.lexical_challenger \
  CSFCUBE_DIR benchmarks/csfcube-lexical-challenger-v1/provenance.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
