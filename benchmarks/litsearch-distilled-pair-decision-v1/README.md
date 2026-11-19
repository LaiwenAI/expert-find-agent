# Distill pairwise source confirmation without an extra model call

Test whether the [4B pair decision](../litsearch-contrastive-confirmation-v1)
can improve the 0.6B path at the same online model budget.
The task is to propose one source or abstain. LitSearch labels
intended papers for each question. Returns outside those labels
are unjudged. [LitSearch](https://aclanthology.org/2024.emnlp-main.840/)

Use the pinned corpus, exact dense retrieval, and four search
routes. Every online route receives the same two sources.

| Route | Source decision | Models after retrieval |
| --- | --- | --- |
| Raw | Rank independent source margins, apply score and gap thresholds | One 0.6B batch |
| Supervised | Shared seven-parameter head, then the same calibration | One 0.6B batch |
| Distilled | Same head, with added soft-target loss | One 0.6B batch |
| Teacher | Compare both sources and a neither option in one prompt | One 4B batch |

The head scores each source against a fixed zero abstention
logit. Inputs are the independent reader margin, the difference
from the other source's margin, dense cosine and its difference,
and query-term coverage in title and abstract. Shared weights
apply the same operation to both source positions. Each route
uses separately fitted score and margin thresholds.

Use the 128 training questions and 96 calibration questions
from disjoint inline partitions and the 246 author-written test
questions. Query strings and released target IDs are disjoint
across partitions. Train on the dense top two with references
present or withheld. When a known reference is in the present
top six, add its pairing with up to two high-ranked alternatives.
Deduplicate pairs within a request. This reduced supervision
does not prove alternatives are irrelevant.

Both heads use the same pointwise features, known labels,
optimizer, and iterations. The supervised loss maximizes the
probability mass of any released target in the pair, or the
abstention mass when the pair has none. The distilled head adds
a KL term at temperature 2, weighted by 0.5 with standard
temperature-squared scaling. The teacher's margins are averaged
across both paper orders to generate the offline soft targets.
The teacher's assessment is never needed in a student request.
The training manifest records paper IDs, query hashes, released
target membership, and teacher margins.

Report the resulting test operating points after separate
2%, 5%, and 10% ceilings in each calibration view. On test,
remove all known targets before reretrieval in the withheld
view. Such returns are unjudged alternatives, whose true
relevance needs further assessment. Paired intervals cluster
by connected sets of reference papers. Serial P95 includes
live retrieval, input construction, model inference, and student
scoring when applicable. Loading and offline teacher work are
reported separately.

## Results

Slurm job 43271 completed on one RTX 5090 on ws5. The fixed
schedule yields 287 training pairs in 20.49 s. The teacher's
unthresholded source preference changes after reversing the
papers in 59 of these pairs. The teacher's highest-scoring
option after averaging disagrees with reference membership
on 146 pairs. This comparison includes unjudged alternatives
whose actual relevance is unresolved. Each decision head
has seven trainable parameters and takes less than 0.18 s to
fit after caching training features and teacher decisions.

Each cell shows **known return / unjudged return with targets
present / unjudged return after withholding**.

| Route | 2% calibration ceiling | 5% calibration ceiling | 10% calibration ceiling | Present P95, ms |
| --- | ---: | ---: | ---: | ---: |
| Raw | 23 / 2 / 3 | 30 / 7 / 9 | 42 / 8 / 14 | 25.21 |
| Supervised | 23 / 3 / 3 | 63 / 8 / 21 | 68 / 13 / 27 | 25.16 |
| Distilled | 32 / 3 / 4 | 54 / 5 / 14 | 61 / 9 / 19 | 25.20 |
| Teacher | 20 / 1 / 3 | 34 / 0 / 2 | 46 / 4 / 8 | 40.26 |

The strict 2% fit shows the largest gain attributable to
teacher targets over the same supervised head. The distilled
route returns 32 known papers versus 23 for either comparator.
The paired known-return difference against supervision is
+3.66 percentage points (95% target-connected interval +1.62
to +6.12). The corresponding interval against the raw
reranker is 0.00 to +7.35 points. The distilled route makes
one more unjudged return in the withheld view than either
comparator. Present P95 stays near 25.2 ms.

At 5% and 10%, supervision returns more known papers than
distillation and accepts more unjudged alternatives. These
are comparisons of observed operating points. An equal-risk
quality claim requires independent judgments of the alternatives
and a held-out comparison at a prespecified risk threshold.
The result is limited to source confirmation on previously
explored LitSearch test questions.

Earlier work distilled listwise rankings into smaller
rerankers. This comparison distills a two-source decision
with an abstention option into a head over one frozen small
model batch. [RankGPT](https://aclanthology.org/2023.emnlp-main.923/)

[Aggregate results](results.json) · [Training decisions](training-pairs.json)

## Reproduce

Use the pinned inputs and open checkpoints from the pair
confirmation experiment. Run inside Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.litsearch.distilled_pair_decision \
  PROJECTION DENSE_CACHE EMBEDDING RERANKER QWEN4B NEW_RESULTS
```
