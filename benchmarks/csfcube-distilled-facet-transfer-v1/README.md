# Transfer distilled source decisions to judged facet pools

Apply the frozen [LitSearch pair-decision heads](../litsearch-distilled-pair-decision-v1)
to the expert-adjudicated [CSFCube](https://openreview.net/forum?id=8Y50dBbmGU)
query-by-example collection. Compare the raw small reranker,
the supervised seven-parameter head, the distilled head of the
same size, and the frozen 4B pair-choice teacher. Head weights,
feature normalization and all model checkpoints remain fixed.

Each of the 50 query–facet pairs has a released candidate pool.
Remove the query paper. Exact dense search using the full query
chooses two candidates from that judged pool. The withheld
view removes every grade 2 or 3 paper before search. All
verification conditions see the same dense pair in each view.

| Confirmation input | Text from the query paper |
| --- | --- |
| Full | Title and complete abstract |
| Matched lead | Title and the first N abstract sentences |
| Facet | Title and up to three sentences with relevant released rhetorical predictions |

All inputs preserve the explicit facet request. A missing facet
prediction falls back to the first abstract sentence. The
matched-lead control uses the same number of sentences as
the facet selection. Sentence lengths may differ. This
intervention changes only what the verifier reads and the
lexical features computed by the decision heads.

Report two threshold protocols. Frozen transfer keeps the
LitSearch score and gap thresholds for each route. Grouped
in-domain calibration fits thresholds on four query-paper
folds and evaluates on the fifth, with frozen model and head
weights. Use the existing grouped fit. Its 5%, 10% and 20%
ceilings limit accepted grade 0–1 papers when targets are
present and any acceptances when strong papers are withheld.
These constraints apply in the fitted folds. Report grades
0 and 1 separately in every held-out fold.

Serial P95 covers query encoding, judged-pool retrieval,
verification and decision. Query-view rendering and corpus
encoding are offline. Paired intervals cluster by query
paper. The benchmark evaluates decisions in known pools
with paper-level grades.

## Results

Slurm job 43279 ran 50 query–facet pairs from 34 query papers
on one RTX 5090 on ws5. Every method received the same top two
within each query and view. Those two include a grade 2 or 3
reference on 35 original requests. With frozen LitSearch
thresholds at 2%, all four full-input routes abstain on all
requests. At 10%, raw, supervised and distilled heads accept
two, three and two strongly relevant papers, respectively.
The teacher accepts none. These frozen results isolate an
acceptance threshold transfer failure from candidate exposure.

### Locally calibrated at the 10% ceiling

| Input | Route | Grade 2–3 | Present grade 1 / 0 | Withheld grade 1 / 0 | Present P95, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | Raw | 13 | 3 / 1 | 4 / 2 | 26.16 |
| Full | Supervised | 9 | 1 / 2 | 1 / 3 | 26.16 |
| Full | Distilled | 10 | 2 / 1 | 2 / 2 | 26.19 |
| Full | Teacher | 6 | 0 / 1 | 2 / 1 | 53.13 |
| Matched lead | Raw | 7 | 1 / 0 | 2 / 1 | 22.15 |
| Matched lead | Supervised | 5 | 0 / 1 | 1 / 2 | 22.22 |
| Matched lead | Distilled | 9 | 1 / 1 | 5 / 2 | 22.19 |
| Matched lead | Teacher | 1 | 0 / 1 | 6 / 1 | 42.06 |
| Facet | Raw | 11 | 3 / 0 | 5 / 1 | 22.48 |
| Facet | Supervised | 10 | 0 / 0 | 2 / 1 | 22.29 |
| Facet | Distilled | 10 | 1 / 0 | 5 / 1 | 22.28 |
| Facet | Teacher | 2 | 0 / 0 | 5 / 0 | 41.97 |

With full-input local calibration, the distilled head returns
ten strong papers versus thirteen for the raw reranker. Its
paired difference is −6 points (95% query-paper interval
−16.07 to +4.17). It reduces withheld grade 1 returns from
four to two and retains two grade 0 returns. The raw reranker
with facet-only context returns eleven strong papers and cuts
P95 from 26.16 to 22.48 ms, while the matched lead returns seven.
Neither result establishes a matched-risk improvement. With
the more permissive 20% local ceiling, short facet plus
distillation returns seventeen strong papers and one grade
0 paper after masking, compared with sixteen and three for
full-input raw ranking. Its paired strong difference spans
zero (−6.25 to +10.87 points). Weak returns and their
uncertainty are reported separately in the aggregates.

Labels select the same sentences as the matched lead for all
sixteen background requests, and different sentences on all
method and results requests. This supports facet selection
as an input-cost experiment. It leaves full-input raw as a
strong control and does not establish transfer of the trained
acceptance policy. The preselected pools and 34 query papers
limit the interpretation.

[Aggregate results, timing, and intervals](results.json)

## Reproduce

Download the four pinned files from the [existing CSFCube
replay](../csfcube-frozen-confirmation). Reuse the released
LitSearch head artifact and the same three Qwen checkpoints.
Run inside Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.csfcube.distilled_facet_transfer \
  CSFCUBE_DIR LITSEARCH_RESULTS EMBEDDING RERANKER QWEN4B NEW_RESULT
```
