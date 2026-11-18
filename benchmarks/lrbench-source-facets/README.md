# Multi-source evidence for LR-Bench reviewer matching

Use the released [LR-Bench](https://huggingface.co/datasets/Gnociew/LR-bench/tree/6b950b4a6f309d4e6fd3749744d2ddd9005fa6f8) paper-centric and reviewer-centric pairs. Both choices in each pair retain their released profile histories. Select up to three owned publications by frozen Qwen3-Embedding similarity. The runner verifies that no candidate history repeats the target with an exact normalized title.

Frozen Qwen3-Reranker-0.6B scores each selected paper for overall fit, method fit and problem fit. General scores are either the dense first source, the maximum over the selected sources, or their mean. The single-source facet score takes the smaller of method and problem fit for each source and keeps the best source. The two-source variant allows one source per facet and uses the same minimum to score the resulting coverage. A single citation wins ties. All aggregation policies share the same source depth. The facet scores are model predictions. The released grades evaluate the whole reviewer profile.

## Results

Slurm job `40470` used an RTX 5090 and covered 896 distinct query-history pairs. The table shows strict pairwise accuracy (ties count as misses) using the released evaluator. The RATE row comes from the authors' released predictions.

| Scoring rule | Reviewer selection, 212 pairs | Paper selection, 1,184 pairs |
| --- | ---: | ---: |
| Dense maximum | 141 (66.51%) | 880 (74.32%) |
| Dense top-three mean | 143 (67.45%) | 899 (75.93%) |
| General verifier, first dense source | 132 (62.26%) | 879 (74.24%) |
| General verifier, maximum | 134 (63.21%) | 885 (74.75%) |
| General verifier, mean | 140 (66.04%) | 860 (72.64%) |
| Facet verifier, one source | 139 (65.57%) | 876 (73.99%) |
| Facet verifier, up to two sources | 139 (65.57%) | 879 (74.24%) |
| RATE released predictions | 163 (76.89%) | 891 (75.25%) |

The two-source rule selects a second source for 11/336 unique reviewer-selection comparisons and 28/796 unique paper-selection comparisons. It changes no reviewer-selection correctness outcomes and gains three correct paper-selection pairs. Compared with dense top-three mean, it changes strict accuracy by −1.89 and −1.69 percentage points in the respective directions. The descriptive 95% anchor-bootstrap intervals are −8.02 to +4.39 and −5.26 to +2.03 points. The two fixed questions rarely lead to complementary evidence under this verifier. The dataset does not adjudicate either facet or individual sources, so these outcomes constrain the ranking claim and leave actual requirement coverage unresolved.

The job embeds profiles and queries in 9.66 s and scores 7,791 document-task pairs in 60.06 s, with zero truncated reader inputs. These are collection-wide batched timings with repeated pairs deduplicated. They do not measure request P95.

## Reproduce

Prepare the pinned inputs in the [LR-Bench comparison](../../docs/merit-suitability-card.md#lr-bench-pairwise-retrieval), including the released RATE predictions and evaluator. Then run [the facet scorer](../../experiments/retrieval/lrbench_source_facets.py) with `LRBENCH_RELEASE EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON` on a Slurm CUDA compute node.
