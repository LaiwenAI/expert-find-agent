# Coverage and ranking under a fixed SE-PEF source budget

The two panels use the same 3,275 experts, pre-2020 source boundary, 256 validation queries and 512 test queries. Both retain the same count per expert, up to eight answers. Reserving one recent answer for each of an expert's most active communities before filling by recency exchanges 4,623 of 26,123 sources. The upstream member and query checksums match the original panel. [Recent panel](../sepef-temporal-intake-v1/README.md) · [Reserved panel manifest](reserved-manifest.json)

| Test outcome | Most recent | Community reservation |
| --- | ---: | ---: |
| Answerer reachable in query community | 448 / 512 | 478 / 512 |
| Text-only BM25 max, community | 12.50% Hit@5 | 8.40% Hit@5 |
| Text-only BM25 sum, community | 13.87% Hit@5 | 7.03% Hit@5 |
| Exact dense, community | 8.59% Hit@5 | 7.03% Hit@5 |
| Expert centroid, community | 11.52% Hit@5 | 8.20% Hit@5 |
| Dense plus BM25, community | 11.91% Hit@5 | 8.20% Hit@5 |
| Full-history prior, community | 38.09% Hit@5 | 38.09% Hit@5 |
| Prior plus validation-selected lexical fusion | 39.06% Hit@5 | 38.28% Hit@5 |

The 448 queries reachable in both panels have a mean of 6.49 versus 3.97 retained in-community answers owned by the labeled answerer. Reservation extends availability to 30 additional queries while reducing the answer depth for 304 of the 448 already reachable queries. Source availability alone therefore overstates retrieval gains. This reusable panel measures released answerer ranking and answer ownership. Citation relevance requires separate judgments.

## Measurement

The dense arms above are independent warm request replays on RTX 4090 GPUs, with individual encoding and serialization in each request. Rank-only paired scoring uses the same union vectors and query embeddings for both panels. The lexical arms compare a maximum answer score against summing scores over the 100 highest scoring candidate answers, adapted to the same local corpus and known community. The fusion grid includes prior-only execution and selects the lexical weight on validation MRR@10. Both panels chose weight 0.05. On test the Hit@5 increments versus the prior were +0.98 percentage points (95% community-bootstrap interval -1.16 to +3.96) and +0.20 points (-2.41 to +3.61). Lexical fusion took 6.08 and 6.27 ms P95 in-process, versus about 0.009 ms for the materialized prior. These results do not establish a ranking gain over the cheaper prior. [Lexical recent](lexical-recent.json) · [Lexical reserved](lexical-reserved.json) · [Reserved warm request replay](reserved-results.json)

## Paired dense assessment

Encoding the union of 30,746 distinct answers once recovers the same test-quality numbers as the independent warm replays. The reservation-minus-recent Hit@5 differences were -1.56 percentage points for maximum answer score (95% community-bootstrap interval -3.05 to -0.32) and -3.32 points for the expert centroid (interval -5.89 to -1.22). Neither dense aggregation placed a labeled answerer in the top five on any of the 30 newly reachable queries. Preserve community availability as an index diagnostic and judge ranking quality separately. [Paired dense results](paired-dense-results.json)

## Reproduce

Build the recent panel, the pinned full-history prior, and the reserved panel:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.sepef_reserved_text_panel \
  RECENT_PANEL NEW_RESERVED_PANEL --workers 4
PYTHONPATH=src:. python -m experiments.retrieval.sepef_prior_fusion prepare \
  EXPERT_TEST_DATA RECENT_PANEL LOCAL_PRIOR NEW_PRIOR_RESULTS
```

Run the CPU lexical control once per panel and the CUDA comparisons inside Slurm compute allocations:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.sepef_lexical_history \
  PANEL LOCAL_PRIOR SCRATCH_DIR NEW_LEXICAL_RESULTS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.sepef_first_query \
  NEW_RESERVED_PANEL QWEN_EMBEDDING_CHECKPOINT SCRATCH_DIR NEW_WARM_RESULTS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.sepef_paired_history \
  RECENT_PANEL NEW_RESERVED_PANEL QWEN_EMBEDDING_CHECKPOINT NEW_PAIRED_RESULTS
```
