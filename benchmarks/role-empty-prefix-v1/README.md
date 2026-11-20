# Dispatch a role-aware search when candidate declarations rule out the original prefix

The [matched-issuance audit](../matched-role-issuance-v1) attributes most
of the four-source role-first gain to a broader set of answered requests
or additional recommendation slots. Test whether a conservative
dispatch rule can capture these gains while preserving the topic
baseline's recommendations on requests it already answers.

Keep the existing topic ranking, embeddings, reranker, cutoff,
same-person binding and read budget. Inspect the declarations of the
first four ranked sources before model verification. Use the following
policies. Both dispatch rules use the topic-first route whenever any
of those declarations is incomplete or unavailable.

| Policy | Candidates sent for verification |
| --- | --- |
| Topic four | First four by topic |
| Topic eight | First eight by topic |
| Person four | First four eligible by topic, using same-person postings |
| Empty prefix | Person four if all original sources are indexed and none credits the requested pair to one person; otherwise topic four |
| At most one witness | Same dispatch when at most one original source has a same-person match |

A source with a missing declaration stays eligible for verification.
The gate refers only to declared roles from the same article, not
general capabilities. It changes candidate selection before the
verifier is called. Every source budget is charged once; there is no
verify-four-then-retry pass. The empty-prefix arm must preserve every
already-issued topic-four recommendation. It may increase coverage
when the four original complete declarations rule out the match.
The one-witness arm has no such guarantee.

Reuse both masks and all queries from [PLOS](../plos-bounded-topic-v1)
and [eLife](../elife-role-nomination-v1), with no split or threshold
changes. Reproduce the earlier reads, issuance and label-backed
counts for all three controls. Report more than the tagged hit count:
track dispatch rate, newly answered requests, off-label citations
and bounds on the support fraction among issued recommendations.
Topic-cluster intervals are descriptive on these reused panels.

## Results

Slurm jobs `43778` and `43781` on an RTX 5090 reproduced all non-timing
results. All controls reproduce their published outcomes. The strict dispatch
preserves every ordered recommendation already issued by topic four.
With all available declarations indexed, its effect depends strongly
on corpus and topic type.

| Panel | Method | Queries dispatched | Requests with recommendations | Tagged / off-label slots |
| --- | --- | ---: | ---: | ---: |
| PLOS, 46 requests | Topic four | 0 | 17 | 37 / 4 |
| PLOS | Empty prefix | 12 | 21 | 43 / 7 |
| PLOS | At most one witness | 18 | 23 | 47 / 13 |
| PLOS | Person four | 46 | 23 | 52 / 16 |
| eLife keywords, 48 requests | Topic four | 0 | 30 | 64 / 30 |
| eLife keywords | Empty prefix | 1 | 31 | 64 / 35 |
| eLife keywords | At most one witness | 8 | 34 | 65 / 40 |
| eLife keywords | Person four | 48 | 39 | 80 / 46 |

Every route in the table uses four reads per request. Strict dispatch
in PLOS retains six of 15 additional tagged slots and three of 12
additional off-label slots relative to always using person nomination.
The tagged P@5 gain over topic four is 2.61 points, with a descriptive
topic-cluster interval of +0.43 to +5.22 points. With the sparse mask,
strict dispatch triggers once and adds one tagged slot in PLOS.
For eLife keywords the strict rule adds five off-label slots and zero
tagged slots under full available coverage. At the sparser mask neither
dispatch rule triggers on that panel. In the eLife heading panel all
four-source methods produce 15 tagged and seven off-label slots.

The added eLife citations remain unjudged. If the known keyword tags
are assumed supported, the strict rule's change in the support fraction
among issued recommendations can range from −5.05 to +1.61 points.
PLOS's corresponding range is −6.00 to +1.76 points.
The first run's warm request P95 for the PLOS topic-four, empty-prefix
and person-four routes is 32.98, 35.98 and 35.95 ms respectively.
The strict gate does not inherit an end-to-end latency advantage
simply by dispatching fewer requests. It is a useful baseline for
preserving issued results, but a lack of same-person declarations
in four complete entries does not transfer as an indicator of useful
expansion. No human or enterprise judgments enter the replay.

[Aggregate results](results.json)

## Reproduce

Use the pinned public projections and the same checkpoints on one
Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_empty_prefix PLOS_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json benchmarks/plos-bounded-topic-v1/results.json \
  ELIFE_PROJECTION benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/elife-role-nomination-v1/results.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
