# Run role-index recovery through the live enterprise first-query stack

Reuse the fixed development corpus, requests, model revisions and
fast shared-prefix parsing from the
[enterprise first-query experiments](../enterprise-prefix-fork-v1).
No sealed request is accessed. Load an exact GPU index of the
52,667 development sources. Build a separate postings index from
the bounded project, patent and named profile claims used in the
[role-pool expansion](../enterprise-role-pool-expansion-v1).
Verify that each role window matches its original owner, source,
and character offsets.

Each arm runs the shared qualification parser, request embedding,
dense expert retrieval, original verifier and recommendation
construction independently for every request. Rotate execution
order per request. The benchmark requires identical initial
qualifications, ranking and evidence in all four arms.

| Arm | Additional action |
| --- | --- |
| Whole | Return the original output |
| Deeper on abstain | When the original output has no recommendation, consider role windows for the original top five candidates |
| Global on abstain | At the same trigger, consider role windows across the full development index |
| Global always | Consider global role windows whenever the initial list has free output slots |

Each attempted fallback considers at most two original role
windows. Verify every parsed criterion on each window, with one
batched Qwen3-Reranker-0.6B forward. Add an expert only when the
same window has a positive margin for every criterion. Existing
recommendations remain first, and the total list is capped at
five experts. The parser and verifier preserve overlength safeguards.
The on-abstention rule uses no learned feedback and does not
alter the first pass.

Timing is warmed execution from original request parsing through
recommendation and citation construction, including fallback
retrieval and verification. Dataset loading, index building,
service queues and network are outside this measure. Report actual
model pairs, rather than treating the maximum two-window budget as
equal realized compute. Compare with paired request bootstrap
intervals for P95. All acceptance counts are provisional model
screening. A separate restricted packet uses exact source offsets
and method-blind forms for external human adjudication.

## Results

Slurm job 43135 builds the source-vector cache in 297.34 seconds.
Job 43137 uses the same RTX 5090 on ws5 and processes all 80
development requests. Every initial recommendation, qualification
span and candidate list matches the earlier frozen enterprise trace
and is identical across all four arms.

| Arm | Requests with a provisional result | Added expert slots | Additional criterion checks | Live P50 (ms) | Live P95 (ms) | Paired P95 change vs whole, 95% interval (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Whole | 34 | 0 | 0 | 187.07 | 1129.47 | 0 |
| Deeper on abstain | 35 | 1 | 132 | 195.12 | 1131.40 | +0.22 to +30.26 |
| Global on abstain | 37 | 3 | 184 | 196.10 | 1156.39 | +15.18 to +63.28 |
| Global always | 37 | 14 | 254 | 198.86 | 1154.99 | +12.84 to +60.18 |

The two global routes answer the same initially unanswered three
requests, with identical evidence and candidates on all initially
unanswered queries. On-abstention uses 70 fewer fallback criterion
checks than always-on, a 27.6% reduction in fallback checks. It
forgoes 11 additional provisional slots on already answered requests.
Neither global route improves P95 over the baseline. The 2-window
cap does not imply equal measured compute: deeper search attempts
46 requests but skips steps when its pool has no eligible window.

All three global-on-abstain additions come from outside the
original shortlist. The exact-span packet has 85 distinct
method-blind views and zero human judgments. With a fixed
five-slot denominator, the three added results bound the possible
strict P@5 increase on these same requests to 0–0.75 percentage
points before adjudication; the unconditional false-promotion
risk is unresolved. The study measures warm, serial first-query
execution and provisional recall. Queue behavior and the sealed
enterprise evaluation remain separate.

[Aggregate results](results.json)

## Reproduce

Create the pinned development JSONL projection as documented in
[the original first-query benchmark](../enterprise-first-query-v1).
Compile role windows locally from the authorized dataset. Build the
dense cache and run the paired experiment inside Slurm CUDA
allocations. Keep the corpus, role windows, vectors and review
packet private.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_role_first_query roles \
  DATASET DEVELOPMENT_CORPUS PRIVATE_ROLE_WINDOWS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_role_first_query vectors \
    DEVELOPMENT_CORPUS MODELS/qwen3-embedding-0.6b PRIVATE_VECTORS.npy PRIVATE_VECTOR_META
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_role_first_query run \
    DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS PRIVATE_ROLE_WINDOWS PRIVATE_VECTORS.npy \
    PRIVATE_VECTOR_META MODELS NEW_RESULTS.json PRIVATE_TRACE
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_role_first_query packet \
  DATASET PRIVATE_TRACE PRIVATE_PACKET
```
