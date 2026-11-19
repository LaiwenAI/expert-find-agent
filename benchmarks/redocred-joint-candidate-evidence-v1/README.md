# Compare evidence for the last candidate before making a replacement

Reuse the source retrieval, extracted entities, fixed 4B reader and
306 exact or transposed requests from the
[automatic-index benchmark](../redocred-automatic-entity-index-v1).
For a given query all policies retrieve the same sources and use the
same initial candidate list. Keep every position except the last
under its original independent evidence check. Select a single
challenger from predicted people and the role-local and boundary
proposal pools. Rank only people outside the original candidate list
with the same lexical proximity scorer. When a challenger exists,
all methods use this same person and the same two-sentence windows.
No candidate proposal or evidence selection uses human labels.

| Policy | Last-position verifier |
| --- | --- |
| Keep | Verify the incumbent independently |
| Replace | Verify only the challenger |
| Independent | Verify incumbent and challenger separately in the same batch and accept the stronger supported candidate |
| Joint | Present the two attributed windows separately within one prompt. Choose candidate 1, candidate 2 or neither. |
| Joint reversed | Reverse the pair order relative to Joint; keep all other inputs unchanged |

The baseline and the first positions apply the preceding exact same
binary score cutoff. In the joint arm, accept the preferred name when
its class-score margin over neither exceeds 0.25. The scales of the
binary and joint prompts need not match. This is a fixed-cutoff
comparison without joint threshold calibration. Both prompts ask for
explicit support for the queried role and work. The joint instruction
prohibits borrowing evidence from the other person's excerpt.

Order candidate labels by a stable query hash. Use the reversed arm
to measure sensitivity to the label and excerpt order. Track prompts,
unpadded tokens and padded tokens per batch. A joint comparison uses
one prompt for two excerpts and can still consume more tokens and
padding than two narrower prompts. All GPU timings include live
retrieval, the cheap proposal decision, evidence windows, prompts
and one warmed model forward per query. They exclude service queues
and model load. Descriptive paired intervals resample reference
documents. The evaluation uses document-level relationship judgments,
with unjudged candidates counted separately.

## Results

Slurm job 43071 on ws5 used one RTX 5090 and the pinned
Qwen3-4B-Instruct-2507 revision recorded in the results. Counts are
accepted identities summed across queries.

| Query view | Policy | Known positive | Annotated competing role | Unjudged | Padded tokens | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Exact (161) | Keep | 98 | 3 | 10 | 133,237 | 58.15 |
| | Replace | 95 | 2 | 41 | 132,283 | 58.12 |
| | Independent | 99 | 3 | 41 | 162,047 | 69.10 |
| | Joint | 100 | 13 | 58 | 209,866 | 95.57 |
| | Joint reversed | 100 | 7 | 62 | 209,861 | 95.83 |
| Transposed (145) | Keep | 76 | 1 | 6 | 118,330 | 59.09 |
| | Replace | 71 | 1 | 30 | 116,989 | 58.56 |
| | Independent | 79 | 1 | 29 | 144,204 | 70.52 |
| | Joint | 81 | 11 | 47 | 186,696 | 96.37 |
| | Joint reversed | 77 | 5 | 61 | 186,696 | 96.40 |

Joint and Keep have identical prompt counts. Joint uses about 15%
more unpadded tokens, 58% more padded tokens and 63–64% higher P95.
Independent uses fewer padded tokens and lower P95 than Joint despite
sending more prompts. Reversing candidate order changes the last
decision, including abstention, on 81/161 exact and 69/145 transposed
queries. After batching, two exact queries also change an earlier
acceptance despite using the same prompts for those positions.

Joint versus Keep changes accepted known-positive P@5 by +0.25 and
+0.69 percentage points with reference-document bootstrap intervals
of [−0.67, +1.15] and [−0.16, +1.54]. This fixed-cutoff test provides
no convincing quality advantage for Joint across the two views. It
adds ten annotated competing-role acceptances in each view, plus
unjudged candidates. Human relationship judgments are incomplete,
and this benchmark measures documentary roles rather than suitability
for a real industrial demand.

[GPU aggregate results](results.json)

### Where a better nomination could help

A separate label-based diagnostic classifies each query before model
verification. It measures the maximum additional labeled positive from
a single replacement if all available baseline positives are kept.
The groups are mutually exclusive.

| Last-slot diagnosis | Exact | Transposed |
| --- | ---: | ---: |
| Last slot already contains an annotated positive | 15 | 20 |
| No additional annotated positive outside the shortlist | 78 | 58 |
| Source for any additional annotated positive was missed | 6 | 17 |
| Retrieved source, but no additional positive identity in the proposal pool | 13 | 12 |
| Additional positive was proposed, but fell below the chosen challenger | 43 | 32 |
| Chosen challenger is an additional annotated positive | 6 | 6 |

An oracle restricted to the single lexical challenger has room to add
one known positive on six queries per view. An oracle over all proposals
can add one on 49 exact and 38 transposed queries. This is headroom
under incomplete judgments, not achievable verifier performance.
The larger measured opportunity is deciding whom to nominate from
the existing pool. The source gap is also larger under transpositions.

[Candidate audit](headroom.json)

## Reproduce

Use the pinned snapshot and NER model from the
[automatic-index benchmark](../redocred-automatic-entity-index-v1).
The reader checkpoint is Qwen3-4B-Instruct-2507. The projection hash
covers the measured input file, including extraction timing metadata.
Run the first command in a Slurm GPU allocation and the second in
the same Python environment on a compute node with CPU access.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.compare_candidate_evidence \
    REDOCRED_DIR NER_PROJECTION.json PINNED_QWEN4B NEW_RESULTS.json
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.docred.audit_candidate_headroom \
    REDOCRED_DIR NER_PROJECTION.json NEW_HEADROOM.json
```
