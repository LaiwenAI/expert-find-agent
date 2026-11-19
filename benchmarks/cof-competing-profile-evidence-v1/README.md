# Query-local source discrimination at a fixed verification budget

Test whether a source is more useful for choosing one expert when
it distinguishes that expert from the other leading candidates.
CoF assigns publications to reviewer profiles and supplies reviewer
relevance judgments. Use these associations to distinguish profiles.
Individual responsibilities and citation entailment require separate
annotations. [CoF dataset and paper](https://github.com/yuz9yuz/CoF)

Encode the complete query once with Qwen3-Embedding-0.6B. Search
the complete inventory using the equal blend of normalized profile
centroid cosine and best owned-paper cosine. Take its top ten
experts. Preserve the first four and compare the fifth expert
with alternatives in the same ten whose unadjusted blend is at most
0.03 below the incumbent. For source scoring, subtract 0.025 per
other leading profile carrying the same source. A replacement must
have a strictly higher adjusted blend and enough sources for the
incumbent's existing verification slots. All constants are fixed
for this comparison.

| Arm | Fifth expert | Source order |
| --- | --- | --- |
| Blend | Original | Raw cosine |
| Global specificity | Apply a fixed 0.025 × log₂ profile-count penalty to the best source when comparing eligible experts | Raw cosine |
| Local sources | Original | Apply the query-local penalty |
| Local rank | Compare eligible experts by adjusted blend | Raw cosine |
| Local joint | Compare eligible experts by adjusted blend | Apply the query-local penalty |

Verify up to three sources per person in one Qwen3-Reranker-0.6B
batch. Every arm evaluates exactly the same number of person–source
pairs per query. Use the original query, a shared prompt, and a
positive yes–no logit margin. Attach the first two accepted
sources when at least two are available. Report verifier-selected
citations together with the corresponding reviewer judgments.
The timed path includes query encoding, full inventory retrieval,
selection, tokenization and synchronization of the reader output.
Corpus preparation, model loading and queueing are separate.

Use the frozen KDD and SciRepEval panels from the [profile-pooling
study](../cof-global-profile-pooling-v1) and all eligible SIGIR
queries after excluding exact title, source ID and full-text
overlap. Hard reviewer labels score expert relevance. Paired
intervals resample queries. Also bound the change in P@5 over
all completions of the unjudged changed rows. The bounds and
bootstrap intervals answer different questions.

## Results

Slurm job 43224 ran all 254 queries on one RTX 5090 on ws5.
The SIGIR panel has complete labels across the returned top five,
and none of its 1,740 source IDs occurs under multiple profiles.
All arms therefore return identical results there. KDD and
SciRepEval have 14,397 and 6,668 shared sources, respectively.
The [profile-sharing audit](profile-sharing.json) counts each
distinct source ID at most once per released reviewer profile.

| Panel | Arm | Known positive at 5 | Known positive with citations | Unjudged with citations | Repeated citation slots | Replacements | P95, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIGIR | Blend | 63 | 18 | 0 | 0 | 0 | 118.42 |
| SIGIR | Global specificity | 63 | 18 | 0 | 0 | 0 | 118.43 |
| SIGIR | Local sources | 63 | 18 | 0 | 0 | 0 | 118.48 |
| SIGIR | Local rank | 63 | 18 | 0 | 0 | 0 | 118.32 |
| SIGIR | Local joint | 63 | 18 | 0 | 0 | 0 | 118.10 |
| KDD | Blend | 24 | 13 | 237 | 142 | 0 | 170.64 |
| KDD | Global specificity | 22 | 11 | 230 | 121 | 56 | 173.34 |
| KDD | Local sources | 24 | 12 | 235 | 118 | 0 | 170.99 |
| KDD | Local rank | 24 | 14 | 232 | 120 | 27 | 170.75 |
| KDD | Local joint | 24 | 13 | 230 | 97 | 27 | 170.88 |
| SciRepEval | Blend | 52 | 25 | 56 | 12 | 0 | 145.22 |
| SciRepEval | Global specificity | 48 | 24 | 58 | 11 | 16 | 145.28 |
| SciRepEval | Local sources | 52 | 26 | 56 | 9 | 0 | 145.37 |
| SciRepEval | Local rank | 51 | 24 | 57 | 11 | 5 | 145.56 |
| SciRepEval | Local joint | 51 | 25 | 57 | 8 | 5 | 145.43 |

The joint method reduces repeated citation slots from 142 to
97 on KDD and from 12 to eight on SciRepEval. Its cited known
positives are unchanged in both panels, and total known positives
stay level on KDD and drop by one on SciRepEval. Paired P@5
intervals for the joint method include zero on both panels.
For unknown labels, the possible P@5 differences span −3.61
to +3.28 percentage points on KDD and −1.36 to +1.02 on
SciRepEval. Adjusting citation order alone changes no expert
identities, both by construction and in the aggregate.

Every arm runs exactly 15 pair evaluations in one batch per
query. All pairs fit the verifier context. Joint serial P95
is within 0.24 ms of the blend on the two panels where sources
are shared. These observations support shared-source
downweighting as a citation redundancy control on these panels.
They give no demonstrated improvement in relevant experts
or independently judged attribution. The temporal measurements
report warmed serial execution. The experiment uses previously
explored CoF panels.

[Related author-association prior work](https://doi.org/10.1145/2911451.2914751) ·
[SIGIR aggregates](SIGIR.json) · [KDD aggregates](KDD.json) ·
[SciRepEval aggregates](SciRepEval.json)

## Reproduce

Use the existing CoF snapshot and vector protocol, and matching
local Qwen checkpoints. Run inside Slurm CUDA from the workspace
with the public snapshots mounted under `artifacts/public/cof`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_competing_profile_evidence \
  SIGIR EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```

Repeat for KDD and SciRepEval.
