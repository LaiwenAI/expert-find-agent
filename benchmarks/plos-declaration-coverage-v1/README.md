# Test where role declarations must be available for early nomination

The [bounded nomination replay](../plos-bounded-topic-v1) measured two
declaration masks. Here the same number of visible declarations can have
different effects depending on which documents carry them. This is a
mechanism sensitivity analysis on the same 78 articles and 46 requests.
All scenarios retain the original labels, checkpoints, query ranking,
topical acceptance cutoff, verifier and owner-binding policy.

At each of five coverage levels, make a nested declaration mask with
each of three independent DOI hash orderings. Relative to each query's
topic rank, also reveal declarations starting from the highest- or
lowest-ranked articles. The latter two are diagnostic interventions with
identical counts, not deployable corpus snapshots. Their masks never
consult the requested roles or publisher topic tags. Hidden declarations
leave a source eligible until verification, where the full declarations
are available as in the earlier replay. This separates index availability
from missing ground truth.

Compare pure topic nomination with four and eight reads, whole-ranked
person nomination with up to four reads and person nomination restricted
to the first eight topic-ranked sources with up to four reads. Within a
query, reuse scores only for exactly the same ordered source batch.
Use a frozen checkpoint with the original batch geometry and run it on
the assigned GPU. Report number of reads and how often the selected set
contains no source that has both the publisher's topic tag and a single
credited person with both roles. That diagnostic uses tags only after
selection. Read counts measure work; the replay does not measure service
latency, human-confirmed relevance or enterprise transfer.

## Results

Slurm jobs `43741` and `43743` on an RTX 5090 reproduced every
reported count for all masks and methods. The topic baselines returned
37 tagged and 4 off-label citations with four reads per request, and
49 tagged and 14 off-label citations with eight. All listed arms use
the same four-read budget, with the bounded arm allowed to use fewer.
The table reports recommendation slots summed over the 46 queries.

| Coverage and location | Whole-ranked person, tagged / off-label | Top-eight person, tagged / off-label | Top-eight person, total reads |
| --- | ---: | ---: | ---: |
| 0 of 78 | 37 / 4 | 37 / 4 | 184 |
| 20 of 78, three hash masks | 38–40 / 4–6 | 38–40 / 4–6 | 184 |
| 39 of 78, three hash masks | 40–41 / 7 | 40 / 7 | 177–179 |
| 59 of 78, three hash masks | 43–49 / 11 | 40–43 / 9 | 154–163 |
| 20 of 78, highest topic ranks indexed | 52 / 16 | 44 / 10 | 125 |
| 59 of 78, lowest topic ranks indexed | 37 / 4 | 37 / 4 | 184 |
| 78 of 78 | 52 / 16 | 44 / 10 | 125 |

At 20 indexed articles per query, switching from the lowest to the
highest ranks cuts the number of selected sets with no tagged
same-person witness from 22 to 11 for whole-ranked nomination.
This identifies a candidate-discovery effect before the model accepts
a source. In the bounded arm, indexing the highest ranks also avoids
59 reads relative to topic four. The whole-ranked arm keeps all 184
reads and admits more off-label examples. An off-label citation is
unresolved under incomplete publisher tags.

These controls attribute the observed differences to declaration
placement on this replay. They motivate reporting known declarations
among top-ranked candidates in addition to global index coverage.
They do not establish that a fixed corpus indexing policy can attain
the query-dependent high-rank mask on unseen demands. That requires
a workload-independent build or a policy trained on disjoint queries,
with extraction costs included.

[Aggregate replay](results.json)

## Reproduce

Rebuild the [pinned holdout](../plos-topic-role-holdout-v1) and use the
same checkpoints as the [bounded nomination replay](../plos-bounded-topic-v1).
The script checks four aggregate fields for every original reference arm
at the original two coverage masks before it writes results.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.declaration_coverage HOLDOUT_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/plos-bounded-topic-v1/results.json NEW_RESULTS.json
```
