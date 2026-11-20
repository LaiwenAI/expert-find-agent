# Verify ordered expert citations after GPU batch compaction

This experiment follows the
[eLife proof-before-rerank comparison](../elife-proof-before-rerank-v1/README.md).
It fixes the original topical shortlist and the author–publication
claims for every arm. Every route uses the same pinned open
embedding model, duty normalizer, semantic reader, relevance
threshold, and five-expert deduplication rule. Absence of a matching
same-author conjunction in a visible complete declaration can
exclude a source from the reader. Unindexed and incomplete
declarations remain in the shortlist.

Hash-partition the 18 eLife topic strings into nine calibration and
nine test topics. Keep both the two-role and three-role requests
for a topic in the same partition. Use the maximum observed
absolute margin difference between the full and shorter batches
on calibration. Apply that radius unchanged to both masks and
four- and eight-source cutoffs on test. The corpus and publisher
roles are shared across the partitions. The partition separates
topic queries for drift calibration, while article-level or
domain-level transfer needs another experiment. Requests for which
the shared normalizer abstains remain abstentions in every arm
and are counted separately from the scored tail-latency panel.

| Route | Reader execution |
| --- | --- |
| Full | Original shortlist in one padded batch. |
| Compact | Compatible sources in one batch padded to their own maximum length. |
| Fixed width | Compatible sources padded to the original shortlist's maximum length. |
| Cutoff guard | Compact, then repeat the full batch if an output-bearing source can cross the cutoff. |
| Prior output guard | Compact, then use the [previous PLOS output guard](../plos-packed-topic-guard-v1/README.md) before considering a full recheck. |
| Witness certificate | Compact, then repeat the full batch only if score perturbations within the radius can change the ordered witnesses. |

The witness certificate examines sources by current score and
original retrieval order. For each source that can still add a new
expert, it checks that the source's lowest score passes relevance
and still precedes the highest possible score of every relevant
competitor. A competitor that can emit only already-emitted
experts has no effect and is removed from consideration. Once the
five slots are fixed, the rest of the shortlist is irrelevant. If
the cutoff can be crossed or two eligible sources can exchange an
emitted citation, the original batch is used. This argument is
conditional on the individual score intervals, the fixed shortlist
and stable role declarations. The maximum error seen on calibration
is an empirical interval radius. Test exceedances are reported.

The replay separately retains the full batch's acceptance mask
while substituting compact scores for ordering, and retains the
full scores for ordering while substituting the compact acceptance
mask. This isolates changes in threshold decisions from changes
in citation order. Neither counterfactual runs at inference time.

Timed preparation includes the verbatim clause packet, duty
normalization, topical ranking, tokenization of the original
shortlist and checking publication-scoped authors. The shared
preparation duration is measured once for each query and charged
to all arms. Reader inference, guard decisions, any repeated
inference and witness construction are timed separately per arm.
The full-batch score is measured for evaluation only. Latency is
warm and local. Index build, model startup, queues, and network
costs are outside the measurement.

## Results

Both tasks in Slurm array `44238` ran on an RTX 5090 on ws5 and
reproduced every non-timing result. There were 32 scored calibration
requests and 517 retained scores. The maximum observed deviation
was 0.25 logits. Of 36 test requests, 33 entered the reader and
three abstained at duty normalization. No retained test score in
either compaction geometry exceeded the calibration radius.

| Indexed declarations / cap | Reader | Changed outputs / 33 | Full rechecks | P95 across runs (ms) |
| --- | --- | ---: | ---: | ---: |
| Partial / 4 | Full | 0 | 0 | 87.93–88.52 |
| Partial / 4 | Fixed width | 0 | 0 | 83.56–83.92 |
| Partial / 4 | Witness certificate | 0 | 4 | 90.44–91.19 |
| Full / 4 | Full | 0 | 0 | 88.02–88.62 |
| Full / 4 | Compact | 1 | 0 | 82.41–82.99 |
| Full / 4 | Cutoff guard | 1 | 4 | 89.25–89.92 |
| Full / 4 | Prior output guard | 0 | 10 | 99.32–100.07 |
| Full / 4 | Witness certificate | 0 | 8 | 97.64–98.36 |
| Full / 8 | Full | 0 | 0 | 103.42–103.72 |
| Full / 8 | Compact | 1 | 0 | 91.48–92.07 |
| Full / 8 | Fixed width | 1 | 0 | 91.50–91.97 |
| Full / 8 | Prior output guard | 0 | 15 | 121.21–121.79 |
| Full / 8 | Witness certificate | 0 | 13 | 121.75–122.45 |

For the cap-eight setting in both masks, substituting only the
compact acceptance mask reproduces the observed output change.
With full coverage at cap four, no score crosses the cutoff.
Substituting only the compact score order reproduces the change
and the cutoff guard misses it. Preserving the original padding
width does not recover the reference output in these settings.

The witness certificate saves two canonical rechecks per
setting relative to the earlier output guard. It restores the
exact expert order and citations on all four settings. At cap
eight with full indexed coverage, the 13 rechecks increase the
total from 157 compact rows to 261 rows, compared with 264 rows
in the full batch. Padded token slots are slightly higher than
full (96,250 versus 95,840), and the paired topic-bootstrap
interval for the P95 difference is +1.14 to +30.86 ms in the
first run and +2.09 to +31.09 ms in the second. Conditional
witness stability therefore survives compaction, but full-batch
fallback is too expensive at this measured operating point.
This study measures numerical agreement with the frozen
reader and does not establish recommendation relevance.

[Full replay aggregates for both runs](results.json)

## Reproduce

Use the same pinned eLife inputs and checkpoints as in the prior
live experiment inside an allocated Slurm GPU job.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.elife_witness_order_guard \
  PROJECTION benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/elife-role-nomination-v1/results.json NEW_RESULT_JSON --order-offset 0
```
