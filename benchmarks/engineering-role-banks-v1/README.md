# Improve one-encoding author gates with learned role representations

Reuse the 54 exact-agreement engineering articles and all-author
evaluation from the [previous gate study](../engineering-all-author-gate-v1).
Use both of its 14-phrase banks and add the 49 independently authored
[NISO task examples](https://zenodo.org/records/18421449). A task
description is evaluated against every author in every article. Some
requests have no credited author, while others have several.
Original, alternate, and NISO banks reuse articles. Their sizes
measure repeated workloads rather than independent samples.

Freeze Qwen3-Embedding-0.6B and compile three role representations:

- **Definition:** original embedding of the role name and definition.
- **Prototype:** normalized mean of six previously authored embeddings
  per role.
- **Ridge:** a 14-way linear projection fitted to the same 84
  [training phrases](../plos-credit-edits-v1/role-training.json),
  with ridge penalty 0.1 and no backbone updates.

All mappings reuse one live query vector. A fourth route, **agreement**,
takes the smallest author-level support margin across the three banks,
after scaling each by the standard deviation of its role scores for
that request. For each bank, a candidate's margin is its best score
among declared roles minus its best score among undeclared roles.
Role membership comes from resolved visible declarations. Structured
tags provide reference labels and never enter the scoring path.
Verify no exact normalized overlap between the 84 fitted phrases
and any evaluation phrase. The glossary definitions are visible;
the NISO task-example texts do not enter training or calibration.

Choose one nonnegative cutoff per route and target publisher using
the other publisher's **original** wording. Maximize credited returns
while admitting an uncredited author on at most 5% of source
requests. Apply cutoffs unchanged to all target wordings. Treat
an author as uncredited only when both agreeing publisher views
omit that role. Measure correct nonempty results, affected requests,
and all authors retrieved. Report paired intervals clustered by
article and a sensitivity analysis clustered by role.

## Results

Both runs of Slurm array 45884 on ws3 RTX 5090 reproduce every
non-timing field. The definition route reproduces prior frozen
thresholds and all original and alternate author-gate counts.
The table shows top-one agreement with the expected role before
author evaluation. Agreement is an author-level gate, without a
single role prediction.

| Wording | Definition | Prototype | Ridge |
| --- | ---: | ---: | ---: |
| Original | 11/14 | 14/14 | 14/14 |
| Alternate | 10/14 | 13/14 | 13/14 |
| NISO | 21/49 | 35/49 | 35/49 |

On both authored wording banks, prototype and ridge have zero
observed uncredited returns in both journals. The following results
use the independent NISO examples. Each cell shows credited author
slots retained and the number of requests with an uncredited author.

| Route | Frontiers: credited / 2,508 | Frontiers: errors / 1,960 requests | Sensors: credited / 856 | Sensors: errors / 686 requests |
| --- | ---: | ---: | ---: | ---: |
| Definition | 1,267 | 218 | 463 | 158 |
| Prototype | 1,907 | 84 | 609 | 32 |
| Ridge | 1,488 | 0 | 418 | 0 |
| Agreement | 1,500 | 83 | 507 | 34 |

Prototype stays under 5% observed request errors in both journals
and retains 74.8% of credited slots overall. Each journal's prototype
errors are concentrated in seven different task examples. Ridge
retains 56.7% of credited slots with no observed uncredited returns,
and returns at least one credited author on 735 Frontiers and
269 Sensors requests. The repeated articles and role phrases cannot
justify a population error guarantee. In particular they should
not be treated as 2,646 independent risk observations.

Warm P95 runs from 7.93 to 8.36 ms, including one query encoding,
all three projections, candidate scoring, and serialization. XML
parsing and source discovery are outside the timing. The result
isolates declaration attachment with known identities and one
requested role. [Aggregate results](results.json) ·
[Compressed live score trace](live-evidence.json.gz).

## Reproduce

Restore the same publisher panel and use the existing normalized NISO
example snapshot, prior training phrase file, and Open-weight Qwen
checkpoint. Repeat the command with offsets 0 and 1 inside a Slurm
CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.engineering_role_banks \
  PANEL NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  QWEN_EMBEDDING NEW_RESULTS --offset 0
```
