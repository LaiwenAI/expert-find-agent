# Calibrate directed recovery with both reachable and withdrawn sources

The [prospective PLOS split](../plos-prospective-proof-split-v1)
shows that natural calibration can expose wrong fallback candidates
while never exposing a representative correct fallback. This
follow-up keeps the same 414-paper catalog and disjoint 12-group
calibration and test partitions. It adds a controlled shallow miss
and tests abstention thresholds with and without these cases.

## Conditions

- **Natural.** Use the original linked shallow search and
  fallback proposals, with both correct papers present and with
  either paper removed from the catalog.
- **Shallow miss.** If the designated paper is in the first six,
  move it to position seven in its own condition's fused ranking.
  Keep the same top-18 membership, raw scores, other condition's
  ranking, and all author/role postings. Apply the same intervention
  once for each side. The source remains in the catalog and can be
  found during recovery.
- **Full withdrawal.** For natural negative trials use the original
  withdrawal from the catalog, including the selected index
  postings. The shallow intervention does not alter these natural
  controls.

The scoring and routing code is the same for natural and staged
requests. The intervention uses the known paper only to construct
the challenge. The selected author and labels, candidate features,
and decisions do not use the reference to route. Construct
interventions separately in each frozen split. The natural test
has already been reported, so the combined replay is exploratory.
Moving a source to position seven measures a controlled shortlisting
failure rather than its natural prevalence.

## Acceptance policies

Compare fixed eLife thresholds, PLOS thresholds calibrated on
natural cases, and thresholds calibrated on natural plus modified
cases. For the target policies, set each cutoff at the maximum
observed wrong fallback score and require strict greater-than
acceptance. If the target split has no adverse proposal for a
route, use that route's eLife cutoff.

Add a fixed lexical guard to the eLife pair scorer. Increase the
logit by half the average of the eLife-standardized minimum
cross-clue lexical separation, uncommon-token coverage, and
weighted term coverage. Minimum is taken across both sources.
No learned coefficients are reset and no new retrieval or model
call is required. Apply the same augmented calibration to this
score. Report correct options, test proofs, and wrong outputs
per route. Compare separately the natural results and the
artificially demoted-source results.

## Results

| Author-role acceptance | Natural keyword strict / wrong, sources present | Natural keyword wrong, source withdrawn | Demoted keyword strict / wrong |
| --- | ---: | ---: | ---: |
| Frozen eLife | 10 / 0 | 2 | 17 / 3 |
| Natural PLOS calibration | 10 / 0 | 4 | 17 / 3 |
| PLOS calibration with shallow misses | 9 / 0 | 0 | 0 / 0 |
| PLOS calibration plus lexical guard | 9 / 0 | 0 | 10 / 0 |
| Shallow search without rescue | 9 / 0 | 0 | 0 / 0 |

The natural keyword test has 36 positives and 36 negatives
with both papers present, plus 144 source-withdrawal requests.
The keyword demotion test covers 22 side-specific interventions,
or 132 requests. Repeated interventions share a source group.
The excerpt demotion test covers 24 interventions, or 144 requests.

In calibration, author-role recovery gets five excerpt options
from two groups and ten keyword options from three groups after
demotion, versus zero under natural search. Only one keyword
group's best correct option clears the worst adverse score under
either ranker. The intervention also exposes a stronger incorrect
keyword proposal. The ordinary pair scorer's zero-adverse cutoff
rises from 0.800 to 0.988, which rejects all 17 reachable
keyword proofs on the demotion holdout. The fixed lexical guard
recovers ten there at zero observed errors, and keeps the nine
natural keyword proofs with zero observed wrong proofs on
withdrawal. It contributes no additional correct natural proof.

The excerpt author-role route keeps ten natural strict proofs
and 18 on the demotion challenge under all four policies. The
lexical guard adds two errors on source withdrawal compared
with unmodified scoring. The global keyword route has no
adverse calibration example and inherits its former permissive
cutoff. It retains four withdrawal errors.

The controlled protocol diagnoses why a low observed error
rate in natural calibration can coexist with uncertain rescue
settings. Added correct options make it possible to evaluate
separation from hard substitutes, but a fixed cross-clue lexical
guard does not achieve a consistent quality/risk gain across
entrances. Role parsing and index access were unchanged. Neither
publisher-level deployment risk nor end-to-end latency is
estimated.

Both jobs in Slurm array 47206 on ws6 (RTX 4090) reproduce the
output counts, cutoffs, and archived natural controls.
[Aggregate results](results.json).

## Reproduce

Restore the PLOS XML, base projection and frozen eLife inputs
from the [prospective split](../plos-prospective-proof-split-v1).
On an allocated Slurm GPU with fresh output paths, run:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m experiments.credit.plos_recoverable_source_calibration \
 ELIFE_CONFIRMATION ELIFE_EXTENDED \
 benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
 benchmarks/elife-held-source-acceptance-v1/panel.json \
 benchmarks/role-first-proof-decoding-v1/results.json \
 benchmarks/elife-directed-source-rescue-v1/results.json \
 benchmarks/plos-open-directed-transfer-v1/manifest.json BASE_PLOS_PROJECTION \
 benchmarks/plos-prospective-proof-split-v1/manifest.json NEW_PLOS_XML_DIRECTORY \
 benchmarks/plos-prospective-proof-split-v1/results.json \
 NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json QWEN_EMBEDDING NEW_RESULTS
```

Repeat with `--reverse`. Both archived natural controls must
reproduce before the intervention results are accepted.
