# Transfer the eLife directed source policy to an open PLOS archive

The [eLife directed-rescue study](../elife-directed-source-rescue-v1)
reduces source-ranking and posting work by indexing declared roles
along with contributors. Its rescue can also issue an incorrect
citation on requests with both sources available. This pilot tests
the same frozen scorer and its cutoffs on a different publisher
without fitting or adjusting thresholds on PLOS.

## Corpus and protocol

Freeze the catalog from cached, publisher-supplied PLOS ONE XML
with the archived CC BY 4.0 license convention. Exclude conflicts in
source versions and overlap with the recorded eLife ORCID hashes.
Unlinked articles can remain as distractors. Use the title and
first abstract portion for indexing and reserve the last portion for
excerpt or keyword clues, following the eLife setup.

The [new groups](manifest.json) exclude papers and recorded identities
in the previous PLOS supplied-source swap and role-edit panels.
Select groups in fixed hash order using only the role declarations
and resolved contributor identities. In each group, construct up to
three supported role pairs. Negative requests swap the two labels or
replace one label with another explicitly declared role lacking the
right same-author association. Test both sources available and then
withdraw each one separately. No model scores enter this construction.
Job 47173 froze the manifest before the GPU runs.

The 366-document archive yields five author-disjoint groups with
15 positive and 15 negative requests per entrance, plus 60
source-withdrawn requests. Replay separately the eight previously
source-supplied role-edit groups with both papers in this archive.
A ninth group lacks a pinned source in the catalog and is excluded
from this panel. The eight groups add 24 positive and 24 negative
requests per entrance and 96 source-withdrawn requests. No recorded
author or paper from that panel appears in the five new groups. This is a targeted
public-archive pilot. People without a resolved ORCID cannot
contribute to the known proof labels.

The eLife-fit route reproduces the archived training-class totals,
adverse calibration group counts, and cutoffs before each PLOS
test. Use the frozen eLife role decoder and the same NISO phrase
inventory. The only PLOS-specific quantities computed are the
catalog vectors, lexical frequencies, and index postings. Compare
shallow search and the three fallbacks, and report source rows,
author posting entries, and verifier pairs separately.

## Results

| Panel and entrance | Shallow strict / wrong | Global fallback | Author fallback | Author-role fallback |
| --- | ---: | ---: | ---: | ---: |
| New groups, excerpt | 2 / 0 | 2 / 1 | 2 / 0 | 2 / 0 |
| New groups, keyword | 1 / 0 | 1 / 0 | 1 / 1 | 1 / 1 |
| Previously source-supplied, excerpt | 6 / 0 | 6 / 0 | 6 / 0 | 6 / 0 |
| Previously source-supplied, keyword | 3 / 0 | 3 / 1 | 5 / 0 | 5 / 0 |

Counts above use both sources available; routes include any shallow
proofs. In the source-withdrawal tests, all methods except global
keyword fallback on the historical groups return zero wrong proofs.
That configuration issues two.

On the new keyword groups, the author-role index makes one exact
proof reachable that the top-18 global fallback cannot reach. The
frozen acceptance cutoff blocks it. A separate request emits one
unsupported result. The previously source-supplied keyword groups
yield three reachable rescues, two of which pass without adding
errors. The index reduces posting visits from 63 to 10 for these
present-source keyword requests, with the same three candidate-pair
scores and the same outputs as author-only fallback. It also
reduces visits on the new groups. The original eLife efficiency
mechanism transfers on the observed postings. The recovered proof
gain depends on the panel, and the small set cannot establish
equal-risk improvement or latency.

The direct-transfer choice should remain separate from a target
calibration study. A follow-up should freeze an independent
publisher-side calibration split with more true and substituted
source pairs that reach the scorer, and keep a physically separate
test. The scope is the 366-paper archive and annotated exact
proofs, without a claim about unjudged alternative papers.

Both tasks in Slurm array 47189 on ws6 (RTX 4090) reproduce the
results and preserve the original new-group counts from array
47177. Job 47190 reproduces the manifest and the intermediate
projection byte for byte from pinned inputs.
[Aggregate results](results.json).

## Reproduce

Use the archived publisher XML supported by the DOI/SHA-256 list
in the [manifest](manifest.json). Restore the eLife projections,
original groups, embedding checkpoint and NISO examples as in the
[directed rescue](../elife-directed-source-rescue-v1).

```sh
export PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.credit.freeze_plos_open_transfer \
  ELIFE_CONFIRMATION ELIFE_EXTENDED \
  benchmarks/plos-selective-review-holdout-v1/role-edits-plos-manifest.json \
  benchmarks/plos-selective-review-holdout-v1/combined-swaps-manifest.json \
  NEW_MANIFEST NEW_PROJECTION PLOS_XML_DIRECTORY \
  --restore benchmarks/plos-open-directed-transfer-v1/manifest.json

python -m experiments.credit.plos_open_directed_transfer \
  ELIFE_CONFIRMATION ELIFE_EXTENDED \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
  benchmarks/elife-held-source-acceptance-v1/panel.json \
  benchmarks/role-first-proof-decoding-v1/results.json \
  benchmarks/elife-directed-source-rescue-v1/results.json \
  NEW_MANIFEST NEW_PROJECTION NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json QWEN_EMBEDDING NEW_RESULTS \
  --source-supplied-panel benchmarks/plos-selective-review-holdout-v1/role-edits-plos-manifest.json
```

Run the evaluation twice on an allocated Slurm GPU, with
`--reverse` and a separate results path for the second run.
