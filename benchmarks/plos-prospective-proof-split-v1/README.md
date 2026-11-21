# Prospectively split PLOS source acceptance by resolved author

The [previous transfer pilot](../plos-open-directed-transfer-v1)
found that an eLife cutoff could reject a reachable correct PLOS
proof. It included five new groups and eight earlier groups,
too few to distinguish threshold transfer from proposal quality.
This study collects fresh groups before using search outcomes.

Cursor through the [Europe PMC Articles API](https://dev.europepmc.org/RestfulWebService)
for PLOS ONE records published in 2025 with an indexed ORCID and
machine-learning or optimization wording in the title or abstract.
Propose paper pairs that share an ORCID in the API metadata.
Process candidates in stable hash order, without repeated papers
or proposing ORCIDs. The publisher XML supplies the identity and role labels.

Confirm every selected link in the original publisher XML.
Require the proposing ORCID on both author lists, a recognized
controlled contribution label on each side, and a supported role
pair with an explicit single-role contrast. Recheck all recorded
ORCIDs and exclude any paper or recorded author already in the
prior PLOS archive, eLife projections, or another new group.
Do not attempt to infer identities for authors without persistent
identifiers. The manifest records article and response digests,
group assignments, and contrast labels without author identifiers.
Apply a separate deterministic hash order and split the publisher-
confirmed groups into calibration and test before any model score.

The existing pinned 366-paper archive provides search distractors.
The new selected articles also enter the search catalog. Keep the
eLife data, decoder, and source-scoring model fixed. Compare
eLife's frozen source acceptance thresholds against thresholds
calibrated solely on the new PLOS calibration groups. The latter
take the maximum wrong fallback score for each route, with strict
greater-than acceptance. A third control takes the stricter of the
two thresholds. If a route has no adverse target examples, keep
its transferred threshold and report that local calibration is
unavailable. Frozen shallow outputs are
shared across policies. Source withdrawal and role edits remain
distinct controls. Count wrong issuances, correct proofs and the
work of applicable author-posting and global fallbacks separately.

## Frozen inventory

| Stage | Count |
| --- | ---: |
| Europe PMC records on the fixed query | 1,409 |
| Candidate pairs after prior exposure exclusions | 80 |
| Distinct paper pairs selected for publisher checks | 45 |
| Publisher-confirmed groups | 24 |
| Calibration groups / role edit families | 12 / 36 |
| Test groups / role edit families | 12 / 36 |
| Distractors plus accepted new articles | 414 |

The publisher check rejects seven metadata links that fail the
same-person check in JATS, and eleven pairs without a suitable
role edit. Of 90 requested XML records, 89 parse and have enough
abstract text. The final manifest pins 48 documents. It contains
only public DOIs, publisher source digests, and derived role
labels. Group allocations and the held set were frozen before
retrieval tests. [Manifest](manifest.json).

## Results

| Entrance and route | Present-source strict / wrong | Withheld-source wrong, eLife cutoff | Withheld-source wrong, PLOS cutoff | Withheld-source wrong, stricter cutoff |
| --- | ---: | ---: | ---: | ---: |
| Excerpt, shallow | 10 / 0 | 0 | 0 | 0 |
| Excerpt, global fallback | 10 / 0 | 4 | 2 | 2 |
| Excerpt, author-role fallback | 10 / 0 | 0 | 0 | 0 |
| Keyword, shallow | 9 / 0 | 0 | 0 | 0 |
| Keyword, global fallback | 9 / 1 | 4 | 4 | 4 |
| Keyword, author-role fallback | 10 / 0 | 2 | 4 | 2 |

Each entrance has 36 positive and 36 hard negative test requests
with both sources available, plus 144 requests with one source
withdrawn. All three cutoffs agree on present-source results.
On keywords, the author-role index returns one more correct
proof than shallow search and reduces posting visits from
70 to 10 versus author-only fallback at identical outputs
and one candidate-pair score. On the missing-source workload,
the same index reduces visits from 209 to 80 while leaving the
wrong-proof counts unchanged relative to author-only lookup.

The independent calibration split has no correct fallbacks
available to score, even before applying an acceptance threshold.
The author-role route has wrong opportunities in four excerpt
groups and one keyword group. Keyword global fallback has none,
so it retains the transferred cutoff. Local calibration lowers
the keyword author-role cutoff and doubles the wrong count under
source withdrawal, from two to four requests. Taking the stricter
cutoff prevents that increase while preserving the two mistakes
from the eLife cutoff. For excerpt global fallback, the target
cutoff reduces wrong outputs from four to two. Threshold transfer
therefore varies by route and entrance. These 12 calibration
groups do not demonstrate a deployable equal-risk operating point.

The next calibration set must deliberately cover both recoverable
positives and realistic substitutes. Freeze thresholds there
before comparing quality at matched risk on a separate holdout.
The present result is an exact-paper retrieval and proof study,
with observed index work and no service latency measurement.
Both ws6 RTX 4090 jobs in Slurm array 47199 agree on all counts,
thresholds and diagnosis fields. [Aggregate results](results.json).

## Reproduce

The exact proof group design and publisher XML checksums are pinned
in the manifest. Restore the raw XML by DOI into a flat directory;
the evaluation rejects content with changed digests. Restore the
base PLOS projection and other inputs as in the
[transfer pilot](../plos-open-directed-transfer-v1).

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m experiments.credit.plos_prospective_calibration \
 ELIFE_CONFIRMATION ELIFE_EXTENDED \
 benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
 benchmarks/elife-held-source-acceptance-v1/panel.json \
 benchmarks/role-first-proof-decoding-v1/results.json \
 benchmarks/elife-directed-source-rescue-v1/results.json \
 benchmarks/plos-open-directed-transfer-v1/manifest.json BASE_PLOS_PROJECTION \
 benchmarks/plos-prospective-proof-split-v1/manifest.json NEW_PLOS_XML_DIRECTORY \
 NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json QWEN_EMBEDDING NEW_RESULTS
```

Run on an allocated Slurm GPU. Repeat with `--reverse` and a
separate output path. The collector is available as
`experiments.credit.plos_prospective_proof_split`. An API rerun can
observe changed upstream metadata; the published manifest locks the
specific inventory accepted in this experiment.
