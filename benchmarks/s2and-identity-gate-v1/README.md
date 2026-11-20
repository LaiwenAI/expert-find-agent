# Selective author linkage for a fixed search catalog

The [same-person role index](../role-pair-placement-v1) depends on the
identity mapping available when it is built. Test how much publication
metadata can recover links when a source has no usable persistent identity.
The experiment uses the ArnetMiner, Qian, and zbMATH gold clusters and
bibliographic metadata distributed with the [S2AND benchmark](https://github.com/allenai/S2AND).
This is controlled catalog assignment. It does not reproduce S2AND's
native clustering task or compare against its full production linker.

## Protocol

Within each supplied name block, treat the earliest one or two labeled
records as the existing publication history for each cataloged identity,
then hold out up to two later records. For roughly one fifth of identities,
chosen by a deterministic hash, omit the whole profile and retain up to
two query records. Those requests have no correct candidate. Exclude gold
clusters spanning distinct supplied blocks, so this measures disambiguation
within the block and leaves name blocking and name variants spanning
blocks outside scope. Exclude profiles already represented by another
author mention of the query paper.

Use a 60% / 20% / 20% hash split by name block within each collection.
Fit the lightweight ranker on the first split and freeze the threshold
on the second. Cross-collection tests exclude any block seen in fitting or
calibration as well as any target block sharing a paper with those splits.
No individual author names appear in aggregate results. The evaluator
does not use ORCIDs, contact information, or S2AND embeddings.

- Fixed controls test uniqueness of exact normalized full name, affiliation,
  or shared coauthor, plus a three-signal rule requiring all three to
  identify exactly one possible candidate.
- Rank every admissible profile with weighted ridge regression over up to
  two anchor publications. Inputs cover name, affiliation, coauthor,
  title tokens, venue, and year. Score each profile by its best anchor.
  Gate the answer on the minimum of its score and its lead over the runner-up.
- The cascade takes a unique shared-coauthor match within the supplied name
  block when one exists and otherwise runs the same ranker. A comparator
  cascade uses the three-signal gate. Calibrate each policy independently
  for a one-sided Wilson upper limit of 5% wrong among issued, with at least
  50 issued examples. If no threshold passes, the policy abstains. Wilson
  estimates and block bootstrap intervals are descriptive due to shared
  names, publications, and threshold selection.

Unsupported affiliations and collaborators are left empty in the native
test. Two further passes remove each field from both the query and its
candidate history while retaining the thresholds. Report correct, wrong,
unmatched-target errors, coverage, profile-anchor comparisons, and warm
local selection time. No index construction or retrieval outside the name
block is timed.

## Results

Both CPU tasks in Slurm array 45589 on ws3 produced identical nontiming
results. CUDA and model training on a GPU were unnecessary. Cells below
are correct / wrong among the emitted links.

| Calibration → evaluation | Queries | Coauthor only | Rank all | Coauthor cascade |
| --- | ---: | ---: | ---: | ---: |
| ArnetMiner → ArnetMiner | 287 | 124 / 0 | 134 / 4 | 145 / 5 |
| ArnetMiner → Qian | 1,127 | 568 / 0 | 601 / 2 | 618 / 5 |
| Qian → zbMATH | 1,937 | 342 / 0 | 437 / 2 | 437 / 2 |
| zbMATH → ArnetMiner | 1,090 | 440 / 2 | 3 / 0 | 441 / 2 |
| zbMATH → Qian | 1,249 | 663 / 0 | 266 / 4 | 700 / 4 |
| zbMATH → zbMATH | 381 | 64 / 0 | 64 / 1 | 82 / 1 |

Across all nine unmasked source–target splits, the cascade reduces
profile-anchor comparisons by 15.8%–61.3% relative to rank all. ArnetMiner
calibration transferred to Qian reduces warm P95 from 0.039 to 0.031 ms
in the primary replay. It increases P95 from 0.009 to 0.010 ms on the
smaller zbMATH name blocks in the same run. On native held ArnetMiner,
the cascade's error rate is 5 / 150 with a one-sided Wilson upper
estimate of 6.7%, above the nominal 5% calibration target. zbMATH's held
split also has an upper estimate above 5%. A higher point coverage
therefore cannot be read as a certified safety improvement.

The strong coauthor-only rule explains nearly all the zbMATH-to-ArnetMiner
transfer gain. On zbMATH-to-Qian, masking coauthors reduces correct links
from 700 to 201. Masking affiliations leaves 681. The method depends on
interpretable collaborator evidence in the current catalog. An identified
author mention still needs separate role and qualification checks. The
benchmark has gold labels for author linkage, not enterprise work or
eligibility.

## Reproduce

Download the three JSON files per dataset from the official
[S2AND release bucket](https://github.com/allenai/S2AND#download-data-or-model).
Preserve its directory layout and filenames. The result records the
original file and derived panel SHA-256 hashes. Use a fresh output path.

```sh
python experiments/identity/prepare.py S2AND_RELEASE_DIR SCRATCH_PANEL_JSON
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python experiments/identity/replay.py SCRATCH_PANEL_JSON NEW_RESULTS_JSON
```

Only code and aggregate results are included here.
