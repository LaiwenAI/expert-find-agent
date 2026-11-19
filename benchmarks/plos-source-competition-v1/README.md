# Confirm an exact study against competing body evidence

When the query describes a particular study, a topically similar paper can
contain the same keywords. This experiment tests whether explicitly comparing
the evidence across the shortlist helps reject such substitutions.

Reuse the [structured-author catalog](../plos-open-person-postings-v1)
and its frozen four-candidate body-confirmation reference. All arms use the
same hybrid nominations and the same source text windows, with
abstract-near-duplicates removed from the body windows. Two source clues
are tested separately: withheld abstract excerpts and the derived keywords.
The intended source is either available or removed before search.

| Policy | Source evidence and decision |
| --- | --- |
| Fixed body top four | Confirm the first candidate in frozen rank order that passes the original dense or maximum-body-overlap cutoff |
| Body peak | Select the maximal body overlap, with a separately calibrated cutoff |
| Body gap | Select the maximal body overlap and multiply it by its lead over the strongest rival |
| Distinctive gap | Divide each matched term's inverse-document-frequency weight by the number of candidates whose best window contains it. Select the maximal discounted support and multiply it by its lead over the strongest rival |
| Deferred distinctive | Accept the first candidate through the frozen dense threshold if its dense lead also clears a calibration cut. Otherwise run distinctive-gap confirmation |

Ranks used to break ties are those from the same hybrid retrieval stage.
A source whose highest-scoring window has no matched term receives zero
body support. This is a shortlist-local contrast and does not assume the
absence of a term in an entire rival document. The dense fast path only uses
the original nomination, with a positive lead required.

All new thresholds are selected on the 24 development papers of the modern
catalog. Each threshold is the largest score among false nominations on
source-present and source-withheld calibration views. Dense leads use false
nominations that already passed the original dense gate, with zero as a
floor. Run the frozen policy on the 24 modern test papers and a separate
197-paper [legacy catalog](../plos-open-role-applicability-v1) containing
72 exact-study test queries. Both the new fallback and the fixed reference keep their modern thresholds
on transfer. Modern evaluation also binds
declared duties to the same author. The legacy transfer evaluates source
identity before any author decisions.

## Results

Slurm job 42917 used one RTX 5090. The fixed route reproduces all published
modern author-level counts. The table gives source confirmations with the
target present, false confirmations with it withheld, and component P95
when it is present. These are source counts before binding authors.

| Catalog and clue | Policy | Correct source | Wrong source, withheld | P95 |
| --- | --- | ---: | ---: | ---: |
| Modern excerpt | Fixed body top four | 19/24 | 2/24 | 8.23 ms |
| Modern excerpt | Body peak | 18/24 | 2/24 | 9.01 ms |
| Modern excerpt | Body gap | 19/24 | 1/24 | 8.96 ms |
| Modern excerpt | Distinctive gap | 16/24 | 1/24 | 9.11 ms |
| Modern excerpt | Deferred distinctive | 16/24 | 1/24 | 8.90 ms |
| Modern keywords | Fixed body top four | 15/24 | 5/24 | 7.50 ms |
| Modern keywords | Body peak | 10/24 | 3/24 | 7.83 ms |
| Modern keywords | Body gap | 7/24 | 2/24 | 7.75 ms |
| Modern keywords | Distinctive gap | 8/24 | 1/24 | 7.72 ms |
| Modern keywords | Deferred distinctive | 14/24 | 4/24 | 7.62 ms |
| Legacy excerpt | Fixed body top four | 66/72 | 0/72 | 8.28 ms |
| Legacy excerpt | Body peak | 61/72 | 0/72 | 8.89 ms |
| Legacy excerpt | Body gap | 64/72 | 0/72 | 8.76 ms |
| Legacy excerpt | Distinctive gap | 63/72 | 0/72 | 8.85 ms |
| Legacy excerpt | Deferred distinctive | 66/72 | 0/72 | 8.51 ms |
| Legacy keywords | Fixed body top four | 56/72 | 11/72 | 7.35 ms |
| Legacy keywords | Body peak | 46/72 | 1/72 | 7.64 ms |
| Legacy keywords | Body gap | 48/72 | 0/72 | 7.55 ms |
| Legacy keywords | Distinctive gap | 32/72 | 0/72 | 7.53 ms |
| Legacy keywords | Deferred distinctive | 50/72 | 6/72 | 7.46 ms |

The plain body-gap policy improves the legacy keyword false-confirmation
count from 11 to zero, with eight fewer correct confirmations. Its descriptive
paired estimate for the withheld error difference is −15.28 percentage
points. The discount based on competing windows loses additional correct
keyword confirmations in the legacy catalog. The deferred policy restores
part of that coverage, but six withheld keyword requests are then confirmed
incorrectly. Its 100 body scans fall below 288 for always comparing four,
while the fixed reference uses only 72.

On modern keywords, the fixed route confirms 15 correct sources and emits
two supported-duty recommendations for an incorrect source when the target is
withheld. Distinctive gap leaves eight correct confirmations and one such wrong
recommendation. The deferred version confirms 14 correct sources and leaves
two withheld wrong recommendations. The gap separates some ambiguity under
both corpora, but no tested policy preserves the fixed route's keyword
coverage while eliminating its wrong-source recommendations at lower cost.

Components include one live query embedding, shortlist retrieval and body
checks. Timings omit index construction, transport and queueing. A body check
scans the candidate's allowed windows. These results concern known-study
disambiguation and do not measure open-ended enterprise suitability.

[Aggregate results](results.json)

## Reproduce

Use the verified source snapshots from the
[structured-author experiment](../plos-open-person-postings-v1) and earlier
[legacy catalog](../plos-open-role-applicability-v1), its modern source result
JSON, and the same Qwen embedding checkpoint. Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. python -m experiments.credit.contrastive_source_confirmation \
  PRIOR_MODERN_MANIFEST EXTERNAL_MODERN_MANIFEST LEGACY_OLD LEGACY_FIRST \
  LEGACY_FRESH EMBEDDING_CHECKPOINT MODERN_RESULTS NEW_RESULTS.json \
  MODERN_XML_DIRECTORY [ADDITIONAL_MODERN_XML_DIRECTORY...]
```
