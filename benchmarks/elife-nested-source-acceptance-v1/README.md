# Calibrate against deeper source substitutes before joining authors

The [new-group source replay](../elife-held-source-acceptance-v1)
found four substitutions after excerpt sources were removed, despite
historical depth-six calibration observing none. A second issue was
that source pair checks scale quadratically with shortlist depth. This
experiment uses hard alternatives from genuine retrieval results and
tests whether source verification before an author join can support
bounded expansion.

## Protocol

Reuse the 2,011-paper catalog and its two fixed entrants, excerpt
and keyword. Learn on the original 18 development groups. Calibrate
on the remaining 42 previously used groups. Replay both sources
present, then with each designated paper withdrawn in turn. Build
calibration alternatives from all author-linked pairs in the top 18
per side. A group's adverse score is the largest score on any
non-designated pair, including when both true sources are present.
Set each gate at the third highest group maximum with strict
greater-than acceptance. This rule is a historical operating point.

Evaluate on the [35 author-disjoint groups](../elife-held-source-acceptance-v1/panel.json)
from the previous comparison. The group panel is excluded from fitting
and threshold selection, but has already been explored. Treat these
comparisons as a follow-up, not a fresh confirmation. The strict proof
label still requires the designated papers, the correct contributor and
both role citations. Other papers may provide useful alternative
evidence under a different task contract.

## Methods and work accounting

Score each candidate source with a small regularized logistic model
over its hybrid rank, absolute and relative dense relevance, dense
leader gap, cross-clue relevance, and the previous weighted lexical
footprints. Fit a control that takes the minimum and maximum of the
same features across both sides and scores each linked pair directly.
Calibrate source acceptance using the weaker of the two side scores.
Both sources must pass before the author join. The query-first Ridge
role decoder and its cutoff are shared by all methods.

Start the nested route with depth six on both sides. If no accepted
linked pair exists, extend to 18. Find the minimum hybrid rank-sum
pair via an inverted map from each author to the first two accepted
right-side positions. A left-side paper can exclude at most one
right-side position by repeating its own DOI. The script checks
the returned pair against exhaustive comparison at both depths on
every test trial. Compare nested access with the same verifier fixed
at six and 18, plus linked and anchored rank rules and the pair
classifier.

Count source classifier scores, author posting entries read, and
the number of expansions. The Cartesian controls enumerate the
cross product, reported as pairs and never equated with author
posting visits. The nested score count assumes reuse of the first
wave's scores. Its join count includes rebuilding the small posting
map after expansion. The replay materializes all retrieval rows in
advance for paired comparison. These work counts do not measure
end-to-end latency or a production retrieval schedule.

## Results

| Entrance | Route | Correct sources / 35 | Strict proofs / 102 | Wrong proofs with sources present | Wrong proofs / 408 with a source withdrawn |
| --- | --- | ---: | ---: | ---: | ---: |
| Excerpt | Linked, depth 6 | 34 | 38 | 0 | 1 |
| Excerpt | Linked, depth 18 | 35 | 40 | 0 | 15 |
| Excerpt | Pair score, depth 18 | 35 | 40 | 0 | 1 |
| Excerpt | Source-first, depth 6 or 18 or nested | 34 | 38 | 0 | 1 |
| Keyword | Linked, depth 6 | 31 | 35 | 0 | 0 |
| Keyword | Linked, depth 18 | 32 | 37 | 0 | 8 |
| Keyword | Pair score, depth 18 | 30 | 35 | 4 | 8 |
| Keyword | Source-first, depth 6 or 18 or nested | 31 | 35 | 0 | 0 |

Deeper retrieved candidates expose non-designated pairs in 15/42
excerpt calibration groups and 3/42 keyword groups. The cutoff now
has actual adverse examples in both views. Naive expansion from
six to 18 gains two correct proofs per entrance and increases
wrong outputs when a source is withdrawn. The depth-18 pair score
preserves the two additional excerpt proofs at the same observed
wrong count as depth-six linked. Its keyword results show that this
operating point does not transfer between the two entrances. The
source-first gate suppresses the extra substitutions from naive
expansion and also gives up all additional correct proofs.

The nested classifier scores 444 excerpt and 516 keyword
candidates versus 1,260 per entrance at fixed depth 18 when both
papers exist. With a source withdrawn it expands on 68/70 excerpt
and 70/70 keyword trials, scoring 2,472 and 2,520 candidates
respectively. It also reads more author posting entries than the
fixed deep gate in both missing-source views. Neither expansion
path finds more correct proofs than the shallow source gate in
this replay. The expensive case is therefore absence, and the
fallback condition needs evidence that a missing source is
recoverable.

Both RTX 4090 jobs in Slurm array 47131 on ws1 agree on all counts,
cutoffs and descriptive paired intervals. The depth-six linked
control reproduces the previous archived counts. The experiment uses
one publisher and known source-withdrawal interventions, with rare
adverse groups. It does not establish deployment risk or end-to-end
efficiency. [Aggregate results](results.json).

## Reproduce

Restore the exact catalog, groups and checkpoint from the
[previous source study](../elife-held-source-acceptance-v1).
On a Slurm GPU allocation run with a fresh output path:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m experiments.credit.nested_source_acceptance \
 CONFIRMATION_PROJECTION EXTENDED_PROJECTION \
 benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
 benchmarks/elife-held-source-acceptance-v1/panel.json \
 benchmarks/role-first-proof-decoding-v1/results.json \
 NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json QWEN_EMBEDDING NEW_RESULTS
```

Repeat with `--reverse` and a fresh output path.
