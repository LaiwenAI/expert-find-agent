# Bound author verification to the relevant contribution clauses

Reuse the 197-paper catalog and 72 held-out requests from the [source expansion replay](../plos-source-expansion-depth-v1/README.md). Freeze the source nominations for the top-one and calibrated depth-four routes. Both routes include the frozen first-source body check. Compare the same 4B checkpoint and paired author prompts over four verification routes.

| Route | Evidence shown to the verifier |
| --- | --- |
| Character cascade | Frozen character gate with the prior whole-note 4B fallback. |
| Full note | Run the 4B reader on the complete contribution note for every candidate. |
| Author scope | Run the 4B reader on the original clauses that name the candidate. |
| Scoped veto | Apply the character cascade, then require an additional positive author-scoped verdict for its recommendations. |

The scope preserves the original clause text and comes from literal matches on byline initials. It uses neither the requested duty nor the target role labels to select clauses. The evaluator derives duty labels separately from the publisher note. On 30 earlier publications with the correct source supplied, find the least restrictive threshold for each alternate route whose wrong-author query count is at most the frozen cascade's count. The test keeps every threshold, model and source decision fixed.

Report clean recommendations, which require a nonempty list of supported authors with no unsupported additions, and both wrong-author and wrong-source recommendations. Supported and wrong-author queries can overlap. Compare paired clean-success and risk differences using author-name components. Factored latency adds retrieval, chosen-source inspection and verification for the selected route. It excludes model initialization, service transport, and concurrent queueing. This is a diagnostic evaluation on reused requests and legacy notes with a recognized initials format.

## RTX 5090 results

Slurm job 42681 on ws5 reproduced the frozen source and character-cascade counts. On 30 calibration papers the cascade made six wrong-author queries. The resulting logit-margin cutoffs were 17.75 for the full note and 21.75 for author scope. The veto calibration admitted all cascade positives at the least restrictive cutoff, so on this split the additional model calls left its decisions unchanged.

| Clue | Top-one author route | Any supported author | Clean success | Wrong author, correct source | Wrong-source recommendation, withheld | 4B calls, source present | Factored P95, source present |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Excerpt | Character cascade | 57 | 37 | 22 | 0 | 103 | 103.58 ms |
| Excerpt | Full note | 48 | 41 | 9 | 0 | 188 | 148.63 ms |
| Excerpt | Author scope | 30 | 26 | 9 | 0 | 188 | 131.24 ms |
| Excerpt | Scoped veto | 57 | 37 | 22 | 0 | 214 | 173.30 ms |
| Keywords | Character cascade | 42 | 27 | 17 | 4 | 81 | 103.00 ms |
| Keywords | Full note | 35 | 31 | 6 | 3 | 158 | 148.34 ms |
| Keywords | Author scope | 24 | 23 | 5 | 1 | 158 | 130.80 ms |
| Keywords | Scoped veto | 42 | 27 | 17 | 4 | 167 | 172.85 ms |

The full-note reader lowers wrong-author counts from 22 to 9 and from 17 to 6. Clean success increases by four queries for each clue, with descriptive paired 95% intervals spanning zero, while support counts fall by nine and seven. Factored P95 rises from about 103 to 149 ms. Author scope reduces wrong authors but gives up more supported recommendations at its independently calibrated threshold. The direct comparison changes both the context and the threshold. The result is a comparison of operating points calibrated by the same risk rule.

With calibrated depth four, clean successes are 37 versus 42 for excerpts and 27 versus 32 for keywords when switching from the cascade to the full-note reader. Comparing depth four to top one under the same full-note reader adds one clean query per clue. The expanded excerpt route also adds one wrong-source recommendation in the withheld view. On this reused panel, changing verification moves clean success more than expanding search depth, and the quality benefit of using the full note for every candidate remains uncertain. These results motivate measuring selective full-note rechecks at separately predeclared operating points. [Aggregate results](results.json)

## Reproduce

Restore the same publisher captures and open checkpoints as in the source expansion replay. Run inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_attribution_intervention \
    OLD_SNAPSHOT FIRST_TRANSFER FRESH_TRANSFER QWEN_EMBEDDING QWEN4B \
    OPEN_ATTRIBUTION_RESULTS BODY_CONFIRMATION_RESULTS CASCADE_RESULTS \
    SOURCE_EXPANSION_RESULTS NEW_RESULTS
```
