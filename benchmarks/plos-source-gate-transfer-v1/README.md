# Fixed-threshold transfer for lexical source confirmation

Freeze a single threshold on the [original PLOS source-confirmation panel](../plos-source-clue-agreement-v1/README.md). For each case where the rank-four join abstains and rank-12 finds a candidate, score the lesser of the two source clues' IDF-weighted overlap with their proposed publications' indexed text. Set the cutoff to the maximum score of all wrong calibration expansions. Keep the encoder revision fixed. The gate accepts a candidate only when its score is strictly above the frozen cutoff. Existing top-four choices pass unchanged.

Capture two new topical PLOS ONE inventories related to electricity demand, microgrids and power generation. Preserve the public query receipts and JATS hashes. Remove papers with any DOI, normalized byline name or validated publisher ORCID from the prior JATS archives. Use title and the first half of the abstract for the index. Generate two held-back clue styles from the last two sentences. The result view gives those sentences. The keyword view uses up to eight nontrivial terms, favoring terms attested in the corpus but uncommon. Connect test pairs only through publisher-validated ORCID identifiers. Evaluate both-source present cases and each of two single-source masks. Thresholds and the shared graph-search procedure remain fixed during the new test.

The threshold was recorded in the research branch before running retrieval on the new source panel. The [preparation manifest](holdout-manifest.json) pins the corpus and clue files. The [primary](capture-primary.json) and [secondary](capture-secondary.json) capture receipts pin the query and source XML. The raw licensed publisher records are available through the DOIs in the manifest. The original and transfer tasks use the same publisher and clue-construction policy, with separate articles and author identifiers.

## Independent RTX 5090 results

The original development run (job 42432) freezes a strict cutoff of 0.1120894758 from five false expansion proposals. The [frozen record](frozen.json) was released under commit `715907ee69f51def3984bb5ea441e161d67ec702` before retrieval on the new test. The test inventory captures 60 articles, removes two with shared normalized byline names and three with insufficient abstract segmentation, and retains 55 articles and six linked paper pairs. The 36 test cases contain 12 both-source and 24 single-source-masked requests.

| Policy | Correct with both sources / 12 | Wrong with both sources / 12 | Wrong with one source missing / 24 | Factored P95 |
| --- | ---: | ---: | ---: | ---: |
| Linked top-four | 9 | 2 | 0 | 8.94 ms |
| Linked expansion | 9 | 2 | 7 | 8.94 ms |
| Frozen lexical gate | 9 | 2 | 3 | 8.98 ms |

Job 42435 observes no correct source expansion on this sample. The lexical gate blocks four of seven wrong additions but still accepts three missing-source substitutes. Two errors among the both-source cases already occur in the top-four route and pass unchanged. This independently measures rejection of wrong additions. All seven added proposals are wrong, so it cannot test whether the frozen gate preserves useful expansions. Keep top-four as the preferred baseline on this panel. A larger independent comparison needs both useful and false expansion proposals. [Aggregate results](results.json)

## Reproduce

Use the development corpus and panel from the [original source confirmation study](../plos-source-confirmation-v1/README.md). Run the initial calibration or use the [frozen result](frozen.json).

```sh
PYTHONPATH=src:. python -m experiments.credit.frozen_source_gate_transfer \
  freeze DEVELOPMENT_PROJECTION DEVELOPMENT_PANEL QWEN_CHECKPOINT NEW_FROZEN
```

Use `experiments.credit.capture_source_gate_holdout` in separate output directories, the second time with `--secondary`. Pass the development projection and the complete prior JATS archive directories to both capture and preparation. Run the same preparation for the two captures, with the second passed as `--additional`.

```sh
PYTHONPATH=src:. python -m experiments.credit.prepare_source_gate_holdout \
  PRIMARY_CAPTURE DEVELOPMENT_PROJECTION NEW_HOLDOUT PRIOR_JATS_DIR [OTHER_PRIOR_JATS_DIRS...] \
  --additional SECONDARY_CAPTURE
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.frozen_source_gate_transfer \
    test NEW_HOLDOUT/projection.json NEW_HOLDOUT/panel.json QWEN_CHECKPOINT NEW_RESULTS \
    --frozen FROZEN
```

Run calibration and inference in Slurm GPU allocations. Report observed source identities and missing-source errors. Factored timing includes online encoding, lexical and dense retrieval, and the two identity joins. The gate's overlap computation is timed on proposed expansions. The protocol excludes model loading, index building, request formation and serving.
