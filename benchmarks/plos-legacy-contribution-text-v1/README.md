# Scope original publisher statements to the named contributor

Use PLOS ONE JATS author notes published in 2010–2015, before standardized CRediT fields were consistently available. Query the publisher's title index for computing topics for calibration and energy topics for test. Retain only titles containing a requested phrase. Supplement the first test capture with more energy titles to reach the planned source count. DOI and byline-name checks separate the two retained panels. The [capture receipt](capture.json) records the included and attempted publications and pins every downloaded XML file.

Read the original openly licensed contribution notes from the publisher, normalizing whitespace. Match the initials to a unique byline spelling. To construct the labels, recognize legacy statements about design, execution, analysis, writing or supplied resources. Keep one paired test per paper, where one author is credited with both requested tasks and another author with exactly one. The model sees identical publisher text and task wording for both authors. Only the queried initials change. The released criterion means explicit credit in a recognized statement. It does not characterize the person's unreported skills.

Compare broad relevance, an explicit conjunction-and-ownership instruction over the whole note, and the same instruction after keeping the sentences containing the target initials. The scoping rule uses a literal whole-token match and does not inspect the requested roles or the adjudication labels. The frozen 4B comparator sees the full note. All methods evaluate the same authors. Use a separate task paraphrase on the test panel. Fit the score cutoff for each route on the calibration panel under a 10% wrong-author acceptance cap. Report pair ordering, accepted correct and wrong owners, and cases where both accept/reject decisions are correct.

These notes are authored publisher text. The eligible legacy format uses concise role sentences and author initials. Results therefore measure scoped attribution for this format, with no implication for unrestricted prose, pronoun resolution or cross-source identity. Timing covers a two-candidate batch and excludes search, storage and model initialization.

## RTX 5090 results

The local capture downloads 42 calibration and 40 test publications. GPU job 42615 verifies their checksums before scoring. The parser retains 30 computing and 20 energy papers, with one same-paper contrast per publication. Only the task wording and held-out source domain change across calibration and test. No captured byline name in a retained test record matches a parsed calibration byline name.

| Route | Relevant author ranks higher / 20 | Accepted supported author / 20 | Accepted wrong author / 20 | Both correct | Paired P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Broad relevance | 15 | 1 | 0 | 1 | 8.83 ms |
| Explicit ownership over full text | 14 | 2 | 0 | 2 | 8.80 ms |
| Match initials before verification | 17 | 0 | 0 | 0 | 8.89 ms |
| Explicit 4B reader | 19 | 15 | 3 | 13 | 23.71 ms |

Scoping the note by initials raises the number of correctly ordered pairs by two relative to broad relevance. The descriptive paired interval for the difference spans −10 to +30 percentage points. The scope-first threshold then accepts zero test authors. The 4B reader trades 2.7 times the paired P95 for much higher coverage and accepts three wrong authors. The test risk counts use only 20 pairs and do not establish that the 10% calibration cap transfers. A useful follow-up should preserve the open reader as a quality comparator, keep unresolved outcomes visible, and seek a format-robust ownership check with more independently annotated documents. [Aggregate results](results.json)

## Reproduce

Recreate the pinned XML snapshot from the public publisher using the included receipt. A changed or unavailable XML file fails the restore check. Run the model comparison inside a Slurm GPU job.

```sh
PYTHONPATH=src:. python -m experiments.credit.legacy_contribution_text restore CAPTURE_JSON NEW_SNAPSHOT
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_contribution_text run \
    SNAPSHOT QWEN_RERANKER QWEN4B NEW_RESULTS
```
