# Search publications and verify their authors' requested contributions

Reuse the [frozen independent author cohort](../plos-independent-cascade-v1/README.md) and the exact 4B and character-gate thresholds. Expand the searchable source inventory to the compatible, openly licensed notes in both publisher captures. Given a report excerpt or its eight rare indexed terms, retrieve the intended publication. The source index uses title and abstract prefix only. It holds back the last two abstract sentences, from which both clue styles are formed. A frozen embedding search and weighted lexical overlap contribute reciprocal source ranks.

After selecting a source, enumerate all unique byline initials that appear in its indexed contribution note. The two requested duties are supplied as task descriptions. A document's author list is available to the search procedure, but the parsed ground-truth role lists are used only to score the exact intended source. Verify candidates in fixed two-author blocks. For the character variant use the previous early accept/reject thresholds and reuse 4B for the remaining decisions. Only emit author IDs with the selected note as their provenance. Test a crossed comparison over author verification and source confirmation. Fit the source cutoff separately per clue style as the maximum dense score among incorrect old evaluation proposals, including proposals when the intended source is withheld. No test decisions enter this fitting step.

Evaluate each held-out request with its intended source available and withheld. Distinguish finding the right source, returning any supported author, returning an unsupported author on the correct source, and returning any author from the wrong source. Every returned author counts as unsupported under the exact-study contract when the intended source is withheld. An alternate study could still be relevant under a broader discovery task. Timing includes live query encoding, source ranking, any 4B calls, and output serialization. It excludes index construction, network serving or demand parsing. The task is open over the available note inventory and targets a specific source; it does not assess general expert retrieval.

## RTX 5090 results

Job 42651 indexes 113 publisher notes and preserves 30 calibration queries and 45 test queries spanning 44 author groups. Before confidence gating, the correct paper is first for 44/45 complete excerpts and 39/45 keyword clues. The two calibrated source thresholds retain 20/30 and 14/26 correctly ranked calibration sources, respectively.

| Clue | Source threshold | Author verifier | Correct source emitted | Queries with supported author | Queries with unsupported author | 4B calls | Complete P95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Result excerpt | None | All 4B | 44 | 38 | 14 | 116 | 125.97 ms |
| Result excerpt | None | Character then 4B | 44 | 41 | 12 | 56 | 99.42 ms |
| Result excerpt | Frozen | All 4B | 34 | 29 | 12 | 89 | 120.44 ms |
| Result excerpt | Frozen | Character then 4B | 34 | 31 | 9 | 41 | 83.97 ms |
| Keywords | None | All 4B | 39 | 34 | 18 | 127 | 129.98 ms |
| Keywords | None | Character then 4B | 39 | 37 | 15 | 58 | 99.71 ms |
| Keywords | Frozen | All 4B | 22 | 20 | 9 | 60 | 124.77 ms |
| Keywords | Frozen | Character then 4B | 22 | 20 | 9 | 31 | 82.63 ms |

All table rows have the intended source available. Supported and unsupported queries can overlap when a returned list includes both kinds of authors. Without a source cutoff, the author gate saves about half of the 4B calls and adds three supported queries for either clue form. The descriptive 95% paired intervals for both support differences include zero. The paired intervals resample author-name clusters.

With the intended source withheld, the character route recommends authors from another source on 40/45 result-excerpt requests and 41/45 keyword requests. Source confirmation drops these to 0/45 and 2/45. The same confidence cutoff loses supported queries when the source is available, from 41 to 31 for result excerpts and 37 to 20 for keywords. On the available keyword requests, both gated routes have the same supported and unsupported query counts, while the character route reduces P95 from 124.77 to 82.63 ms.

The transfer result separates source presence from author attribution. Accurate attribution conditional on a candidate source does not settle whether that source is the intended one. The benchmark remains an exact-study task with constructed clues, supplied duty boundaries and publisher-note labels. [Aggregate results](results.json)

## Reproduce

Restore both snapshots using the documented receipts in [the independent attribution benchmark](../plos-independent-cascade-v1/README.md). Use the pinned open checkpoints and the recorded author-verification settings in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_open_attribution \
    OLD_SNAPSHOT INDEPENDENT_SNAPSHOT QWEN_EMBEDDING QWEN4B TRANSFER_RESULTS CASCADE_RESULTS NEW_RESULTS
```
