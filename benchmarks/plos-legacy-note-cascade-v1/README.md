# Route uncertain compiled contribution claims to the full-note reader

Start from [compiled legacy contribution notes](../plos-compiled-legacy-notes-v1/README.md). Reuse the same 30 calibration papers, 20 energy test papers, phrase variants, note cache, and reader checkpoint and cutoff. The 4B reference runs a two-candidate batch on every note. Test whether lexical or semantic owner-contrast scores can settle easy decisions before the 4B call.

Compare a character gate, a semantic gate, and a gate that requires both. For each gate, choose lower and upper candidate-score quantiles from a predefined grid on calibration. At or below the lower threshold the candidate is rejected. Above the upper threshold the candidate is accepted. Scores between the thresholds go to the reader. For the combined gate, both scores must satisfy the corresponding early condition. Uncertain pairs use the original two-candidate reader batch, retaining only pending decisions from it. This preserves checkpoint input shape and avoids giving a cheaper one-candidate batch to the cascade.

On calibration, require at least as many supported decisions and no more wrong-author decisions than the original 4B output. Among feasible settings keep the route with the lowest mean estimated latency, measured as the fast stage plus the 4B stage when needed. The always-read control can win the calibration. It remains the selected route in that case. Report the best gate as a separate diagnostic. Freeze parameters before the timed test. Measure each live route consecutively from the cached note and supplied task labels through its decision. Timing includes the fast scoring and any actually executed reader inference, and excludes corpus search, index loading, transport and model initialization.

For the agreement route, also execute the same frozen decisions using lazy stage ordering. If either candidate is inconclusive on the lexical score, the reader is mandatory. Run it first. The semantic stage can then affect only a candidate with a definite lexical result that differs from the reader decision. If both lexical decisions are definite, compute semantic agreement first and call the reader only if the two signals cannot settle both decisions. The equivalence rule holds for fixed stage verdicts. Verify actual decision parity and mandatory reader-call parity, because numerical inference can depend on execution shape. The pair uses the same shapes and candidate presentation in both schedules.

The design uses previously explored publisher-note cohorts. Counts measure observed attribution under the registered publisher statement grammar, not a risk guarantee under free-text or real enterprise search. The cutoff grid uses calibration labels only.

## RTX 5090 results

Job 42632 reproduces the earlier 4B outcome counts. On calibration the 4B route accepts 23 supported authors and three wrong authors. The character, semantic, and agreement gates select 10, 11 and 17 reader calls out of 30 respectively and meet both calibration constraints. On the 20 test papers, the live measurements are:

| Route | Supported accepted | Wrong owner accepted | Exact pairs | 4B calls | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always 4B | 15 | 3 | 13 | 20 | 21.37 ms | 24.04 ms |
| Character gate | 17 | 5 | 13 | 8 | 8.73 ms | 21.77 ms |
| Semantic gate | 14 | 4 | 11 | 7 | 15.65 ms | 30.60 ms |
| Agreement, eager | 16 | 2 | 14 | 15 | 24.14 ms | 30.75 ms |
| Agreement, lazy | 16 | 2 | 14 | 15 | 20.89 ms | 30.75 ms |

The character gate makes fewer 4B calls and improves mean latency, but the development constraints do not transfer. The agreement gate accepts one more supported author and one fewer wrong author than always reading. Each difference depends on one test publication, and its paired interval contains zero. The eager and lazy agreement schedules match on all 20 decisions and 4B calls. The lazy schedule reduces semantic forwards from 20 to 12. Exhaustive synthetic testing over all 324 two-candidate gate and reader verdict combinations also produces the same decisions and required reader calls.

The schedule removes warm mean-time overhead relative to eager agreement on this panel. Its mean is close to always-read, and its P95 remains 6.71 ms slower. A request that needs both models still pays for both. The result supports eliminating redundant stage execution under a fixed decision contract. It does not demonstrate a P95 speedup under equivalent attribution risk on an independent cohort. [Aggregate results](results.json)

## Reproduce

Use the same pinned XML snapshot and checkpoints as [the compiled-note benchmark](../plos-compiled-legacy-notes-v1/README.md). Supply its aggregate result and run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_note_cascade \
    SNAPSHOT QWEN_EMBEDDING QWEN4B COMPILED_RESULTS NEW_RESULTS
```
