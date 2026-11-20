# Separate window and reader effects on Robust04 rescue

The [calibration transfer](../followir-robust04-rescue-risk-v1)
on Robust04 found a low ceiling for the short-window 0.6B margin.
Hold the same first-pass threshold, candidate set, and overlap
screen fixed. Score the out-of-fold Core17 and News21 rescue
candidates for calibration, then transfer once to the 90 pinned
Robust04 candidates. The runner checks the frozen original cutoff
and lead-window verifier decision counts.

Cross two open checkpoints and three views. The 0.6B reranker
scores yes and no using its retrieval prompt. Qwen3-4B-Instruct
scores support and rejection using the same prompt as the previous
[reader diagnosis](../followir-rescue-reader-diagnosis-v1). Model
and prompt change together, while all evidence windows are shared
across readers.

- Lead uses the title and first 850 body characters.
- Criterion window uses the title and one equal-length window
  selected by inverse document frequency of instruction terms
  outside the topic query. Window selection uses no relevance labels.
- Long input uses the title and document body, truncated to fit
  the 1,024-token prompt budget. It is not guaranteed to expose
  all evidence. Record capped inputs.

For each arm, choose the smallest threshold satisfying both
development constraints. Allow at most 10% of known negatives in
each collection and at most 10% of pooled documents invalidated by
the amended instruction. On Robust04, report the observed counts
and the retrospective scalar threshold ceiling under both 10%
risk limits. The latter uses the holdout labels and measures
score separation only.

Measure P95 per rescued request, including window selection,
tokenization, and verification of two already selected candidates.
The measurements exclude initial candidate generation, offline
IDF computation, model loading, and service costs. Human relevance
labels apply to whole documents. The experiment can identify an
effect of showing a different window. A score improvement alone
does not identify the exact supporting span.

## Results

Slurm job `43658` ran on one RTX 5090. The test reproduces the
previous 0.6B lead cutoff and its outcomes. All six arms score
the same 90 candidates over 45 rescued requests. There are 33
known positives, 49 known negatives including 18 invalidated
documents, and eight unjudged candidates.

| Reader and input | Transferred positive / negative / invalidated | Retrospective positive at both 10% caps | P95 |
| --- | ---: | ---: | ---: |
| 0.6B, lead | 3 / 3 / 2 | 1 | 13.77 ms |
| 0.6B, criterion window | 3 / 4 / 3 | 3 | 14.14 ms |
| 0.6B, long input | 2 / 0 / 0 | 2 | 21.62 ms |
| 4B, lead | 4 / 0 / 0 | 9 | 52.31 ms |
| 4B, criterion window | 4 / 0 / 0 | 7 | 52.98 ms |
| 4B, long input | 3 / 0 / 0 | 8 | 91.45 ms |

No arm accepts an unjudged candidate at its transferred cutoff.
No held-out input is truncated at the configured model limit.
The 0.6B criterion window does not improve accepted positives over
lead and admits one more known negative. Its retrospective ceiling
increases from one to three. Reading the full document produces
zero known negative acceptances at the 0.6B transferred cutoff,
but issues only two positives. With the 4B model and prompt,
both short views accept four positives and no known negatives.
The lead view finds one more positive than the 0.6B lead with
a descriptive paired 95% interval of −0.06 to +0.12 positives per
topic. The effect is uncertain. Its higher retrospective ceiling
shows that changing the reader and prompt improves ordering
on this cohort, while a threshold chosen outside Robust04 does
not realize all that headroom.

The 4B lead costs 3.8 times the 0.6B lead in local verification
P95. Its long input costs an additional 39.14 ms at P95 and
does not improve the accepted positive count. Use the 4B short
reader as a selective escalation comparator. Neither long input
nor the larger reader is supported as an unconditional default by
these results. The same fixed cohort and document-level judgments
cannot establish case-by-case evidence sufficiency or enterprise
recommendation quality.

[Aggregate results](results.json)

## Reproduce

Use all three pinned prepared FollowIR collections and the frozen
embedding, reranker, and Qwen3-4B-Instruct checkpoints. See the
[Robust04 anchor](../followir-robust04-rescue-risk-v1) for input setup and
the [earlier reader comparison](../followir-rescue-reader-diagnosis-v1)
for the 4B prompt and checkpoint layout. Run inside a Slurm CUDA
allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.followir.robust04_reader_factorial \
    CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS \
    ROBUST04_JSON ROBUST04_VECTORS QWEN_EMBEDDING QWEN_RERANKER QWEN4B \
    benchmarks/followir-constraint-rescue-scope-v1/results.json \
    benchmarks/followir-robust04-rescue-risk-v1/results.json NEW_RESULTS.json
```
