# Fit synthetic contrast scores for fixed real rescue candidates

The [rescue threshold transfer](../followir-rescue-threshold-transfer-v1)
shows that the cutoff learned on initial topical candidates can reject
every positive candidate proposed during rescue. Recalibrating on
human-labeled rescue candidates transfers unevenly between Core17
and News21. This study fits a lightweight verifier head on the public
[FollowIR synthetic training data](https://github.com/orionw/FollowIR)
and tests transfer to the same human-judged rescue workload.

Hold fixed the first four topical candidates, initial verification
threshold, and up to two full-pool constraint rescue proposals.
Use the same Qwen3-Reranker-0.6B and original 850-character lead views.
The previously reported human-calibrated raw threshold is a reference,
and the runner verifies its accepted positive, negative, invalidated,
and unjudged counts on both panels.

Train three linear heads with the same public synthetic examples.
Screen the training set against the target topic and instruction
wording and split it by normalized topic. Calculate feature scales
on training only. Freeze all thresholds from a disjoint synthetic
calibration split at a 10% empirical cap on known negatives.

- **Synthetic raw** transfers a cutoff on the unchanged verifier margin.
- **Lexical head** adds topic term coverage and coverage of instruction
  terms absent from the query. It requires no extra verifier forward.
- **Contrast head** uses the unchanged verifier margin and the difference
  between the full instruction margin and a score for the same document
  with only the query. It adds one verifier batch on the rescue route.
- **Matched contrast head** uses those same features. Its loss adds a
  pairwise term for matched positive and negative synthetic documents
  with the same topic, instruction and contrast identifier.

The heads make no change to which documents are opened. Report candidate
acceptances by human judgment, distinguishing documents invalidated by
the instruction change and unjudged cases. Pair comparisons across all
topics, with zero changes where rescue is inactive. The fixed pools
evaluate document relevance, without establishing expert ownership.
Measured route cost separates the existing retrieval and rescue passes
from the lexical feature overhead and extra topic-only verifier call.

## Results

Slurm job `43644` ran on one RTX 5090. The overlap screen removes
34 synthetic rows. The resulting split has 1,391 fit rows,
389 matched pairs within fit, and 351 calibration rows with 185
negatives. Each fitted head accepts 18 of the 185 calibration
negatives and 162 of 166 positives. The online transfer test has
30 rescue candidates in News21 and 34 in Core17.

| Rescue acceptance policy | News21 positives / negatives / invalidated / unjudged | Core17 positives / negatives / invalidated / unjudged |
| --- | ---: | ---: |
| First-pass raw | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Human rescue raw | 3 / 5 / 2 / 0 | 4 / 0 / 0 / 0 |
| Synthetic raw | 11 / 17 / 12 / 2 | 13 / 16 / 6 / 4 |
| Synthetic lexical | 10 / 16 / 11 / 2 | 12 / 12 / 6 / 3 |
| Synthetic contrast | 11 / 17 / 12 / 2 | 13 / 16 / 6 / 4 |
| Synthetic matched contrast | 11 / 17 / 12 / 2 | 13 / 16 / 6 / 4 |

The synthetic raw cutoff is −5.8125, versus human rescue cutoffs
of 0.375 for News21 and 2.5 for Core17. The two contrast heads
and the synthetic raw cutoff have identical counts on both panels.
The lexical head rejects four more known negatives in Core17
than synthetic raw, but still accepts 12 of 17 known negatives.
These comparisons locate the problem after candidate expansion.
The 10% synthetic calibration operating point does not transfer
to real rescue negatives. Matched synthetic contrast training
does not correct that shift.

For requests that reach rescue, adding the lexical features changes
the local P95 from 67.95 to 68.06 ms in News21 and 46.89 to 47.00 ms
in Core17. Adding the extra verifier forward gives 76.77 and
55.71 ms. These warm replay times include the existing retrieval
and verification path, and exclude model startup and serving.
The test uses small reused public document cohorts; synthetic
false examples are not guaranteed to resemble topical candidates
that fail the amended instructions. A future policy should use
independently labeled real near misses for calibration, and report
risk at similar issuance before adding extra verification.

[Aggregate results](results.json)

## Reproduce

Use the pinned [FollowIR paper and release](https://aclanthology.org/2025.naacl-long.597/),
prepared document pools and caches from the
[rescue-scope study](../followir-constraint-rescue-scope-v1), and the
same frozen open checkpoints. Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.followir.rescue_contrast_head \
    FOLLOWIR_TRAIN_JSONL CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS \
    QWEN_EMBEDDING QWEN_RERANKER \
    benchmarks/followir-constraint-rescue-scope-v1/results.json \
    benchmarks/followir-rescue-threshold-transfer-v1/results.json NEW_RESULTS.json
```
