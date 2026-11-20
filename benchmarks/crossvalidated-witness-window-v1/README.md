# Cross answer windows and verifier on fixed known-witness proposals

The [equal-budget study](../crossvalidated-matched-rescue-v1) finds
62 queries with a known answer among the checked sources, with 38
retained after short-window verification. Reuse its same-author arm
and the exact 526 input positions. The selected answers are fixed
from the dense ordering before either reader runs. Every method
uses the same initial and extra batch boundaries and title.

Cross the original frozen Qwen3-Reranker-0.6B and an open
Qwen3-4B-Instruct checkpoint with three windows:

- The title and the first 850 answer characters.
- The title and an equal-length answer window with the highest
  IDF-weighted overlap with the request, using the existing
  fixed-stride selector.
- The title and answer body passed to the reader's 1,024-token
  packed-input limit. Capped inputs are counted.

Both readers receive up to 950 request characters. The 0.6B uses
the original retrieval instruction. The 4B prompt explicitly
asks whether the answer text, beyond the canonical title, addresses
the request. The reader comparison changes checkpoint and prompt
together. Each issues at a strict positive margin, repeats the
same final top-five author rule, and cites one answer per author.
The zero threshold is an unchanged diagnostic for 0.6B and a fixed
diagnostic for 4B. It is not a calibrated risk equivalence.

Count exact retained witnesses, total citation output, accepted
witnesses subsequently displaced, and rejected queries whose
proposed known answers already fit in the lead window. P95
includes window selection, tokenization, model forwards and
assembled decisions, with candidate retrieval held outside the
timed stage. Use the same per-query connected components as the
original benchmark for paired intervals.

The endorsements are limited to the known accepted answers.
Other citations and authors remain unjudged. A complete available
answer body does not prove that the shortened request or a
specific answer span independently suffices.

## Results

Job `43704` ran on one RTX 5090. The 0.6B lead arm reproduces
every published aggregate for same-author rescue. All six
conditions use the same 526 checks and propose a known witness
on the same 62 queries.

| Reader and window | Queries retaining exact witness | Total cited answers | Capped inputs | Local P95 |
| --- | ---: | ---: | ---: | ---: |
| 0.6B, lead | 38 | 58 | 0 | 67.02 ms |
| 0.6B, targeted | 38 | 62 | 0 | 73.15 ms |
| 0.6B, long | 40 | 67 | 151 | 103.14 ms |
| 4B, lead | 15 | 23 | 0 | 318.46 ms |
| 4B, targeted | 16 | 21 | 0 | 346.33 ms |
| 4B, long | 18 | 24 | 160 | 480.42 ms |

At the same length, the targeted window recovers five queries
and loses five relative to lead. Their observed union is 43
queries, an available-label diagnostic that assumes hindsight.
It is not a measured routing policy. The longer 0.6B input
gains two queries without losing any at the same threshold, with
a descriptive paired 95% interval of 0.00 to +7.46 percentage
points. It adds 36.12 ms at P95. Already during lead reading,
four of the 24 rejected queries presented the complete answer
body for every proposed known witness. Missing answer text alone
cannot explain these four decisions. A shortened request,
the assessment prompt, or actual evidence alignment could matter.

The 4B arms emit many fewer citations at the fixed diagnostic
cutoff. Even with longer input they retain fewer known witnesses.
Without assessed alternatives, these counts do not establish
a precision ordering or favor a threshold adjustment. The lead
and targeted window complement each other on this panel, so a
test of selective window choice needs held-out queries and an
equal-time comparison. Reading all longer inputs is a costly
control. [Aggregate results](results.json)

## Reproduce

Use the frozen snapshot and verified open checkpoints. The 4B
reader uses the same checkpoint layout as
[the FollowIR diagnosis](../followir-rescue-reader-diagnosis-v1).
Run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.community.witness_window_factorial \
    SNAPSHOT_DIR/snapshot.json SNAPSHOT_DIR/manifest.json \
    benchmarks/crossvalidated-matched-rescue-v1/results.json \
    QWEN_EMBEDDING QWEN_RERANKER QWEN4B NEW_RESULTS.json
```
