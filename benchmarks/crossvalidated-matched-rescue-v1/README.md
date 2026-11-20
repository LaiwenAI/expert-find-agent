# Match evidence-check budgets for author breadth and depth

Use the same 68 Cross Validated queries, 364 answer records, and
frozen Qwen3 embedding as the
[author-and-witness benchmark](../crossvalidated-witness-retrieval-v1).
Each request starts with the best-ranked answer from each of the
five highest-ranked authors. A frozen 0.6B reranker checks whether
the answer directly addresses the query. It reads the canonical
question title and the first 850 characters of answer text, with
up to 950 request characters. The reranker threshold is zero
on the yes-minus-no margin, fixed as a diagnostic operating point.

Use at most three additional checks per query. Cap the realized
number for all arms at the minimum of three and the number of
initial authors with a second answer. No arm checks an answer
already read or checks the same author twice in the extra batch.
Within each arm, visit candidates in descending dense score.

- **Same authors** reads another answer from an initial author.
- **New authors** reads a best answer from someone outside the
  initial list.
- **Unrestricted dense order** admits either type.
- **Unresolved initial evidence** admits all new authors and
  initial authors whose first answer failed the verifier threshold.

After both batches, sort authors by the best positive verifier
margin. Return up to five authors with one exact answer each.
Record a known witness first among checked proposals and then
among actual final citations. Keep unlisted witnesses and authors
unjudged. The reranker can overwrite a known answer with another
answer by the same person; exact cited-answer scoring exposes
this loss. This diagnostic keeps the same checks for every arm.
It does not learn a policy that decides whether to invoke rescue.

Paired intervals resample components sharing a known answer.
Measure P95 from online query encoding through initial verification,
the policy choice, extra verification, and result assembly. Arms
share the same measured common work, and extra arms run in rotated
order. Offline document encoding and service transport sit outside
the measurement.

## Results

Job `43700` ran on one RTX 5090. All methods use 526 checks, including
the same 340 initial checks and 186 additional checks. The realized
extra budgets are one for two queries, two for 14, and three for 52.

| Scope | Queries proposing a known witness | Known witness in final citations | Proposed witness rejected by verifier | Extra new authors | Local P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial only | 59 | 38 | 21 | 0 | Unmeasured |
| Same authors | 62 | 38 | 24 | 0 | 72.58 ms |
| New authors | 59 | 38 | 21 | 186 | 71.44 ms |
| Unrestricted dense order | 62 | 38 | 24 | 133 | 71.47 ms |
| Unresolved initial evidence | 62 | 38 | 24 | 142 | 71.48 ms |

The additional same-author reads recover a known witness for
three more queries, compared with zero for expansion to new
authors at the same budget. The paired difference in proposal
hit rate is +4.41 points (descriptive 95% interval 0 to +10.14).
The unrestricted and initial-evidence-dependent routes recover
the same aggregate number of known witnesses before verification.
None of the extra known witnesses changes the final output.
No query has a positive-scored known witness that is entirely
lost in final citation selection. At this operating point the
lost known-witness queries are due to nonpositive verifier
margins, rather than displacement after verification.

The designated answer can be a community-endorsed witness without
its lead excerpt independently containing enough evidence. The
benchmark does not identify whether the nonpositive margins are
caused by missing context or verifier misclassification. It
motivates comparing equal-length answer windows or full answers
on the fixed proposals, before increasing breadth or adding a
scope router. The panel is reused and has sparse positive
judgments, so changes in unjudged outputs are not error estimates.

[Aggregate results](results.json)

## Reproduce

Use the same frozen snapshot and verified Qwen3 checkpoints.
Run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.community.matched_witness_rescue \
    SNAPSHOT_DIR/snapshot.json SNAPSHOT_DIR/manifest.json \
    benchmarks/crossvalidated-witness-retrieval-v1/results.json \
    QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS.json
```
