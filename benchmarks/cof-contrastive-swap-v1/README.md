# Transfer two-order reviewer comparison to fixed-issuance CoF search

Transfer the frozen comparison prompt and decision rule from
[LR-Bench](../lrbench-contrastive-evidence-v1/README.md) to
[CoF](https://github.com/yuz9yuz/CoF). Use the same SIGIR, KDD and
SciRepEval cohorts as the
[competing-profile study](../cof-competing-profile-evidence-v1/README.md).
These panels are independent of LR-Bench and reused from earlier CoF
experiments.

Retrieve from the complete released reviewer inventory using the
existing equal blend of profile centroid and best owned-source
cosine. Preserve the first four reviewers. Only the fifth reviewer
and the sixth-ranked challenger may occupy the last slot, and all
arms issue exactly five reviewers. The challenger and incumbent
each provide up to two source abstracts in the replacement
comparison. The model uses the same prompt as on LR-Bench and
receives no reviewer IDs or baseline scores. Choose the first
presentation order by a hash of query and unordered reviewer IDs.
Do not fit a CoF routing threshold or use any target labels for
decision making.

| Arm | Replacement rule |
| --- | --- |
| Dense five | Keep the initial five. |
| Pointwise | Swap if the challenger has a higher maximum score on its first two sources from the frozen 0.6B reranker. Score all six profiles together and reuse the scores for citations. |
| One order | Swap if one presentation to Qwen3-4B-Instruct-2507 favors the challenger. |
| Both orders | Compare both presentations in a batch. Swap only when both favor the challenger. |
| Deferred reverse | Start with one presentation. Run the reverse presentation only if the first favors the challenger. Swap only when both favor the challenger. |

The last two arms implement the same decision rule for identical
reader predictions. Because neural scores may vary with batch
shape, compare their actual ordered recommendations and citations.
Every route uses the shared 0.6B citation reader for the selected
five reviewers. A reviewer receives the first two above-threshold
citations when at least two of its three source scores exceed
zero. The pointwise arm reads the third source for both fifth-slot
options in its one shared citation batch. This is an execution
comparison, with distinct total model costs across arms.

Measure official hard-label positive reviewers at five, swap gains
and losses, and how many of the fifth/sixth pairs admit a provable
improvement under the available labels. Resample paired requests
for the known-positive difference. Separately report the possible
P@5 delta over all completions of missing labels. Also count
known-positive reviewers with model-selected citations. Reviewer
judgments do not establish that those citations support a required
claim personally. Timing covers query encoding, full retrieval,
candidate comparison, and producing the recommendation and
citation structure, with measured common retrieval charged to
every arm. The workload is warm and local, with no serving queues
or network delay.

## Results

Slurm array `44261` uses one RTX 5090 per task on ws5. The
retrieval inputs, hard-label files, and selected query manifests
match the earlier CoF study. A rotated SIGIR repeat reproduces every
non-timing aggregate. The table reports the first run for each panel. A cited positive is a positively judged reviewer with two
model-selected sources. This metric does not validate the sources.

| Panel | Arm | Known positives at five | Cited positives | Swaps | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| SIGIR (73 queries) | Dense five | 63 | 18 | 0 | 114.46 |
| SIGIR | Pointwise | 66 | 22 | 37 | 146.22 |
| SIGIR | One order | 60 | 21 | 34 | 181.04 |
| SIGIR | Both orders | 62 | 18 | 9 | 234.82 |
| SIGIR | Deferred reverse | 62 | 18 | 9 | 239.29 |
| KDD (122 queries) | Dense five | 24 | 13 | 0 | 165.26 |
| KDD | Pointwise | 20 | 12 | 49 | 201.27 |
| KDD | Both orders | 25 | 14 | 7 | 318.47 |
| KDD | Deferred reverse | 25 | 14 | 8 | 293.08 |
| SciRepEval (59 queries) | Dense five | 52 | 25 | 0 | 140.62 |
| SciRepEval | Pointwise | 51 | 24 | 28 | 179.94 |
| SciRepEval | Both orders | 52 | 25 | 2 | 256.20 |
| SciRepEval | Deferred reverse | 52 | 25 | 2 | 250.31 |

On SIGIR all fifth/sixth pairs have hard labels. There are nine
possible positive-for-negative swaps, a ceiling of +2.47 points
in P@5 for this restricted action. Both orders recovers none,
replacing one known positive with a known negative. Pointwise
recovers seven such swaps and loses four other positives. Its
net change is +0.82 P@5 points, with a paired 95% interval of
−0.82 to +2.74 points. The full-baseline ranker is stronger
than the dual comparison on this panel and faster than all
replacement variants.

Only four fifth/sixth pairs have both labels on each of the
other two panels. Completing the missing labels could place the
dual route's P@5 effect between −0.98 and +0.82 points on KDD
and −0.34 and +0.34 on SciRepEval. On those panels the known
counts and bounds give different information. Across all three
panels, deferring the reverse order reduces the number of 4B
prompts by 27–32%, while SIGIR P95 rises relative to the
batched comparison. There is one changed ordered output on
KDD between the two executions. The two-order improvement
observed on LR-Bench did not transfer to this fixed-issuance
experiment at the tested cost.

[Full GPU aggregates](results.json)

## Reproduce

Use the existing CoF snapshot and vector protocol, the pinned
embedding and reranker checkpoints, and the open comparison
checkpoint from the linked LR-Bench experiment. Run in a CUDA
allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_contrastive_swap \
  SIGIR EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON --order-offset 0
```

Repeat for KDD and SciRepEval.
