# Frozen evidence policy transfer to HotpotQA and QASPER

This follow-up freezes both heads from the
[MuSiQue selection-policy study](../musique-adaptive-policy-v1). Job
45511 recreated the original training panel and state pool before
reading either target dataset. The bundled NPZ holds numeric model
parameters and feature normalization only. The original MuSiQue
held-out replay reproduces all seven policies' complete-set counts
at all three budgets. No target labels enter training or selection.

## Transfer panels

- [HotpotQA](https://aclanthology.org/D18-1259/) contributes 768 queries
  selected by fixed question-ID hash from its public distractor
  development split, with 620 bridge and 148 comparison questions.
  Deduplicate the sampled contexts into a 7,576-paragraph shared
  search pool. A complete result contains the exact original
  paragraphs housing all labeled supporting sentences. A different
  paragraph on the same Wikipedia page gives no credit.
- [QASPER](https://aclanthology.org/2021.naacl-main.365/) contributes one
  fixed-hash query per eligible test paper, 130 in total. Retain cases
  whose smallest extractable text-only answer annotation requires
  two to four paragraphs. Build a shared pool of 19,710 paragraphs
  from all 416 test papers. Count any of the alternative extractable
  annotations as sufficient. The cued arm supplies the associated paper
  title, which is task metadata on QASPER.
- The source-bound QASPER control gives the same exact identified paper
  to every policy and searches only that paper. Its matched raw and
  cued controls use the same questions and the global pool. Fixed
  graph propagation requires at least 16 eligible passages, so the
  source-bound comparison omits shorter papers.

All comparisons use frozen Qwen3-Embedding-0.6B, the original candidate
pool, rank features, and exact title-edge policy from MuSiQue. Policy
inference uses the incremental implementation with identical weights.
HotpotQA and QASPER are separately authored datasets. Sharing a
Wikipedia title or passage across datasets may occur. The results
evaluate evidence set completion, with no downstream verifier or
expert eligibility label.

## Main transfer

| Test and evidence budget | Dense | Graph propagation | Learned static | Learned with selection state |
|---|---:|---:|---:|---:|
| HotpotQA 4 | 432 | 578 | 558 | 627 |
| HotpotQA 6 | 488 | 655 | 619 | 668 |
| HotpotQA 12 | 582 | 720 | 682 | 710 |
| QASPER raw 12 | 8 | 9 | 5 | 5 |
| QASPER with title cue 12 | 29 | 28 | 19 | 16 |

At four reads, the frozen state-conditioned policy finishes 49 more
HotpotQA evidence sets than graph propagation, +6.38 percentage
points with a paired descriptive 95% interval of +3.78 to +9.11.
The gain over learned static at that depth is +8.98 points with a
paired interval of +6.64 to +11.33. At six reads, the difference
from graph propagation crosses zero. At twelve reads, graph
propagation finishes ten more sets. HotpotQA post-ranking P95 is
0.356 ms for incremental selection, and the shared retrieval and
feature-building stage takes 10.98 ms. These are local component
times, with index construction and external source reads excluded.

For QASPER, adding the exact title as query text increases the
number of cases with a complete evidence set reachable inside the
learned candidate pool from 24 to 94. Both frozen heads still lose
to dense at twelve reads. This helps distinguish missed sources
from selection losses when evidence is already available.

## Source-bound QASPER control

The matched comparison retains 128 queries after omitting two
papers with fewer than 16 passages. The candidate pool contains
complete reference evidence for 23 raw, 92 title-cued, and 115
source-bound queries. When restricted to the designated paper,
dense retrieval finds 26 complete sets at six reads, compared
with 15 for the frozen adaptive head. At twelve reads, hybrid and
propagation each find 56 complete sets, compared with 40 for the
adaptive head. Source binding improves reach and a selection gap
persists. Train and evaluate any proposed within-paper policy
separately from cross-document state selection.

## Reproduce

Use the official asset snapshots and pinned Qwen3-Embedding
checkpoint from the earlier studies. Asset checksums are in the
two scripts. All jobs ran on one GPU in ws3. Job 45517 evaluated
HotpotQA, QASPER and the earlier MuSiQue panel with the same
frozen NPZ. Job 45521 added the source-bound QASPER control.

```sh
study=benchmarks/musique-policy-transfer-v1
dependencies=benchmarks/musique-evidence-gap-v1:benchmarks/musique-adaptive-policy-v1
export PYTHONPATH=src:.:"$dependencies"
python "$study/freeze_policy.py" MUSIQUE_ARCHIVE EMBEDDING frozen-heads.npz
python "$study/transfer.py" hotpot HOTPOT_DEV EMBEDDING frozen-heads.npz hotpot.json
python "$study/transfer.py" qasper QASPER_TEST EMBEDDING frozen-heads.npz qasper.json
python "$study/transfer.py" qasper QASPER_TEST EMBEDDING frozen-heads.npz bound.json --bound-qasper
python "$study/transfer.py" replay MUSIQUE_ARCHIVE EMBEDDING frozen-heads.npz replay.json
```

The released numeric head file has SHA-256
`5880f1de28de4ef5c7d34a43c87ab1f399fdec5af2cf1b1392a4dc75a14202f6`.
The report includes no input passages or private expert records.
