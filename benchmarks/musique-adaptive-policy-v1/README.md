# Reading-conditioned policy versus one-shot graph retrieval

This experiment follows the [MuSiQue evidence-gap audit](../musique-evidence-gap-v1).
That audit found that a fixed graph propagation pass recovered more complete
evidence sets than the original sequential rule at six and twelve reads.
Here we test whether a learned policy benefits from knowing which passages
it has already selected.

## Design

Use the [official MuSiQue release](https://github.com/stonybrooknlp/musique).
Select 768 questions by fixed ID hash from the official training split.
Select a different 768 by fixed ID hash from the public development split,
excluding all 464 queries in the previous audit. Each panel has its own
deduplicated pool of title and paragraph pairs. The two panels do not share
question strings. The official test labels are not used.

Reuse the same frozen Qwen3-Embedding-0.6B model and exact title mention
graph. One-shot baselines are dense, lexical-dense reciprocal rank fusion,
and three graph propagation steps. A second one-shot graph baseline fuses
the dense and propagation rankings. The prior sequential link heuristic
also runs. All policies emit at most four, six, or twelve paragraphs.

The two learned policies share the union of each base ranking's first 28
passages. They also share the same training prefixes, frozen input models,
weighted binary loss, and 32-unit MLP architecture. Prefixes come from
dense ranking, one-shot propagation, and their fusion. Annotation marks
the remaining supporting passages during training. The static model sees
ranking signals, lexical overlap, passage features and graph degree. The
state-conditioned model also sees title repetition, links to the previous
selections, and as-yet-uncovered query terms. All inference features come
from the index and selection history. Neither model sees gold passages,
an independent verification score, or external observations at inference.

The incremental implementation maintains coverage, repeated titles, and
link counts after each new selection. It reuses the same neural weights
and is checked against full recomputation on the ordered twelve-passage
sequence. Per-policy timings cover the twelve-choice CPU stage.
Embedding, initial retrieval, graph propagation and candidate feature
construction belong to the shared query stage. This is a pooled
evidence retrieval test with fixed read counts. It does not measure
reader acceptance or a production search service.

## First held-out retrieval run

| Policy | Complete at 4 reads | Complete at 6 reads | Complete at 12 reads | Post-ranking P95, 12 choices |
|---|---:|---:|---:|---:|
| Dense | 105 | 136 | 200 | 0.001 ms |
| Hybrid | 97 | 135 | 200 | 0.001 ms |
| Graph propagation | 140 | 182 | 272 | 0.001 ms |
| Dense plus propagation | 112 | 151 | 239 | 0.001 ms |
| Linked reads | 134 | 178 | 251 | 0.157 ms |
| Learned static | 168 | 223 | 305 | 0.097 ms |
| Learned with selection state | 184 | 239 | 312 | 0.916 ms |

The training and evaluation pools have 11,267 and 9,718 exact
paragraph records, respectively, with 283 text records in common.
At six reads the adaptive gain over learned static is +2.08 percentage
points, with a paired descriptive 95% interval of +0.52 to +3.65.
The gain over one-shot propagation is +7.42 points, with a paired
interval of +5.08 to +9.77. The adaptive-versus-learned-static interval
at twelve reads ranges from −0.52 to +2.34 points. Both model heads
fit from the same 346,063 state–candidate training rows per arm.
The full shared query stage took 11.86 ms at P95. In the main held
pool, 497 of 768 queries have all annotated support in the union
available to the two learned heads. This is a candidate-pool upper
bound and leaves room for better first-round recall.

## Incremental replay

On all 768 held-out requests, the incremental implementation
returned exactly the same ordered twelve-paragraph sequence as
full state recomputation. Its P95 was 0.371 ms versus 0.911 ms in
the same run, including setup of query-local masks for the
incremental arm. The complete-evidence counts and paired intervals
also exactly reproduced the first run. This test changes only the
implementation of the learned dynamic selector.

## Four-paragraph reader replay

Freeze the previous audit's four-paragraph 0.6B cutoff at 3.375.
No threshold is fitted to this panel. Compare the same state-independent
and state-conditioned selections through that verifier. Also replay
the matching official missing-evidence contexts from MuSiQue-Full.
Clipped inputs are excluded from positive acceptance and counted
separately.

| First-pass retrieval | Unclipped complete | Complete and accepted | Incomplete and accepted | Clipped | Reader P95 |
|---|---:|---:|---:|---:|---:|
| Graph propagation | 140 | 94 | 194 | 20 | 12.63 ms |
| Learned static | 164 | 116 | 212 | 18 | 12.58 ms |
| Learned with selection state | 179 | 132 | 225 | 35 | 13.03 ms |

The adaptive gain in complete-and-accepted outcomes over learned
static is +2.08 percentage points across all 768 requests, with a
paired descriptive 95% interval of +0.52 to +3.65. Requiring acceptance
of a complete positive and rejection of its official negative gives
81, 100, and 113 paired successes, respectively. The frozen reader
accepts 91 of the 759 unclipped official missing-evidence contrasts
(12.0%). A natural view missing an annotated paragraph may still have
alternative evidence, so its acceptance is a diagnostic coverage
warning. The higher clipping count and the measured contrast
acceptance keep the verifier operating point open for further work.
This reader comparison uses four paragraphs. The six-paragraph
improvement above measures evidence recall before verification.

## Reproduce

Use the archive and verified model snapshots listed by the
[previous audit](../musique-evidence-gap-v1). Run on a Slurm CUDA node.
The policy heads are fitted on CPU and have one hidden layer.
The embedding and optional reader use frozen CUDA checkpoints.

```sh
script=benchmarks/musique-adaptive-policy-v1/musique_policy.py
study_dependencies=benchmarks/musique-evidence-gap-v1
PYTHONPATH=src:.:"$study_dependencies" python "$script" ARCHIVE EMBEDDING incremental.json
PYTHONPATH=src:.:"$study_dependencies" python "$script" ARCHIVE EMBEDDING reader.json --reranker RERANKER
```

The first retrieval result used job 45494. The equal-output
incremental replay used job 45498. Job 45502 added the reader. All ran on ws3.
