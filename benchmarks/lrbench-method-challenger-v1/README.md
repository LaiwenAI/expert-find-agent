# Nominate a method-focused second source for reviewer matching

Test the two-slot nomination strategy from the
[CSFCube diagnostic](../csfcube-facet-challenger-v1) on released
paper–reviewer familiarity comparisons. Use [LR-Bench and RATE](https://aclanthology.org/2026.acl-long.2035/)
with all 212 paper-centric and 1,184 reviewer-centric pairs.
The paper-centric direction is primary. It asks which of two
supplied reviewers is more familiar with a paper. The reverse
direction asks which of two papers a given reviewer knows better.
The benchmark uses self-assessed familiarity grades.

Use each released candidate's original publication history.
The full-query cosine ranks owned sources. Preserve the highest
source and nominate a second from ranks two through six. The
plain control takes the original second. The lead and method arms
compute another cosine from shorter views on both sides. Reuse
the [CoF sentence-selection rule](../cof-method-query-view-v1),
which takes up to three leading sentences or three sentences
with the most distinct method cues. Keep the title and original
sentence order. The sentence budget is matched. Token counts can
differ. Unlike CSFCube, no query facet or rhetorical labels are
supplied. The cue marks a hypothesized method view.

Encode the full and compact query in a single forward for each
compact route. A frozen 0.6B Qwen reranker scores the same number
of source pairs across arms, using complete query and source
abstracts. Use the maximum score per profile. Each paired
comparison uses one verifier batch. The same official evaluator
scores the dense top-three mean and the published RATE predictions
as controls. Comparisons that tie count as misses. No ranking
threshold is tuned on LR-Bench. Bootstrap paired accuracy
differences by the paper or reviewer anchor for the respective
direction.

Serial P95 covers online embedding, source scoring, and the
batched reranker. It excludes offline indexing, network and
queueing. Candidate histories are supplied by the benchmark.
A selected publication provides profile evidence for the
model, while source-level ownership and constraint satisfaction
have no independent judgments here.

## Results

Slurm job 43338 ran on one RTX 5090 on ws5. All three verifier
routes issue exactly 839 source pairs in the paper-centric split
and 4,656 in the reviewer-centric split.

| Policy | Paper ranks reviewers, correct / 212 | P95 | Reviewer ranks papers, correct / 1,184 | P95 |
| --- | ---: | ---: | ---: | ---: |
| Dense two, same verifier | 136 | 44.48 ms | 889 | 50.37 ms |
| Lead challenger | 137 | 48.00 ms | 888 | 57.29 ms |
| Method challenger | 136 | 48.14 ms | 892 | 57.45 ms |
| Dense top-three mean | 144 | — | 900 | — |
| RATE released predictions | 163 | — | 891 | — |

Method cue selection changes the evidence shortlist on 95 of 212
paper-centric comparisons, but makes only one incorrect comparison
correct and one correct comparison incorrect. Its descriptive
paired interval against dense two is −1.43 to +1.32 percentage
points. The interval in the reverse direction is −0.36 to +0.91
points. The simpler dense mean has higher point-estimate accuracy
than the verifier routes in both directions. Both compact routes
add serial latency. RATE's released predictions come with no
matched serving measurement in this experiment.

The strict-risk operating point on CSFCube does not establish a
reviewer-ranking benefit here. The task, selection signal, and
quality endpoint differ, so this comparison cannot isolate which
factor limits the transfer. Keep the inexpensive full-profile
controls for reviewer search and require a fresh matched-budget
gain before adopting the compact challenger. The aggregate
`changed_winner` field counts changes in pairwise correctness,
including changes involving ties. [Aggregate results](results.json)

## Reproduce

Prepare the pinned release and evaluator from the
[existing LR-Bench comparison](../../docs/merit-suitability-card.md#lr-bench-pairwise-retrieval)
and the official frozen Qwen3 embedding and reranker checkpoints.
Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.lrbench_method_challenger \
  LRBENCH_RELEASE QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
