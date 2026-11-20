# Route a bounded follow-up budget over deeper expert histories

Extend the [packed rescue comparison](../cof-packed-evidence-rescue-v1)
from four to twelve sources. Use the same frozen KDD and title-filtered
SciRepEval query panels and the same full-catalog mean/max reviewer
ranking. Keep the first five nominees and their order fixed. The
first wave reveals the union of the top three sources of those
nominees. Each additional source can support any shortlisted owner
that has that source in its depth-12 inventory. Two positive model
verdicts are required for attaching citations. Any citations already
established by the first wave keep their original IDs and order.

Run the frozen Qwen3-Reranker-0.6B once per unique query-source pair
in the depth-12 pool. Every score uses the same four-row, 1,024-token
shape, padding the final batch by repeating its first row. Replay the
recorded verdict only when a policy selects a source. Pinned model
verdicts isolate selection decisions from numerical batch effects.

Fit a ridge-regularized logistic model on the other collection's
deeper sources. Drop training queries whose normalized titles are
present in the evaluation collection. The input features are inverse
dense source rank, title overlap, query-word coverage, IDF-weighted
overlap, length, and shared ownership. The outcome is the verifier's
sign. Reviewer labels are reserved for evaluation.

For a candidate with one accepted source, an unread source receives its
estimated chance of completing the pair. For a candidate with zero
accepted sources, multiply that estimate by the strongest available
second-source estimate. Sum the scores when a source is shared across
candidates. The probabilities are a scheduling heuristic. The following
ablations use identical read budgets and share all work other than the
order of additional reads.

| Policy | Pending-source scores | Recompute after each observed verdict |
| --- | --- | --- |
| Dense fixed | Reciprocal dense rank | No |
| Dense state | Reciprocal dense rank | Yes |
| Lexical state | IDF-weighted overlap | Yes |
| Learned fixed | Other-collection logistic model | No |
| Learned state | Other-collection logistic model | Yes |

At equal reads, every policy uses exactly the available portion of a
1-, 3-, 5-, or 10-source follow-up cap. A separate diagnostic allows at
most five additional sources under 1,536 or 2,560 actual prompt-token
slots, and reports realized work. Report known positives, judged
negatives and unjudged nominees among cited slots. Equal-issuance
comparisons retain the same number of cited nominees per query in
their fixed rank order. The partial-label intervals assign unjudged
changed positions the best and worst possible labels. CoF grades
reviewer relevance. Each attached citation still reflects a model
verdict on its title and abstract, with personal contribution and
citation relevance unjudged.

## Results

GPU array 45626 on a ws3 RTX 5090 scored 6,506 KDD pairs across
122 requests and 3,424 SciRepEval pairs across 59. Only four prompts
reached the cap. The two tasks in CPU array 45629 reproduce every
reported decision and work count.

The depth-12 reference increases known-positive cited nominees
from 12 to 13 in KDD and from 22 to 30 in SciRepEval. The following
figures use exactly five additional reads per query.

| Panel | Policy | Cited slots | Known positives | Judged negatives | Follow-up tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| KDD | Dense state | 291 | 12 | 44 | 320,095 |
| KDD | Learned fixed | 297 | 12 | 45 | 309,377 |
| KDD | Learned state | 302 | 12 | 45 | 308,961 |
| SciRepEval | Dense state | 105 | 26 | 23 | 127,147 |
| SciRepEval | Learned fixed | 110 | 29 | 24 | 121,926 |
| SciRepEval | Learned state | 111 | 29 | 24 | 122,873 |

On SciRepEval, the three extra known-positive citations from the
learned policy come with additional issuance. At matched issuance
the known-positive change is zero, with one additional judged
negative for Learned state. The same number of known positives is
attained by Learned fixed and Learned state in both panels. At the
2,560-token cap, SciRepEval has 29 known positives with either
learned policy and 26 with Dense state. Realized reads are 294
versus 293. KDD remains at 12 for all three policies.

The model can help prioritize additional text for citation coverage.
These data do not justify per-read policy updates for precision or
default deep verification. With five reads, the selection-only P95
in the first replay is about 0.050 ms for Learned fixed and 0.16 ms
for Learned state. The prompt token counts are actual model input
lengths before padding. Selection timing starts from precomputed
lexical features, lengths and frozen verdicts. Service latency,
queueing, verification throughput and independently judged evidence
remain separate measurements.

## Reproduce

Use the frozen datasets and checkpoints from the packed rescue
comparison on Slurm CUDA. Capture one private intermediate per panel.
The CPU replay can reuse those captures. Only aggregate results are
released here.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_deep_source_capture \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_CAPTURE

PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_deep_source_routing \
  KDD_CAPTURE SCIREPEVAL_CAPTURE NEW_RESULTS
```

Capture `SciRepEval` with the same GPU invocation and a fresh path.
