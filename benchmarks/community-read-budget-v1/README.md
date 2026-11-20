# Transfer first-request verification budgets to Unix & Linux

Capture a new Stack Exchange community with a larger answer catalog.
Use 2024–2025 Unix & Linux questions closed as duplicates, linked to the
accepted answers on their earlier canonical questions. Apply the
[existing temporal and direct-link screens](../crossvalidated-witness-retrieval-v1).
Use up to 16 closed-question pages and 16 pages of high-vote historical
questions with accepted answers as background. Evidence creation and
last edit must predate 2023. This index is a sampled historical catalog;
it includes eligible known answers. [Stack Exchange capture API](https://api.stackexchange.com/docs/advanced-search)
and its [original-question relation](https://api.stackexchange.com/docs/types/original-question).

The manifest pins captured bytes and lists public post IDs without
storing post text or account profiles. Original posts, attribution,
revisions and applicable licenses are available at
`https://unix.stackexchange.com/q/<question-id>` and
`https://unix.stackexchange.com/a/<answer-id>`. On top of the
capture filters, remove queries with shared known-answer text or
near-identical title tokens across the three previous panels.

## Compare read budgets

Retrieve one historical answer from each of the first ten distinct
accounts, ranked by the best answer's frozen dense similarity.
Verify the first 3, 4, 5 or 10 answers in one GPU batch with the same
open-weight Qwen3-0.6B verifier and one common issuance threshold.
Return at most five verified accounts. An adaptive arm uses five
reads on queries whose rank-4 or rank-5 predicted verifier margin is
above the development median, otherwise three. Both the student
and that median are fit on Cross Validated 2024 before target use.
Training sees frozen teacher logits but no accepted-answer labels.
The [prior selector comparison](../community-distilled-reads-v1)
studied which answers to verify at fixed budget. [Ranked List Truncation](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/meng-2024-ranked.pdf)
provides prior comparisons of variable candidate counts for re-ranking.

Report actual reads and warm one-client P95 including encoding.
Partition requests by the rank of the first known witness before
verification. Report the paired known-supported P@5 delta relative
to both fixed four and fixed five, and match positive issuance
per query in a diagnostic against fixed four. A random replay
allocates the same number of three- and five-read requests as the
adaptive arm, reusing live fixed-arm outputs. That replay estimates
the value of budget assignment, separate from the budget itself.
The site labels reflect the question authors' acceptance. Other
outputs remain unjudged.

## Results

Both tasks of array `45682` completed on the ws3 RTX 5090 and
reproduced all non-timing results. The
[public manifest](unix-manifest.json) identifies 195 requests
and 1,412 historical answers from 614 accounts. Cross-site checks
removed no additional requests.

| Arm | Answers read | Known selected | Known issued | All issued | Warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 3 | 585 | 124 | 91 | 237 | 33.85–33.94 ms |
| Fixed 4 | 780 | 131 | 95 | 274 | 38.77–38.88 ms |
| Fixed 5 | 975 | 139 | 99 | 297 | 47.14–47.27 ms |
| Adaptive 3 or 5 | 935 | 136 | 99 | 296 | 47.23–47.50 ms |
| Fixed 10 | 1,950 | 157 | 104 | 349 | 83.16–83.23 ms |

Fixed three drops eight known issued answers relative to five
(−0.82 percentage points in known-supported P@5, paired 95%
interval −1.43 to −0.31). Fixed four raises known recovery
relative to three at roughly 5 ms more P95. The fixed and
adaptive five-read comparisons with fixed four issue four more
known answers, but their advantage disappears with per-query
matched issuance. Unlisted candidates still need independent
judgments for a precision comparison.

The development median sends 175 of 195 queries to five reads
on this community. Adaptive scheduling saves 40 answer checks
against fixed five and does not reduce P95. At the same realized
budget, random allocation averages 98.18 known answers, with a
central 95% replay range of 96–99. The adaptive result of 99 falls
within that range.

The first known answer is in the first three for 118 requests,
at ranks 4–5 for 12, at ranks 6–10 for 15, and absent from the first
ten distinct accounts' selected answers for 50. With ten reads,
95 of the 145 requests with a visible known answer issue at least
one such answer. These two gaps motivate separate comparisons of
alternate sources from the same account and verifier behavior.

[Aggregates for both runs](results.json).

## Reproduce

```sh
PYTHONPATH=src:. python -m experiments.community.crossvalidated_witness UNIX \
  --site unix --year 2024 --end-year 2025 --duplicate-pages 16 --background-pages 16

PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.community.first_query_read_budget \
  CV_2024 CV_2025 MATHOVERFLOW UNIX QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS \
  --order-offset 0
```

Use the frozen receipts for exact replication. A fresh API capture
may observe edited or removed posts and different background votes.
Rotate inference order with `--order-offset 1` for the second run.
