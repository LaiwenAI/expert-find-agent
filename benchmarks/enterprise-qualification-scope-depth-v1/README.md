# Cross prior qualification scope with expert and evidence depth

Use the 25 multi-condition enterprise development requests on which
the original system abstained. Cross the
[headroom experiment](../enterprise-joint-headroom-v1)'s first five
or first twenty people and first two or up to eight passages per
person with the qualification interpretations from the
[fixed-evidence replay](../enterprise-qualification-replay-v1).
Keep the source inventory, candidate ordering, role windows, topic
condition and Qwen3 0.6B verifier fixed. A candidate must have a positive
verifier margin for the topic and every qualification in the arm.

The evidence search uses the original requirement interpretation in
every cell. Alternative interpretations only change which evidence
checks are required. An alternate-only clause is scored against the
same inspected passages, in a separate forward batch. The original
baseline batch layout is unchanged. The alternative parser assessed
all original qualifications and two other sampled clauses per request.
It did not exhaustively annotate the requests.

The union requires every qualification accepted by either parser,
and the intersection requires only those accepted by both. The
topic-only arm puts an upper bound on model approvals within each
fixed evidence pool. These are model approvals, with neither human
determination of applicant eligibility nor service latency measured.
The development panel cannot estimate effects on an independent
enterprise sample.

## Results

Both Slurm tasks in array `44923` ran on `ws3` with the same
checkpoints and reproduced the original headroom counts. Reversing
query order preserved every aggregate. Cells count requests with at
least one model-approved person out of 25.

| People × passages | Inspected passages | Original | Union | Direct gate | Intersection | Topic only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 × 2 | 250 | 0 | 0 | 7 | 7 | 12 |
| 5 × up to 8 | 869 | 1 | 1 | 9 | 9 | 14 |
| 20 × 2 | 997 | 2 | 2 | 13 | 13 | 17 |
| 20 × up to 8 | 3,344 | 2 | 2 | 16 | 16 | 20 |

With the original qualifications, adding deeper reads after reaching
twenty people left request coverage at 2/25, although model-approved
people rose from four to seven. In the widest pool, 16/25 cleared the
direct gate and 20/25 had topic evidence for at least one person.
Thus fourteen more requests are sensitive to qualification scope in
that pool, and five still have no model-positive topic evidence even
under the topic-only arm. A missing topic pass can reflect retrieval,
passage selection or verification. Neither the approval gap nor the
topic-only ceiling determines which qualification interpretation is
correct.

The crossed design supports prioritizing independent source-text
review of scope before widening the default search. If adjudication
endorses the alternative for some requests, shallow expansion to
more people is a comparison to carry forward alongside deep reading
of the first five. The excerpt counts describe workload. Service
timing and human-grounded accuracy remain separate endpoints.

[Aggregate results](results.json).

## Reproduce

Run on a Slurm GPU with the pinned headroom inputs, a new public
aggregate and a new private trace output. Repeat with
`--order-offset 1` to reverse query order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_joint_headroom \
  QUERIES CORPUS ROLE_WINDOWS VECTORS VECTOR_META MODELS SCRATCH \
  FIRST_ROUND_TRACE FIRST_ROUND_AGGREGATE NEW_AGGREGATE NEW_PRIVATE_TRACE \
  --scope PRIVATE_SCOPE_TRACE SCOPE_AGGREGATE
```
