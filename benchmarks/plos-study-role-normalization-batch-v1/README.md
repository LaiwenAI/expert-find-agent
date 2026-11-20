# Batch the reusable normalization of two requested roles

Replay the same 118 authored two-duty requests from
[the normalization study](../plos-study-role-normalization-v1).
Each request has two duties and two menu orderings for each duty.
Use the same 4B checkpoint and role agreement policy. Compare:

- Four serial calls using the original reader and its full logits.
- Four serial calls computing only the last-token vocabulary logits.
- Two batches of two, grouping the two menus for each duty.
- One batch of four across both duties and both menus.

For the last-token paths, left-pad with an explicit attention mask
and compute positions within each unpadded prompt. Retain the same
menu token map, tokenizer, prompt text, and inference precision.
Monitor every menu choice, final agreed role pair and report outcome.
The GPU timings cover input construction, tokenization, GPU execution
and output mapping for one query. All four prompt evaluations occur
for each request without a cross-query result cache. Warm each path
and rotate its execution order. The reported speed is for query
normalization alone; retrieval, serving and queueing are outside it.

## Results

Both Slurm tasks in array 44158 completed on an RTX 5090. Fixture
and checkpoint digests match the preceding study. All three
optimized paths match all 472 reference menu choices and all
118 query mappings on both runs. Across the downstream report
pairs, all four paths keep 54/59 supported views. They accept no
in-schema negatives and no outside-schema views on this panel.

| Query execution | Local normalization P95, ms |
| --- | ---: |
| Original four serial calls | 58.13–58.43 |
| Last-token four serial calls | 56.86–57.14 |
| Two last-token batches | 44.93–45.33 |
| One last-token batch | 38.69–39.01 |

One batch lowers the measured P95 by about 33% in each replay.
Last-token-only logits account for a small part of the reduction.
Matching decisions are empirical for this checkpoint, precision,
hardware and prompt panel. The unchanged error counts inherit the
previous benchmark's limits, including reused authored paraphrases.
Reserve this result for the query-normalization component.

[Both GPU runs](results.json)

## Reproduce

Reconstruct the pinned fixture from the
[normalization study](../plos-study-role-normalization-v1).
Use an allocated Slurm GPU and the pinned checkpoint.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.study_role_normalization_batch PINNED_FIXTURE NEW_RESULTS \
  --order-offset 0
```

Repeat with order offset 1 and a fresh result path.
