# Reallocate the last TREC evidence check at exactly matched cost

The [joint planning replay](../trec-joint-owner-source-plan-v1)
found no supported-ranking gain from widening the nomination pool.
Those runs also change which sources are read. This comparison
isolates the last candidate decision while keeping the selected
source and exact amount of verification work fixed.

Reuse the 49 [TREC 2006 Enterprise](https://trec.nist.gov/data/t15_enterprise.html)
expert and support judgments. Rebuild historical graphs with the
held topic removed and fit the original fixed-slate and wider
pair models as in the prior study. Replay the strongest earlier
fixed dispatcher through the next-to-last check. Keep the same
last document for all arms. Compare its original action with
the best unchecked owner according to the wide score, first among
the existing 20 nominees and then among the 60 historical candidates
with a link to that same document. If the prefix already confirms
five positions, leave the last action unchanged. Prior held labels
are only revealed as replay observations. The final candidate
choice does not see its label. Each variant seals the plan after
the final check and keeps the fixed priority of earlier nominees.

| Route | Reads | Checks | Ungraded checks | Supported / issued | Changed last actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed dispatcher | 80 | 266 | 220 | 10 / 11 | 0 |
| Reallocate within the fixed slate | 80 | 266 | 220 | 10 / 11 | 6 |
| Reallocate within the wide menu | 80 | 266 | 220 | 10 / 11 | 17 |

Both ws6 Slurm runs (array 47377) reproduce the archived fixed
dispatcher and the entire new topic signature. An outside owner
is eligible at the last source in 30 topics and actually checked
in 14. Neither reallocation changes supported P@5 on any topic.
The comparison is conditional on the reference trace, historical
candidate menus, and pooled labels. Ungraded checks remain
unconfirmed. It provides no live timing or claim about unseen
support.

[Aggregate results](results.json)

## Reproduce

Use the same pinned topics and expert and source qrels as the
[earlier replay](../trec-joint-owner-source-plan-v1#reproduce).
Run on an allocated Slurm compute node.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.trec_enterprise.final_witness_reallocation \
  TREC_DATA benchmarks/trec-joint-owner-source-plan-v1/results.json NEW_RESULTS
```

Repeat with `--reverse`.
