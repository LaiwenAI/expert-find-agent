# Test author-bound role postings under a three-source budget

Reuse the frozen PLOS and eLife heading requests and live hybrid top-64
pools from the [live request replay](../live-joint-source-request-v1).
Each arm opens exactly three distinct publisher declarations. The
verified reply is selected with the same rank rule and original
declaration fields across arms. Topic labels only generate and score the
requests. A missing topic tag remains unresolved.

| Arm | Facts exposed before source selection |
| --- | --- |
| Sequential role-gap | Author identifiers and the first two opened declarations |
| Joint byline | Author identifiers only |
| Article roles | Per-article union of roles, without binding a role to its author |
| Personal roles | Per-author roles on a deterministic sample of sources |

For personal roles, a source outside the index sample and a contributor
with no role fields are both unknown. For a visible contributor with
parsed role fields, the role must occur in those fields to prequalify the
pair. The same checked identity must satisfy both source conditions.
Rank feasible pairs by number of unknown roles, worst topic rank, sum
of topic ranks, then a stable identifier tie break. A third source
maximizes new fully indexed pairs with the first two, then possible
pairs. Full declarations are still read to verify the output. Both arms
using roles fall back to the topical plan if no eligible pair exists.

The two-list selector only needs each owner's two best distinct sources
on each side. The first pair would use both first choices if they differ.
If they coincide, replacing either one with the next entry on that side
preserves the best possible value. The benchmark compares implementation
outputs with an exhaustive oracle in 1,000 seeded mixed-visibility
instances and checks 1,000 additional plans against the byline planner
when every posting is unknown.

## Results on RTX 5090

Both tasks of Slurm array `44574` on ws5 reproduced the same nontiming
results. Each pair reports known / issued. Actual indexed source
fractions differ from the nominal hash cutoffs.

| Before-read information | PLOS (160) | eLife (51) |
| --- | ---: | ---: |
| Sequential role-gap | 28 / 33 | 0 / 0 |
| Joint byline | 49 / 78 | 14 / 16 |
| Full article role union | 60 / 95 | 19 / 21 |
| Personal roles, 25% cutoff | 44 / 81 | 22 / 25 |
| Personal roles, 50% cutoff | 62 / 106 | 19 / 25 |
| Personal roles, all sources | 82 / 131 | 28 / 30 |

At the full personal-role setting, PLOS has 49 unresolved outputs
versus 29 for joint byline. eLife has two unresolved outputs in each arm.
Among requests both arms answer, full personal roles changes three
PLOS experts, increasing known matches from 49 to 51. It does not change
any eLife answer in the overlap. At the same panel-wide issuance as
joint byline, the PLOS known count rises from 49 to 57 out of 78 and the
eLife count from 14 to 16 out of 16. This diagnostic matches total count
using retrieval rank without labels. Its paired intervals per request
span zero in both panels. Against sequential role-gap on the 33 PLOS
requests both answer, full personal roles yields 26 known matches versus
28. These results support coverage gains with complete structured roles.
They do not establish stronger expert ranking.

The partial masks expose 30 and 60 of 130 PLOS source records and 97
and 166 of 322 eLife records. Gains are not monotone across cutoffs.
In eLife, the 50% cutoff loses three known matches versus joint byline
on the 16 shared requests. Complete indexing should be treated as a
publisher-field diagnostic until extraction, missingness and attribution
are evaluated in the target domain.

After retrieval, the warm source selection, three reads and
declaration matching have P95 of 0.216–0.219 ms for byline and
0.274–0.276 ms for full personal roles on PLOS. The ranges on eLife are
0.307–0.317 and 0.386–0.391 ms. Timing excludes query encoding, index
retrieval, response assembly, fetching publisher records and extracting
roles from text. Index construction measurements cover only the in-memory
projection from existing structured declarations.

## Reproduce

Restore the frozen publisher XML inventories and embedding checkpoint.
Run on a Slurm compute node, then repeat with `--order-offset 1` using a
different output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.live_joint_role_postings \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

[Aggregate results and matched-issuance diagnostics](results.json).
