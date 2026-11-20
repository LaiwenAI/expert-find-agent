# Keep identical author initials bound to the specified study

The [synthetic prose intervention](../owner-scope-narrative-v1)
found that accurate numeric decisions could conceal wrong evidence.
This follow-up isolates a related error in original PLOS ONE
contribution statements: the same initials can refer to a different
credited contributor on a different study.

Restore the pinned XML receipts from the
[legacy notes](../plos-legacy-contribution-text-v1),
[independent cascade](../plos-independent-cascade-v1), and
[body-transfer](../plos-body-transfer-v1) studies. The builder checks
XML hashes and requires explicitly open licensing and recognizable
legacy declarations. It pairs notes with the same unambiguous
initials but disjoint normalized full byline names. One note credits
that contributor for both requested tasks. The other note credits
the contributor for exactly one. Greedy hash-order selection retains
unique DOI and disjoint normalized byline-name groups across pairs.

Assign two report aliases to the original, whitespace-normalized
notes. Both benchmark views have the same notes, initials, and
requested task descriptions. Only the target report alias changes.
A positive means both contributions are explicitly credited in that
particular note. The two reports are assembled for this counterfactual
task; the underlying contribution statements are authentic.

Compare the frozen 4B decision with both notes, with only the named
note selected by exact report alias, and with only its literal
initials-bearing clauses. The pooled symbolic policy intentionally
disregards the report identifier and answers yes if either note
credits both duties. Report paired correctness, supported
coverage, and unsupported attribution to the contributor in the
negative report. The scoped route includes clause selection in
local decision time. The report lookup uses the given alias,
without open search or automatic report-name resolution.

## Results

Slurm tasks in job `44124` reproduced every decision count in
two ws5 runs. The construction retains 59 matched pairs from
118 distinct source documents and 1,054 eligible pair--role--initials
combinations. Each pair has one supported and one unsupported view.

| Route | Both decisions correct / 59 | Accepted supported / 59 | Accepted unsupported / 59 | Local P95 |
| --- | ---: | ---: | ---: | ---: |
| 4B, both reports | 11 | 58 | 48 | 20.76 ms |
| 4B, exact report only | 5 | 59 | 54 | 15.62 ms |
| 4B, exact report and initials clauses | 3 | 59 | 56 | 14.62 ms |
| Pooled symbolic union | 0 | 59 | 59 | — |

An exact report lookup and literal initials narrowing reduce
latency, while the frozen decision prompt accepts more unsupported
cases in this format. Correct source selection is necessary for
attribution. The reader still needs to establish that both roles
were credited within that source. The paired challenge deliberately
selects initials collisions. These rates do not estimate their
prevalence in open expert search.

[Aggregate results](results.json)

## Reproduce

Keep the reconstructed evidence fixture local. The script releases
aggregate counts and digests, without author names or excerpt text.
Use the pinned Qwen3 4B model on a Slurm GPU allocation.

```sh
PYTHONPATH=src:. python -m experiments.credit.study_bound_initials build NEW_PRIVATE_FIXTURE \
  LEGACY_SNAPSHOT INDEPENDENT_SNAPSHOT BODY_TRANSFER_SNAPSHOT
PYTHONPATH=src:. python -m experiments.credit.study_bound_initials run PRIVATE_FIXTURE NEW_RESULTS
```
