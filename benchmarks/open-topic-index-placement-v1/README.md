# Choose which contributor-role records to extract before first-query search

The [observed-role repair benchmark](../open-topic-observed-repair-v1)
found that changing the third read after opening two declarations did
not recover additional proofs. Here the planner and three reads stay
fixed. Only the set of declarations materialized in the role index
changes. Use the same 452-article mixed catalog, 160 PLOS requests,
51 eLife requests, top-128 hybrid pools per condition and original
declaration verifier.

Use hash-derived nominal budgets at 2%, 5%, 10%, 25%, and 50%.
The record experiment gives each method the exact same number of
slots per publisher as the size-matched hash mask. It models a
limit on submissions for role extraction. The separate byte
experiment gives each method the same upper bound on serialized
role payload per publisher, established by the same hash sample.
That experiment packs records that have already been extracted.
It allows unused bytes when no remaining record fits. An omitted
source stays searchable with opaque roles.

Placement uses article identifiers, validated bylines and titles.
The byte experiment also uses encoded record sizes. It does not
use the roles in those records or publisher topic labels to decide
which records to keep.

| Policy | Placement rule |
| --- | --- |
| Hash | Archived deterministic mask |
| Authors | Largest number of identifiable contributors per unit of cost |
| Owner degree | Highest sum of log contributor document frequencies per unit of cost |
| Pair cover | Repeatedly select a same-owner source pair that completes the most new two-source contributor histories per unit of additional cost |
| Diverse pair | Same marginal coverage objective, breaking ties first by title token-set distance within the pair |

For owner $e$, let $D_e$ be the sources linked to that owner by
the byline. Pair cover greedily raises the following proxy subject
to the per-publisher capacity $B$:

$$
F(S)=\sum_e \mathbf{1}(|D_e\cap S|\ge 2),\qquad \sum_{d\in S} c_d\le B.
$$

A step selects the feasible pair with largest increase in $F$ per
additional cost. The first source alone adds no two-source history;
a second linked source can make both useful. A pair may introduce
one source if its partner is already selected.
For pair cover, a tie favors more newly represented contributors per
unit of cost, then stable IDs. If no pair completes a new history,
fill the remaining slots from owner degree. The byte cap charges the
role payload. Parser time, XML volume, shared bylines, content indexes
and runtime metadata are outside the measured budget.

Use the natural, common-request, and matched issuance quota
comparisons. Record proof reachability after reading, final selected
proofs, and publisher-tag missing-label bounds. Measure warm request
P95 and offline placement time.
At the 25% budgets, replay the complete role index as an additional
reference. Compare exact cited responses as well as counts. Rotate
arm order and reverse query traversal in the replay.
The data and tags have been used in the prior experiments.
No independent expert-suitability judgments are part of this replay.

## Exact linked-source index

For the two-distinct-source task, define
$D^+=\{d:\exists e\in B(d),\ |D_e|\ge 2\}$ using complete bylines
$B(d)$. Materialize personal-role records only for $D^+$.
Also test excluding all other sources from role lookup while
preserving their original topical ranks and fallback eligibility.

Any feasible pair $(a,b)$ uses the same indexed person $e$ on
distinct sources, so $a,b\in D^+$. The first-pair key is therefore
unchanged by deleting roles outside $D^+$. A third source outside
$D^+$ cannot contribute a compatible pair with either anchor and
retains its zero gain. The unchanged rank lists preserve fallback
choices. Given identical selected reads, the original verifier and
final tie breakers produce the same citation response. This argument
requires a fixed inventory, stable contributor identifiers and
complete bylines in the original planner. It covers this
two-source same-contributor route. Single-source evidence uses a
separate path. A corpus update that gives an identifier a second
source must make the earlier and newer roles available together.

## Results

Slurm job 47553 ran the placement sweep in both orders on ws6.
All 20 decision signatures match across replays. At the 25% byte
caps, owner degree and pair cover reproduce all full-index
recommendations and citations on both publishers. The 211 total
requests keep the same read count. The matched hash mask has lower
labeled coverage.

| Publisher | Hash payload cap | Hash known / issued | Owner degree known / issued | Full index known / issued |
| --- | ---: | ---: | ---: | ---: |
| PLOS | 7,012 B | 32 / 68 | 85 / 140 | 85 / 140 |
| eLife | 52,182 B | 34 / 42 | 46 / 49 | 46 / 49 |

In both publishers, only 15 contributors occur on at least two
articles. The 25% record and byte budgets can accommodate their
support. Tighter budgets expose the difference between graph
coverage and actual expert selection. At the 5% eLife byte cap, pair
cover raises the number of indexed two-source contributors from
8 to 11 versus owner degree, while known outputs fall from 38 to 31.
At the common quota of 35, they are 32 and 31. At the 5% PLOS record
cap, pair cover raises known outputs from 57 to 68. At the common
quota of 98 they are 57 and 67, with possible change from −7 to +17
under missing judgments. There is no consistent advantage over
owner degree at the tighter budgets.

Slurm job 47554 ran the exact index in both orders on an RTX 4090.
The original index and both linked variants preserve every ordered
plan, nominated expert, citation and abstention on all 211 requests.

| Publisher | Full / linked records | Full / linked role payload | Full P95 | Compact P95 |
| --- | ---: | ---: | ---: | ---: |
| PLOS | 130 / 28 | 29,389 / 6,424 B | 17.66–31.65 ms | 17.34–30.81 ms |
| eLife | 322 / 27 | 172,129 / 16,423 B | 18.79–18.98 ms | 18.32–18.40 ms |

The exact linked index retains 55 of 452 role records and cuts
serialized role payload by 88.7%. The byline index and source data
remain shared. Warm local P95 includes live encoding, retrieval and
reads. The replay excludes service transport and does not measure
how much parsing time could be avoided at build time. The catalogs
contain few repeated contributors, and denser graphs may allow a
smaller footprint reduction.

[Aggregate results and replication hashes](results.json)

## Reproduce

Use the XML and checkpoints from
[the observed-role benchmark](../open-topic-observed-repair-v1).
Repeat from the same snapshots with `--reverse` on a Slurm GPU.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.open_topic_index_placement \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING \
  benchmarks/mixed-publisher-joint-v1/results.json \
  benchmarks/live-joint-source-request-v1/results.json \
  benchmarks/open-topic-observed-repair-v1/results.json NEW_RESULTS.json
```


Replay the exact index variants separately.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   python -m experiments.credit.open_topic_link_certificate   PLOS_XML benchmarks/plos-credit-cross-source-v1   ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING   benchmarks/mixed-publisher-joint-v1/results.json   benchmarks/live-joint-source-request-v1/results.json NEW_CERTIFICATE.json
```
