# Allocate three evidence reads with weighted alternative proofs

The [ranking-signal control](../open-topic-joint-agreement-v1)
localized more PLOS losses to three-source allocation than to
final ranking within the read set. Here all routes reuse the
same top-128 hybrid candidates per condition, personal-role
index, three distinct declaration reads, and final original-source
verifier. The altered planner can consider two alternative pairs
at once rather than committing to the first ranked pair before
considering the third source.

A graph edge joins distinct documents if the indexed roles make
a same-person proof possible across the two conditions. Its weight
is the best compatible product of the two rank factors
`60 / (60 + rank)`. Each opaque role contributes a factor of one
half. Duplicate contributors and both orientations of the same
two sources do not create separate edges. These are rank weights,
with no probabilistic interpretation. Score a set of three sources
by its largest edge weight plus half its second-largest distinct
edge weight. Ties favor a larger best edge, better source ranks,
and stable source identifiers.

| Route | Candidate set of three-source plans |
| --- | --- |
| Role sequential | Existing two independent role-compatible anchors and a targeted third read |
| Role joint | Existing first pair and third source by compatible-pair counts |
| Anchored score | Keep the existing first pair; compare its neighboring third sources and the existing plan |
| Free score | Include the same anchored alternatives and extend the 16 largest-weight graph edges with their neighbors |
| All edges | Seed with every graph edge. Extend with neighbors, or add the most topical unused source if the edge has no neighboring source |

Always include the existing plan. Generate candidates from
adjacent edges, so planning never requires opening declarations.
Anchored and free scoring separate third-source priority from
joint replacement of the first two sources. The all-edge route
tests a broader frontier with the same score and read budget.
It gives every compatible pair at least one containing three-source
plan. Peak work can grow with author publication counts, since
building the graph joins an author's occurrences across two lanes.

Evaluate the frozen 160 PLOS and 51 eLife open-topic requests.
Test the complete index and a deterministic publisher-record mask
at roughly 50% declaration coverage. Masked articles remain eligible
as opaque until read. Require the historical role sequential and
role joint decisions to reproduce on the complete index. Report
natural, common, and equal-quota issuance. Publisher tags evaluate
topical support after reads; they are not inputs to any route.
Record possible count differences under unresolved judgments and
topic-pair bootstrap intervals separately. Using the complete index
and labels only for diagnosis, count requests with a supported proof
in the retrieval pool, in any enumerated set, and in the chosen set.
That separates failures of the bounded plan generator from failures
of the score within the enumerated plans. Timing covers warm
local retrieval, planning, and reads, without transport or queues.

## Results

The final RTX 4090 array 47493 on ws6 reproduces all decisions
under reversed traversal, as well as the first two planner
comparisons from array 47483. The half-index mask exposes
226 of 452 declaration records. Reads remain fixed at three
per request in every arm and condition.

| Panel and role coverage | Role joint | Anchored score | Free score | All edges |
| --- | ---: | ---: | ---: | ---: |
| PLOS complete | 85 / 140 | 85 / 140 | 84 / 136 | 82 / 136 |
| eLife complete | 46 / 49 | 46 / 49 | 40 / 44 | 36 / 40 |
| PLOS half | 56 / 114 | 56 / 114 | 58 / 111 | 59 / 106 |
| eLife half | 35 / 45 | 35 / 45 | 35 / 42 | 29 / 36 |

Cells show tag-supported proofs over issued responses at natural
issuance. Anchored scoring changes two PLOS half-index plans and
preserves every issued proof. At equal issuance, all edges ties
joint planning on PLOS complete (82 / 136 each) and loses one on
eLife complete (36 versus 37 at 40 issuances). On PLOS half it
gains four at the quota of 106, with the interval from −10 to +14
proofs under unresolved judgments. The latter is a sensitivity
bound, separate from the sampling interval.

| Panel | Proof exists in top-128 pool | In free-score candidate sets | In all-edge candidate sets | In chosen all-edge set | Returned with all-edge score |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLOS complete | 115 | 92 | 115 | 86 | 82 |
| eLife complete | 49 | 46 | 49 | 36 | 36 |
| PLOS half | 115 | 63 | 115 | 63 | 59 |
| eLife half | 49 | 41 | 49 | 29 | 29 |

Adding every compatible pair closes the enumerator's ceiling gap
on both corpora. The rank-weighted score then fails to choose a
supported plan on many of those requests. On complete indexes it
also reaches fewer supported proofs after the same reads than the
existing joint planner (86 versus 89 on PLOS and 36 versus 46 on
eLife). The next comparison should validate evidence scores on
independently judged pairs before expanding the planner further.

The all-edge route evaluates P95 of four to seven sets across
these panels on a small catalog. In paired runs, warm local P95
changes by +0.17 to +0.46 ms. Absolute timing varies across the
two runs. This provides neither a large-catalog work bound nor
independent evidence of a general recommendation quality gain.
Keep the existing joint plan as the first-query candidate.

[Aggregate results and replay hashes](results.json).

## Reproduce

Restore the authorized frozen public publisher XML and model using
[the same-role planner benchmark](../open-topic-role-matched-v1#reproduce).
Repeat with `--reverse` on an allocated Slurm GPU.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.open_topic_three_source_plan \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING \
  benchmarks/mixed-publisher-joint-v1/results.json \
  benchmarks/live-joint-source-request-v1/results.json \
  benchmarks/open-topic-role-matched-v1/results.json NEW_RESULTS
```
