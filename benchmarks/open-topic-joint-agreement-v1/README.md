# Test agreement between topical ranking signals in joint evidence allocation

The [same-role sequential control](../open-topic-role-matched-v1)
showed that joint nomination retrieves more publication-tag-supported
evidence at natural issuance. Its PLOS advantage remains uncertain
when the number of issued results is equal. This replay asks whether
requiring stronger agreement between lexical and semantic ranks can
improve which evidence the agent selects before reading.

Freeze the same 452 PLOS and eLife publications, 211 open-topic
requests and the original top-128 hybrid candidates per condition.
All routes read exactly three distinct declarations and use the
same original-statement verification, final citation ranking and
output format. The three joint routes see the same role postings.
They differ only in the source order used to nominate the initial
pair and the tie-break for the third source.

| Route | Three-source plan |
| --- | --- |
| Topic | Two topic-leading hybrid anchors and the highest-ranked unused item by the existing topical rule |
| Role sequential | Two independently selected role-compatible anchors, followed by one targeted read |
| Role joint | Join the two lanes by contributor identity and role, using the archived hybrid ranks |
| Lexical joint | Same join, with lexical ranks inside the frozen hybrid pool and the original rank as tie-break |
| Agreement joint | Same join, prioritizing the better worst rank across lexical and dense channels. Break ties by the sum of the two ranks and then by original hybrid order |

An absent lexical rank is placed after the complete inventory for
both alternative orders. Every arm uses the same live lexical and
embedding computation. Measuring channel agreement does not add
a model forward. The planner uses title/abstract rankings and
role postings. Publisher topic labels enter only after retrieval.

Report natural issuance, common issuing requests, and retrospective
equal quotas. For label-supported proofs give a clustered interval over
topic pairs and separate identification bounds for unjudged outcomes.
Count whether a label-supported proof exists anywhere in the three
reads, and whether the final response chooses it. The topic-based
route is a mechanism control. The role joint and role sequential routes
must reproduce their complete archived decisions before comparisons
are scored. This is an exploratory replay of previously viewed requests.

Timing measures warm local common retrieval plus per-arm plan,
read, verification and serialization. The comparison excludes
index construction, networking and independent expert-fit judgments.

## Results

Both RTX 4090 tasks in Slurm array 47476 on ws6 reproduce the
archived decisions and agree on every new output under reversed
traversal. The runs read 480 declarations per arm for PLOS and
153 per arm for eLife.

| Route | PLOS tag-supported / issued | PLOS reachable after reads / 160 | eLife tag-supported / issued | eLife reachable after reads / 51 |
| --- | ---: | ---: | ---: | ---: |
| Topic | 1 / 1 | 1 | 0 / 0 | 0 |
| Role sequential | 30 / 45 | 30 | 0 / 3 | 0 |
| Role joint | 85 / 140 | 89 | 46 / 49 | 46 |
| Lexical joint | 84 / 140 | 88 | 43 / 49 | 43 |
| Agreement joint | 84 / 140 | 88 | 47 / 49 | 47 |

The joint routes issue on the same requests in each panel. Against
the archived joint route, agreement changes eight PLOS experts. Three
changes gain a tagged proof and four lose one. The PLOS difference is
−0.63 percentage points over 160 requests (95% topic-pair interval
−4.73 to +2.74), with missing-judgment bounds of −5 to +4 proofs.
On eLife it changes one expert and gains one tagged proof; the
identification interval is zero to one. Warm local P95 ranges from
17.0 to 18.5 ms across routes and panels.

The comparison identifies evidence selection as the larger observed
bottleneck on PLOS. The [archived shared pool](../mixed-publisher-joint-v1)
contains some tag-supported proof on 115 requests. The joint planner's
three declarations preserve that possibility on 89, and the final
ranker returns a supported proof on 85. A perfect selector restricted
to the same three declarations could add at most four tag-supported
proofs on these requests. With the same candidate pool, different
read plans have headroom on another 26. On eLife the corresponding
counts are 49, 46 and 46.

Keep the original hybrid joint ordering for the first-query candidate.
The large differences against topic-only and sequential allocation are
coverage differences at the same reading budget. The matched-issuance
comparison and independent judgments remain the tests of recommendation
quality. The cheap channel-agreement order gives no consistent gain
across publishers. [Aggregate runs](results.json).

## Reproduce

Restore the frozen XML and inputs from
[the matched planner experiment](../open-topic-role-matched-v1#reproduce).
Run with an allocated Slurm GPU and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.open_topic_joint_agreement \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING \
  benchmarks/mixed-publisher-joint-v1/results.json \
  benchmarks/live-joint-source-request-v1/results.json \
  benchmarks/open-topic-role-matched-v1/results.json NEW_RESULTS
```
