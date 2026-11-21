# Score source relevance before allocating three evidence reads

The [all-edge experiment](../open-topic-three-source-plan-v1)
puts every compatible publication pair in a candidate read set.
Its rank-weighted score still misses many tag-supported proofs.
This study holds that candidate frontier and the read budget fixed
and inserts an open Qwen3-Reranker-0.6B relevance score before
choosing which declaration files to read.

## Independent facet judgments

Use the official [CSFCube](https://github.com/iesl/CSFCube/tree/7ffe012c1d92c470f185fc086086caf5b8e05ef3)
release at the linked commit. For each of the 50 expert-judged
query-facet cases, take the top six documents by Qwen embedding
similarity from its judged candidate pool. Compare the dense order
with the frozen reader using a matching instruction for background,
method or result. Count strongly relevant top results (grade 2–3)
and concordance for pairs that cross the grade-2 boundary,
with half credit for tied scores. Group
the paired top-result interval by query paper. These judgments
measure article relevance within expert-annotated candidate pools,
without testing author identity or applicant qualifications.

## Open-topic allocation

Freeze the original PLOS and eLife requests, full contributor-role
index, hybrid candidate depth of 128, and exactly three declaration
reads. Compare four routes.

| Route | Planning before reads |
| --- | --- |
| Role joint | Existing logical first pair and third-source rule |
| All-edge rank | Existing graph frontier with rank-weighted set scores |
| Reader six | Use the all-edge frontier. Fit at most six distinct condition–document inputs by preserving the baseline edges first, then adding remaining edges in the original rank order |
| Reader full | Use the all-edge frontier and score every compatible condition–document input |

The reader sees the original title and abstract, not a declaration.
It scores topical relevance, while compatibility comes from the
same contributor-role index in all arms. A plan's score is the
best complete pair's weaker relevance margin. Preserve the joint
plan on ties. Skip the reader when there is no decision to make.
The margin ranks plans. It does not suppress any issued response
or act as a calibrated confidence value. All routes verify the
chosen declarations and use the same final proof ranking. Compare
reachable proof and issued proof separately. Publisher topic tags
are available only after the reads and do not certify real-world
expert suitability.

Report natural, common, and equal-quota issuance. The paired
interval groups by topic pair, and the missing-label count bounds
are a separate sensitivity measure. Compare warm local P95 and
reader work. Timing includes retrieval, allocation, title and
abstract scoring, declaration reads and response serialization.
Index construction, transport and serving load are excluded.
Both open-topic panels reuse previously studied requests.

## Results

Both RTX 4090 tasks in Slurm array 47503 on ws6 reproduce
every discrete result under reversed traversal. On CSFCube the reader ranks a strongly relevant paper first on
29/50 query-facet cases, up from 27/50 in the fixed dense pool.
The paired change is +4 percentage points, with a query-paper
interval of −7.84 to +16.98. Across 297 pairs that cross the
strong-relevance threshold, the grade ordering score rises from
186 to 197.5. These measures give a separate check on the
scorer and leave its top-result advantage uncertain.

| Planning route | PLOS tag-supported / issued | PLOS reachable after reads / 160 | eLife tag-supported / issued | eLife reachable after reads / 51 |
| --- | ---: | ---: | ---: | ---: |
| Role joint | 85 / 140 | 89 | 46 / 49 | 46 |
| All-edge rank | 82 / 136 | 86 | 36 / 40 | 36 |
| Reader six | 87 / 132 | 91 | 45 / 49 | 45 |
| Reader full | 87 / 132 | 91 | 45 / 49 | 45 |

Equalizing PLOS at 132 issued responses gives 80 supported
proofs for role joint and 87 for either reader. For the six-input
route, the paired interval over all 160 requests is −2.58 to
+11.77 percentage points. The separate missing-judgment bound
is −22 to +29 proofs. On eLife equal issuance gives 46 versus
45 tagged proofs in 49 responses. The reader can undo some of
the earlier all-edge ranking losses. A gain over the simpler
joint planner remains unresolved across panels.

| Panel | Role joint P95 | Reader six P95 | Reader full P95 | Condition–document inputs, six / full |
| --- | ---: | ---: | ---: | ---: |
| PLOS | 17.7–17.9 ms | 55.1–55.9 ms | 95.0–95.5 ms | 566 / 759 |
| eLife | 18.1–18.5 ms | 54.2–54.7 ms | 78.1–78.4 ms | 209 / 261 |

Bounding the model to six inputs retains all observed aggregate
coverage results of full scoring and lowers its latency. It still
roughly triples local P95 against the current joint route. Keep
the existing reader-free plan as the default. The promising use
of the scorer is limited to decisions where a relevance judgment
can alter a competitive source allocation. The experiment does
not establish that such a gate can identify those cases.
[Aggregate results](results.json).

## Reproduce

Rebuild the authorized PLOS and eLife snapshots, embedding and
reranker, and use the four CSFCube files consumed by its loader
at the release commit above. Run in Slurm CUDA and repeat with
`--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.open_topic_plan_reader \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING \
  benchmarks/mixed-publisher-joint-v1/results.json \
  benchmarks/live-joint-source-request-v1/results.json \
  benchmarks/open-topic-three-source-plan-v1/results.json QWEN_RERANKER CSFCUBE_DIR NEW_RESULTS
```
