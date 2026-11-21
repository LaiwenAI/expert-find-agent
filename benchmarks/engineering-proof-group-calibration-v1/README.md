# Calibrate risk on full same-author proof groups

The [original-text transfer](../engineering-raw-conjunction-v1) used
a threshold calibrated on single-author, single-role judgments from
Frontiers for the linked Sensors proof task. This comparison keeps
the four score functions frozen and changes the thresholding unit
to a complete two-source conjunction.

## Data and split

Freeze 30 distinct article pairs from the
[eLife confirmation projection](../elife-source-guard-confirmation-v1)
for calibration. Freeze 30 more from the
[extended eLife projection](../elife-extended-role-holdout-v1) for a held
comparison. Enforce disjoint papers and recorded, resolved authors
within and between the pools. No selected eLife contributor identifier
matches an ORCID-derived identity in the linked Sensors test. The
assignment does not attempt to link authors without persistent IDs.

Each pair supplies up to three designated positive conjunctions.
A negative reverses the two source-specific roles while leaving
the two papers and candidate identities unchanged. Each swapped
role exists somewhere on its new paper; the same-author requirement
makes the swap unsupported. Select pair order and role contrasts
deterministically before model scoring. Labels describe publisher
declarations and the controlled edits. Restore original XML and
verify both the SHA-256 and Git object hash before reading the
author-specific footnotes. The eLife original footnotes contain
role lists. Add numeric aliases to represent their association to
JATS authors. Scope and ownership are supplied, and only resolved
contributors appear in the eLife inference text. Query phrases
in the eLife panels exclude exact normalized phrases used in the
Sensors linked test.

## Calibration

Score authors with the original frozen shared text, author-scoped
text, indexed-role and fused routes. Keep the same Frontiers fit and
matched-register calibration as the earlier benchmark, reproducing
its thresholds and linked Sensors answers before evaluating the
new eLife panels. A candidate's proof score is the smaller of its
two clause scores. For every calibration paper pair, take the
maximum score of any unsupported author across its positive and
swapped requests. For each route, choose a strict cutoff from
these maxima, permitting at most one of 30 calibration groups to
issue an unsupported proof. A wrong group has at least one wrong
issued author anywhere in its requests. Apply both sets of frozen
cutoffs to the held eLife groups and reused Sensors source pairs.

The 5% rule is an observed calibration criterion. It provides no
distribution-free bound on operational risk. Both eLife panels
contain curated contrasts from one publisher, and same roles and
wording banks recur across groups. Sensors supplies a cross-publisher
transfer check on a previously explored four-pair panel. The
collection fixes source choice, scope and author linkage and
measures verification with a warm component timer, separate from end-to-end service.

## Results

Both calibration and held eLife panels have 89 positive and 89 swap
requests across 30 distinct paper pairs each. The Sensors panel has
15 positives and 15 swaps across four paper pairs. Each cell gives
correct positive requests and groups with any wrong issued author.

| Route | Held eLife, atomic | Held eLife, proof-group | Sensors, atomic | Sensors, proof-group |
| --- | ---: | ---: | ---: | ---: |
| Shared text | 0 / 0 | 1 / 4 | 0 / 0 | 0 / 2 |
| Scoped text | 8 / 2 | 9 / 6 | 0 / 0 | 0 / 0 |
| Indexed roles | 81 / 1 | 83 / 1 | 7 / 1 | 8 / 2 |
| Fusion | 83 / 4 | 80 / 1 | 8 / 1 | 7 / 1 |

Proof-group calibration holds each route to one adverse eLife
calibration group. On the held eLife data, it reduces fusion from
four adverse groups to one, with three fewer correct positives.
Indexed roles find 83 positives at the same observed held risk.
The low false-return counts of the raw readers under atomic
thresholding come with low coverage. The proof-group thresholds
increase their adverse groups in transfer within eLife.

On Sensors, the eLife-calibrated index gives four unsupported
requests across two groups, compared with two requests in one group
under the atomic threshold. Fusion issues two unsupported requests
in one group under either threshold. This cross-publisher result
does not reproduce the held eLife operating point. A deployment
threshold would need calibration and auditing on the actual source
formats and task distribution.

### Per-request component replay

Rerun each held request independently. Encode its live phrases, score
the frozen role banks, apply the threshold and serialize the proof.
Fusion also invokes the reranker on the original author-scoped texts.
The text and membership views are resident and prepared before the
timer. This excludes source retrieval, author linking, context
assembly, HTTP transport and queueing.

| Panel | Route | Local P95 (ms) | Reranker checks per replay | Correct / wrong requests |
| --- | --- | ---: | ---: | ---: |
| Held eLife, 178 requests | Indexed roles | 10.7–10.8 | 0 | 83 / 1 |
| Held eLife, 178 requests | Fusion | 24.2–24.6 | 548 | 80 / 1 |
| Sensors, 30 requests | Indexed roles | 10.5–10.9 | 0 | 8 / 4 |
| Sensors, 30 requests | Fusion | 20.4–21.2 | 60 | 7 / 2 |

The two RTX 4090 runs on ws1 (Slurm array 46986) agree on results,
cutoffs and model work. Both live arms preserve all batch replay
decisions. The final results reproduce the score-only pilot (46980).
The index has a lower observed inference cost and more correct
outputs at the same observed risk on held eLife. On the four
reused Sensors groups, it has higher support and higher risk than
fusion with these fixed thresholds. [Aggregate results](results.json).

## Reproduce

Use the archived [frozen eLife selection](frozen-pairs.json). It
can be regenerated with `freeze_conjunction_calibration` and the two
parent eLife projections and manifests. Restore the verified notes:

```sh
PYTHONPATH=src:. python -m experiments.credit.restore_conjunction_notes \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json RAW_NOTES
```

Use the frozen engineering inputs from
[the preceding study](../engineering-raw-conjunction-v1) and run
under Slurm with a GPU and a fresh output path:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.conjunction_group_calibration \
  ENGINEERING_PANEL LINKED_XML_DIRECTORY CROSS_SOURCE_PANEL \
  benchmarks/engineering-cross-source-search-v1/source-manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
  RAW_NOTES benchmarks/engineering-raw-conjunction-v1/results.json NEW_RESULTS \
  --component-timing
```

Repeat with `--reverse` and a separate output path.
