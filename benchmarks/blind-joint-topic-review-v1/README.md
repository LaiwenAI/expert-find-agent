# Independently review the matched-issuance topic-check comparison

Replay the frozen
[selective topic-check experiment](../selective-joint-topic-check-v1)
on the original GPU inputs. Refuse to prepare a packet if its
checkpoint receipts, calibration cutoffs, per-arm issuance or
matched known counts differ. Take the same 32 proposed answers
per method from PLOS and 12 from eLife, matching the full
reader's issuance without using their topic labels to select
these outputs. Include every distinct request–person–citation
bundle in the union of all methods. Equal-sized lists can
come from different queries.

A card shows both requested topic–role conditions, the two
source titles and abstracts, credited role text and publisher
byline evidence. The same card can represent multiple systems.
The reviewer copy omits system identity, scores, thresholds and
publisher topic tags. The answer key maps systems and requests
to the anonymous cards and must stay separate from the reviewers.
The original publication records are linked by DOI. The
materialized packets and any reviewer data stay out of Git.

Record topic and role support for each cited paper, plus same
contributor identity. Independently label each field supported,
contradicted or unclear. The combined scorer treats a bundle
as supported only with unanimous support on all five fields.
It treats a bundle as contradicted when every reviewer identifies
the same contradicted field. Missing labels, inconclusive evidence
and other disagreement remain unresolved. Report the support
interval for each method and the exact additional support range
versus the full reader. Shared unresolved cases cancel in the
paired difference. These are deterministic missing-judgment
bounds, not sampling intervals.

## Reproduce

On a Slurm GPU with the same two publisher snapshots and
frozen checkpoints:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.blind_joint_topic_review build \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/selective-joint-topic-check-v1/results.json PACKET_JSON PRIVATE_KEY_JSON
```

Render the reviewer-only file and combine completed exports
locally without loading a model:

```sh
python -m experiments.credit.blind_joint_topic_review render PACKET_JSON REVIEW_HTML
python -m experiments.credit.blind_joint_topic_review score \
  PRIVATE_KEY_JSON NEW_AGGREGATE_JSON REVIEWER_ONE_JSON REVIEWER_TWO_JSON
```

Send only the rendered HTML to each reviewer. It runs offline
and exports locally. No human judgments are included in this
benchmark at this stage.
