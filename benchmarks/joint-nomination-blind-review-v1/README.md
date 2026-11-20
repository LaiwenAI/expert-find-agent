# Isolate first-round nomination at matched issuance

The [live request replay](../live-joint-source-request-v1) compares source
plans at three declaration reads. This follow-up isolates the part of the
coverage gain that might improve a recommendation on a request where
both policies return an expert. The sequential plan opens two topical
sources and selects a third by the observed contributor-role gap. The
joint plan selects sources using contributor identity before opening
them. Both use the same live query embeddings, top-64 hybrid pools,
source snapshots, and declaration checks. The diagnostic encodes
each query once and alternates plan execution order. It compares
outputs without timing the shared path.

The views answer separate questions:

- **Natural issuance** counts matches and abstentions on every request.
- **Common issuance** restricts the comparison to exactly the same
  requests where both plans recommend an expert. Identical
  person-and-citation outputs are shared judgments. This estimates
  differences within the selected overlap only.
- **Matched aggregate quota** takes as many recommendations as the
  lower-issuance plan returns, using the weaker evidence rank, then
  rank sum, then a stable request hash. No topic labels select results.
  Methods can return recommendations on different requests. Replay
  knows the empirical quota and does not define an online threshold.
  If either plan issues nothing, there is no matched-quota comparison.

A known result has both requested declared roles and both publisher
topic tags. A missing tag remains unresolved. Descriptive intervals
resample ordered topic pairs. They measure uncertainty on the proxy
labels and do not adjudicate missing judgments.

## Results on RTX 5090

The two execution orders were repeated on Slurm array `44570`.
The following counts reproduced. Each cell reports known / issued
recommendations.

| Panel | View | Sequential | Joint |
| --- | --- | ---: | ---: |
| PLOS (160 requests) | Natural issuance | 28 / 33 | 49 / 78 |
| PLOS | Same issuing requests | 21 / 24 | 19 / 24 |
| PLOS | Same aggregate quota | 28 / 33 | 27 / 33 |
| eLife headings (51 requests) | Natural issuance | 0 / 0 | 14 / 16 |
| eLife headings | Same issuing requests or quota | unavailable | unavailable |

In PLOS, 20 of the 24 shared issuing requests return the same expert
and citations. The four changed requests return a different expert.
Two lose a known match under the publisher proxy. At the same aggregate
quota, the methods overlap on 16 requests. The paired known-match
difference across all 160 requests is −0.63 percentage points with a
descriptive 95% interval from −10.17 to +8.51 points. The main observed
gain is coverage. There is no established ranking improvement on
the shared requests. eLife supplies no shared issuing requests for
this particular comparator.

## Independent review

The packet contains 107 distinct request–person–citation bundles from
natural issuance across both panels. Both matched views map to subsets of
the same cards. Eight cards cover the four PLOS cases that change the
expert on the same request. Reviewing those eight is sufficient to
resolve the paired difference for that view if all judgments are
conclusive, because the remaining 20 identical answers cancel.
It does not establish either method's absolute precision. No independent
human judgments have been completed.

The reviewer-only HTML hides policy, rank and topic tags. The private
key binds case IDs to each policy and view. Each card shows two
requirements, source titles and abstracts, contributor-role excerpts
and publisher bylines. The independent reviewers check each topic and
role and confirm whether both documents identify the same person.
The scorer reports support intervals and bounds for the paired
difference against the sequential plan, including shared-case
cancellation. The private key, rendered cards and label exports stay
outside Git.

## Reproduce

Restore the frozen publisher snapshots and embedding checkpoint for
the live request replay. Run twice on a Slurm GPU with separate output
paths, including a repeat with `--order-offset 1`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.joint_nomination_blind_review \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json \
  SUMMARY_JSON PACKET_JSON PRIVATE_KEY_JSON

PYTHONPATH=src:. python -m experiments.credit.blind_joint_topic_review \
  render PACKET_JSON REVIEW_HTML
PYTHONPATH=src:. python -m experiments.credit.blind_joint_topic_review \
  score PRIVATE_KEY_JSON AGGREGATE_JSON REVIEW_ONE_JSON REVIEW_TWO_JSON
```

[Aggregate replay results](results.json).
