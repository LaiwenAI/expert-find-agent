# Test citation rescue against independent person-specific judgments

The [CoF packed rescue experiment](../cof-packed-evidence-rescue-v1) increased attachment coverage at fixed expert ranks, with fewer known-positive gains on KDD than on SciRepEval. Those datasets do not judge the individual citations. This audit tests a separate question on the public TREC 2006 Enterprise qrels: how frequently can a topically supporting document fail to support the specific expert being recommended? The official judgments distinguish positive support (2), no support either way (1) and negative support (0) for the candidate and topic. [TREC assessment protocol](https://trec.nist.gov/pubs/trec15/papers/ENT06.OVERVIEW.pdf)

Fix the same leave-one-topic-out five-expert ranking as in the [person-specific evidence audit](../trec-ownership-metric-audit-v1). For each expert, shuffle only their judged documents using 32 deterministic orders per topic. Reveal the first three documents' target-expert judgments to represent initial verification. The rescue action applies when exactly one of those documents is positive and at least one judged additional source exists. Every route may attach at most one additional document from the same next-four pool. Unknown pairs, corpus retrieval, opening costs and proof of actual personal contribution are outside this replay.

Two stages deliberately use qrels-derived signals. An oracle topical screen accepts a document if it supports anyone for that topic. A person check reveals the official document grade for the queried expert. The selective route checks only documents that appear under multiple judged experts for the topic. Its ambiguity flag is **pool visibility**, not verified exclusivity in the full corpus. A single judged association implies a true person match only under the oracle topical signal. Compare single-source checks, sequential checks until first support, first topical hit, checking this hit with or without the ambiguity gate, and checking topical candidates with backfill until one passes. The exhaustive oracle counts all target checks in the four-source window. A separate sensitivity arm simulates 90% topical sensitivity and 5% false-positive rate with one seeded draw shared across all views of a topic–document per replay. These rates are assumptions, not measured model behavior. Person checks still reveal the human labels.

## Results

The 49 topics and 32 repeated orders produce 782 eligible slot-replays, with a true completion available in 467. Repeated orders are dependent. Bootstrap intervals resample topics after averaging their orders. Count positive and non-supporting additions separately.

| Oracle topical condition | Valid new citations | Neutral or contradictory additions | Person checks |
| --- | ---: | ---: | ---: |
| Check only the next ranked source | 194 | 0 | 782 |
| Attach first topical hit | 420 | 155 | 0 |
| Check first topical hit only if shared | 420 | 0 | 409 |
| Check every first topical hit | 420 | 0 | 575 |
| Check shared topical hits and backfill | 467 | 0 | 497 |
| Check all topical hits and backfill | 467 | 0 | 677 |
| Sequential person checks until first support | 467 | 0 | 2,179 |
| Exhaustive person checks | 467 | 0 | 3,028 |

Among the 155 unsuitable first topical additions, 146 are neutral and nine are contradictory. With noisy topical decisions the shared-only backfill accepts 436 valid and 22 invalid additions in 513 person checks. Those 22 errors include 18 neutral and four contradictory additions. Checking all topical positives and backfilling recovers 446 valid additions and zero invalid in 713 perfect person checks, against 467 and zero in 2,179 sequential checks. This establishes a useful cost decomposition for a real test. The numbers exclude computing the topical signal, and perfect person checks mean the replay cannot estimate deployment precision or latency. It tests evidence supporting a specific person on a topic, not whether a new citation satisfies a distinct industrial constraint.

[Aggregate results and sensitivity analysis](results.json)

## Reproduce

Use both pinned public qrels from the [TREC 2006 data page](https://trec.nist.gov/data/t15_enterprise.html). The CPU runner validates their hashes before evaluation.

```sh
PYTHONPATH=src:. python -m experiments.trec_enterprise.citation_rescue_audit \
  TREC_QRELS_DIRECTORY NEW_RESULTS.json
```
