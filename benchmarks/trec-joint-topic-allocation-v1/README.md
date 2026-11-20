# Allocate three sources on pairs of independently judged TREC topics

Use the official [TREC 2006 topics, expert judgments and support
judgments](https://trec.nist.gov/data/t15_enterprise.html). A known
pair requires one grade-2 expert on both topics, with two distinct
documents independently judged to support that person on their
respective topics. Original topics are composed without rewriting.
Their individual judgments define the conjunction. This is a
public graded-evidence transfer diagnostic. It does not assign a
new judgment of the combined request, project participation, or
authorship.

Take all unordered topic pairs with at least one known proof.
Select a second analysis panel by greedy matching in stable hash
order so that no original topic appears twice. The main pairs share
topics and documents and are not independent observations.

For each pair, simultaneously withhold both topics' expert and
document judgments from the discovery index. Embed the original
title and description for every topic using the frozen Qwen3-0.6B
model. Rank the remaining 47 topics by cosine similarity. Score
each source by the sum of reciprocal ranks of historical topics
in which it was pooled. The ranker reads topics and historical
membership only. It does not retrieve raw W3C text or use a
published run's ranking for that target.

Construct two historical person--document association views on
these same source lists. The broad view uses all pooled
associations regardless of grade. The stricter view requires at
least one grade-2 support judgment on a historical topic.
Associations in these views need not be author bylines.
Neither view obtains any associations from the two held topics.
A perturbation check verifies that changes confined to those
topics leave the candidate ordering and posting view invariant.

Compare topical selection, sequential repair, and same-person
joint nomination. Each exposes exactly three distinct documents
under a pooled judgment oracle. Given a source, the oracle can
reveal held-topic grade-2 support for eligible historical
person--document pairs. Sequential repair may use the first
two disclosures to allocate the third. No other policy looks
at the held judgments before assigning the reads. The outcome
marks whether the three documents contain a known full proof.
Unjudged candidates receive no positive support. The counts are
observed proof recovery under an oracle, not calibrated system
precision. The three policies share a document budget. They expose
different numbers of candidate person--document pairs per read.

The protocol uses grade 2 on both dimensions according to the
[track report](https://trec.nist.gov/pubs/trec15/papers/ENT06.OVERVIEW.pdf).
Results should be interpreted as allocation under historical
pooled associations. Testing a live retriever against W3C documents
requires the original document collection or a separately
verified projection.

## Results on ws5

Both tasks of Slurm array `44674` reproduce the same aggregates.
The 49 separately judged topics yield 749 positive topic pairs
and 24 pairs in the disjoint-topic subset. The union of pooled
source IDs contains 48,650 documents. Each arm exposes 2,247
documents on the main panel.

| Historical association | Depth per topic | Pairs with known proof reachable | Topical | Sequential repair | Joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| All pooled links | 64 | 88 | 6 | 12 | 7 |
| All pooled links | 256 | 206 | 7 | 12 | 7 |
| Prior grade-2 links only | 64 | 37 | 3 | 5 | 3 |
| Prior grade-2 links only | 256 | 71 | 4 | 6 | 3 |

The broader source depth makes substantially more known proofs
reachable. It does not improve the joint plan's recoveries within
the tested three-read budget. At depth 256, the grade-2-only
history reduces candidate associations exposed by the joint
route from 425,870 to 102,559, but loses reachable and recovered
proofs. No arm recovers a labeled pair in the 24-pair disjoint
panel, so that subset gives no directional replication.

The publisher result therefore does not transfer to historical
pooled associations in this test. Even the strongest sequential
control recovers a small fraction of all known proofs. The
difference motivates separating candidate access, historical
identity links and support for the two requested topics in any
future raw-document comparison.

## Reproduce

Get the three official files from NIST. The runner checks their
SHA-256 hashes. Restore the frozen embedding checkpoint and run on
a compute node with a Slurm GPU. Use a fresh output path and
`--order-offset 1` for the repeat.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.trec_enterprise.joint_topic_allocation \
  TREC_QRELS_AND_TOPICS_DIRECTORY QWEN_EMBEDDING NEW_RESULTS
```

[Replicated aggregate results](results.json).
