# Measure same-author proof on original publisher contribution text

The [multisource filter study](../credit-multisource-proof-filter-v1)
generated statement text from the role labels. This evaluation uses
original contribution paragraphs from *Frontiers in Robotics and AI*
and *Sensors*. Each admitted article has exact agreement between a
parser of the visible publisher text and structured JATS role tags.
The JATS labels score each requirement; the note, initials and visible
role index drive the methods. Agreement is across distinct recorded
representations from the same publisher. It does not independently
verify the contribution itself.

Reuse the [dual-format contribution panel](../engineering-contribution-transfer-v1)
and [verified two-source engineering links](../engineering-cross-source-search-v1).
Restore the six additional article files with their recorded checksums.
Use the original request families and all identities linked across each
designated source pair. Each request asks for one role on each source.
A matched negative swaps those bindings. Source identities, candidate
identities and condition-to-source mapping are given.

Compare four routes using one shared Qwen3-Reranker-0.6B and the
previously frozen query role banks.

| Route | Evidence score |
| --- | --- |
| Shared text | Full contribution text and target-author initials |
| Author scope | Original blocks mentioning the target author and their initials |
| Indexed roles | Frozen ridge role margin, contrasting parsed roles held by the author with those absent |
| Fused | Regularized logistic combination of scoped reranker margin and frozen ridge and definition margins |

Build indexed memberships from the visible note's parsed declarations.
The embedding role banks use the prior 84 training descriptions. The
reranker and banks are frozen. Fit the score fusion on a hash-ordered,
article-disjoint two-thirds split of Frontiers. On the remaining
Frontiers papers, select each threshold to accept at most 5% of known
uncredited atomic author–role cases, with strict inequality to handle
ties. Apply the thresholds unchanged to both the original Sensors
single-paper cases and the linked Sensors panel. The single-paper cases
use one authored phrase per role. The linked cases use [NISO example
phrases](https://zenodo.org/records/18421449). Some Sensors articles
occur in both test panels. Neither panel is used for fitting.

After the first experiment, repeat using unused NISO phrases for
Frontiers training and thresholding. Exclude any phrase in the linked
test after normalization. Match this phrasing in the Sensors atomic
panel. Two roles lack an unused phrase and are omitted from this
variant's atomic panels. Keep the same article split and the same
linked test. This diagnostic changes both phrasing and the mix of
atomic roles, so it cannot isolate phrasing alone.

For the linked requests, issue a named author only if both original
scope claims pass. Report correct proof recovery and adverse identity
outputs separately. The reranker computes batched evidence scores and
the decision routes reuse them. This test evaluates attribution quality
under supplied source and ownership metadata; it does not measure
service latency or open-corpus retrieval.

## Results

The linked corpus contains 15 positive and 15 matched negative
requests drawn from four source pairs. Two Slurm tasks in array
46963 ran on RTX 4090 GPUs on ws1, reversing inference batch order
with the same batch composition. Thresholds, counts and output
signatures agree. The original authored-phrase pilot (46959)
reproduces the linked counts. Repacking the complete run changes
some single-source counts and thresholds; both sets are archived.

| Route | Sensors atomics, authored phrases, correct / wrong accepts (137 each) | Linked proof, authored calibration, correct / wrong requests | Linked proof, unused NISO calibration, correct / wrong requests |
| --- | ---: | ---: | ---: |
| Shared text | 7 / 3 | 0 / 0 | 0 / 0 |
| Author scope | 10 / 3 | 0 / 0 | 0 / 0 |
| Indexed roles | 137 / 8 | 15 / 7 | 7 / 2 |
| Fused | 137 / 2 | 10 / 4 | 8 / 2 |

Each wrong linked request is a source-swap negative. Independently
of the threshold, compare the verified author's weaker-side score
on its positive and negative request. Shared text ranks the positive
higher in 1 of 15 comparisons, candidate scope in 11, and the indexed
and fused routes in 14 each. There is one tie for shared text, scope,
and indexing and none for fusion. The reused contrast families do
not supply independent samples.

Calibrating with unused NISO phrases lowers adverse issuance for the
indexed and fused routes while reducing correct proof recovery. Under
this calibration, fusion and indexing produce the same two wrong
request counts, with eight versus seven correct recoveries. The raw
reader still abstains on all linked requests despite the improvement
in ordering from scope. The effect of applying a threshold chosen
on atomic author checks must be evaluated separately from the
quality of the ordering. A subsequent comparison should freeze a
conjunction-level calibration protocol on independent source groups
and measure actual service cost against the index-only route.
[Aggregate results](results.json).

## Reproduce

Restore the original [engineering contribution panel](../engineering-contribution-transfer-v1)
and [cross-source panel](../engineering-cross-source-search-v1).
Retain the six linked XML records with the published intake hashes.
Run on a Slurm GPU with a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.raw_conjunction_audit \
  ENGINEERING_PANEL LINKED_XML_DIRECTORY CROSS_SOURCE_PANEL \
  benchmarks/engineering-cross-source-search-v1/source-manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```

Repeat with `--reverse` and a new output path. The linked directory
must contain both `sources.json` and `xml/`, as generated by the
original restoration routine.
