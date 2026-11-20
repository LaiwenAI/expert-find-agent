# Share exact evidence prefixes while preserving atomic prompts

Reuse the three pinned Re-DocRED panels, global-name windows and
frozen Qwen3-4B-Instruct-2507 reader from the
[joint scoring experiment](../redocred-joint-reader-v1/README.md).
Each request uses four binary prompts with the original work,
person, role, and two assigned source sentences. Reuse the exact
token IDs as emitted by the original chat template. The two
questions belonging to a candidate remain independent until the
conjunction is scored.

| Route | Execution |
| --- | --- |
| Independent | Four separate rows in the original padded batch. |
| Common prefix cache | Prefill the prefix common to the four rows and use the established read-only KV broadcast for their suffixes. |
| Common prefix tree | Share that common prefix in one packed forward. Keep suffixes on distinct branches. |
| Full prefix tree | Share every exact token prefix within a request. Read one original per-claim score at each leaf. |
| Guarded full tree | Re-execute the original padded batch if a packed score is close enough to a calibrated decision boundary to change the outcome. |

In a depth-first preordered tree, a token can attend to node `j`
only if the token lies in the subtree interval rooted at `j`.
The node position is its depth in the original prompt. This
supplies the same ancestor tokens and positions as causal
attention in an independent prompt. Prefix reuse and tree-shaped
attention are established techniques, e.g.
[Hydragen](https://arxiv.org/abs/2402.05099). This implementation
uses a dense custom mask and measures single-forward verification
with four prompts per request.

Configure the same original margin cutoff using the development
batch. The full-tree guard radius is the maximum absolute margin
difference seen in that development panel. Recheck when an atomic
margin is within the radius of zero or the conjunction cutoff.
The bound is empirical. Report exceptions on test and compare
binary decisions as well as maximum score error. Both original
binary decisions and candidate conjunctions must match for a
request to count as unchanged.

Time warm local requests, including window selection, tokenization,
mask construction, prefix materialization, inference and rechecks.
Charge measured common preprocessing equally to each route. Token
counts record physically processed input slots, including padded
suffixes and full fallbacks, while the attention matrix cost can
depend on layout. These are within-document verification timings,
outside corpus retrieval, request queues and transport.

## GPU results

Both Slurm tasks in array `44327` completed on an RTX 5090 on ws5.
They reproduce all non-timing aggregates. The independent route
reproduces the previous acceptance counts on all three panels.
Exact trie token counts match the preceding CPU audit. The
table covers the 213-request title-disjoint holdout.

| Route | Changed requests / 213 | Changed candidate decisions | Known / unjudged approvals | Input slots | P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent | 0 | 0 | 50 / 14 | 170,172 | 53.99–54.70 |
| Common prefix cache | 5 | 1 | 50 / 15 | 113,265 | 48.27–48.32 |
| Common prefix tree | 6 | 2 | 49 / 15 | 101,795 | 38.96–39.02 |
| Full prefix tree | 8 | 3 | 48 / 15 | 83,750 | 33.19–33.21 |
| Guarded full tree | 0 | 0 | 50 / 14 | 108,734 | 73.38–73.39 |

The full tree removes 50.8% of processed input positions and
reduces local P95 by about 38%. All routes use the original
prompt tokens. Different batch geometry and numerical execution
can still move margins across a decision boundary. One held-out
atomic score exceeds the development maximum error of 2.75.
The restored-decision route performs 32 full rechecks, saving
input slots overall and increasing tail latency.

A structural regression test reconstructs every prompt from
its leaf's interval ancestors, including equal and nested
prompts. A float32 Qwen3 toy-model check also reproduces
independent next-token logits to within 0.000001. Exact
logical attention and finite-precision execution agreement are
separate properties.

### Prewarm the immutable instruction prefix

Both tasks in Slurm array `44332` reproduce all non-timing results.
Compile the 81-token prefix once from the fixed prompt template,
without using calibration or test text. Each request verifies that
its original tokens start with that prefix. The cache prefill is
charged once at startup. Score the remaining text either as
separate suffixes or as a full prefix tree rooted at the shared
cache. The static-tree guard uses its own development margin
radius of 2.5. The table uses the same 213-request panel.

| Warm route | Changed requests | Changed conjunctions | Known / unjudged approvals | Input slots | P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent | 0 | 0 | 50 / 14 | 170,172 | 54.24–54.47 |
| Static prefix cache | 5 | 1 | 50 / 15 | 101,160 | 40.79–40.81 |
| Static prefix plus tree | 6 | 3 | 50 / 15 | 66,497 | 30.26–30.29 |
| Guarded static prefix plus tree | 0 | 0 | 50 / 14 | 89,225 | 69.82–69.93 |

The guarded static tree removes 47.6% of input slots and
matches every atomic and conjunctive decision. It still
rechecks 29 of 213 requests and remains slower at P95 than
the independent reference. No static-tree test margin exceeds
its development radius on this panel. The guard's observed
agreement remains conditional on that empirical envelope.
The warm results motivate cheaper rechecks or a canonical
execution geometry shared by the reference and optimized
reader before deploying exact-output acceleration.

[Fresh-prefix runs](results.json) · [Static-prefix runs](static.json)

## Reproduce

Use the pinned Re-DocRED snapshot and Qwen3-4B-Instruct-2507
checkpoint from the linked benchmark. Start on Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.conjunctive_trie_reader \
  REDOCRED_DIR QWEN4B NEW_RESULTS --order-offset 0
```

The prewarmed variant uses the same arguments and the
`experiments.docred.conjunctive_static_prefix` module.
