# Bind topics and roles before budgeted proof search

Use the same [held eLife catalog](../elife-prompt-holdout-v1) and
[screened single-verification search](../budgeted-source-role-service-v1).
Each case keeps the original keyword clues, NISO activity phrases,
expected papers and contributor reference. Render the two pairs into
one prose request using three constructions: topic then role for each
paper, topics first with reversed former/latter references, or role
then topic for each paper. The four input fragments are delimited by
quotes. This isolates binding once the spans are observable. It does
not measure discovering fragment boundaries in unconstrained text.

Compare five input paths. Supplied pairs are the exact input anchor.
Adjacent pairing assumes spans 1 and 3 are topics and the following
spans are their roles. The structural parser finds `about` markers
and follows relative or local paper references. Its cue vocabulary
matches the controlled English forms. The Qwen3-4B span parser sees
the prose plus numbered spans. It selects one of 12 topic-to-role
assignments or an unclear answer by reading a single output token.
A factorized model passes two shorter prompts in one batch: identify
the topic spans, and find the span paired with the first quote.
Recover the second pair by elimination. Inconsistent outputs abstain.
The selected strings are copied directly into the search request.

Record exact span binding, strict proof, wrong-source and adverse
issuance, and latency including the parser. The verifier budget
still applies after parsing. Each model parser adds one call before
retrieval. All arms compose the same search function with the same
frozen thresholds. Requests in this run keep both target papers
available. A small set of publication pairs and phrasing templates
does not establish broad language performance.

## Results

Each form has 132 requests and 66 positives. The paired values
below are correct bindings and strict proofs, respectively.

| Parser | Topic then role | Reversed references | Role then topic |
| --- | ---: | ---: | ---: |
| Supplied pairs | 132 / 36 | 132 / 36 | 132 / 36 |
| Adjacent | 132 / 36 | 0 / 0 | 0 / 0 |
| Structural cues | 132 / 36 | 132 / 36 | 132 / 36 |
| Flat span model | 132 / 36 | 110 / 28 | 0 / 0 |
| Factorized span model | 119 / 33 | 39 / 9 | 117 / 33 |

For all forms, the structural path reproduces the supplied-pair
counts, including one adverse outcome per form, at about 22 ms
local P95. This is expected for explicit markers and manufactured
inputs aligned with the rule vocabulary. It only supports a fast
path when the relevant structural cues are available.

The flat span model identifies the exact two topic spans on
128/132 reversed-reference requests but binds both roles correctly
on 110. Eight strict proofs are lost relative to supplied pairs,
and adverse results increase from one to three. On role-first
inputs, the flat model fails all bindings. Factorization restores
117 bindings and 33 strict proofs there, but drops to 39 bindings
and nine strict proofs for reversed references. No parser issues
a wrong-source result on this fixture. Exact span-pair quality
must be reported by form because the two model layouts fail on
different syntax.

For correctly explicit bindings, adding a model approximately
doubles local tail latency. The flat model with interleaved
inputs has 45.3–45.5 ms end-to-end P95; the factorized model
with role-first inputs has 46.5 ms. Both jobs in Slurm array
46068 ran on ws3 RTX 5090 and reproduce every non-timing field
in reverse order. Timings include one-token parser inference and
the real search implementation, excluding HTTP, index construction
and text segmentation beyond quote extraction. These are matched
synthetic forms over the same 13 article pairs. [Full results](results.json)

## Reproduce

Restore the held projection and frozen cutoffs, then run in a
Slurm CUDA allocation with the pinned models. Repeat with
`--reverse` and a new output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.prose_budgeted_source_role \
  ELIFE_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/independent-source-confirmation-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_RESULTS
```
