# Extract unquoted requirements before budgeted expert search

The [quoted-span benchmark](../prose-budgeted-source-role-v1) gave
parsers the boundaries of the two paper topics and two required
activities. Here all four fragments appear without delimiters.
Use one deterministic role-swap family per target publication pair
from the [held eLife catalog](../elife-prompt-holdout-v1), taking the
lexicographically first family hash per pair. Each family has a
positive and a control with swapped roles. Render each into three
synthetic English forms, for 26 requests and 13 positives per form.
The target publications are present in the 393-paper catalog.

The two marker rules were written for the local-marker and reversed-reference
forms. The shifted form uses different connectives and presents each
activity before its topic. Rules abstain there. Both Qwen3-4B prompts
ask for verbatim topic–activity pairs in JSON. One prompt permits an
empty list and the other tells the model both pairs are present.
A local validator requires four unique, nonoverlapping substrings and
orders the pairs by topic position. The hybrid uses a matching rule or
falls back to the corresponding model prompt. Supplied exact spans
provide a search-only reference. All accepted extractions go through
the same [screened single-verification path](../budgeted-source-role-service-v1).
The parser call is charged separately from that verifier cap.

Exact binding requires both complete spans in each topic–activity pair.
A strict proof requires the correct sources, contributors and roles.
An adverse issuance includes any incorrect contributor or mismatched
source or role. A parser can pass copy validation and still mis-bind
a valid substring. The aggregate also reports emitted proofs, wrong
sources, verifier calls, and parsing token costs.

## Results

Both jobs of Slurm array 46079 on ws3 RTX 5090 completed and
matched on all non-timing fields in forward and reverse request order.
Each cell reports exact bindings / strict proofs and warm serial P95.

| Parser | Local markers | Reversed references | Shifted markers |
| --- | ---: | ---: | ---: |
| Supplied spans | 26 / 11; 25.2–26.2 ms | 26 / 11; 23.0–23.4 ms | 26 / 11; 23.0–23.3 ms |
| Marker rules | 26 / 11; 23.2–23.6 ms | 26 / 11; 22.3–22.9 ms | 0 / 0; 0.0–0.0 ms |
| Qwen, empty allowed | 1 / 1; 655.0–662.5 ms | 0 / 0; 26.1–26.3 ms | 0 / 0; 26.1–26.3 ms |
| Qwen, extraction required | 4 / 5; 846.1–860.6 ms | 5 / 3; 986.8–1002.3 ms | 0 / 0; 1011.6–1021.5 ms |
| Rules + Qwen, empty allowed | 26 / 11; 22.5–22.6 ms | 26 / 11; 21.8–22.3 ms | 0 / 0; 26.1–26.3 ms |
| Rules + Qwen, extraction required | 26 / 11; 21.9–22.5 ms | 26 / 11; 21.8–22.2 ms | 0 / 0; 1011.6–1021.5 ms |

The rule path was fast on its two written forms and had no coverage
on the shifted form. Requiring extraction is a post-pilot prompt
diagnostic on this same panel. Added decoding restores a few proofs
on the reversed form, but none on the shifted form. On the local form,
five proofs are correct even though just four extractions match all
gold boundaries exactly. Even without
an empty-list instruction, every shifted request fails the strict
copy gate. The model's partial recovery is insufficient for a default
fallback on these requests. The few model issuances and repeated
paper pairs also give little evidence about population risk.

Next compare a span boundary and pairing scorer that returns
original offsets without token-by-token generation. Hold out source
families and connective families separately, and compare with
independently worded requests before promoting any parser to the
general first-query path. Score downstream strict and adverse
issuance as well as binding, since exact-span errors can differ in
their effect on the search.

Reported P95 is warm serial execution from request text to proof or
abstention, including generation when invoked and excluding index
build, HTTP and queueing. Models are frozen. The parser result is
reused when evaluating its hybrid arm and the measured parser time
is charged whenever that arm would call the model.

## Reproduce

Restore the held projection and frozen cutoffs, then run in a Slurm
CUDA allocation with the pinned checkpoints. Repeat with `--reverse`
and a fresh output. The source snapshot and checkpoint hashes are in
[the aggregate results](results.json).

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.unquoted_budgeted_source_role \
  ELIFE_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/independent-source-confirmation-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_RESULTS
```
