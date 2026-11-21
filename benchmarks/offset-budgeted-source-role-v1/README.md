# Recover first-query requirements with one offset-tagging pass

The [unquoted benchmark](../unquoted-budgeted-source-role-v1)
found that generating two verbatim topic–activity pairs could take
approximately one second without producing a valid fallback on its
shifted English form. This experiment trains a small bidirectional
GRU to assign each token to one of four bound fields or background.
The word-lookup model uses a single CPU forward pass. A second
model supplies frozen Qwen3-Embedding-0.6B subword states to the
same recurrent width and runs on CUDA. Both return original
character spans without generating text. Inference
requires one contiguous run for each field and sorts the two pairs by
the first and second topic. It sends the original substrings into the
same screened single-verification search.

## Split and fitting

Use the same eLife target pairs and synthetic test requests as the
earlier benchmark. Exclude their articles and any article sharing
a recorded author identifier with them from tagger development.
Generate source keyword fragments from the remainder, discarding
exact test keyword strings. Split training and calibration articles by
publication hash and remove duplicate strings across those pools.
Draw training and calibration activities from separate slices of the
previous training phrase bank. The test requests use the earlier
held NISO phrases. The underlying search models and cutoffs are frozen.

Generate 2,048 training and 320 calibration requests, with the
two-pair requests using 16 training and three distinct calibration
templates. A quarter of draws use shared development templates
with fewer than two requirements. The three test templates were
excluded from fitting and calibration. Their general sentence
structures have analogues in training. Train only the tagger on CUDA. Apply word dropout to the lookup
model. Cache the frozen embedding states while fitting the contextual
head. The comparison changes contextualization, tokenization and
execution device together. It does not isolate their individual
effects. For the calibrated route choose the lowest
development probability threshold that admits no examples with
fewer than two requirements and attains at least 90% exact-copy
binding among admitted development examples. The selection permits
abstaining on every request. The test panel reports binding accuracy,
strict proofs and adverse issuances for the tagger with and without
that threshold, the fixed rules, and the rules with calibrated tagger
fallback. Supplied spans provide a search-only reference.

## Results

Slurm array 46094 completed all four tasks on an RTX 5090. Each
cell reports exact bindings / strict proofs / adverse issuances
over 26 requests with 13 positives, followed by warm serial P95.
Ranges span the two seeds and execution orders.

| Route | Local markers | Reversed references | Shifted markers |
| --- | ---: | ---: | ---: |
| Supplied spans | 26 / 11 / 0; 21.7–23.2 ms | 26 / 11 / 0; 21.4–22.3 ms | 26 / 11 / 0; 21.4–22.7 ms |
| Marker rules | 26 / 11 / 0; 21.6–22.5 ms | 26 / 11 / 0; 21.3–22.5 ms | 0 / 0 / 0; 0.0 ms |
| Lookup tagger | 0 / 0 / 0; 1.2–1.3 ms | 0 / 0 / 0; 1.2 ms | 0 / 0 / 0; 1.1 ms |
| Contextual tagger | 0 / 0 / 0; 9.1–9.6 ms | 0 / 0 / 1–2; 19.3–29.0 ms | 2–4 / 2 / 0; 19.5–28.5 ms |
| Calibrated contextual tagger | 0 / 0 / 0; 9.1–9.6 ms | 0 / 0 / 0; 9.0–9.4 ms | 0 / 0 / 0; 8.9–9.4 ms |
| Rules + calibrated contextual tagger | 26 / 11 / 0; 21.4–22.4 ms | 26 / 11 / 0; 21.2–22.6 ms | 0 / 0 / 0; 9.0–9.4 ms |

The randomly initialized lookup model rejected every test request.
The contextual tagger took about 9 ms P95 to parse. On the shifted
form its uncalibrated version returned two strict proofs per seed,
from two total issuances per seed. The supplied-span search returned
11. For reversed references, every tagger issuance was adverse
(one or two per run). The low issuance counts give little evidence
about population risk.

Both contextual runs selected a threshold of 1.0 after failing the
development precision criterion at every nonempty operating point.
The calibrated fallback therefore returned nothing on the shifted
form. Loss decreased to below 0.002, while exact binding among
valid development proposals at threshold zero was 14/79 and 10/57.
This illustrates template overfit and weak transfer of the minimum
token-confidence score. A useful selective fallback needs to recover
more proposals and detect incorrect topic-to-activity assignments.

Digits describe warm serial parser-plus-search latency or time to
abstain. Serving, queueing, and index build are outside the timed
scope. The test article pairs and authored templates are reused
from the earlier unquoted diagnostic. [Full results](results.json)

## Reproduce

Restore the held projection, phrase banks and frozen search cutoffs.
Run inside a Slurm allocation and change the output path for the
second training seed. Use `--context` for the frozen contextual features
and `--seed 29 --reverse` for the alternate run order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.offset_budgeted_source_role \
  ELIFE_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/independent-source-confirmation-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_RESULTS --seed 17
```
