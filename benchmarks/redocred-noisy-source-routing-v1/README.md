# Stress test work-alias routing on imperfect query names

Search the same 1,000 public documents and candidate entity mentions as
[global role search](../redocred-global-role-search-v1).
Keep the requested role and the original person–document judgments fixed.
Replace the query's work string with one of three variants. The scorer
receives only this replacement string, including in the reader query and
evidence selector.

| Variant | Construction |
| --- | --- |
| Observed alias | Select a different annotated surface form in the reference document, excluding a short fixed list of generic mentions |
| One token removed | Delete one deterministically selected token from a multi-token work name |
| Transposition | Swap two adjacent, unequal interior letters in one alphabetic token of length five or more |

Variant selection uses a stable hash. Eligibility differs across panels.
An observed alias is drawn from the document annotations and can have
an exact match by construction. The other variants are synthetic
stress cases. They do not estimate rates of spelling mistakes,
abbreviations or ambiguous descriptions in real queries.

Compare five source policies with identical limits of four documents
and five person–source candidates. All use the previous mention and
role-proximity scorer for candidate order.

| Policy | Source selection |
| --- | --- |
| Flat four | BM25 over the noisy work string and role cues |
| Exact then flat | Exact work alias postings when available, otherwise BM25 |
| Fuzzy only | Exact alias or the strongest fuzzy alias, with BM25 when neither is available |
| Guarded fuzzy | Use a sole exact, multi-token alias directly. Otherwise retain the top BM25 document, add fuzzy documents and fill from BM25. Skip fuzzy documents below the guard. |
| Swap then guard | When the exact input is absent, try one adjacent swap in a single alphabetic token. Open the directly matching document only when there is exactly one indexed alias and one document. Otherwise use guarded fuzzy. |

The fuzzy lookup indexes character trigrams of known aliases. Use
overlap to shortlist up to 48 aliases, then use normalized sequence
similarity. Require 0.65 for the fuzzy-only route. The guarded route
requires 0.78; it reserves a place for the second alias's best
document when the two similarity scores differ by at most 0.07.
These thresholds are fixed heuristics. The swap path was added after
inspecting the first typo results and is exploratory on the reused panel. They are not estimates of
the probability that a source is correct.

Run the resident index on CPU and repeat search with
Qwen3-4B-Instruct-2507 on RTX 5090. The reader receives at most two
selected sentences per proposed pair and uses the previous
alias-boost cutoff of 0.25 for every route and variant. Do not
recalibrate on these perturbed views. The test reuses the original
work–role judgments and is an exploratory sensitivity study.
Report known support and competing roles, keeping unjudged
acceptances visible. Bootstrap query differences by reference
document. GPU service timing includes resident retrieval, span
selection, prompt preparation and inference under warm serial
execution. Index construction and service queueing are excluded.

## Results

RTX 5090 job 43041 reproduced the four original policy decisions of job
43038 and added the exact-swap route. CPU source and candidate results
also reproduce the GPU proposals before verification.

| Query variant | Policy | Queries with known source | Opened documents | Known positives accepted | Known rivals accepted | Unjudged accepted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Observed alias (24) | Flat four | 23 | 96 | 15 | 0 | 0 |
| Observed alias (24) | Exact then flat | 24 | 32 | 23 | 0 | 0 |
| Observed alias (24) | Fuzzy only | 22 | 26 | 23 | 0 | 0 |
| Observed alias (24) | Guarded fuzzy | 24 | 51 | 20 | 0 | 0 |
| Observed alias (24) | Swap then guard | 24 | 51 | 20 | 0 | 0 |
| Token removed (136) | Flat four | 109 | 544 | 53 | 0 | 26 |
| Token removed (136) | Exact then flat | 109 | 523 | 53 | 0 | 26 |
| Token removed (136) | Fuzzy only | 101 | 192 | 70 | 0 | 24 |
| Token removed (136) | Guarded fuzzy | 110 | 523 | 53 | 0 | 26 |
| Token removed (136) | Swap then guard | 110 | 523 | 53 | 0 | 26 |
| Transposition (145) | Flat four | 91 | 580 | 48 | 1 | 7 |
| Transposition (145) | Exact then flat | 91 | 580 | 48 | 1 | 7 |
| Transposition (145) | Fuzzy only | 145 | 147 | 111 | 1 | 9 |
| Transposition (145) | Guarded fuzzy | 145 | 580 | 73 | 1 | 7 |
| Transposition (145) | Swap then guard | 145 | 151 | 111 | 1 | 9 |

On transpositions, the exact-swap guard matches the fuzzy-only
route's 111 accepted positives and one annotated rival error.
Relative to BM25, it accepts 63 additional known positives and sends
601 person–source candidates to the reader instead of 725. It takes
the cheap correction branch in 143 of 145 views. Its resident CPU
retrieval P95 is 0.035 ms, versus 1.69 ms for fuzzy-only. Warm GPU
path P95 remains similar to BM25 (57.21 vs 57.88 ms).
The fewer expansions and verifier inputs do not imply the same
fractional reduction in tail latency.

On token deletion, fuzzy-only finds fewer known sources than BM25
(101 vs. 109) but exposes a more concentrated set of candidates from
the retained sources, with 70 accepted positives versus 53.
The guarded route finds 110 known sources but matches BM25's 53
accepted positives. These are distinct source-level and person-level
trade-offs. Only four observed or truncated aliases had exact
postings in multiple documents. Flat found a known source for two of
these; exact, fuzzy-only and guarded each found one for three.
This small slice measures multiple-document aliases, not a reliable
error rate for homonymous works.

These findings support separating exact repair from ambiguous partial
name lookup. The correction branch was motivated by a synthetic
one-swap benchmark, and the collection still supplies entity mentions.
No inference about real industrial query variation or automatically
extracted names follows from this experiment.

[Aggregate results](results.json)

## Reproduce

Use the pinned splits and reader from the
[global search experiment](../redocred-global-role-search-v1).

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.docred.noisy_role_source_routing REDOCRED_DIRECTORY CPU_RESULTS.json
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.noisy_role_source_routing REDOCRED_DIRECTORY GPU_RESULTS.json \
    --gpu-model PINNED_QWEN4B
```
