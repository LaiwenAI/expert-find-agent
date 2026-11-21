# Decode topic–activity offsets jointly

The [offset tagger](../offset-budgeted-source-role-v1) rejected most
held queries when the most likely token labels fragmented a span or
omitted a field. Hold its contextual features, training data and
checkpoint recipe fixed. Decode the same token scores two ways.
Greedy decoding checks the independent token argmax. Joint decoding
maximizes their sum over paths containing each of the four fields in
one contiguous span, with topic 1 preceding topic 2. The activities
may precede their topics or appear in either order. The selected
character ranges must be nonempty and disjoint. Both parsers return
the original text substrings.

Keep the top two complete paths and the best incomplete path. Compare
two gates, based on the score gap to the runner-up complete path or
the gap to the best incomplete path. Choose the highest-coverage
threshold on the 320 development queries with at least 90% exact
binding and zero accepted incomplete requests. If no nonempty
threshold qualifies, the gate abstains. As a follow-up diagnostic,
train a second head with an alternate random seed and issue a single
parse only when the heads agree on both topic and activity strings.
Both heads share one frozen Qwen3-Embedding-0.6B forward pass.
The agreement diagnostic applies no fitted threshold.

All accepted parses use the same screened single-verification search
and frozen models. The 13 target publication pairs and the three
unquoted synthetic templates are unchanged. Each form has 26 cases,
including 13 positive cases. For conditioned fallback results, report
only requests where greedy did not produce a legal parse. Warm serial
P95 includes tokenization, encoder, decoding and any search actually
executed. It excludes HTTP and queueing.

## Results

All tasks of Slurm arrays 46105 and 46109 completed on ws3 RTX 5090.
Joint decoding without gates reproduced its counts in both arrays.
The table uses the final runs. Each entry is strict / adverse /
issued, followed by warm serial P95 over both execution orders.
The positive denominator is 13 in each form.

| Route | Local markers | Reversed references | Shifted markers |
| --- | ---: | ---: | ---: |
| Supplied spans | 11 / 0 / 11; 22.1–22.5 ms | 11 / 0 / 11; 21.5–22.1 ms | 11 / 0 / 11; 22.3–23.9 ms |
| Greedy offsets | 0 / 0 / 0; 9.3–9.4 ms | 0 / 1–2 / 1–2; 20.0–27.5 ms | 2 / 0 / 2; 19.9–27.3 ms |
| Joint offsets | 7–9 / 0–1 / 8–9; 31.0–32.1 ms | 0–2 / 9 / 9–11; 31.1–33.9 ms | 9–11 / 0–3 / 11–12; 32.2–33.6 ms |
| Joint with exact agreement | 3 / 0 / 3; 22.7–22.9 ms | 0 / 8 / 8; 32.6–33.1 ms | 7 / 0 / 7; 32.7–33.3 ms |

On the shifted form, joint decoding recovered most of the strict
proofs available to the supplied-span search. It also widened exposure
to flawed parses: the reversed-reference form had nine adverse
issuances in each run. The single-head margins provided no feasible
nonempty acceptance threshold on the development pool, so both
prespecified gates rejected every held request.

Agreement is a useful comparison because both heads can make the
same wrong assignment. It retained seven strict proofs and no
observed adverse issuance on the shifted form, but retained eight
adverse issuances on the reversed form. The development check also
accepted incomplete requests. This experiment shows that neither
span feasibility nor cross-seed agreement validates the reference
semantics in these inputs. The next discriminator should be tested on
explicit competing assignments to the same spans and independently
written requests.

Absolute precision and tail timings are limited to this synthetic
panel on one GPU. The diagnostic reuses the original paper pairs and
wording and is not a new held cohort. [Full results](results.json)

## Reproduce

Restore the input artifacts listed in the
[offset tagger benchmark](../offset-budgeted-source-role-v1#reproduce).
Add `--context --structured --ensemble` to the existing runner
command. Repeat with `--seed 29 --reverse` and a fresh output.
