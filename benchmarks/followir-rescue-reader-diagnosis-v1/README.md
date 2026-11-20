# Factor reading window and verifier on unsuccessful rescue proposals

Reuse the candidate selection from
[FollowIR constraint rescue](../followir-constraint-rescue-scope-v1).
Run the original 0.6B lead verifier on the four initial candidates
and replay the full-pool constraint route when it abstains. Score
the same two rescue proposals under each of six reader settings.
The new readers never change which requests trigger rescue or which
documents are proposed.

Cross the frozen Qwen3-Reranker-0.6B and Qwen3-4B-Instruct-2507
with three text views: title and the 850-character lead, title and
an 850-character constraint-term window, or title and a longer
body subject to the 1,024-token input limit. Both readers preserve
the prompt suffix. The 4B variant is a standalone instruction
classifier using the logit margin for digits 1 and 0. Model and
prompt change together, so a difference does not isolate model size.
The long-body condition is clipped where needed. Each reader and
view uses its own threshold calibrated to at most 10% known-negative
acceptance on a separate collection's first six topical candidates.
Rescue proposals are held out from calibration.

Time includes window selection, prompt preparation and scoring for
each reader call on a two-document batch. The vectors and document
text are resident for this diagnostic. Initial retrieval and trigger
evaluation are outside the verifier timing. The dataset grades
the document. Human passage labels would be needed to attribute a
specific error to a reader that saw sufficient evidence.

## GPU results

Job `43582` ran on an RTX 5090. The recreated full-pool rescue
policy proposed 30 sources for 15 News21 requests (11 judged
positives, 17 known negatives and two unjudged) and 34 sources
for 17 Core17 requests (13 judged positives, 17 known negatives
and four unjudged). No tested input hit the token cap.

The table reports accepted judged positives / negatives and
warm verifier P95 for each view. Unjudged acceptances are
listed separately in the aggregate results.

| Reader and window | Core17 to News21 | News21 to Core17 |
| --- | --- | --- |
| 0.6B lead | 0 / 0 · 12.94 ms | 0 / 0 · 12.36 ms |
| 0.6B constraint window | 0 / 0 · 13.46 ms | 0 / 0 · 12.58 ms |
| 0.6B full | 0 / 0 · 21.78 ms | 0 / 0 · 20.78 ms |
| 4B lead | 1 / 4 · 50.80 ms | 2 / 1 · 48.79 ms |
| 4B constraint window | 1 / 2 · 53.14 ms | 0 / 0 · 49.35 ms |
| 4B full | 3 / 7 · 91.99 ms | 1 / 0 · 85.71 ms |

Giving the 0.6B reader the full document under its separately
recalibrated threshold does not recover any previously rejected
positive in either transfer. The exact reasons for individual
errors are unresolved. The 4B full pass recovers three positives
on News21 but also accepts seven known negatives and one unjudged
source. Its verifier P95 is about four times the 0.6B component
time across these matched window comparisons. Topic-level paired
intervals for increased positive acceptances include zero.

Short-window evidence loss is insufficient to explain the aggregate
failure of this fixed 0.6B scoring and calibration path. This reused
candidate slice does not support wider reads or replacing the small
reader as default. A held-out passage-labeled collection is needed
to distinguish model interpretation, model calibration, and evidence
visibility with direct labels.

[Aggregate results](results.json)

```sh
PYTHONPATH=src:. python -m experiments.followir.rescue_reader_diagnosis \
  CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS EMBEDDING RERANKER CHAT4B \
  PRIOR_RESCUE_RESULTS NEW_RESULTS
```
