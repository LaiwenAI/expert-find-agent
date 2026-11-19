# Locate the loss from confusing duty wording

The [shared-vector repair experiment](../plos-open-role-vector-v1) returned
nothing on a second phrasing of the same exact-study requests. This study
traces source confirmation, role applicability and correct shortlist
availability, then tests a bounded wording intervention.

Use the same 197-article PLOS catalog and all 72 supported/unsupported and
source-present/withheld contrasts for each source clue. The source and author
labels are identical across wording banks. All embedding, decision and
reranker weights and operating thresholds are frozen. The instrumented
reference replays all prior quality and dispatch counts.

- **Primary-clause projection** removes a trailing clause introduced by
  “rather than,” “without,” “while,” or “before,” when a preceding clause has
  at least three tokens and no explicit negation or conditional marker. The
  projection is used to encode the requested duty. The semantic repair sees
  the full original text. This heuristic assumes the query asks for one duty.
- **Gate bypass** removes only the role applicability gate for diagnostic
  replay. It keeps source confirmation, the shortlist, and the author
  intersection. Use the observed false returns to assess the effect of bypass,
  rather than treating the extra coverage as a deployment gain.
- **Wording transfer** freezes the same projection on an earlier independent
  wording bank. Those requests often put another person's activity first.
  Publications and source clues still match the other conditions, so this
  isolates phrasing and does not test transfer to a new corpus.

## Results

The original stress wording yields 64/72 and 50/72 correct source confirmations
for excerpt and keyword clues. The correct duty is in the two-role shortlist
on 51 and 41 of those requests, respectively. The frozen role applicability
gate passes none. The loss therefore occurs before the student and reranker.

Prototype projection changes 62/72 supported stress requests and recovers
16 and 10 clean results without a verifier call. It leaves one wrong-source
recommendation on the withheld keyword view; duty projection cannot compensate
for incorrect source identification. Bypassing applicability with the
combined policy recovers 40 and 31 clean results but returns four and three
unsupported duties when the source is present. The bypass also returns two
wrong sources on the withheld keyword view.

The same projection changes only 3/72 supported queries in the independent
wording bank and yields unchanged clean counts of 8/72 and 5/72. It also
leaves all original NISO examples and their measured outputs unchanged.
The projection recovers explicit suffix contrasts on the construction bank,
while the separate wording bank shows the heuristic's coverage limit.
A broader parser needs to identify whose activity is asserted before mapping
it to a role. Simply expanding the accepted role inventory trades away the
controls seen here.

Slurm jobs 42900 and 42903 on an RTX 5090 agree on every shared quality,
dispatch and stage count. Job 42903 adds the independent wording bank.

Component timings are measured on a Slurm GPU node and exclude transport and
queues. Wording examples repeat across papers. These are publisher attribution
checks with exact study identities, not labels of industrial capabilities.

[Aggregate results](results.json)

## Reproduce

Reuse the [shared-vector evaluation](../plos-open-role-vector-v1) command and
pinned inputs. Select a fresh output path and append:

```sh
--scope-reference benchmarks/plos-open-role-vector-v1/results.json \
--external-phrases benchmarks/plos-credit-external-v1/boundary-phrases.json
```
