# Allocate verification to new people or deeper evidence on held TREC topics

Use the 49 judged topics from the [TREC 2006 Enterprise Track](https://trec.nist.gov/data/t15_enterprise.html).
The [prior abstention replay](../trec-abstention-rescue-scope-v1) allowed
the evaluated topic's pool membership when nominating candidates.
This replay constructs the entire candidate and source list without
that information. For each topic, the other 48 supply the only
discovery graph. The query's original title and description are
encoded with frozen Qwen3-Embedding-0.6B. The
[historical index](../trec-joint-topic-allocation-v1) ranks documents
by reciprocal-rank accumulation over semantically related topics.
Score every owner in the 256-document result by the best three
historical source-link values. Compare all historical associations
and associations supported at grade 2 at least once.

Check the top source of each of the first five candidates. Apply a
follow-up budget of 2, 4 or 8 candidate–document checks. Compare
deepening the first five candidates, widening to ranks 6–10 or 6–20,
alternating across scopes, and choosing from both scopes by estimated
historical value. All policies prefer unconfirmed candidates, use
the same historical value and discount it by tests already spent on
the candidate. Follow-up also runs when the initial pass found some
support, so the report includes both complete coverage and recovery
after abstention. Each association is checked at most once. Charge
both association checks and unique documents.

The simulated reader receives the held-topic label only after
choosing an association. A missing grade remains unknown and yields
no recommendation. Compare a grade-2 support oracle and a sensitivity
condition with 90% true-positive and 5% false-positive probabilities
on judged associations. The latter shares the random draw across
owners of the same source. Expert grade 2 and cited-document grade 2
are both necessary to credit a grounded slot. Score 1 remains a
neutral grade under the official [track definition](https://trec.nist.gov/pubs/trec15/papers/ENT06.OVERVIEW.pdf).

These experiments measure allocation over archived pooled
associations under withholding. The original W3C documents or
official submitted rankings would be needed to evaluate a live
retriever and verifier with this protocol. The matched-check
diagnostic uses only paired replays with the same realized check
counts. Resampling averages correlated runs within a topic first.

## Results

Both tasks in GPU array 45650 on ws3 reproduce every aggregate.
At eight follow-up checks per topic there are 392 additional
association checks in every arm. Rank 20 supplies the outer limit
for widening.

| Historical links | Reader | Deepen first five | Widen to ranks 6–20 | Alternate | Historical value |
| --- | --- | ---: | ---: | ---: | ---: |
| All pooled associations | Grade-2 oracle | 2 | 3 | 4 | 3 |
| Grade-2 history | Grade-2 oracle | 1 | 8 | 5 | 8 |
| Grade-2 history | Correlated-error scenario | 0.92 | 7.17 | 4.83 | 7.17 |

Cells count grounded recommendation slots over 49 topics. For the
grade-2 history, widening raises supported P@5 by 2.86 percentage
points versus deepening under the oracle (paired topic bootstrap
95% interval +0.41 to +6.12). The four-check result has a confidence
interval spanning zero. Under the correlated-error setting the
eight-check change is +2.55 points (+0.37 to +5.75). The heuristic
that chooses from both scopes gives the same counts as widening
in that setting.

The historical grade-2 index has six reachable known supported
experts within the first five per topic and 43 within the first
twenty. Fourteen topics have some reachable support only outside
the first five. Even with scope widening, 533 of the 637 initial
and follow-up pairs have no held-topic grade. The remaining
unrecovered labels can reflect historical access, shortlist selection,
or routing. A missing judgment gives no evidence of a wrong match.
Use the eight-check widening result as a conditional comparison
under the historical index. Absolute quality and deployment latency
need a live document collection and independent verification.

## Reproduce

Restore the official topics and qrels and the frozen embedding
checkpoint from the historical index experiment. Use a Slurm CUDA
allocation and a fresh result path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.trec_enterprise.trec_historical_rescue \
  TREC_DATA QWEN_EMBEDDING NEW_RESULTS
```
