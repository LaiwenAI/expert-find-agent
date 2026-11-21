# Train nomination for accessible citations under a fixed read budget

The [TREC nomination waterfall](../trec-nomination-waterfall-v1)
found that many judged expert–topic positives had accessible
support in the historical graph, but few survived nomination and
source allocation together. Compare nomination targets on the
49 judged TREC 2006 Enterprise topics while keeping source menus
and the reuse-first dispatch algorithm fixed for any given owner.

## Protocol

For each held topic, remove all of its qrels from the discovery and
training graphs. Each training topic separately masks its own links
when constructing examples. Retain the same first 60 ranked people
from the historical expert-first graph as the nomination universe,
and nominate exactly 20. Each owner uses their original 16-source
menu. Topic-title and description similarity is the frozen lexical
feature used by the earlier study.

Both fits use the same eight inputs. They encode candidate rank,
source inventory size, distinct historical topic support within the
menu, strongest lexical topic link, first-source and top-three
votes, and two shared-owner counts. The first fit targets grade-2
expert relevance. The second requires both grade-2 expert relevance
and a grade-2 held-topic citation within the historical source menu.
Other rows receive zero observed credit, including unjudged rows.
The target concerns recoverable judgments in this archive.

| Route | Nomination rule |
| --- | --- |
| Expert-first | Keep the original first 20 owners and order |
| Learned topic | Choose 20 of 60 using the relevance fit |
| Learned witness reorder | Use the evidence-access fit to reorder the original 20 |
| Learned witness expand | Choose 20 of 60 using the evidence-access fit |
| Reserved witness | Keep the original first 15 in order, then fill five slots by the evidence-access fit |
| Reserved witness, shared source | Keep the same 15. Prioritize the remaining owners who share at least one of their first four sources with the first four of any retained owner, then fill any empty slots by fitted score. |

The two reserve arms were added after the initial four-arm
comparison. They are exploratory follow-ups on the same 49
topics. The reuse filter uses the historical source menu and
owner links only.

All routes use reuse-first dispatch and are capped, topic by topic,
by the **realized** distinct reads and owner–document checks of the
original expert-first route. A check reveals the held source grade.
Unknown pairs remain unconfirmed. A supported slot requires grade 2
for the cited source and the expert. The read oracle uses the same
owner and source menu with the stated read cap and perfect label
information. It is an upper bound under that read cap and does
not additionally enforce the check cap.

## Results

| Nomination | Grade-2 experts in menus | Experts with reachable labeled evidence | Read oracle slots | Reads | Checks | Supported slots / issued |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert-first | 304 | 76 | 49 | 103 | 285 | 9 / 9 |
| Learned topic | 296 | 73 | 49 | 97 | 273 | 8 / 8 |
| Learned witness reorder | 304 | 76 | 49 | 99 | 278 | 10 / 10 |
| Learned witness expand | 249 | 71 | 52 | 81 | 268 | 3 / 4 |
| Reserved witness | 276 | 76 | 52 | 98 | 272 | 8 / 9 |
| Reserved witness, shared source | 293 | 76 | 50 | 85 | 281 | 7 / 8 |

Evidence-trained reordering preserves the nominee set, recovers one
additional supported slot and uses four fewer reads and seven fewer
checks. Its paired supported P@5 gain is +0.41 percentage points,
with a 95% topic-bootstrap interval of 0.00 to +1.22. The interval
includes zero at the observed resolution.

Full evidence-driven replacement changes 417 of the 980 nomination
slots. It raises the read oracle ceiling from 49 to 52, while the
replayed supported result falls from nine to three. None of its
supported results comes from an added nominee. Reserving the first
15 limits the loss, and the shared-source variant promotes one
supported added nominee. Neither reserve improves on the original
expert-first route in this replay.

The ceiling-to-realized gap increases when candidate selection
optimizes source reach independently of the dispatch it will receive.
A stronger comparison needs to measure joint nomination and source
allocation under the same read and check caps. The current numbers
are deterministic archive replay over historical associations and
partial judgments. They do not measure live corpus search, source
access time, or verifier accuracy.

Slurm tasks 46792_0 and 46792_1 on ws3 reproduce all aggregates
and the per-topic comparison signature. The original four routes
also reproduce the initial four-arm comparison.

## Reproduce

Use the official
[TREC 2006 Enterprise topics and qrels](https://trec.nist.gov/data/t15_enterprise.html)
and the original input hashes. Run on a Slurm compute allocation,
then repeat with `--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.trec_enterprise.supported_owner_nomination TREC_DATA NEW_RESULTS
```

[Complete aggregate results](results.json)
