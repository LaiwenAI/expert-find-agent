# Measure certified two-source stopping and third-source completion

Use the two frozen topic-role panels and three-source byline joint plan
from the [live request replay](../live-joint-source-request-v1).
Keep that plan's first two sources. Open their original declarations
before assigning the final read.

- **Fixed triple** opens all three preassigned sources.
- **Staged byline** retains the fixed third source when the opening
  pair supports both roles. Otherwise it chooses a source that could
  supply a role to someone already supported on the other condition.
  Use the same minimax topic rank, rank sum and stable tie break.
- **Staged roles** ranks potential third sources first by matching
  indexed roles, then by unknown roles, then by nonmatching indexed
  roles. It still uses original declarations to verify the response.
- **Certified prefix** applies a stopping bound to the opening pair
  before deciding whether to fetch the third source. Unread role
  postings are not trusted by the bound.

The bound assumes exact contributor identities and at most one further
read within the fixed ranked pools. Any newly feasible two-source proof
must pair one already verified role with a compatible contributor in an
unread source. Assign that unread contributor the role optimistically
and rank such pairs by the same output key as actual proofs. Return
after two reads if no possible pair can improve the current winner, or
if there is neither a current winner nor any possible pair.
This certifies the identity and both cited sources. It does not certify
topic labels, role extraction from free text, or outcomes after the
pools expand. It checks only the best observed anchor for each contributor,
condition and proposed third source because an earlier anchor rank
dominates later ones for that fixed completion.

In 1,500 generated examples, 918 stops triggered. Each matched the
winner from reading every possible third source separately. In the
structured replay, each certified answer is checked against the
corresponding three-read staged answer. A separate warm local latency
pass compares the fixed triple and certified byline prefix from live
JSON parsing through serialization, alternating their order on each
request. It checks response bytes and includes the query encoder,
retrieval, actual local declaration reads and matching. It excludes
transport, concurrent queueing, cold storage and ingestion.

## Results on RTX 5090

Both tasks in Slurm array `44591` ran on ws5 and reproduced the same
nontiming aggregates. P95 reports the range across the two task orders.

| Panel | Fixed / certified declaration reads | Stop with proof / no possible completion | Byte-identical responses per run | Fixed / certified warm P95 |
| --- | ---: | ---: | ---: | ---: |
| PLOS, 160 requests | 480 / 331 | 72 / 77 | 160 | 9.63–9.68 / 9.61–9.66 ms |
| eLife headings, 51 requests | 153 / 102 | 16 / 35 | 51 | 10.01–10.06 / 10.03–10.08 ms |

The certificate avoids 149 and 51 reads, respectively, without
changing a response. The 11 uncertified PLOS requests use their third
read; six return a verified pair and five still abstain. The gain is
in declaration accesses. Local request P95 shows no material speedup
with warm local files. The read savings do not establish a speedup
with remote sources.

With the first two sources fixed, no tested staged policy changes
the chosen third source, with author identities alone or with clean,
masked or rotated role postings. The 78 PLOS and 16 eLife issued
responses stay identical to the fixed triple, including each person
and both citations. Publisher labels support 49 and 14 respectively.
On these panels the previously observed benefit of
[author-role postings](../live-joint-role-postings-v1) comes from
choosing the initial pair. Moving the same signal to the final read
does not reproduce that benefit.

## Reproduce

Restore the pinned publisher snapshots and embedding model used by the
live request replay. Run twice on a Slurm GPU, with fresh result paths
and `--order-offset 1` for the second run.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.joint_prefix_completion \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

[Replicated outcomes and timings](results.json).
