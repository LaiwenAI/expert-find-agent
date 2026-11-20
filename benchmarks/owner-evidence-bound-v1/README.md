# When does a third source change a declared-role match?

Test whether reading a declaration before choosing the next source improves
a first-query expert match. Reuse the 160 PLOS ONE and 51 eLife heading
requests, frozen XML manifests, and mixed 452-paper collection from the
[live joint-source benchmark](../live-joint-source-request-v1). Each request
specifies two topic–role conditions. An output must cite two distinct
declarations that contain the requested roles for the same person with a
checksum-valid ORCID. A **known** output also matches both publisher topic
annotations. An issued output with missing topic annotations remains
unresolved.

All arms share live lexical/Qwen3-Embedding-0.6B retrieval and a
declaration verifier. The per-topic pool has 64 or 128 articles. A stable
hash determines which sources expose personal role fields before
reading. Byline identities are always available. Hidden role fields and
contributors without parsed roles are treated as unknown. Full postings
measure the potential of declared fields under this snapshot.

| Arm | Selection and read budget |
| --- | --- |
| Joint byline | Static joint plan with only author identities, three reads |
| Static two | First two articles of the partial-role static plan, two reads |
| Adaptive two | Same first article as Static two, then choose a second using the observed roles and the partial index, two reads |
| Static three | Partial-role pair and a complementary article selected in one batch, three reads |
| Adaptive forced | Same first two articles as Static three, then choose a third using observed roles, three reads |
| Adaptive with stop | Same as Adaptive forced, with permission to skip the third read when no remaining article can improve the verified result |
| Full roles | Static three with complete access to the published structured role fields |

For the stop check, consider each unopened article in the ranked pools.
Allow any byline contributor with an unknown relevant role to fulfill it
and join that possibility only to an already verified witness for the same
person and the other condition. If no pair improves the current result
under the verifier's ranking (worst topic rank, sum of ranks, identifier
and citations), a third read cannot change its output. A scope check
replays every skipped alternative using its true declarations and checks
the returned person and both citations. This guarantee applies to the
fixed first two reads, the current ranked pools, their declared roles,
and at most one extra read. Source ranking, unreported roles and other
forms of eligibility remain outside that guarantee.

## Results

Slurm jobs 45566 and 45568 completed on ws3, each with one CUDA GPU. The
second adds the two-read adaptive arm and reproduces the first job's
other nontiming results. The table shows known outputs by depth and
nominal role visibility. The forced and stopped third-read paths returned
identical outputs, as did the stopped and static three-read paths.

| Depth | Publisher | Index visibility | Static two | Adaptive two | Static three / adaptive with stop | Third reads skipped |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | PLOS ONE | 0% / 25% / 50% / 100% | 40 / 40 / 51 / 68 | 40 / 40 / 51 / 68 | 43 / 42 / 52 / 68 | 155 / 158 / 159 / 160 of 160 |
| 64 | eLife | 0% / 25% / 50% / 100% | 14 / 14 / 11 / 17 | 14 / 14 / 11 / 17 | 14 / 14 / 11 / 17 | 51 of 51 at every cutoff |
| 128 | PLOS ONE | 0% / 25% / 50% / 100% | 42 / 28 / 55 / 85 | 42 / 28 / 55 / 85 | 45 / 31 / 56 / 85 | 154 / 154 / 156 / 160 of 160 |
| 128 | eLife | 0% / 25% / 50% / 100% | 14 / 30 / 31 / 46 | 14 / 30 / 31 / 46 | 14 / 30 / 31 / 46 | 51 of 51 at every cutoff |

The stop check saves 205–211 of 633 fixed three-read fetches per
depth/visibility pair (32.4–33.3%), with 251,348 alternative-read checks
across both depths and all visibility settings. Its value over a fixed
two-read budget is limited to recovering 1–3 known PLOS matches at
partial visibility. eLife and the full-visibility arms retain every
observed match with the static two-read plan. Adapting the second read
never changes the known-match count in either collection.

At the 50% cutoff, moving from partial to full postings increases known
matches from 52 to 68 on PLOS and from 11 to 17 on eLife at depth 64.
At depth 128 the corresponding changes are 56 to 85 and 31 to 46. At full
visibility and depth 128, the pool contains annotated proofs for up to
115 PLOS and 49 eLife requests. The verifier selects annotated proofs
for 85 and 46. Topic annotations are incomplete, so the difference
includes candidates with unjudged topics.

P95 combines a shared live query parse and retrieval time with the
warm local arm's planning, declaration reads, verification and JSON
serialization. The offline correctness checks and index initialization
are excluded. The stopped path lowers measured P95 by only 0.04–0.27 ms
against the fixed three-read plan in the seven-arm replay. These reads
open local structured records. The experiment does not measure
retrieval of publisher full text, queueing, model verification of
natural-language qualifications, or enterprise relevance.

## Reproduce

Restore the frozen XML snapshots from the linked inventories and the
pinned embedding checkpoint. Use a Slurm compute allocation with a
CUDA GPU. Repeat with depth 128 and order offset 1.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false python -m experiments.credit.owner_evidence_bound \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json \
  NEW_RESULTS --depth 64 --order-offset 0
```
