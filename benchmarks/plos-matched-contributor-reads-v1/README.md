# Compare joint and sequential nomination with matching role information

The [mixed-publisher comparison](../mixed-publisher-joint-v1)
shows that joint nomination with indexed personal roles
recovers more label-supported proofs than a byline-only route.
The comparison changes both access to role metadata and planning.
This replay separates those factors on the 13 new PLOS groups from
the [frozen role service evaluation](../plos-frozen-role-recheck-service-v1).

All routes share 440 sources, top-64 hybrid lists, supplied role
labels, and three reads of distinct local declaration files. The
recorded contributor inventory consists of ORCID-resolved people
with a declared role. The byline routes can see these identities
but the role values are available only after a read. The two
role-aware routes share the full personal-role metadata. This is
a controlled comparison in which indexed fields agree with the
declarations. Neither route trains on the test groups.

| Route | Initial selection | Third declaration |
| --- | --- | --- |
| Topical | First distinct source from each clue list | Highest-ranked unused source |
| Byline sequential | Same as topical | Prioritize a source linked by identity to an observed contributor with the first role |
| Byline joint | First compatible pair joined by contributor | Prefer a source with more potential cross-list links |
| Role sequential | First distinct source listing each role, chosen independently | Prefer a source with a matching role for an observed contributor |
| Role joint | First pair joined by contributor and role | Prefer a third source with more eligible same-contributor links |

All methods choose the final proof by the same worst citation rank,
rank sum, and stable identifiers. A correct proof uses the two
designated sources and one recorded contributor with both requested
roles. Evaluate the keyword view on the intact catalog and after
withdrawing each designated paper. A replacement paper is wrong
for this exact-source task even if its contribution declaration is
valid. The role contrasts are paired with the positive requests.

Report natural issuance, paired effects with bootstrap by paper
group, and retrospective equal issuance. For the latter, truncate
the proposed proofs to the smaller issuance count by worst citation
rank and rank sum before using a request-based tie-break. The two
routes may then issue on different requests. Also report the result
restricted to the requests where both routes already issue.

Timing re-encodes each request and includes ranking, planning, local
JSON file access and response serialization. Source and query
initialization and HTTP transport are outside the measurement. The
fixed files and rotated order keep I/O and order effects comparable.

## Results

Both ws6 RTX 4090 tasks in Slurm array 47384 reproduce every
read plan, proof, and accuracy count in opposite traversal orders.

| Policy | Intact exact / issued (78 requests) | Wrong substitutes (156 withdrawals) | Intact P95 (ms) |
| --- | ---: | ---: | ---: |
| Topical | 21 / 25 | 0 | 17.46–17.57 |
| Byline sequential | 36 / 40 | 5 | 17.61–17.69 |
| Byline joint | 39 / 43 | 24 | 17.66–17.68 |
| Role sequential | 39 / 43 | 8 | 17.68–18.08 |
| Role joint | 39 / 54 | 50 | 17.40–18.03 |

Each route reads 234 intact and 468 withdrawal declarations.
Among the 43 intact requests answered by both role routes, each
has 39 exact proofs. At the same panel-wide quota of 43, the
rank-truncated role joint route has 36 exact proofs, compared with
39 for role sequential. It adds eleven intact issuances at natural
coverage, all on cases with no exact proof under the role contrast.
The byline joint control also attains all 39 exact proofs, but has
three times as many withdrawal errors as role sequential. The
intact and withdrawal suites distinguish proof retrieval from
substitute risk. The pre-specified tests on these 13 groups use
keywords tied to designated sources and supplied CRediT roles.
They do not estimate topical expert relevance or human-verified
qualification on open enterprise queries.

Convenient source and role cues can let a sequential control find
all exact proofs. Wider compatible pairing then mainly increases
misattribution to substitute sources. Independently established
source identity takes priority in that setting. Joint planning
remains an experiment on open topic requests, where evidence
may span several compatible but differently ranked sources.

[Both complete executions](results.json)

## Reproduce

Restore the [frozen 2024 and 2025 PLOS XML inventories](../plos-frozen-role-recheck-service-v1)
with matching digests, the two original eLife exclusions, the base
PLOS projection, the NISO role bank, and Qwen3-Embedding-0.6B.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.plos_matched_contributor_reads \
  ELIFE_CONFIRMATION ELIFE_EXTENDED BASE_PLOS_MANIFEST BASE_PLOS_PROJECTION \
  OLD_PLOS_MANIFEST OLD_PLOS_XML FRESH_PLOS_MANIFEST FRESH_PLOS_XML \
  NISO_EXAMPLES ROLE_TRAINING QWEN_EMBEDDING NEW_OUTPUT
```

Run in an allocated Slurm GPU job and repeat with `--reverse`.
