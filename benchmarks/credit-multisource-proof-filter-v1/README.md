# Filter contributor checks without treating missing bylines as negative evidence

The [MathOverflow service replay](../community-expected-support-service-v1)
found that a structural filter halves verifier checks under known
single-owner answer metadata. Test related filtering on multi-author
article records and requests spanning two or three specified articles.
A person must satisfy every source-scoped contribution clause with
the same recorded identity. This benchmark starts after source and
clause scope selection.

Use the [pinned PLOS projection](../plos-scoped-search-v1) and
[independently sampled eLife confirmation projection](../elife-source-guard-confirmation-v1).
Evaluate the cross-paper universe of authors with resolvable identifiers
and a recorded declaration. In eLife, use articles with a complete
recorded contribution declaration. Treat unresolved identity linkage
as outside the evaluation universe. Author role labels from the
publisher are gold. Render declaration text by concatenating the
corresponding CRediT definitions, and render queries with separate
example phrases from the same [NISO collection](https://zenodo.org/records/18421449).
This controls role wording and does not estimate the performance of
extracting contributions from original publication prose.

For each publisher and source count, find article sets linked by at
least one identity. For each accepted set, freeze one positive
request and one matched negative in which one clause changes to a
different role declared somewhere in that article. The new conjunction
must lack a common supporting person. Use two clauses in the first
source and one for each subsequent source. Deterministically pick
up to 12 source-disjoint linked sets per panel. Panels can reuse
articles and contributors across source counts.

Compare five routes.

| Route | Eligible declaration checks |
| --- | --- |
| Scoped dense | Every author declaration in the clause's own article |
| Complete filter | Only registered identities present in every source |
| Stale filter | Strict intersection after hiding one real link on a single article |
| Guarded filter | Intersect only articles with complete metadata. Verify all declarations on the flagged article if a common identity remains eligible in the other articles. |
| Guarded staged | Verify eligible declarations on the complete articles first. On the flagged article, only verify people who passed all the first-stage clauses. |

The injected link and flagged source are chosen by hashes of article
and contributor identifiers, without access to request role labels.
Keep all authoritative declaration rows available for verification.
The guard assumes that the stale article is flagged and all other
bylines are authoritative. It issues only when a person passes
every clause, using the identity attached to each checked declaration.

Use the same Qwen3-Reranker-0.6B and margin threshold across routes.
First compare the planned subsets with the scoped-dense decisions
held fixed. Then compare independent live inference to capture
batch-shape effects. Time the verifier calls and prompt preparation
after routing, in a resident sequential GPU loop. Count actual
unpadded prompt tokens and verifier forwards. The benchmark does
not include live article retrieval, byline refresh, network,
queueing, or online source identification.

## Results

Both reversed-order runs on ws1 (RTX 4090, Slurm array 46934)
reproduce every work count and output signature. The first four
routes reproduce the pilot outcomes from array 46929. Each family
has one positive and one matched negative request. Entries in the
four central columns count attempted clause–declaration checks.

| Publisher, sources | Families | Scoped dense | Complete filter | Flagged, guarded parallel | Flagged, guarded staged | Known positives found by dense and both guards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PLOS, two | 8 | 114 | 60 | 114 | 84 | 3 |
| eLife, two | 12 | 262 | 84 | 262 | 131 | 4 |
| PLOS, three | 1 | 28 | 8 | 14 | 6 | 0 |
| eLife, three | 3 | 60 | 24 | 32 | 18 | 0 |

Complete filtering preserves every issued author set and uses
176 rather than 464 checks. In the two-source eLife panel, resident
verifier P95 falls from 71.7–72.0 ms to 21.9–22.3 ms. The stale
intersection excludes at least one necessary check for 22 gold
person–request slots. It removes six correct author returns that
scoped dense found, as well as one incorrect output on a negative
request. Falling issuance therefore overstates its safety.

Both guard routes preserve the same issued author sets under frozen
and live margins. Staging reduces work from 422 to 239 checks
relative to the guarded parallel plan. For two-source eLife, P95
falls from 71.5–71.9 ms to 39.6–39.8 ms, after the flag is already
available. Staged scheduling can increase GPU forwards while doing
less work, as on two-source PLOS (25 versus 20 forwards). The
staged guard's ability to avoid checking a person only applies
relative to the fixed verifier's decision on the earlier clauses.

The verifier recovers only seven of 24 constructed positive
requests. It recovers none of the four three-source positives.
The equality result applies to saved verification work, with
absolute recall and generality still limited. The corruption
scenario models a missing link that is correctly flagged. Real
cache inconsistency, unrecognized author identities, and prose
interpretation require separate evaluation. [Aggregate results](results.json).

## Reproduce

Use the pinned input digests in the aggregate JSON, including the
[original PLOS projection instructions](../plos-scoped-search-v1)
and the [archived eLife confirmation manifest](../elife-source-guard-confirmation-v1/manifest.json).
Run with a GPU under Slurm and a fresh output path.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.multisource_proof_filter \
  PLOS_PROJECTION ELIFE_PROJECTION NISO_EXAMPLES QWEN_RERANKER NEW_RESULTS
```

Repeat with `--reverse` and a distinct output path.
