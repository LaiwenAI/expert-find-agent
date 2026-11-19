# Compare live enterprise search routes with a blinded review packet

Use the 80 development requests and development-only inventory of
52,667 owned sources from the
[first-query benchmark](../enterprise-first-query-v1). Run each method
independently with the same cached, cross-precision qualification
parser, embedding model, exact GPU index and provisional evidence
verifier. Rotate the method order. Warm all methods before timing.

| Method | Candidate and source policy |
| --- | --- |
| Whole | Rank experts by their highest source score for the complete request. Verify their two highest-scoring sources. |
| Criterion RRF | For requests with an objective and prior qualifications, retain the first two whole-request experts and fill the remaining three positions using mean reciprocal rank over criterion projections. Verify the two best whole-request sources for each expert. |
| Criterion cover | Use the criterion-RRF candidates. Greedily select up to two owned sources by coverage of their best reciprocal source ranks across criteria. |

All paths use the whole-request policy for requests without a prior
qualification. Verify at most five experts with two owned sources each.
The shared verifier reads the same maximum source prefix and uses the
same threshold and truncation handling. A recommendation requires a
provisional witness for each criterion. The experiment records the
number of actual criterion–source checks, because the same cap does
not always produce identical counts. Timing covers parsing, query
encoding, retrieval, source selection and verification and excludes
queued load, transport and index construction.

Store the request, expert and exact citation offsets in a local
restricted execution trace. Bind each displayed passage to the
immutable corpus and validate the packet using the existing evaluation
pipeline. The packet hides method membership, rank and model grades.
It includes the original requirement and the prior qualifications
shared by all methods. Deduplicate exact evidence sets for the same
query and expert, while different sets for that expert get distinct
review cases. This packet grades the displayed evidence set. Its joint
constraint field does not identify which model-assigned citation
supports each individual criterion. Group identifiers in the packet
are development placeholders.

For blinded review, use separate copies for two reviewers and review
the original requirement even when the qualification list is empty.
Judge technical fit, joint hard-constraint support and personal
contribution separately. Judge only the displayed passages, check
every applicable prior qualification and cite the relevant portion
in the rationale. Missing information remains uncertain. Treat future
targets as past requirements only when the request explicitly demands
prior experience. Coauthorship or ownership of a source record does
not establish the named person's role in the work. Reconcile the
reviews and use the
[existing scoring protocol](../../docs/annotation-protocol.md).
Keep private materials and completed labels in the local workspace.

## Live results and review status

Slurm job 43013 on the RTX 5090 processed all 80 development requests.
Each method used 1,280 criterion–source checks with no clipped
comparisons. Model support is a screening decision.

| Method | Requests with provisional recommendations | Provisional slots / 400 | Changed ranked lists / 80 | First-query P50 | First-query P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole | 34 | 70 | 0 | 194 ms | 1,182 ms |
| Criterion RRF | 34 | 71 | 24 | 198 ms | 1,271 ms |
| Criterion cover | 34 | 71 | 24 | 198 ms | 1,272 ms |

Both criterion routes return the same evidence views. Each shares all
70 exact views from the whole-request route and contributes one new
view. The validator checks 71 distinct review cases and their exact
spans. With the two reviewer copies still blank, the paired strict
hit-rate difference is bounded between 0 and +0.25 percentage points
on this finite panel. That upper endpoint is one potentially
supported slot among 400. It is an identification bound, not
a sampling confidence interval. The new view may also increase
the adverse-group rate by up to one request out of 80 under the
development grouping.

The candidate routes alter the top five without materially increasing
the observed recommendation coverage, which stays at 42.5%.
The additional evidence view can be adjudicated blind. The sealed
123-group panel was unused.

[Aggregate results](results.json)

## Attribution signals in the development inventory

Audit the first 800 characters and the complete text of every
development source. Normalize the canonical name with NFKC, case
folding and whitespace removal, then test literal inclusion. Scan
for a small fixed set of English and Chinese author-contribution
heading markers. Match paper metadata by exact source ID and owner,
and verify its snapshot checksum.

| Source type | Records | Name in the first 800 characters | Name in complete stored text | Contribution heading in complete text |
| --- | ---: | ---: | ---: | ---: |
| Paper | 33,320 | 8 | 10 | 0 |
| Profile | 19,347 | 2,174 | 2,465 | 0 |

The paper sidecar links all 33,320 records. It contains 32,938 author
lists, 5,257 records with a DOI (5,106 distinct identifiers), and
zero verified personal roles. DOI coverage marks a possible lookup
path and does not show that the publisher makes contribution evidence
available. The literal-text audit misses aliases, implied identities
and other formats. A name occurrence also cannot establish a role.
These observations distinguish bibliographic identity and actual
personal contribution. In the realized blind packet, only one of
71 cases contains the canonical name in any displayed passage, and
no passage matches a contribution heading. They motivate a separate role-evidence
retrieval action when the ordinary profile or abstract lacks that
information.

## Prospective coverage and risk

For the separately sequestered 123-group candidate panel, an
illustrative 60% coverage floor and 5% conditional adverse-risk
ceiling with simultaneous 95% confidence require at least 85 issued
groups. At 85 issued groups there can be no adverse or unresolved
issued groups. Even issuing on all 123 allows at most one such group.
These are exact binomial design calculations with independent
group assumptions. With actual coverage 70% and adverse-or-unknown
risk 1%, prospective power for passing both gates is 25.6% at 123
groups, 77.2% at 250, and 87.5% at 300. These scenarios do not
establish the needed sample size for a paired quality effect. They
motivate measuring development coverage and planning an independent
follow-up if tight bounds are needed.

```sh
python -m experiments.enterprise_review_gate 123 --coverage-floor .6 \
  --risk-ceiling .05 --actual-coverage .7 --actual-risk .01
```

## Reproduce

Build the same authorized development projections described in
the [first-query benchmark](../enterprise-first-query-v1). The GPU
runner verifies both projection checksums and uses the same pinned
checkpoints. The local packet builder checks all source snapshots
against the private dataset manifest.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_routed_review run \
    DEVELOPMENT_QUERIES DEVELOPMENT_SOURCES CHECKPOINT_DIRECTORY PRIVATE_TRACE
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_routed_review build \
  PRIVATE_DATASET PRIVATE_TRACE PRIVATE_REVIEW_DIRECTORY
```

```sh
PYTHONPATH=src:. python -m experiments.retrieval.audit_enterprise_person_evidence \
  PRIVATE_DATASET NEW_AUDIT_JSON --packet PRIVATE_REVIEW_DIRECTORY \
  --paper-metadata PRIVATE_PAPER_METADATA_DIRECTORY
```
