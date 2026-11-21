# Count the effective contributor graph and stress partial role rows

Use the original PLOS XML fixture and frozen eLife projection from
the live publisher task. Count people with checksum-validated
identifiers, distinct same-person source pairs and connected
components among linked sources. Reuse the original query
constructor to count topic pairs, fully tagged witness pairs, and
per-person query participation. A person can contribute to several
queries, and a query can have more than one eligible person.

To simulate a nonempty but incomplete role posting, remove one role
from a deterministic 50% or 100% sample of multirole author–source
rows. The hash depends only on source and author, not query labels.
Use the original topic tags and complete declarations offline to
enumerate each query's eligible proofs over its whole publisher
catalog. Count queries still having at least one original supported
pair under the incorrect trusted-negative interpretation. The
conservative partial-row interpretation suppresses no original
proof on this candidate-eligibility test. It may admit other
possibilities for verification. This experiment does not select
three reads, measure the ranker, or model real omission rates.

Both hash seeds on ws3 produce byte-identical counts. Recover the
inputs referenced by the live publisher benchmark and run:

```sh
PYTHONPATH=src:. python -m experiments.credit.declared_role_evidence_audit \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 ELIFE_PROJECTION \
  benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

[Aggregate result](results.json)
