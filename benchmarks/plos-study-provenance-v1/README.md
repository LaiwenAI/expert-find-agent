# Bind indexed contribution evidence to the frozen report content

Reuse the 59 authentic PLOS study pairs from the
[initials collision benchmark](../plos-study-bound-initials-v1).
Index recognized legacy contribution headings, their credited
initials, and the character offsets of their source sentences.
Evaluate the requests with controlled role wording and an
explicit report alias. The expected content digest comes from
the frozen original study record.

Each pair supplies five views. Keep both authentic reports for the
supported and unsupported views. For the missing-report view, make
only the other report available. For the substituted-report view,
retain the requested alias but supply the other report's content.
For the unknown-role view, request a role outside the closed
vocabulary. The simulated substitutions test alias-to-content
consistency and do not estimate a deployment failure rate.

Compare a pooled lookup across all visible notes, a lookup
restricted to the report alias, and an alias lookup that also
verifies the frozen digest. Unavailable sources, unknown roles
and mismatched digests remain unresolved. The pooled lookup may
attach two witnesses from different reports.

## Results

Two CPU runs on ws5 reproduce all non-timing counts.

| Lookup | Correct / 295 | False support | Wrong-source witnesses | Local lookup P95 |
| --- | ---: | ---: | ---: | ---: |
| Pooled | 118 | 118 | 236 | 0.88 microseconds |
| Report alias | 236 | 59 | 118 | 0.69 microseconds |
| Report alias and digest | 295 | 0 | 0 | 1.07 microseconds |

Restricting the lookup to the requested report fixes the original
59 cross-study attribution errors. The alias still accepts all 59
substituted reports. Checking the frozen content digest leaves
these unresolved and retains all 59 supported views. The
digest-checked route also leaves all 59 missing reports and all 59
unknown-role requests unresolved.

The controlled role labels share the legacy declaration grammar
with the compiler. This is a provenance and abstention test, with
no independent estimate of role interpretation or open retrieval
accuracy. The timings cover an in-memory lookup and hash over
short notes. They exclude parsing, storage, discovery and model
calls. The frozen digest must come from a trusted acquisition.

[Aggregate results](results.json)

## Reproduce

Build the same private fixture described in the
[initials collision benchmark](../plos-study-bound-initials-v1).
Only aggregate results and digests need to leave the workspace.

```sh
PYTHONPATH=src:. python -m experiments.credit.study_provenance_checks PRIVATE_FIXTURE NEW_RESULTS
```
