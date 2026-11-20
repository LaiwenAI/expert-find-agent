# Audit whether parsed clauses are prior applicant qualifications

Reuse the 25 multi-condition development requests and exact
qualification offsets from the
[first-round enterprise comparison](../enterprise-joint-first-round-v1).
The audit includes every clause originally marked as a prior
qualification, plus up to two originally unselected clauses
from each request. Prefer previously defined applicant cues when
sampling unselected clauses, with a fixed hash tie-break.

Compare full-request four-role classification with a direct binary
prompt on the existing 144 [constructed controls](../enterprise-requirement-roles-v1).
These controls include historical performance by a different party,
future targets, existing conditions, and prior applicant evidence.
Reuse the former full-context predictions on controls at their
validated offsets. On development compare the actual first-round
parser's frozen marks with the direct binary model on the same
sampled excerpts. Both use the Qwen3-4B-Instruct-2507 checkpoint,
but the prompt, decision labels and execution path differ.

The development packet shows the original request and exact
excerpt without candidate experts, cited evidence, method or
qualification predictions. The reviewer distinguishes a prior
applicant requirement from future deliverable, operating
condition, background, mixed or uncertain. A prior qualification
requires a past achievement, experience or credential applicable
to the applicant. Future project metrics and past claims about
a vendor or issuer do not themselves impose such an eligibility
condition. The original request text and character offsets remain fixed.

No method reviews its own outputs. Model disagreement locates
cases for independent review; it is not an estimated error rate
or authority to remove a requirement. The 25-request group and
unselected clauses are purposive. The audit does not estimate
qualification recall across all requests or retrieval effects.

## Prepared and verified review inventory

The packet has 98 excerpts from 25 requests: all 48 parser-selected
qualifications and 50 comparison excerpts. Both independently usable
blank forms validate against the same frozen source. No direct-gate
result or human label was used to construct this packet. The direct
gate was submitted as Slurm array `44835` on `ws5`. Its model results
are pending and are outside this preparation report.

[Preparation counts](preparation.json).

## Reproduce

Use the pinned development trace, private original controls trace
from the requirement-role benchmark, published control suite and
official open model checkpoint. Keep all raw development records
and reviewer copies outside Git. Run the audit inside a Slurm GPU
allocation and use fresh output paths.

To freeze the same form without model results:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_qualification_scope_review freeze \
  DEVELOPMENT_QUERIES PRIVATE_FIRST_ROUND_TRACE FIRST_ROUND_AGGREGATE \
  CONTROLS PRIVATE_PREVIOUS_CONTROLS_TRACE PRIVATE_PACKET
```

To run the direct gate or create another, separate packet bound to its
private predictions:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_qualification_scope_audit \
  DEVELOPMENT_QUERIES PRIVATE_FIRST_ROUND_TRACE FIRST_ROUND_AGGREGATE CONTROLS \
  PRIVATE_PREVIOUS_CONTROLS_TRACE QWEN_CHECKPOINT NEW_AGGREGATE PRIVATE_AUDIT
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_qualification_scope_review build \
  PRIVATE_AUDIT NEW_AGGREGATE PRIVATE_PACKET
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_qualification_scope_review validate \
  PRIVATE_PACKET FIRST_REVIEW SECOND_REVIEW
```

Use two separate reviewer copies and independently adjudicate
unresolved or conflicting assessments before using the results.
