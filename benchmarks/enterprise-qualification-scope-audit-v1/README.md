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

## Lexical cue controls

| Rule on 144 constructed excerpts | Correct prior qualifications / 36 | False prior qualifications |
| --- | ---: | ---: |
| Applicant cue in target excerpt | 24 | 0 |
| Applicant cue in target or nearest preceding colon heading | 36 | 12 |

The heading condition recovers the 12 remaining qualifications
and also accepts 12 future-work controls. Use the cue to prioritize
review, not as a hard eligibility decision. None of the 48
development clauses selected by the original parser contains the
published cue in the target or nearest preceding colon heading.
This describes the cue list and sample. It does not establish that
any particular development parser decision is incorrect.

## Prepared and verified review inventory

The packet has 98 excerpts from 25 requests: all 48 parser-selected
qualifications and 50 comparison excerpts. Both independently usable
blank forms validate against the same frozen source. No direct-gate
result or human label was used to construct this packet. The initial
`ws5` allocation was superseded by a verified `ws3` run. The model
predictions are bound to the unchanged packet in a private sidecar.

[Preparation counts](preparation.json).

## Model comparison on RTX 5090

Both tasks of Slurm array `44870` on `ws3` reproduce the exact
predictions. On the constructed controls the four-role classifier
accepts 24 of 36 actual qualifications and 20 nonqualifications.
The direct qualification prompt accepts 34 of 36 actual
qualifications and nine nonqualifications. Six of its nine errors
are future-work controls. The comparison holds the checkpoint and
marked source text fixed, with different prompts and label spaces.

On the 98 sampled development excerpts, the direct gate accepts six
of the 48 original qualifications and one of the 50 comparison
clauses. The classifiers disagree on 43 excerpts in 21 requests.
This shows sensitivity to how the scope question is framed.
The disagreements remain unjudged, so they cannot establish
parser error or justify changing a request before retrieval.

[Aggregate results](results.json).

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
# Bind model results to the original frozen form without changing it.
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_qualification_scope_review attach \
  PRIVATE_PACKET PRIVATE_AUDIT NEW_AGGREGATE PRIVATE_METHOD_SIDECAR
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_qualification_scope_review validate \
  PRIVATE_PACKET FIRST_REVIEW SECOND_REVIEW
```

Use two separate reviewer copies and independently adjudicate
unresolved or conflicting assessments before using the results.
