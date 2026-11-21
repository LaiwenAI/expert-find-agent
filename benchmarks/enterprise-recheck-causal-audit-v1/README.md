# Replay the clause that changes one enterprise recommendation

Recover the exact non-timing signature of the [layout
comparison](../enterprise-recheck-layouts-v1) before analyzing
differences. Keep the 80 original requests, 52,667 sources, parser
checkpoints, dense retrieval and packed guarded verifier fixed. On
every affected request, replay each relevant full label sequence
through the same search. Then change one disputed qualification label
at a time and repeat with the same text and search settings.

## Results

Slurm job 46540 reproduced the original search signature. The
variable-length layout adds four qualification instances and removes
one across four requests. The four-row layout adds two instances
across two requests. Both layouts omit the same single recovery
recommendation.

For both layouts, adding the same disputed criterion to the reference
makes that recommendation disappear. Reverting only that criterion in
the alternative recovers the original output. The expert no longer
receives one of the two recovery windows when the criterion is present.
A separate fixed-view probe also returns a nonpositive margin for the
new criterion against the original excerpt. The intervention
establishes the cause of the output change in this frozen pipeline.
Whether the clause belongs to applicant eligibility, and whether the
recommendation was well supported, remain questions for independent
review.

All six disputed source anchors already exist in the [154-clause
packet](../enterprise-scope-review-union-v1). The consequential anchor
is outside the first 16-case batch. Its existing alias is linked
privately to one separate evidence card, with no duplicate clause
annotation needed. The evidence card binds the exact original source
span. Both review tracks remain unjudged.

[Aggregate replay and review counts](results.json)

## Reproduce

Run the replay on Slurm CUDA with the inputs from the preceding
[layout comparison](../enterprise-recheck-layouts-v1#reproduce).
Use the preceding aggregate as the exact signature reference.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_recheck_causal_audit \
  QUERIES CORPUS ROLE_WINDOWS VECTORS META MODELS PACKED_REFERENCE LAYOUT_RESULTS SUMMARY PRIVATE_TRACE
PYTHONPATH=src:. python -m experiments.retrieval.export_enterprise_recheck_causal_review \
  QUERIES PRIVATE_DATASET SUMMARY PRIVATE_TRACE PRIVATE_REVIEW
```
