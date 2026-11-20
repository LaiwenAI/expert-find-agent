# Check whether scientific verification transfers to requirement wording

Use the existing [document-disjoint SciFact calibration](../scifact-criterion-calibration-v1)
from the [official corpus](https://aclanthology.org/2020.emnlp-main.609/).
Evaluate each annotated claim twice with its evidence held fixed:
as a direct claim and with the deterministic prefix
`Prior work must show that:`. The underlying relation labels
stay fixed. The unknown control uses the previously constructed
cited-abstract pairs without annotated evidence. This corpus has
appeared in earlier experiments.

Compare the current frozen enterprise Qwen3-Reranker-0.6B with
a constrained Qwen3-4B-Instruct-2507 reader. In the 4B route,
support, contradiction and unknown compete for the last-token
decision. Keep their presentation order fixed and change the
label tokens from 0/1/2 to 2/0/1 for an additional requirement
route. The agreement score takes the lower of the two
support-vs-next-class margins. Fit separate cutoffs on the
public training panel by maximizing supported claims subject
to the existing 5% ceiling for contradiction and unknown.
Freeze all cutoffs for the document-disjoint evaluation panel.
Zero-cutoff comparisons isolate wording and token remapping.

Evaluate these frozen requirement scorers on the same 40
[private method-blind enterprise views](../enterprise-owned-gap-sources-v1).
Reconstruct each query's objective and prior requirements
from the existing frozen parse. Score each criterion against
each displayed excerpt. For explicit same-object cues, require
one source group to cover every criterion. Otherwise allow
separate cited sources to cover the criteria. This measures
model-based technical completeness of an evidence bundle.
It does not assess verified individual contribution. No
recommendations change and the diagnostic does not include
human annotations. The evaluated criterion–excerpt pairs
differ from the gap-only retrieval checks of the prior study.

Timings use warm, rotated batches of up to 12 from the public
panel. The agreement rule needs both label mappings. Private
scores and identities stay in the authorized workspace. The
public output contains only aggregate counts.

## Results

Job `43504` completed on one RTX 5090. Fit uses 309
claim–abstract pairs. Evaluation uses 226 pairs with 97
annotated supports, 47 contradictions, and 82 constructed
unknown controls.

| Method | Supports accepted after calibration / 97 | Contradictions accepted / 47 | Unknown accepted / 82 | Batch P95 |
| --- | ---: | ---: | ---: | ---: |
| 0.6B, claim wording | 4 | 1 | 1 | 114.3 ms |
| 0.6B, requirement wording | 7 | 0 | 1 | 114.8 ms |
| 4B, claim wording | 43 | 3 | 3 | 574.2 ms |
| 4B, requirement wording | 41 | 2 | 1 | 574.9 ms |
| 4B, permuted digits | 53 | 3 | 1 | 574.9 ms |
| 4B, agreement score | 41 | 2 | 1 | Two passes |

For the reranker at zero cutoff, adding requirement wording
raises binary accuracy by 3.98 points (paired descriptive
95% interval +0.88 to +7.08), mostly by reducing acceptance
of negative controls. It still accepts 39 contradictions.
The 4B changes its three-way verdict in 116 of 226 cases
when the same requirement and excerpt use a different label
mapping. The permuted route retains more true supports after
calibration, with 3/47 false accepts on contradictions,
exceeding the training ceiling on this evaluation panel.

For the private 40 enterprise views, the original operating
point accepts 46 of 169 criterion–excerpt pairs. The
public-calibrated 0.6B threshold accepts zero. The 4B
requirement, permutation, and agreement routes accept 11,
3, and 8 individual pairs, respectively. All five policies
leave every full bundle unresolved. These are model verdicts
on fixed evidence views, with no human labels, and support
an audit of disagreement rather than changing production
recommendations. [Aggregate results](results.json).

A separate private-pair replay, job `43510`, reproduces the
accepted pair counts. The original ranker and the calibrated
4B agreement route share two supportive decisions. The
ranker alone accepts 44 pairs, the agreement route alone
accepts six, and both defer on 117. The differences cover
38 of 40 evidence views. A separate private ID order
prioritizes those cases without altering the two frozen
review forms or showing a model verdict on either form.
[Disagreement counts](disagreements.json).

## Reproduce

Use the same SciFact archive and open checkpoints as the earlier
public calibration. In the private compilation step, point to
the saved review directory and the frozen
[role-first query trace](../enterprise-role-first-query-v1).
Execute the second command within a Slurm CUDA allocation.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.scifact.requirement_verifier_transfer prepare \
  PRIVATE_DATASET PRIVATE_REVIEW_PACKET FROZEN_ENTERPRISE_TRACE PRIVATE_INPUTS
python -m experiments.scifact.requirement_verifier_transfer run \
  SCIFACT_ARCHIVE PRIVATE_INPUTS MODEL_ROOT NEW_AGGREGATES PRIVATE_DISAGREEMENTS
python -m experiments.scifact.prioritize_requirement_review \
  NEW_AGGREGATES PRIVATE_INPUTS MODEL_ROOT PRIVATE_CASE_ORDER PRIVATE_PAIR_FLAGS NEW_COUNTS
```
