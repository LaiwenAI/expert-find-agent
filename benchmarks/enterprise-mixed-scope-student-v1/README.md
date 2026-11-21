# Train on requests with mixed qualification scope

The [shared-encoding pilot](../enterprise-shared-scope-student-v1)
reduced the cost of extracting clause features, but the fitted head
largely stopped marking qualifications on enterprise requests.
This experiment tests whether training on several roles in the
same request helps the head learn to localize a qualification.

Start from the 168 bilingual
[scope-transfer controls](../enterprise-scope-transfer-v1). Form
three-section requests inside a single language, split, and
layout. Rotate topic order and bind each unchanged target clause
to prior applicant qualification, future work, current site
conditions, or third-party background. A request can have
zero, one or two prior qualifications. Within each topic order
there are four sets of paired alternatives. A pair changes
the governing language in one section and leaves its literal
target and both other sections unchanged. Original offsets
identify which text is classified. The data generator preserves
the original separation of layout templates and topics across
training, calibration and test. Multiple views share topics
and templates. The sample size in anchors is not the number
of independent requests.

| Route | Ridge fitting examples | Features |
| --- | --- | --- |
| Single | 96 original single-target anchors | Shared forward and reverse anchor and document vectors |
| Mixed, matched count | 96 mixed anchors with the same 25% positive share | Same |
| Mixed | All 576 mixed training anchors | Same |
| Mixed + neighbors | All 576 mixed training anchors | Same plus the two adjacent clause vectors on each side, when present |

All routes use the same frozen Qwen3-Embedding-0.6B model and
two encoding rows for each request. The neighbor features
use hidden states from the same forwards. They do not add model
calls. Use a separate 48-request, 144-anchor mixed calibration panel to fix
each high-confidence cutoff at the largest absolute error score.
The held mixed panels test exact qualification sets and whether
all variants of a counterfactual anchor are correct. The 72
heading controls are a separate one-anchor-per-case check using
paired language and target identifiers.

On the 80 private development requests, score every clause and
compare with the whole-block 4B parser's output. These are model
disagreements, as the private projection contains no human
labels. The fallback simulation replaces the student's entire
qualification set with the 4B set when any clause falls below
the cutoff. It adds synchronized parser times, without retrieval,
serving, or a proven quality guarantee.

The replay keeps the distinctions between document exactness,
anchor accuracy and counterfactual consistency. On this design,
a rule that always predicts no qualification gets 216/288 anchors
and 36/96 documents right but fails every counterfactual anchor.
Therefore overall anchor accuracy by itself does not indicate that
the model identifies the correct clause. The key comparisons are
qualification recall, false marks, and sensitivity to changing the
governing text around an unchanged target.

## Results

Both tasks in Slurm array 47307 on ws6 reproduce the decision
counts with reversed enterprise query order. Each uses an RTX
4090. The held set has 96 mixed requests and 288 target anchors.

| Training route | Held qualifications found / 72 | Held false marks / 216 | Full held requests exact / 96 | All views correct / 36 mixed anchors |
| --- | ---: | ---: | ---: | ---: |
| Single | 62 | 88 | 35 | 0 |
| Mixed, matched count | 43 | 61 | 31 | 0 |
| Mixed | 16 | 18 | 40 | 0 |
| Mixed + neighbors | 19 | 30 | 39 | 0 |

At matched training size, mixed composition trades away 19
correct qualification marks to remove 27 false marks on the
held mixed requests. Training on more mixes reduces both
further. The failure to keep all views correct for any of the
36 mixed-context anchors shows that the extra exact documents
are insufficient to establish successful scope binding.

On the private requests, the full mixed head retains 36 of
the 4B parser's 51 qualification marks, compared with two for
the single head. It also marks 840 of the remaining 1,401
clauses, compared with six for the single head. These marks
are disagreements with a model, so they give no estimate of
true false positives or negatives on enterprise requests.
On the separate heading controls, both answers in a pair are
correct for 21, 18, 10 and eight of the 36 pairs in table order.
At the frozen cutoffs no route passes any held mixed anchor.
Only the single head (four requests) and matched mixed head
(one request) could skip 4B in the private replay. The estimated
parser P95 increases for every gate. The whole-block
reference takes 1,292–1,305 ms at P95. The simulated gates
take 1,315–1,331 ms across both runs.

[Aggregate results and execution digests](results.json)

## Reproduce

Use the pinned enterprise development projection from the
[hybrid benchmark](../enterprise-hybrid-rescue-v1) and the
same two checkpoints as the
[shared-encoding pilot](../enterprise-shared-scope-student-v1).
Run on allocated Slurm CUDA and repeat with `--reverse`. Use a
fresh output path. The mixed challenge is generated from the
public scope JSON without accessing the private request content.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_mixed_scope_student \
  DEVELOPMENT_QUERIES benchmarks/enterprise-heading-roles-v1/controls.json \
  benchmarks/enterprise-scope-transfer-v1/controls.json \
  QWEN3_EMBEDDING_CHECKPOINT QWEN3_INSTRUCT_CHECKPOINT NEW_RESULTS.json
```
