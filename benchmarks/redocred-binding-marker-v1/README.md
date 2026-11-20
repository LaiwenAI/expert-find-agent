# Learn person–work binding under a fixed evidence window

Use the pinned [Re-DocRED](https://aclanthology.org/2022.emnlp-main.580/)
and [Re²-DocRED](https://aclanthology.org/2026.eacl-long.213/) releases.
Construct two-person by two-work cases for a single documented
role. The four queries share the same unedited four-sentence
window and candidate names. Source selection uses name and role
cues, without reference evidence. Cases come from the previously
studied [work ownership diagnostic](../redocred-work-owner-contrast-v1).
Use the upstream training set to fit new heads. Exclude normalized
calibration and evaluation titles from fitting. Fit cutoffs on
development documents, then freeze them for the official test
documents. The fit, calibration and held panels contain 237, 64
and 62 cases from 204, 47 and 49 documents.

For each case, train on annotation presence as an approximate
decision target. The paired loss prefers a known relation to an
unlabeled candidate differing by exactly one person or work.
Preserve shared-role positives in the expanded annotation inventory.
Unjudged candidates can be true relations. Their acceptance counts
measure unresolved review workload, without estimating false
acceptance risk. All model inputs use only the fixed window and names.

## Comparisons

- **Generic reader:** frozen Qwen3-Reranker-0.6B. Its relative
  alternative uses the smaller margin difference against swapping
  the person or the work.
- **Mention geometry:** seven exact-match features for mentions of
  the work, person and role cues and their sentence-level co-occurrence.
  Fit a linear head on these features.
- **Named and marked text:** encode the claim and window together
  with frozen Qwen3-Embedding-0.6B, append the same geometry features,
  and fit the same linear head. Marked text replaces exact canonical
  name matches with target-person, target-work and other-entity markers.
  Entity markers are an established relation-extraction technique.
  [Matching the Blanks](https://aclanthology.org/P19-1279/)
- **Marked with contrasts:** train on the same features and examples,
  adding a paired ranking loss for the single-coordinate alternatives.

The four learned heads use 200 full-batch AdamW steps on CUDA
with a shared seed. Backbone weights stay frozen. Fit a threshold
for each arm at the same 5% or 10% development cap on accepting
unjudged links. For each observed evaluation output, separately
count overlap with every original annotated evidence sentence.
Evidence present in the window does not establish that the model
used it, and other genuine evidence may exist.

## Results

Slurm array 45840 runs on ws3 RTX 5090. Both tasks reproduce all
non-timing fields, and no reader inputs are clipped. The held
set has 166 labeled links and 82 unjudged ones. Only 60 labeled
links have all originally annotated sentence text in the supplied
window. Results below use the 5% calibration cap.

| Method | Labeled accepted / 166 | Unjudged accepted / 82 | Accepted with all reference sentences / 60 | Warm local P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Generic reader | 33 | 4 | 17 | 15.11–15.26 |
| Generic reader, relative margin | 32 | 3 | 13 | 15.11–15.26 |
| Mention geometry | 40 | 4 | 16 | 0.85–0.86 |
| Named text | 21 | 5 | 9 | 11.79–13.21 |
| Marked text | 19 | 1 | 8 | 10.03–11.83 |
| Marked text with contrasts | 7 | 0 | 1 | 10.03–11.83 |

Mention geometry accepts seven more known relationships at the same
observed unjudged count. The paired difference is +0.11 accepted
known links per case, with a title-clustered 95% interval from −0.26
to +0.46. Its extra output does not increase the count carrying
all reference evidence. At the 10% calibration cap, marked text
accepts 46 known and four unjudged links, of which 22 known links
have all reference evidence. The generic reader accepts 44 known and
five unjudged links, with 21 of the known links carrying all reference
evidence. This difference is uncertain. Marking and paired training
did not give a stable improvement.

Component timings cover selection of the same shared four sentences,
each model or head call, and serialization. The marked arms share an
encoder pass and their timings include scoring both heads. The
experiment supplies the source and candidate identities and does not
establish how well the methods work on unseen domains or full expert search.

[Replicated decisions and source-evidence overlap](results.json).

## Reproduce

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.docred.binding_marker_training \
  REDOCRED_DIR RE2DOCRED_DIR QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --offset 0

PYTHONPATH=src:. python -m experiments.docred.binding_window_audit \
  REDOCRED_DIR RE2DOCRED_DIR NEW_RESULTS NEW_AUDIT
```

Run training and live checks in a Slurm CUDA allocation. Repeat with
`--offset 1` to rotate execution order.
