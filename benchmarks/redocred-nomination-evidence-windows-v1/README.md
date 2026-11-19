# Pass nomination-time role evidence to a frozen verifier

The [fixed-slot nomination experiment](../redocred-contextual-role-nomination-v1)
found cases in which a contextual ranker nominated more labeled
candidates than the geometry ranker but its final list retained fewer.
Test whether routing the role verifier to the nomination-time mention
improves this handoff. Reuse both frozen development procedures and
the same 306 open-source-document queries. Run each shortlist policy
with two evidence selectors.

- **Lexical** independently chooses the existing two sentences per
  candidate from full-name, work-name and role-word matches.
- **Anchored** is identical except for a replacement in the last slot.
  Recompute the nominator score on individual mentions of that
  candidate. If the highest-scoring mention is outside the lexical
  window, read its sentence and keep whichever lexical sentence
  better names the work. When the best mention is already visible,
  keep the original prompt.

The two policies must emit an identical candidate list on every
query. They use the same number of sentences per candidate, one
warmed forward per query, the same open-weight reader, and the
same decision threshold. Different sentences may have different
lengths. Measured P95 covers source retrieval, nomination, prompt
construction and GPU verification and excludes model loading and
queued load. Prompt layouts and model inputs for all earlier
positions are shared, but padding can change their numerical output.

The mention score is a local probe of the pooled nomination model.
It is not a proof of which sentence caused the original choice.
The evidence selector does not use human relationship labels or annotated
aliases. They serve nominator training, validation, and evaluation.
For replaced positives with sentence annotations, report the number
of windows that contain at least one or all annotated sentences.
Annotations are incomplete and their presence does not independently
validate a response citation. Bootstrap intervals resample complete
reference documents within each query view.

## Results

Slurm job 43089 ran on one RTX 5090 on ws5. The fixed lexical
arms reproduced the earlier verified counts and nominee swap counts.
Each view uses 747 and 662 slots, respectively, with two sentences
per slot.

| Test view | Nominator | Last-slot replacements | Changed windows | Replacements with all annotated evidence / available | Verified positives, either window | Lexical P95 (ms) | Anchored P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact, 161 queries | Geometry | 57 | 0 | 3 / 3 | 101 | 58.17 | 58.34 |
| | Context | 55 | 0 | 3 / 3 | 99 | 58.50 | 58.86 |
| Transposition, 145 queries | Geometry | 42 | 0 | 2 / 2 | 79 | 58.50 | 58.52 |
| | Context | 45 | 0 | 2 / 2 | 77 | 58.90 | 58.84 |

Across all 199 query–nominator replacements, lexical selection already
included the nomination-selected mention sentence. For all annotated
positive replacements, both windows included every available annotated
sentence. The paired input was therefore identical and every
acceptance was reproduced. The result identifies overlap between
these two selectors in this panel; it cannot measure the effect of
showing materially different evidence. The annotated replacement
subsets are small. An evaluation on queries where lexical selection
and a candidate-specific index disagree is needed to test evidence
choice beyond this short-document setting.

[Aggregate results](results.json)

## Reproduce

Use the pinned dataset, NER projection and reader described in the
[nomination benchmark](../redocred-contextual-role-nomination-v1).
Run on a CUDA compute node through Slurm.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.nomination_evidence_windows \
    REDOCRED_DIR NER_PROJECTION.json PINNED_QWEN4B NEW_RESULTS.json
```
