# Select a separate source clause for each requested contribution

Use the exact 59 real PLOS report pairs, source identifiers and Qwen3-4B
checkpoint from the
[binary encoding study](../plos-study-label-order-v1).
The chosen report is given. Reproduce its direct binary prompts as
controls. For the factored method, ask the model once per task to
point to a numbered contribution clause or to abstain. Keep the
source pre-bound and reject clauses that do not contain the exact
requested initials. The same clause cannot satisfy both distinct
requested roles. Compare the full note with an initials-prefiltered
list, and place the abstention digit first or last. A four-call
agreement route requires the two encodings to select the exact same
clause for each task.

The literal reference uses the complete publisher note vocabulary,
including the known task-to-role map used to construct the fixture.
It validates the parsing and identity gate for this structured family.
The model selector sees only the raw clauses and requested paraphrase.

## Results

Both Slurm tasks in array 44146 ran on an RTX 5090. The fixture and
checkpoint hashes match the previous study. All non-timing counts
agree across the two tasks. Each class contains 59 views.

| Route | Supported accepted | Unsupported accepted | Both sides correct / 59 | Local P95, ms |
| --- | ---: | ---: | ---: | ---: |
| Direct, 1 means yes | 59 | 52 | 7 | 15.49–15.64 |
| Direct, 0 means yes | 38 | 15 | 25 | 15.43–15.56 |
| Report clauses, abstention first | 57 | 25 | 34 | 30.36–30.74 |
| Report clauses, abstention last | 58 | 24 | 35 | 30.54–31.05 |
| Report clauses, both choices agree | 53 | 17 | 39 | 60.87–61.77 |
| Initials-prefiltered, abstention last | 58 | 27 | 32 | 29.05–29.15 |
| Literal reference | 59 | 0 | 59 | 0.02 |

Explicit clause selection improves paired correctness and recovers
supported cases over the conservative binary encoding, while still
accepting many unsupported cases. The report-scope agreement route
still makes 43 wrong-role pointers across 236 individual requests,
even after enforcing exact initials. Source and author binding
therefore do not solve role entailment. On this schema, recognized
fields offer a deterministic low-cost path. Contrastive tests on
unseen paraphrases and free prose are needed for claims about the
automatic semantic verifier. GPU timings measure local decision
steps and exclude search, service queueing and startup.

[Both runs](results.json)

## Reproduce

Reconstruct the pinned fixture using the previous label-order
benchmark. Use an allocated Slurm GPU and the pinned 4B checkpoint.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.study_clause_selection PINNED_FIXTURE NEW_RESULTS
```
