# Calibrate person-marked evidence nomination

[The preceding study](../redocred-dense-evidence-nomination-v1)
found that forcing a reserve replacement lost evidence-backed
relations, and that all people mentioned in a high-ranked
sentence can inherit the same score. This experiment builds
one extra view per automatic mention. Replace that mention's
tokens with `[FOCAL PERSON]`, leave the rest of the sentence
intact, and encode the resulting text offline. The verifier
always receives the unmodified original sentences.

Use the same frozen Qwen3 embedding model for both dense
scores. The unmarked arm scores the sentence. The marked
arm scores the mention-specific view. Both use one live
query embedding, with the same work and role. Rank each
candidate by its maximum mention score. Compare the best
excluded candidate with the current last candidate. Keep
the other slots, the two lexical sentences per candidate,
the reader threshold, and the source budget fixed.

Compare the automatic shortlist against four reserve
policies. Each dense representation has a positive-margin
control and a calibrated margin policy. Calibration uses
queries and retrieval from the 500 development documents.
Exclude every exact normalized work name represented in
the test requests. Replay candidate switches on that
development pool. Choose a threshold above which the switch
improves the number of accepted relationships with complete
annotated evidence, changes at most 30% of calibration requests,
and does not increase accepted unjudged links in aggregate.
If no threshold qualifies, freeze the policy to retain
the existing shortlist. The 30% ceiling is a calibration
criterion. Test replacement rates are measured independently.

Use the same 161 exact-name and 145 transposed queries as the
earlier study. Source lookup uses the full public 1,000-document
index at evaluation. The aligned Re-DocRED and Re²-DocRED
releases supply the reference labels. Missing relation labels
remain unjudged. The exact and perturbed cohorts share some
works. Report both separately and pair by their anchor documents.
Latency includes online lookup, nomination, evidence selection
and the batched reader. Offline index construction is separate.

## Results

Slurm job `43453` completed on one RTX 5090.
Encoding 22,735 distinct mention views took 15.61 s beyond
6.10 s for the unmarked sentence bank. Neither representation
met the selection rule on the 172 calibration requests.
Both calibrated policies therefore kept every automatic
shortlist and reproduced the baseline quality counts.

| Route | Exact replacements / 161 | Exact fully evidenced / unjudged accepted | Transposed replacements / 145 | Transposed fully evidenced / unjudged accepted | P95 exact / transposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Automatic | 0 | 53 / 13 | 0 | 43 / 7 | 58.3 / 58.9 ms |
| Positive sentence margin | 136 | 48 / 25 | 122 | 36 / 16 | 64.0 / 64.6 ms |
| Positive person-marked margin | 156 | 50 / 34 | 137 | 37 / 18 | 63.7 / 64.1 ms |

Person marking improves the positive-margin point estimate
against the sentence control by two and one fully evidenced
links, respectively. It also replaces more incumbents and
returns more unjudged links, so this is not a comparison at
the same realized replacement rate. Against the automatic
baseline, person marking loses three and six fully evidenced
links. The descriptive paired intervals for fixed-denominator
grounded P@5 are −1.20 to +0.28 percentage points on exact
names and −1.76 to −0.12 points on transpositions.

Keep the automatic shortlist and lexical evidence windows.
These results rule out promotion based on a positive
margin for this representation and calibration setup.
They leave room for explicitly attributed evidence, such
as contribution declarations or project role assignments.
[Aggregate results](results.json)

## Reproduce

Use the pinned snapshots, automatic projection, and
checkpoints from the [previous experiment](../redocred-dense-evidence-nomination-v1).
Run on a Slurm CUDA compute node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.owner_anchored_reserve \
  REDOCRED_DIR RE2DOCRED_DIR PROJECTED_MENTIONS QWEN_EMBEDDING QWEN4B NEW_RESULTS
```
