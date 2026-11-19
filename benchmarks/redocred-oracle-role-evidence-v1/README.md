# Separate evidence selection from person-specific verification

This diagnostic replays the nonempty-evidence subset of the
[Re-DocRED person-specific benchmark](../redocred-person-evidence-v1).
Each pair holds the work, requested role and source fixed, and contrasts
the credited person with a person labeled for a different role in the
same work. Every verifier sees exactly the same source text for both
people under a given window policy. The question changes only the name.

| Window | Input to both people |
| --- | --- |
| Matched | A lexical selection with the same sentence count as the annotations. Maximize coverage of both names, the work and role cues, breaking ties by the weaker name's coverage and then source order. |
| Gold | Every sentence annotated for the supported person and target role. |
| Full | The entire document in its original order. |

The matched policy receives the oracle sentence count but no sentence
locations. Gold receives the supported person's evidence locations,
but both questions get the same sentences and neither question reveals
how the window was selected. This isolates an evidence-location
intervention at equal sentence count. It is a diagnostic with access
to labels, and the gold text may focus mainly on the supported
person. It does not attest that all gold sentences are self-contained
or that another person's missing relation label is an explicit negation.

Both models are frozen. Qwen3-Reranker-0.6B uses the same evidence
instruction as the original experiment. For Qwen3-4B-Instruct-2507,
score the difference between the first-token logits for `1` (explicit
support) and `0` (otherwise). Pretime a warm-up, rotate window order
and counterbalance person order. Separately calibrate each model-window
threshold to accept at most 5% of development rivals. Strict exceedance
determines acceptance. Report both-correct pairs and the two component
outcomes, plus descriptive paired intervals resampling whole documents.
Inference timing includes tokenization, score computation and device
transfer; it excludes window selection and model loading.

## Results

Slurm job 42980 on RTX 5090 reproduced the decisions from job 42975.
The development panel contains 96 pairs from 39 documents. The test
panel contains 80 pairs from 36 documents. Each gold window averages
1.68 test sentences. The lexical selector matches the exact annotated
window in 45 of the 80 test pairs.

| Verifier and window | Ordered / 80 | Accepted supported / 80 | Accepted rival / 80 | Both correct / 80 | Paired inference P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B, matched | 62 | 26 | 3 | 24 | 8.91 ms |
| 0.6B, gold | 73 | 29 | 1 | 28 | 8.92 ms |
| 0.6B, full | 48 | 5 | 4 | 5 | 12.34 ms |
| 4B, matched | 69 | 59 | 2 | 58 | 25.97 ms |
| 4B, gold | 76 | 64 | 2 | 63 | 25.71 ms |
| 4B, full | 74 | 65 | 3 | 64 | 52.45 ms |

For the reranker, the gold window rescues seven pairs and loses three
relative to the length-matched control. The both-correct difference of
+5.0 percentage points has a descriptive 95% interval from −3.95 to
+13.89. The reranker still ranks the correct person lower or ties on
7 of 80 pairs when given gold evidence, and accepts only 29 at its
frozen development threshold. Moving the evidence location and
choosing a deployment operating point are distinct interventions.

The 4B reader with the lexical window gets 58 pairs right, versus 63
with gold and 64 with full text. Full text minus matched has an
interval of −5.06 to +20.62 points, so these results leave the size of
the full-context benefit uncertain. On identical windows, the 4B reader improves both-correct rates by
42.50 points on lexical selections (95% interval 27.03 to 58.14)
and 43.75 points on gold selections (28.15 to 60.29). The model
comparison changes the checkpoints and their respective scoring prompts.
On gold windows, the ordering advantage alone has an interval of −1.52
to +11.11 points. The larger observed difference concerns the calibrated
acceptance decisions.

Gold evidence contains both exact canonical name strings in 21 test
pairs and omits even the supported person's exact name in three.
Mentions may instead use aliases or pronouns. The experiment diagnoses
within-document verification on the subset with sentence annotations.
It does not measure source discovery, overall expert ranking or
endpoint latency. Evidence selection with an oracle count is an
analysis control. A policy that routes uncertain cases to the stronger
reader would need a query-independent window budget or a budget inferred
from text, alongside independent calibration of the routing decisions.

[Aggregate results](results.json)

## Reproduce

Use the split files and reranker snapshot from the
[person-specific benchmark](../redocred-person-evidence-v1). Use
`Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554` with local provenance. Run
inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.oracle_role_evidence \
    REDOCRED_DIRECTORY QWEN_RERANKER QWEN_READER NEW_RESULTS.json
```
