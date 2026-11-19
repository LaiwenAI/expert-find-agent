# Evidence windows for lightweight scientific claim verification

Use the fixed [SciFact calibration and development panels](../scifact-small-nli-v1/README.md), the same mDeBERTa checkpoint, and the 512-token no-truncation rule. The full abstract reproduces the earlier verdict and over-limit counts exactly. Split each abstract at the published sentence boundaries. Hold the complete claim fixed for every arm.

| Evidence view | Selection rule |
| --- | --- |
| Full | All abstract sentences |
| Lead three | Up to three initial sentences |
| Lexical three | Contiguous three-sentence window covering the most distinct claim tokens. Break ties by source order. |
| Two lexical windows | Add a disjoint three-sentence window only when it contributes a previously uncovered claim token. Concatenate the selected spans in original order. |
| Annotated rationale | Smallest complete judgment rationale for annotated pairs. Use the lexical-three excerpt for the unknown cases without annotated evidence. |

Only the last route uses human evidence locations. For each route, calibrate on the same 309 training pairs with separate 5% false-support caps for contradiction and unknown. Evaluation uses the same 226 development pairs. A contained rationale means at least one complete set of annotated sentences is present in the view, before token-limit checks. The diagnostic does not require every additional sentence to be relevant.

## RTX 5090 results

| Evidence view | Contains rationale / 144 | Over limit / 226 | Raw accepted support / contradiction / unknown | Calibrated accepted support / contradiction / unknown | Verifier batch P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 144 | 34 | 38 / 3 / 7 | 23 / 0 / 0 | 41.2 ms |
| Lead three | 34 | 0 | 17 / 1 / 1 | 30 / 6 / 6 | 18.9 ms |
| Lexical three | 92 | 0 | 40 / 1 / 3 | 20 / 0 / 0 | 19.1 ms |
| Two lexical windows | 111 | 1 | 41 / 2 / 2 | 25 / 0 / 0 | 42.2 ms |
| Annotated rationale | 144 | 0 | 36 / 1 / 3 | 43 / 5 / 4 | 17.7 ms |

The class denominators are 97 support, 47 contradiction, and 82 unknown. Three lexical sentences eliminate all over-limit cases and halve model batch P95 relative to the full abstract. With the default decision boundary they retain two additional true supports and accept six fewer negatives. Threshold calibration changes the ordering. With zero observed false supports, calibrated lexical three retains 20 true supports versus 23 for full and 25 for two windows. The short-view advantage is therefore primarily latency and input coverage. Even the complete annotated rationale yields just 36 accepted supports with the raw model. Finding a present rationale and judging it remain distinct bottlenecks.

Verification time includes tokenization, device transfer and scoring for batches of up to 16. Building all five views together separately took 1.54 ms at the 95th percentile over batches. These are warm batch diagnostics, with a reused English benchmark corpus. They do not measure enterprise retrieval, the validity of a candidate's attributed work, or Chinese verification. [Aggregates](results.json)

## Reproduce

Run on a Slurm CUDA allocation using the archive and mDeBERTa checkpoint from the linked study:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.nli_evidence_windows \
  --archive-path SCIFACT_ARCHIVE --checkpoint MDEBERTA_CHECKPOINT --output NEW_RESULTS
```

Job 41241 ran all five views on ws5.
