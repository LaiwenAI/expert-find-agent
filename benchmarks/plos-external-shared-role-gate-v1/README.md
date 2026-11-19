# External transfer of the shared-embedding role gate

Freeze the query encoder, definition and ridge heads, and 10% calibration operating points from the [linked-witness experiment](../plos-shared-embedding-witness-v1/README.md). Use the [external contribution edits](../plos-credit-external-v1/README.md) and their original boundary wording. These phrases describe adjacent tasks, often mentioning work done by another contributor. Verify no normalized phrase overlap with the NISO example inventory or the 84 original synthetic training phrases. The external benchmark has been used for earlier role classifiers.

Exclude all publications used in the linked-witness calibration and test cases, and any paper sharing a publisher identifier with them. Verify each retained publisher declaration hash against the same projection. Validate the labels by recomputing them from the authored roles before running models. Each group applies one requirement edit to each of two authors on the same publication. The edit flips the expected outcome for one author and leaves it unchanged for the other. The matched rows balance the flip direction. A group counts as exact only when both authors and both versions receive the expected verdicts.

Use the same candidate records and two source-scoped role phrases for each request version. Encode all four phrases for a group in one forward. Compare the shared-embedding definition and ridge routes, a role-instruction ridge comparator using the same 84 training phrases, and a paper-wide role-union ablation with the definition scores. Each arm verifies membership using the best declared-versus-undeclared role margin. The shared cutoffs transfer without retuning. Dedicated ridge uses a positive margin. The ablation attributes the union of all coauthor roles to both authors. Component timing includes batch encoding and in-memory checks for both versions and both authors. It excludes source retrieval, parsing, transport and preprocessing. Paired intervals resample publications and are descriptive because the wording bank repeats across papers.

## RTX 5090 results

Job 42398 excludes one article used in the linked-witness study. The retained 78 edit groups span 23 publications and yield 154 unsupported and 158 supported author–request decisions. The frozen gate thresholds are zero.

| Route | Both authors and edits exact / 78 | Unsupported returns / 154 | Supported returns lost / 158 | Two-version component P95 |
| --- | ---: | ---: | ---: | ---: |
| Shared definitions | 18 | 17 | 74 | 8.48 ms |
| Shared ridge | 41 | 7 | 37 | 8.47 ms |
| Dedicated role-instruction ridge | 39 | 16 | 40 | 8.55 ms |
| Paper-pooled definitions | 0 | 125 | 35 | 8.52 ms |

The shared definition head led the preceding linked-witness study, but misses 21 more complete edit groups than the dedicated ridge comparator here. Its paired difference in exact groups is −26.9 percentage points, with a publication-resampled interval of −41.0 to −12.9 points. The shared ridge head improves by 2.6 points, with an interval of −6.9 to +12.5 points, and has nine fewer observed unsupported returns. Paper-level pooling preserves more positive evidence than the author-scoped definition route while assigning coauthors' roles to the wrong person. Accordingly, preserve author scope and keep both lightweight mapping heads as separate candidates for a new domain. The results quantify wording sensitivity on supplied source records. They do not measure open search, estimate production risk, or demonstrate an accuracy win over the dedicated head. [Aggregate results](results.json)

## Reproduce

Use the pinned publisher projection, NISO example file, previously verified embedding checkpoint and the original training phrases from the linked-witness experiment. Supply its aggregate JSON and the external fixture directory. Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.external_shared_role_gate \
    PUBLISHER_PROJECTION EXTERNAL_FIXTURE NISO_EXAMPLES ROLE_TRAINING QWEN_EMBEDDING PRIOR_RESULTS NEW_RESULTS
```
