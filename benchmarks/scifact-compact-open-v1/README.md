# Frozen short-window verifier on held-out SciFact-Open pairs

The [compact verifier development study](../scifact-compact-verifier-v1/README.md) found the same number of accepted true and false supports for full abstracts and two lexical windows. This check freezes the reader and both support-margin cutoffs before evaluating a separate set of annotated claim–abstract pairs from [SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/).

The projection excludes claim IDs and normalized claim text from the original SciFact training and development sets and the earlier SciFact-Open source-conflict projection. It also excludes papers cited or annotated in those original splits and papers in the earlier projection. Abstracts must be nonempty and at most 2,500 characters, as in development. All retained labels are published support or contradiction judgments. The set contains 37 claims and 30 papers, of which 33 pairs entered SciFact-Open through citation and four through pooling. The sample represents these retained pairs. There are no unknown judgments in this projection.

| Outcome | Complete abstract | Two lexical windows | Paired change |
| --- | ---: | ---: | --- |
| Annotated supports accepted | 12/26 | 10/26 | One gained, three lost |
| Annotated contradictions accepted as support | 1/11 | 0/11 | One false support removed |
| Verifier P50 | 26.73 ms | 19.99 ms | 25.2% lower |
| Verifier P95 | 41.67 ms | 28.32 ms | 32.0% lower |

The 95% paired bootstrap interval for the support acceptance difference is −23.1 to +7.7 percentage points, resampling connected claim–paper components. The P95 speedup ratio is 1.47× with a 95% component bootstrap interval of 1.06–1.74×. Both readers missed all four pooled supports. Quality noninferiority is unresolved. The observed missed supports warrant retaining the full-abstract reference when testing any shorter default route.

The RTX 5090 measurement starts with a supplied claim and abstract. It includes window construction, tokenization and inference at batch size one. Arm order alternates by pair. The experiment measures abstract-level support acceptance and component latency. Expert retrieval, personal attribution and end-to-end service latency require their own evaluation. [Aggregate results](results.json)

## Reproduce

Use the [SciFact-Open dataset](https://github.com/allenai/scifact-open) and the original [SciFact data archive](https://github.com/allenai/scifact). The runner checks the SHA-256 of both archives and the regenerated earlier projection. Use a new output path for each command. The last command needs a CUDA Slurm allocation and the [pinned Qwen3-4B reader](../scifact-live-selective-v1/README.md).

```sh
python -m experiments.scifact.open_source_conflict prepare \
  --input OPEN_ARCHIVE --sha256 8f4dc238b2eff422bab5c7918fcbddb6996b29d37cb887942ef6ee1a16fb3415 \
  --output PRIOR_PROJECTION
python -m experiments.scifact.compact_open_holdout prepare \
  OPEN_ARCHIVE ORIGINAL_ARCHIVE PRIOR_PROJECTION NEW_PAIRS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.compact_open_holdout compare \
  NEW_PAIRS QWEN_4B_CHECKPOINT NEW_RESULTS
```

Job 41321 ran on ws5.
