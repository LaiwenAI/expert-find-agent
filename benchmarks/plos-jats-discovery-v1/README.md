# Discover sources before checking contributor roles

Extend the [given-source role check](../plos-jats-linked-gate-v1/README.md) with [source discovery controls](../plos-source-confirmation-v1/README.md). The frozen index covers 171 PLOS articles. The source panel has 25 linked pairs in 15 connected components, crossed with two clue styles and three visibility states. The 178 role requests reuse 11 pairs. A wrong pair counts as a source error even when it shares an author with the target pair. A source abstention is tracked separately from role uncertainty after an exact source match.

Compare linked top-4, linked expansion to rank 12 with one source in the top 4, absolute text reranking, a cheap ridge gate over dense rank and lexical coverage features, and known-source diagnosis. Fit source gates while holding out the connected component. Use the same frozen role model, cutoff, and publisher declaration postings after every accepted pair. Compile those postings from the original JATS before replay.

| Source route | Correct positive with excerpts / 89 | Correct positive with keywords / 89 | Wrong source accepts on excerpt missing-source replay / 356 | Wrong source accepts on keyword missing-source replay / 356 |
| --- | ---: | ---: | ---: | ---: |
| Linked top-4 | 41 | 21 | 0 | 0 |
| Linked expansion | 51 | 37 | 20 | 8 |
| Absolute text | 25 | 11 | 0 | 0 |
| Cheap ridge gate | 41 | 7 | 20 | 8 |
| Known source | 51 | 51 | 0 | 0 |

On job 41739 (RTX 5090), linked expansion recovered more present-source positives and admitted wrong pairs when a target source was removed. Of its 28 wrong pair accepts, one produced an unsupported expert recommendation. The cheap gate did not improve the balance. The absolute text gate abstained frequently. The cases reuse articles and role requests, so the counts are scenario totals, not independent source-level estimates. Factored timing combines measured, reused retrieval and role stages, with text reranking included for the absolute text arm. It excludes request serving and offline compilation. See the [aggregate output](results.json) for timing and error decomposition.

Rebuild the pinned source projection and visibility cases using the source discovery controls above. Place each publisher snapshot in the raw directory as `<DOI suffix>.xml`. Then use the same NISO and training fixtures referenced by the given-source role check. Run replay with verified open checkpoints on a Slurm CUDA node.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.jats_discovery_replay prepare PROJECTION RAW_DIRECTORY POSTINGS
python -m experiments.credit.jats_discovery_replay run \
  PROJECTION CASES POSTINGS NISO_EXAMPLES ROLE_TRAINING EMBEDDING RERANKER NEW_RESULTS
```
