# Test a sparse source check before contributor verification

Reuse the [25 linked pairs in 15 components](../plos-source-confirmation-v1/README.md) in the 171-document PLOS index. Score overlapping content terms by inverse document frequency. Compare independent top-one selection, author-linked expansion of the sparse rankings, and expansion with an independent lexical check on each source. Confirmation requires two or more corpus-unique clue terms supporting the selected document on each side and zero corpus-unique clue terms pointing to a competing available document. The rule uses only the frozen index and current clue, with no training labels or dense inference. Document frequencies remain frozen when a source is masked.

| Clue | Policy | Exact / 25 present | Wrong / 25 present | Wrong / 50 missing |
| --- | --- | ---: | ---: | ---: |
| Result excerpt | Sparse independent | 8 | 17 | 48 |
| Result excerpt | Sparse linked | 17 | 1 | 4 |
| Result excerpt | Sparse linked and confirmed | 1 | 0 | 0 |
| Keywords | Sparse independent | 3 | 22 | 50 |
| Keywords | Sparse linked | 14 | 2 | 8 |
| Keywords | Sparse linked and confirmed | 1 | 0 | 0 |

The lexical check abstains on 24 of 25 present cases in each clue view. Removing terms unique to one indexed document drops keyword exact matches from 14 to 8 before confirmation. The confirmation rule necessarily abstains under that stress condition. These cues come from the abstract result text and reuse articles, so the results characterize this replay and support keeping the strict rule out of the mandatory confirmation path. An author link improved sparse retrieval, but a linked candidate alone still left missing-source errors.

The replay ran in a Slurm CPU allocation on ws5. P95 for executing all three policies in the shared in-memory loop is 0.186 ms for original excerpts and 0.041 ms for original keywords. These timings exclude index construction, embedding, role verification, and serving. [Aggregate results](results.json)

Rebuild the frozen projection and panel following the [source confirmation instructions](../plos-source-confirmation-v1/README.md).

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.credit.source_anchor_gate \
  PROJECTION PANEL NEW_RESULTS
```
