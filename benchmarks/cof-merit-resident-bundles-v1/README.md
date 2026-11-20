# Full-catalog retrieval with resident evidence bundles

Transfer the [LR-Bench resident bundle design](../merit-resident-profile-bundles-v1)
to the established CoF KDD and SciRepEval full-catalog
panels. Their exact target exclusions and labels are
the ones used by the [profile pooling benchmark](../cof-global-profile-pooling-v1).
All methods score the full reviewer inventory. A query
whose ID, normalized title, or exact text matches a
reviewer publication is outside the selected panel.

The dense baseline ranks by the best owned source
from frozen 0.6B embeddings. A stronger pooling control
blends that score with the reviewer's profile centroid.
The proposed index computes one or up to three bundles
per reviewer without target access. Select representatives
by incremental coverage of the reviewer's publications.
For the multi-bundle route, link every representative
to its two nearest owned sources. Encode each bundle
with the released MERIT 8B adapter. For every target,
embed the title and abstract once and take the largest
bundle cosine for each reviewer. The winning bundle
supplies sources for verification. Audit token lengths before
embedding the catalog. Batch ordinary views at up to 4,096
tokens and encode any longer view alone, up to 8,192 tokens. The union variant
reserves candidate slots for the dense and bundle channels
and combines their ranks when ordering those candidates.
Its evidence includes sources from both channels.

All arms share the original 0.6B source verifier, the
ten-candidate reporting limit, and the five-slot
recommendation and verification cap. Each person may
present up to three sources. The results report published
positive retrieval, explicit label coverage, and how many
known-positive people retain verifier-selected citations.
The labels concern reviewer expertise. They do not
independently validate the attached evidence. Unjudged
candidates stay unknown in the paired P@5 bounds.

Rotate arm order. Report warm serial latency, with and
without the common verifier, along with the separately
measured offline build time and vector bytes. These
measurements do not cover network transport or concurrent
service traffic.

## Results

Both panels completed on an RTX 5090. Jobs 45281 (KDD)
and 45277 (SciRepEval) use the same implementation.
The 7,956-token catalog view on SciRepEval ran alone,
and every text was encoded without truncation. The KDD
rerun reproduces all reported quality and output-agreement
aggregates from job 45267.

| Panel | Route | Known positives at 10 | Known-positive P@5 | Positive and cited P@5 | Warm P95 through verification |
| --- | --- | ---: | ---: | ---: | ---: |
| KDD | Dense best source | 33 / 242 | 3.77% | 2.13% | 135.29 ms |
| KDD | Centroid and best source | 45 / 242 | 3.93% | 1.97% | 144.27 ms |
| KDD | MERIT one bundle | 22 / 242 | 1.97% | 0.00% | 161.12 ms |
| KDD | MERIT multiple bundles | 25 / 242 | 2.13% | 0.49% | 171.97 ms |
| KDD | MERIT and dense union | 35 / 242 | 3.11% | 1.31% | 197.43 ms |
| SciRepEval | Dense best source | 78 / 153 | 16.61% | 8.47% | 129.03 ms |
| SciRepEval | Centroid and best source | 82 / 153 | 17.63% | 7.12% | 131.01 ms |
| SciRepEval | MERIT one bundle | 46 / 153 | 10.85% | 0.68% | 133.44 ms |
| SciRepEval | MERIT multiple bundles | 51 / 153 | 14.24% | 3.05% | 140.01 ms |
| SciRepEval | MERIT and dense union | 76 / 153 | 16.61% | 7.46% | 165.90 ms |

The resident designs are slower here than the compact
dense control, even after moving bundle encoding offline.
The complete request is 27% slower with multiple bundles
on KDD and 8.5% slower on SciRepEval. The union recovers
some known positives lost by the standalone bundles at
added latency. Every bundle route reports fewer known-positive
recommended reviewers with model-selected citations than the
compact dense baseline. CoF judges only a subset of the returns,
and its labels do not judge those citations. These comparisons
change both expert nomination and source selection, so the next
diagnostic should hold the nominees and order fixed.

[Full KDD aggregates](kdd.json) · [Full SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the CoF archive and cached frozen source embeddings
from the [profile pooling experiment](../cof-global-profile-pooling-v1).
Use the same MERIT weights, verified text templates,
and isolated dependency versions as the
[matched MERIT baseline](../merit-author-retriever-v1).
Run on Slurm CUDA with a fresh output.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_merit_resident_bundles \
  KDD MERIT_SCRIPT_DIR QWEN8B_CHECKPOINT MERIT_ADAPTER \
  QWEN06B_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```

Pass `SciRepEval` for the second panel.
