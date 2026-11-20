# Test precision as a way to reduce prefix-tree rechecks

Reuse the pinned Re-DocRED requests, evidence and Qwen3-4B checkpoint
from the [exact-prefix experiment](../redocred-exact-prefix-v1/README.md).
Every route scores the same four original prompts. The common
instruction prefix is prefilled once in the matching precision.
The FP16 checkpoint is the same BF16-loaded weights cast to FP16
at initialization. No new weights or prompt calibration data are
introduced. The two model copies reside on the same allocated GPU.

| Route | Initial execution | Canonical recheck |
| --- | --- | --- |
| BF16 batch | Original padded rows | None |
| BF16 tree | Prewarmed prefix cache plus packed tree | None |
| BF16 guard | BF16 tree | BF16 batch inside the BF16 error band |
| FP16 batch | Original padded rows cast to FP16 | None |
| FP16 tree | Prewarmed FP16 prefix cache plus packed tree | None |
| FP16 guard | FP16 tree | FP16 batch inside the FP16 error band |
| Cross-precision guard | FP16 tree | BF16 batch inside the cross-precision error band |

Fit the common conjunction cutoff on the 44 development requests
using the unmodified BF16 batch. Fit one error radius per guard from
the largest absolute tree-vs-batch margin difference in development.
These are separate calibration quantities. Each radius compares the
fast route against the batch it would fall back to. Recheck the
request if any atomic score approaches zero or the shared cutoff
within that radius. Count both atomic and conjunctive differences
at test time against the route's own batch reference and against
the original BF16 batch. The error radii are empirical envelopes.
The two held-out panels do not participate in calibration.

Report known and unjudged approvals separately. A missing label
does not establish lack of support. The reader comparison holds
source discovery and entity resolution fixed. Serial request times
include the common window and prompt construction, all inference
and any recheck. The fixed instruction cache is already warm. The
workload excludes network, queues and GPU model initialization.

## Reproduce

Use the pinned release and reader checkpoint from the preceding
experiment. Select a compute node with enough GPU memory for the
two model copies and run inside an allocated Slurm GPU job.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.conjunctive_precision_guard \
  REDOCRED_DIR QWEN4B NEW_RESULTS --order-offset 0
```

## Results

Both tasks of Slurm array `44341` completed on an RTX 5090 on ws5.
Rotate the execution order by one slot in the repeat. The two runs
agree on all non-timing aggregates. On the 213 requests held out
by source title, the guarded FP16 tree matches all atomic signs and
candidate decisions of the FP16 batch with five rechecks. Its P95
is 31.50–31.64 ms versus 55.48–55.67 ms. It processes 70,217
token slots including rechecks versus 170,172 for the FP16 batch.

Across precisions, the FP16 and BF16 batches disagree on three
candidate decisions. The cross-precision guard matches the BF16
batch on these held-out requests with 21 rechecks. Its P95 is
66.32–66.33 ms versus 54.43–54.65 ms for the BF16 batch.
The BF16 guard needs 29 rechecks and reaches 70.18–70.31 ms.
The raw deviation exceeds its development envelope on five
requests for the FP16 guard and eight for the cross-precision
guard. Output agreement in these panels is empirical.

On the 38 native held-out requests, the FP16 guard agrees with
the FP16 batch without rechecks at 30.08–30.21 ms P95 versus
52.61–52.98 ms. See [aggregate results](results.json) for the
other routes, input hashes and both execution orders.
