# Test selective expansion of a short evidence window

Use the original frozen MiniLM and the two verified rationale-pair checkpoints from the [short-window supervision study](../scifact-minilm-window-supervision-v1/README.md). No weights change. Use the same 76 calibration and 226 evaluation claim–abstract pairs with disjoint papers. The unknown label is an annotation-based proxy from cited abstracts without released evidence.

Begin with the three-sentence lexical window. An available second window is disjoint and covers claim terms absent from the first. Compare four policies. The first always uses the first window. The second sends both windows in source order to a single model call whenever available. The third scores each window separately and uses the larger support margin. The fourth scores the first window, then sends the combined view only if the first margin lies inside a learned interval. The combined verdict supersedes the initial verdict in that interval. The runner falls back to the first verdict if expansion does not fit.

Calibrate a single threshold separately for the fixed policies. For the selective policy, choose its threshold and interval on the development panel to maximize true support with separate 5% false-support caps and at most 25% additional calls. Ties favor fewer additional calls. Keep the first-only route in the candidate family. All policies use the same available views, and gold sentence locations are used only for the coverage audit.

All model forwards are timed at batch size one. Report factored P95 by summing the measured per-view times on paths that use multiple calls. Results exclude document retrieval, view selection and serving transport. A longer combined view can raise single-forward time even when the route uses one forward.

## RTX 5090 results

Among 226 test cases, 88 have a disjoint second window. Complete annotated evidence is present in 92 first views and 111 combined views. All combined views fit the reader in this run. Counts below are true support / false support on refutation / false support on unknown, with denominators 97 / 47 / 82.

| Reader | First only | Combined one call | Maximum of separate calls | Selective combined | Selective second calls | First / selective P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | 33 / 4 / 4 | 30 / 4 / 3 | 37 / 4 / 5 | 37 / 5 / 6 | 20 / 226 | 1.38 / 2.65 ms |
| Rationale-pair seed 20261119 | 29 / 2 / 5 | 31 / 4 / 4 | 31 / 2 / 5 | 31 / 2 / 5 | 3 / 226 | 1.37 / 1.37 ms |
| Rationale-pair seed 20261120 | 29 / 2 / 4 | 30 / 4 / 4 | 32 / 2 / 4 | 30 / 2 / 4 | 2 / 226 | 1.36 / 1.38 ms |

Job 42256 repeats job 42251 and reproduces its thresholds and decisions, while recording calibration review counts and length-limit checks. The fitted intervals request 11, 6, and 6 second calls among 76 calibration cases, respectively. Extra evidence adds 19 complete annotated sets but does not reliably improve the fixed combined verdict. In the trained seeds selective expansion recovers two and one true supports with unchanged observed false-support counts relative to their first-window controls. The paired intervals include zero. Retention remains below frozen first-window verification. The broader separate-read policy doubles the number of evaluated views on 88 requests and pushes factored P95 above 2.57 ms. These operating points do not establish a risk-constrained improvement over the strongest control. [Aggregate results](results.json)

Use the pinned SciFact archive and the weights and training aggregate from the [short-window supervision study](../scifact-minilm-window-supervision-v1/README.md). Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.selective_window_expansion \
    SCIFACT_ARCHIVE TRAINING_RESULTS WEIGHTS MINILM_CHECKPOINT NEW_RESULTS
```
