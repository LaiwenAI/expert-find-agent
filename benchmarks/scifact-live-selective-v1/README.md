# Live selective evidence verification, batched and single-request

Use the same 309 calibration pairs and 226 development pairs as the [SciFact verifier study](../scifact-criterion-calibration-v1/README.md). The short verifier is frozen mDeBERTa on the [three-sentence lexical window](../scifact-nli-windows-v1/README.md). The slow verifier is the direct Qwen3-4B three-way reader on the full supplied abstract. Both are resident on the same RTX 5090.

Calibrate each standalone model with a support logit margin and separate 5% false-support caps for contradictions and unknowns. The selective policy learns two short-verifier margin cutoffs from a fixed quantile grid: accept strong supports, withhold support at low margins, and send the middle band to the full reader. Calibrate that policy to retain the most true supports subject to the same 5% caps and a 50% fallback-call cap. The capped comparison uses a case-ID hash to make half the pairs eligible for fallback, then chooses its own band under the same calibration criteria. Calibration limits are empirical and do not fix test-time rates.

## Live evaluation

Run every route with fresh model forwards. Rotate the execution order per batch and per single request. Include lexical window construction, tokenization, selection, inference and actual fallback. Calibrate in batches of up to 16 and reuse all thresholds unchanged for single requests. Times cover the evidence-verification component with supplied abstracts. Retrieval, queueing, networking and startup are excluded.

| Route | True support / false support, batch | Full reader inputs, batch | Batch P95 | True support / false support, single | Full reader inputs, single | Single P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short only | 20 / 0 | 0 | 22.5 ms | 21 / 0 | 0 | 4.66 / 4.84 ms |
| Full only | 48 / 6 | 226 | 796.7 ms | 49 / 6 | 226 | 26.32 / 37.83 ms |
| Band with hash eligibility | 31 / 1 | 93 | 529.8 ms | 31 / 1 | 93 | 4.82 / 40.26 ms |
| Selective band | 39 / 4 | 101 | 426.2 ms | 39 / 4 | 103 | 4.82 / 39.81 ms |

True and false support denominators are 97 and 129. The selective band reduces batch P95 by 46.5% relative to full verification, with nine fewer true supports and two fewer false supports. It retains eight more true supports than the hash-eligibility control, alongside three additional false supports and eight additional full reader inputs. Separately, on single requests, the selective band lowers the median by 81.7% and increases P95 by 5.3% relative to full verification. Near-threshold decisions change with batch geometry: the selective policy dispatches 101 inputs in batches and 103 serially. The pilot batch results in job 41290 reproduce the final support counts and fallback counts exactly.

The A/B timing modes expose different trade-offs. Under single-request execution, the fallback overhead offsets the P95 savings from fewer full-reader calls. The reported cutoffs and error rates do not establish equal quality or safe transfer to enterprise expert attribution. [Aggregate results](results.json)

## Reproduce

Use the pinned [mDeBERTa checkpoint](../scifact-small-nli-v1/checkpoints.json), the pinned Qwen reader and the official SciFact archive. Run within a CUDA Slurm allocation:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.selective_evidence_verifier \
  --archive-path SCIFACT_ARCHIVE --short-checkpoint MDEBERTA_CHECKPOINT \
  --full-checkpoint QWEN_4B_CHECKPOINT --output NEW_RESULTS
```

Job 41293 ran both timing modes on ws5. This release contains aggregate metrics and benchmark code.
