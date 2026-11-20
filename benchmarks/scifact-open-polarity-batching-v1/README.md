# Test length-aware batching at fixed scientific verification quality

After the [verification cascade](../scifact-open-polarity-cascade-v1)
lost correct decisions, retain the full Qwen3-4B reader for every
evidence item. Use the same 195 human-judged claim–abstract pairs,
input prompts and calibrated cutoffs as the
[original polarity benchmark](../scifact-open-polarity-gate-v1).
The FIFO reference must reproduce the earlier aggregate decisions.

Process windows of 32 consecutive evidence items, with forward
batches of up to 16. Three methods share the model, prompts, data
order on input, and output order.

- **FIFO:** retain input order throughout.
- **Length sorted:** within each window, group similar token lengths
  to reduce left padding. Restore the original order after scoring.
- **Guarded length sorted:** perform the same grouping. When either
  stance margin is within a fixed distance of its decision cutoff,
  re-run the original FIFO batch that contains that item and use
  all outputs from that replay. Set the guard distance to the
  maximum absolute change in either margin between FIFO and
  length sorting on the 263 calibration items. This is an empirical
  guard and is not a bound on all possible outputs.

Measure processed padded tokens including replays, decisions
against FIFO for both requested stances, and window completion
time from tokenization through output assembly. The timer starts
with all items already available. It excludes arrival queueing,
model loading, document retrieval and service transport. Report
the per-window P95 descriptively across seven windows.

## Results on RTX 5090

Both tasks of Slurm array `44748` on `ws5` agreed on all decision
and token counts. Timing ranges span reversed-order runs.

| Method | Executed padded tokens | Changed decisions / 195 | Replayed FIFO batches | Window P95 (ms) | Sum of window times (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FIFO | 122,164 | 0 | 0 | 1249–1257 | 6877–6920 |
| Length sorted | 106,500 | 3 | 0 | 1033–1037 | 5912–5936 |
| Guarded length sorted | 207,044 | 0 | 10 | 2192–2197 | 11561–11581 |

Sorting removes 12.8% of padded tokens and speeds up the ready
windows. It changes three evidence decisions, including one
additional opposite-stance acceptance, while the count of correct
acceptances stays at 99. The guard width of 1.75 logit points
replays 10 original batches, reproduces every FIFO decision on
this panel, and costs more tokens and time than FIFO. Decision parity does not imply identical scores. Twenty items still
have a different guarded margin. This makes the guarded route
unsuitable as a drop-in acceleration here, although the workload
shows useful padding headroom for alternative execution designs.

[Aggregate results](results.json).

## Reproduce

Use the pinned evidence projection, 4B reader and original gate
results. Run on a Slurm GPU and repeat with `--order-offset 1` to
reverse timed method order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.polarity_batching \
  PROJECTION QWEN_4B_READER \
  benchmarks/scifact-open-polarity-gate-v1/results.json NEW_RESULTS
```
