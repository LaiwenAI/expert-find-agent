# Sensitivity to clarification reply errors

Replay the [majority-gated clarification benchmark](../plos-role-clarification-v1/README.md) over the same 120 requests. The gate asks on 62 requests. Nine asked clauses have the intended role outside the proposed options. Enumerate all reply outcomes and calculate expected counts exactly under assumed response channels.

- With a wrong reply, choose either the highest-scoring incorrect option or one uniformly at random from the incorrect options.
- With no reply, leave the original response unresolved.
- Open reply allows the intended role outside the menu. Menu-only reply records no answer when the intended role is absent.
- Keep-rival verifies both the reply and the strongest admissible alternative. Confirmation asks again only if accepting the first reply would resolve the request. It requires the same answer twice. The independent channel draws again. The partially shared channel copies the first response half the time and draws independently otherwise.

## Results

Assume a 10% reply error rate, always receiving an initial reply and errors selecting the strongest rival. Counts are means over the specified channels on this fixed panel.

| Route | Fully supported positive | Unsafe positive | Correct no match | False no match | Unresolved | Follow-up questions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trust open reply | 59.50 | 4.70 | 15.60 | 0.30 | 39.90 | 0 |
| Trust menu reply | 58.60 | 3.80 | 12.00 | 0.30 | 45.30 | 0 |
| Keep strongest rival | 49.40 | 1.90 | 3.00 | 0.00 | 65.70 | 0 |
| Confirm, independent | 55.63 | 0.47 | 14.34 | 0.03 | 49.53 | 49.10 |
| Confirm, partially shared | 57.56 | 2.59 | 14.97 | 0.17 | 44.72 | 49.10 |

With accurate open replies, all per-request outcomes reproduce the original gate. Restricting accurate replies to the menu loses one supported positive and four correct no-match responses. Under 20% nonresponse and 10% top-rival error, open replies yield 53.20 supported positives and 3.76 unsafe positives. Retaining the strongest rival lowers unsafe positives to 1.52 with 45.12 supported positives. Additional confirmations depend strongly on error correlation and consume another interaction. The benchmark measures sensitivity to specified assumptions. Actual reply, nonresponse and repeat-error rates remain unmeasured.

## Reproduce

On a Slurm GPU compute node, use the same public inputs as the linked clarification benchmark:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.clarification_reply_sensitivity \
  PUBLISHER_FIXTURE NISO_EXAMPLES ROLE_TRAINING EMBEDDING_CHECKPOINT \
  SCRATCH_DIRECTORY NEW_RESULT_JSON
```

[Aggregate results](results.json) include the complete error and nonresponse grid and pinned implementation hashes. Per-request responses stay local.
