# Treat partial relevance as adverse in verification routing

Reuse the [disjoint RELISH panel](../relish-risk-routing-v1), the
same 100 held requests and the same candidate generation. The
previous evaluation treated only grade-0 issuance as adverse. This
experiment treats a request as adverse if **any issued source is
grade 0 or 1**. A strong return still requires an issued grade-2
source. A request can contribute to both counts. This models a
requirement for fully relevant evidence in each issued slot.

Use the same fit/risk/held split. Refit the cosine and compact
cross-encoder thresholds with both lower grades in the negative
class. Use the stricter adverse indicator in the learned
benefit target, with the same half-point penalty and 30% fit
fallback target as before. Calibrate every arm against the same
10% adverse-request ceiling on the independent risk subset.
Keep the held set fixed until the policies are frozen.
Add fixed dense eight so no adaptive method gains by comparing
only with verification at half the candidate depth.

The held equal-risk diagnostic retires already issued results
in order of low normalized confidence until routes meet a common
upper bound. The common cap is the smallest adverse count among
learned, dense four, dense eight, pure cosine and live hash
routing. It does not substitute a new result for a retired one.
Report both paired raw and risk-matched strong-return contrasts
against the four alternative policies. Resample by seed query.

The live hash threshold is fitted to the same intended fallback
rate, so its realized compute can differ. As a second diagnostic,
perform 2,000 random reallocations of the learned policy's
exact number of four-read fallbacks. Use live pure-cosine and
compact-four score observations with the same learned issuance
shift. This estimates the distribution of strong and adverse
counts at the same check budget without generating timing figures
for those replays. Warm timing refers to live routes only.

## Results

Both strict tasks of Slurm array `45777` on the ws3 RTX 5090
complete, with two rotated execution orders per task. All four
passes reproduce every non-timing result. All schedules meet
five adverse requests out of 55 on the calibration subset. On 100 held requests, the shared matched
cap is five adverse requests. The strict definition produces
substantially fewer strong returns than the earlier grade-0-only
comparison.

| Arm | Strong queries, frozen | Adverse queries, frozen | Strong queries, matched cap | Verified requests |
| --- | ---: | ---: | ---: | ---: |
| Dense eight | 27 | 8 | 16 | 100 |
| Dense four | 27 | 5 | 27 | 100 |
| Compact four | 25 | 5 | 25 | 100 |
| Cosine | 46 | 9 | 30 | 0 |
| Hash routing | 47 | 9 | 29 | 31 |
| Margin routing | 36 | 9 | 24 | 44 |
| Learned routing | 36 | 9 | 30 | 24 |

At the frozen thresholds, learned routing has 36 strong requests
versus 27 for fixed four but incurs nine adverse requests versus
five. After matching risk, its advantage over fixed four is
three percentage points (paired 95% interval −6 to +13).
The learned and pure-cosine paths match on every request's
strong-return outcome after trimming. Warm P95 is 8.59–8.68 ms
for pure cosine, 28.11–28.29 ms for learned routing, and
31.43–31.54 ms for fixed four. Fixed eight takes 50.37–52.15 ms. Compared with hash routing
the matched change is +1 point (−5 to +7). Learned allocation's
exact-budget random replay has a central strong-return range
of 28 to 35 and adverse range of 5 to 10. Learned routing returns
36 strong results and nine adverse results at its frozen point.

The grade-0-only positive quality–cost result does not carry
over as a clear gain against the pure-vector control when partial
relevance is adverse. Keep the stronger low-cost control.
This corpus supplies paper relevance judgments. Whether an
expert satisfies an individual qualification still needs original
source evidence and an independent judgment.

[Results for both strict runs](results.json).

## Reproduce

Use the original frozen RELISH panel and checkpoints.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.csfcube.relish_risk_routing \
  PANEL MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --adverse-grade 1
```

Run on a Slurm GPU. With the default grade-0 definition, the
same runner preserves the previous comparison.
