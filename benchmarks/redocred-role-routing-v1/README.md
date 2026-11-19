# Route person-specific verification without annotated evidence at inference

Replay the development and test pairs from the
[Re-DocRED person-specific benchmark](../redocred-person-evidence-v1).
Compare the purpose-built 0.6B reranker, 4B instruction reader and a
lexical control. Use the same two sentences and two named-person questions
for all policies within each pair. The selector maximizes lexical
coverage of both names, work and role cues and breaks ties by the weaker
name's coverage and then source order. The window length is fixed at two,
including when an annotated evidence location or length is unavailable.
Selection and routing receive neither evidence annotations nor gold roles
for the people.

The lexical score for each person is the maximum over exact name
occurrences in the window of one divided by one plus the distance in
tokens to the nearest same-sentence role cue. Distance is measured from
the nearer edge of the name. With no same-sentence match the score is zero.
These cues are a baseline for proximity, which may be high when a
sentence attributes the actual role to another person.

The cascade starts with either lexical scores or the frozen 0.6B
reranker. If either person's score falls in its calibrated ambiguity
interval, the 4B reader scores the pair in one batch and its cutoff
determines both decisions. Otherwise the fast scorer accepts only scores
above the upper bound. Fit the two gates for development call ceilings
of 50% and 75% of pairs. Search symmetric bands centered on observed
development scores, with radii from distances to those scores. Choose the
4B cutoff based on the remaining allowed rival acceptances. Optimize
both-correct pairs, breaking ties by fewer rival acceptances, more
supported acceptances and fewer reader calls. All policies have the same
ceiling of eight development rival acceptances, including the standalone
baselines. Decisions require strict threshold exceedance. Record the
held-out routing and acceptance rates without tuning.

Rerun the route with its frozen thresholds on the GPU. Measured time
covers two-sentence selection, prompt preparation, inference and
synchronous serial fallback where used. Model loading, corpus search,
queueing and batching across requests are outside the timed interval.
Report model calls as pair-batch forwards. A reduction in calls or mean
latency is separate from a reduction in tail latency.

Use a deterministic hash control to test whether the lexical ambiguity
band identifies useful cases for review. Calibrate it to the same
observed development reader-call count as the 75% lexical gate, use
the same fast cutoff and recalibrate its reader cutoff under the same
rival-acceptance cap. Held-out call counts can differ. This comparison
was added after inspecting the first run and is exploratory on the
reused test panel.

## Results

Slurm job 42989 on RTX 5090 reproduced the original seven policies'
decisions and call counts from job 42985. Test contains 163 pairs from
41 documents. Ceilings in the policy names apply on development.

| Policy | Both correct / 163 | Supported accepted / 163 | Rivals accepted / 163 | 4B calls | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical only | 23 | 23 | 0 | 0 | 0.24 ms | 0.33 ms |
| 0.6B only | 12 | 14 | 9 | 0 | 8.47 ms | 9.10 ms |
| 4B only | 92 | 93 | 1 | 163 | 22.40 ms | 25.40 ms |
| Lexical, 50% ceiling | 83 | 84 | 1 | 77 | 10.69 ms | 25.11 ms |
| Lexical, 75% ceiling | 97 | 98 | 1 | 108 | 15.06 ms | 25.59 ms |
| Hash control | 78 | 81 | 4 | 99 | 13.59 ms | 25.21 ms |
| 0.6B, 50% ceiling | 68 | 69 | 6 | 81 | 19.84 ms | 34.60 ms |
| 0.6B, 75% ceiling | 91 | 94 | 5 | 125 | 25.71 ms | 34.35 ms |

The lexical gate at the higher ceiling reduced 4B calls by 33.7% and
mean latency by 32.8% against always using 4B. Observed P95 was nearly
unchanged. Both policies accepted one rival. The both-correct gain was
+3.07 percentage points (paired 95% interval −6.04 to +12.73, clustered
by source document). This panel does not establish a quality gain or
noninferiority.

The lexical gate and hash control each called 4B on 119 of 172
development pairs. On test, the lexical gate solved 19 additional
pairs and made three fewer rival acceptances, while calling 4B nine
more times. The exploratory paired difference was +11.66 points
(interval +3.23 to +20.55). The held-out call-count difference limits
the interpretation as a same-budget advantage. At the lower lexical
ceiling, the 4B call count drops by 52.8% relative to always using 4B,
with nine fewer both-correct decisions.

The sequential 0.6B-first routes ran the 0.6B model on every test
pair and increased P95 by about 9 ms. The fast stage's extra GPU call
therefore needs to be included in the comparison. This is a
within-document attribution test with two named candidates. The
rival's requested role is absent from the revised label inventory,
which can still contain omissions. The test panel was reused from
the preceding diagnostics and provides a candidate gate to evaluate
on a new corpus or expert-search panel.

[Aggregate results](results.json)

## Reproduce

Use the dataset and pinned checkpoints from the
[oracle evidence diagnostic](../redocred-oracle-role-evidence-v1).
Run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.route_role_verification \
    REDOCRED_DIRECTORY QWEN_RERANKER QWEN_READER NEW_RESULTS.json
```
