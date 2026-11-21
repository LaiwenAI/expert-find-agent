# Calibrate source existence and two-role proof separately

Freeze both gates on PLOS papers and transfer them to the
[53-paper engineering corpus](../engineering-cross-source-search-v1).
The evaluation uses 15 positive requests, 15 role-to-source swaps,
and 60 versions with one of the requested sources hidden. The 15
families come from four article pairs. Each query supplies two source
descriptions and their clause boundaries. The gold proof requires the
designated sources, exact contributor set, and both requested roles
supported by the declarations.

## Calibration

- **Source existence.** On the [171-paper PLOS projection](../plos-scoped-search-v1),
  rank each of 24 clue probes with the target available and with it
  hidden. Fit separate class-balanced ridge scores for excerpts and
  keywords on the fused top-six candidates. Features are cosine
  similarity, distance to the dense leader, IDF-weighted lexical
  coverage, the leading dense score gap, and reciprocal fused rank.
  Threshold each score on the maximum for each hidden-target probe
  to allow at most one of 24 probes to accept a candidate. At test
  time apply the threshold before the ORCID join. An anchored
  baseline requires one original fused leader and puts both sources
  within 0.04 cosine of their respective dense leaders. All routes
  resolve the source pair before scoring the requested roles.
- **Two-role proof.** Keep the [ridge and prototype projections](../engineering-role-banks-v1)
  frozen. Standardize the role scores for each clause and convert them
  into probability weights with temperature scaling. Multiply the
  weights assigned to an author's declared roles across the two
  sources. Cite the highest-weight declared role on each source.
  Choose temperature and threshold on 48 scripted PLOS cases over
  three linked paper pairs, including scope swaps and within-source
  role replacements. Calibration uses 14 original role descriptions.
  The engineering requests use independently authored NISO examples.
  Count a result as adverse when any issued contributor or cited role
  is wrong. Allow at most 5% adverse calibration cases; both banks
  retain 24 strict positives with zero observed adverse cases in
  calibration.

The fitted calibration counts describe operating points on the same
PLOS data used to fit them. Paper pairs, roles and query wordings repeat.
The counts cannot support an independent risk estimate.

## Transfer results

With the intended roles supplied, measure strict positive proofs and
wrong-source outputs in the hidden-source views. Each hidden-source
column pools cases with the left and right paper removed separately.

| Source policy | Excerpt proofs / 15 | Excerpt substitutions / 60 | Keyword proofs / 15 | Keyword substitutions / 60 |
| --- | ---: | ---: | ---: | ---: |
| Linked | 15 | 5 | 15 | 5 |
| PLOS-calibrated + linked | 15 | 0 | 5 | 0 |
| Anchored | 15 | 0 | 10 | 0 |

With the two designated sources supplied, measure the following strict
role results. Exact authors alone gives credit when the right people
are returned for the wrong roles.

| Role policy | Exact authors / 15 | Strict proofs / 15 | Adverse / 30 | Swap outputs / 15 |
| --- | ---: | ---: | ---: | ---: |
| Ridge single-role band | 1 | 1 | 0 | 0 |
| Prototype single-role band | 4 | 4 | 0 | 0 |
| Ridge two-role score | 10 | 5 | 8 | 3 |
| Prototype two-role score | 8 | 4 | 6 | 2 |

The fitted source filter matches the anchor on excerpt correctness and
loses five additional keyword positives. The learned two-role score
also fails the transferred risk target even with correct source
locations. Keep the simple source anchor as the baseline. Use
held-out paper pairs with coauthor and role confounders to test
clause-level role judgments before combining the two sources. Evaluate
whether a low-confidence source should trigger a bounded backup read
by counting both recovered proofs and new wrong-source outputs.

Both runs of Slurm array 45907 on ws3 RTX 5090 reproduce every
non-timing field. Warm local P95 is approximately 10–13 ms. The
timer includes a live batch encoding both clues and both role
phrases, ranking, source selection and contributor scoring. Index
construction, network transport and unconstrained query parsing
are outside the timer. Each arm is timed as a factored path through
shared encoding and ranking. These results describe four target
paper pairs with publisher-declared responsibilities.
[Full results](results.json)

## Reproduce

Restore the engineering panel and PLOS projection using the linked
benchmark instructions. The runner checks the PLOS projection hash.
Use the previous engineering search results for the frozen single-role
bands. Run on a Slurm CUDA node and use a new output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.engineering_joint_calibration \
  ENGINEERING_PANEL PLOS_PROJECTION NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  QWEN_EMBEDDING NEW_RESULTS --offset 0
```

Repeat with offset 1 to reverse the request order.
