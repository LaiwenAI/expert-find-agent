# Adaptive evidence depth across supplied trial sections

The [fixed-budget comparison](../budgeted-evidence-selection-v1/README.md) found that many NLI4CT evidence sets exceed the initial sentence budget. This study asks whether a small offline-trained controller can choose which source to expand and when to stop, while improving complete evidence coverage at the same aggregate read cost. It operates on the trial IDs and section type supplied by the benchmark. [NLI4CT, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.1041/)

Start with the same lexical windows. Each expansion adds up to three unread source lines to one trial. The lexical control ranks expansions by newly covered query terms and proximity. The learned controller uses a 14-feature ridge regressor to rank candidates by estimated marginal benefit per added sentence. Its training target combines the fraction of annotated evidence added and an indicator that the expansion completes the source's annotated set. Input features include query overlap, matched numbers, proximity, and the amount of source text left unread. Both routes can take up to three expansions. A single cutoff for each route is selected on development data to maximize complete coverage at a mean cost no greater than one fixed lexical expansion there. The cutoff and coefficients are fixed for test-time decisions.

The official train split contains trial IDs that also occur in development or test. Exclude every training statement that refers to any trial in either split. Fit on 1,200 remaining statements and 45,013 candidate expansions generated along lexical rollouts. The resulting train, development, and test sets cover 618, 100, and 249 distinct trials with zero pairwise trial overlap. Test labels are used after the selection paths have been generated. The same test split served as an aggregate fixed-budget reference in the preceding study.

## Gold test

| Route | Mean sentences per statement | Complete evidence / 500 |
| --- | ---: | ---: |
| Initial windows | 6.76 | 80 |
| Lexical, adaptive depth | 8.55 | 110 |
| Learned, one fixed expansion | 9.31 | 179 |
| Learned, two fixed expansions | 11.44 | 264 |
| Learned, three fixed expansions | 13.24 | 317 |
| **Learned, adaptive depth** | **9.93** | **268** |
| Best equal-cost query-independent mixture | 9.93 | 204.8 expected |

The learned adaptive policy recovers complete evidence for 132/229 single-trial and 136/271 two-trial statements. It reads 13.2% fewer sentences on average than two learned expansions and recovers four additional sets. The matched-cost reference selects the strongest query-independent mixture of any two fixed-depth baselines on the test set. Its best mixture runs two learned expansions with probability 0.678 and zero expansions otherwise. With that mixture held fixed, the paired difference is +12.6 percentage points, with a 95% bootstrap interval of +9.0 to +15.5 points across 113 connected trial components. This is a descriptive interval conditional on the selected test-cost mixture. The lexical adaptive policy underperforms its own equal-cost fixed-depth mixture, suggesting that remaining query overlap is insufficient as a stopping score on this task.

This experiment evaluates gold evidence coverage with supplied sources, before model verification. The full three-step controller replay has a component P95 on the order of 0.4 ms on the development machine. The timing excludes document retrieval, reader calls and serving overhead. Complete annotation coverage is a proxy for available evidence. Expert recommendation quality and end-to-end latency require their own paired evaluation.

## Reproduce

Use the [pinned official archive](https://github.com/ai-systems/nli4ct/blob/8e8e5fc506b916984011873403b9ea5ed6dd96c2/Complete_dataset.zip) and a new output path. This experiment runs on CPU.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.nli4ct_adaptive_depth NLI4CT_ARCHIVE NEW_RESULTS
```

[Aggregate results, coefficients and calibration cutoffs](results.json)
