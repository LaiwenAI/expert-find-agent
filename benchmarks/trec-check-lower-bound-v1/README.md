# Minimum verification work on judged enterprise-search pools

This replay uses the 49 judged [TREC 2006 Enterprise topics](https://trec.nist.gov/data/t15_enterprise.html) and their distinct expert and supporting-document judgments. It holds ten candidate experts per topic fixed under the existing leave-topic-out prior and uses the same 12 seeded source orders as the [certified dispatch replay](../trec-expert-certified-stop/README.md). Only grade 2 supports a candidate. The perfect verifier exposes that label on each check.

For a returned confirmed expert, exact citation order requires checking the entire prefix through its last cited witness. For an unconfirmed expert who could displace the current output, ruling out a requirement of `s` accepted sources requires at least `max(0, number_of_sources - s + 1)` negative checks. An expert below the last of five confirmed candidates can be ignored. Add these per-expert requirements to obtain the minimum possible number of checks for the realized labels. Each label is an independent possible outcome before observation. The oracle knows the answers, so its count can be below that of any implementable online schedule.

| Required sources | Oracle minimum checks | Prior-order checks / waves | Two-source prefix checks / waves | Four-source prefix checks / waves | Supported precision at five |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 224.58 | 227.50 / 75.98 | 226.85 / 62.24 | 229.40 / 57.59 | 46.53% |
| 2 | 254.32 | 257.78 / 81.59 | 259.01 / 70.03 | 261.86 / 65.59 | 42.45% |

The most economical fixed schedule is within 1.01% and 1.36% of the lower bound, respectively. The four-source prefix uses fewer waves and more checks. All schedules reproduce the oracle expert order and citations. The bound counts checks of expert–document associations, even when fetching a document can be shared across experts. The qrels define the visible candidate-source inventory, and the benchmark uses no original source text or model. These are counts under a perfect-verifier replay, rather than end-to-end latency. [Aggregate results](results.json)

## Reproduce

Download both expert qrels files from the NIST track page and run:

```sh
PYTHONPATH=src:. python -m experiments.trec_enterprise.check_lower_bound QRELS_DIR NEW_RESULT_JSON
```
