# Native suitability factorization: negative result

A complete610-pair experiment reruns native binary classification against a factored native call on the same official MERIT target/profile inputs. Both use gpt-6-luna, alternate arm order, three workers and no retries. Rating fields are excluded from inference. This is exploratory after earlier aggregate benchmark observations, not a pristine final test or reproduction of MERIT's trained checkpoint.

The factors ask whether at least one core requirement is demonstrated and whether at least two distinct relevant requirements are demonstrated. Three-valued conjunction produces a positive only when both pass, a negative when either fails, and otherwise an abstention. The factored arm adds an uncertainty option to the primitive answer space; this comparison does not isolate factorization alone. Semantic consistency between the predicates is not mechanically guaranteed.

| Arm | Valid /610 | Valid accuracy | Valid precision | Valid recall | Mean attempt seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contemporaneous binary | 597 | 73.37% | 66.54% | 72.69% | 1.301 |
| Factored | 600 | 68.50% | 58.50% | 84.00% | 1.397 |

There are13 and10 unresolved requests, respectively; neither arm explicitly abstains. The jointly valid587-pair accuracy difference is−4.43points, exact-paper-cluster interval[−7.52,−1.34]. Full-cohort accuracy bounds are[71.80%,73.93%] versus[67.38%,69.02%], with factored-minus-binary bounds[−6.56,−2.79]points. Thus favorable completions of the missing factored outputs cannot reverse the observed comparison under these bounds. FP rises from91 to149. Mean time difference+0.096s has interval[−0.021,+0.211]s; no speed improvement is demonstrated.

The implementation remains an exploratory comparator, not a production default. It tests whether a fast explicit task-rule decomposition helps, and finds that independent predicate judgments are insufficient here. It does not establish that every factorization fails, that either model's probabilities are calibrated, or that self-rated reviewer suitability is industrial transfer truth.

Code: `src/expert_search/factored_suitability.py`, `experiments/merit_factored_comparison.py`, `experiments/analyze_merit_factored.py`. Authority: `artifacts/merit-factored-comparison-v1/{protocol,evaluation,progress}.json` and all1,220 terminal prediction records. Missing outputs remain missing and are not imputed as negative.
