# Research Reproduction Entry Points

Run all commands from the project root; model training is not required.

| Artifact | Command or module | Prerequisite |
|---|---|---|
| Author-recovered baseline | python -m expert_search benchmark | corpus, SQLite index |
| SciFact lexical | python -m expert_search.research_cli public-lexical | Download and adapt the data |
| SciFact dense/RRF/progressive quality | python -m expert_search.research_cli public-hybrid | API corpus and query vectors |
| Progressive computation latency | python experiments/progressive_latency.py | Encoded vectors, single-threaded BLAS |
| CMU lexical | python -m expert_search.research_cli cmu-lexical | Fixed upstream d_20_1 |
| CMU frozen profiles | python -m expert_search.research_cli cmu-dense | CMU API vectors |
| CMU cards | python -m expert_search.card_benchmark | Generated and encoded cards |
| CMU online evidence | cmu_evidence_benchmark.run(limit=58) | Model service configured |
| Same-subset model comparison | python -m expert_search.cmu_compare | Recorded request results |
| Controlled-task construction | python -m expert_search.controlled_data | Local teacher corpus |
| Controlled-task evaluation | python -m expert_search.controlled_benchmark | Constructed data and fast index |
| Local BGE | python -m expert_search.local_encoder_benchmark | Optional local dependencies, download frozen model |
| Results-table synchronization | python -m expert_search.research_report | Completed experiment files |

Commands require setting `PYTHONPATH=src`. Local encoding timings use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, with model ONNX threads=1. Retesting is required if the CPU, service, or data scale changes.

See public-benchmark.md for data acquisition and licensing information. Model prompts and cache versions are in the source code; model outputs, failures, and hashes are under artifacts. This directory includes third-party data and should not be added wholesale to an open-source repository. Extracted text from the original PDFs and teacher materials are also not part of this project's code license.


## Additional Reproduction Entry Points

|Artifact|Command|Scope|
|---|---|---|
|CoF classic language model|python -m expert_search.cof_language_benchmark --dataset NIPS|Score all profiles; evaluate on the judged set|
|Exact compiled pair timing|python experiments/compiled_lm_benchmark.py|Run SIGIR/NIPS sequentially; three repetitions do not count as new quality samples|
|Complete teacher index|python experiments/teacher_compiled_lm.py|Includes saving, loading, and three randomized pairs; no industry-accuracy gold standard|
|Persistent retrieval|python -m expert_search.compiled_expert_search 'robot motion control'|Initialization and query reported separately; measure topic relevance|
|Shared-evidence statistics|python -m expert_search.packing_analysis --dataset NIPS --serialization shared|Completeness, failure fallback, paired differences, P95|
|Local-set aggregation|python -m expert_search.cof_late_interaction --dataset NIPS|Requires local environment; existing MaxSim component, not a claim of originality|
|Semantic synthetic candidates|python -m expert_search.semantic_synthesis_pool|local environment; no model API during preparation|
|Candidate-generation review|python -m expert_search.semantic_synthesis_pool --execute|Real model calls, with source and task hash checks|
|Per-condition re-review|python -m expert_search.scoped_dataset_audit|Complete condition IDs, unique source references, and requirement references; still silver|
|Re-review export|python -m expert_search.scoped_dataset_audit --export|Re-validate model outputs and source artifacts; does not export original text|

Frozen local model commands use `.venv-local/bin/python` or an environment installed with the project's optional local dependencies. All timing commands require an explicitly set single-threaded environment; running timing tasks in parallel while sharing the CPU changes the measurement conditions.
