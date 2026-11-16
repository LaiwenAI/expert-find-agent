# Source release preparation

The repository contains runnable retrieval, synthesis, evaluation and public-dataset adapters under the MIT software license. It has not been published as an open-source release; the configured target `LaiwenAI/expert-find-agent` was verified as a private, empty repository before source synchronization. The code license does not grant rights to third-party datasets or literature.

Build a local source review archive with:

```bash
python3 experiments/build_release.py
```

The archive includes source, tests, experiment scripts, paper drafts and usage documentation, plus SHA-256 hashes for each file. It excludes credentials, local environments, downloaded corpora, model weights, real enterprise/teacher data, evidence quotations and model traces. This archive is for reviewing the implementation; it is not a complete benchmark/data release.

After installing the package, run offline unit checks with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

Public benchmark adapters download SciFact and CMU Gold separately. Their annotations and source texts retain their original terms; an adapter does not grant redistribution rights. The local synthetic candidates are model-audited silver diagnostics, not human gold. Original industrial requirements, narrowed capability subtasks, unknown evidence labels and source references must remain distinguishable in any eventual dataset release.

Before public distribution, document dataset-specific rights and provenance, and publish a reproduction manifest identifying exact artifact versions. The review archive deliberately makes no claim that these remaining release requirements are already met.
