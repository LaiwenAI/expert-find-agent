# Source release preparation

The repository is public at [LaiwenAI/expert-find-agent](https://github.com/LaiwenAI/expert-find-agent), with `research/prototype-20261117` as its default branch. It contains runnable retrieval, synthesis, evaluation and public-dataset adapters under the MIT software license. Unauthenticated access was verified after publication; the publication snapshot and history audit are recorded in [source-publication-20261117.json](exploration/source-publication-20261117.json). This is a research prototype, with no independently demonstrated joint expert-accuracy and end-to-end speed advantage. The software license does not grant rights to third-party datasets or literature.

Build a local source review archive with:

```bash
python3 experiments/build_release.py
```

The archive includes source, tests, experiment scripts, paper drafts, usage documentation and the [OpenAlex temporal author-recovery proxy](../benchmarks/openalex-author-recovery-v1/README.md), plus SHA-256 hashes for each file. The proxy contains only selected OpenAlex CC0 metadata and known-author associations; it is not an enterprise-fit benchmark. The archive excludes credentials, local environments, other downloaded corpora, model weights, real enterprise/teacher data, evidence quotations and model traces. It is not a complete enterprise benchmark/data release.

After installing the package, run offline unit checks with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
```

Public benchmark adapters download SciFact and CMU Gold separately. Their annotations and source texts retain their original terms; an adapter does not grant redistribution rights. The local synthetic candidates are model-audited silver diagnostics, not human gold. Original industrial requirements, narrowed capability subtasks, unknown evidence labels and source references must remain distinguishable in any eventual dataset release.

Before distributing source datasets, document dataset-specific rights and provenance, and publish a reproduction manifest identifying exact artifact versions. The review archive deliberately makes no claim that these remaining release requirements are already met.
