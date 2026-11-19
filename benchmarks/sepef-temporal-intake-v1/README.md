# SE-PEF temporal expert retrieval intake

The official SE-PEF release contains historical answer ownership, validation and test relevance judgments, and text for the underlying questions and answers. The relevance label identifies the released answerer for a question. It does not judge whether a prior answer is adequate evidence for a specific requirement. We use only activity counts, answer text and reply ownership. [Author dataset](https://zenodo.org/records/8332748) · [Author preprocessing](https://github.com/pkasela/SE-PEF/blob/main/se-pef_data/data_creation.py)

The audit verifies that all stored profile-answer timestamps precede the authors' 2019 training cutoff and that no validation and test question IDs overlap. The released author universe was selected by the original authors before their test boundary. Thus the validation results use the same published candidate universe as test. No profile-answer feature uses validation or test answers.

Compare two text-free priors over the released historical metadata. Global ranks experts by historical answer count. Community ranks only experts with answers in the query's site, by their answer count in that site. Both are computed from either full pre-cutoff history or a cap of eight most recent answers per expert. Score every released validation and test qrel, retaining zero scores for missing labels. Community scope is taken from the qrel question ID.

## Metadata results

The released universe contains 3,306 experts, of whom 3,275 have at least one historical answer in the cutoff snapshot. Their 564,690 answers span 50 communities. The recent cap retains 26,123 answers. Validation contains 16,710 qrels and test has 19,995.

| Test prior | Test labels reachable / 19,995 | Hit@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| Full history, global | 19,090 | 6.69% | 0.0462 |
| Full history, community | 18,100 | 38.79% | 0.2455 |
| Recent eight, global | 19,090 | 0.015% | 0.00005 |
| Recent eight, community | 17,020 | 5.72% | 0.0291 |

The full-history community prior is a strong text-free control and loses access to 990 labeled test questions relative to global scope. Capping history skews the activity prior and lowers community reachability further. Preserve full-history counts when bounding the answer text index. [Aggregate audit](results.json)

## Text retrieval

The stream builder reads only required ZIP members from the official archive, checks member CRCs and validates question timestamps and label agreement. It hash-samples 256 validation and 512 test questions whose labeled experts have historical profiles. It retains eight recent source answers per expert, writes local text assets, and records source hashes. The subsequent GPU harness compares BM25, exact dense over the full collection, exact dense and centroid scoring within community, and a community hybrid. Each returns an owned historical answer. Results use one RTX 4090 and a single warm request at a time. Timing includes query cleanup, encoding where used, document and expert scoring, and serialization. It excludes network, queueing, initialization and offline index construction.

| Test route | Hit@5 | MRR@10 | P95 (ms) |
| --- | ---: | ---: | ---: |
| BM25, global | 7.81% | 0.0480 | 16.56 |
| Exact dense, global | 7.03% | 0.0428 | 14.93 |
| Exact dense, community | 8.59% | 0.0513 | 14.48 |
| Expert centroid, community | 11.52% | 0.0648 | 14.76 |
| Dense plus BM25, community | 11.91% | 0.0719 | 26.15 |
| [Full-history community prior](../sepef-matched-prior-v1/README.md), matched queries | 38.09% | 0.2413 | — |

The strongest text-only arm trailed the full-history community prior by 26.17 percentage points in Hit@5. A labeled expert had selected source text in the query's community on 87.5% of test queries, so losses from community membership account for at most 12.5 percentage points against a perfect unrestricted ranking. The prior uses more historical records per expert, with counts only. These controls compare complete retrieval routes under their stated inputs and do not isolate the effect of text or the answer cap. The SE-PEF authors' baselines aggregate answer scores by summation after BM25 filtering and use asker history in the personalized variant. This panel instead evaluates ranking with query text, known community and historical answer evidence. [Author baseline](https://github.com/pkasela/SE-PEF#to-create-the-baseline)

[Aggregate GPU results](text-results.json) · [Verified panel manifest](text-panel-manifest.json)

## Reproduce

For the metadata audit, extract `expert_test_data.json`, `val/qrels.json` and `test/qrels.json` under the author release's inner `SE-PEF` directory. Then run:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.sepef_history_prior METADATA_DIR NEW_RESULT_JSON
```

For text retrieval on Slurm compute nodes:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.sepef_temporal_panel NEW_PANEL_DIR --workers 4
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.sepef_first_query PANEL_DIR QWEN_EMBEDDING_CHECKPOINT \
  SCRATCH_DIR NEW_GPU_RESULT_JSON
```

The release excludes question texts, answer texts and individual author IDs. The code and aggregate audit reproduce against author-hosted inputs.
