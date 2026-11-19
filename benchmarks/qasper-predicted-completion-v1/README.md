# Predicted anchors and competing papers in QASPER

This experiment transfers published selector weights from a [single missing paragraph with verified anchors](../qasper-completion-selector-v1) to predicted anchors and competing sources. No fitting or threshold selection uses the new outcomes. The experiment reuses the official QASPER v0.3 development and test questions; development and test papers are disjoint.

For every question, take the highest-ranked unseen paragraph by frozen Qwen3-Embedding-0.6B as the first anchor. Return exactly two more paragraphs where available. At each step, provide the selector with the four highest dense-ranked paragraphs it has yet to select. The same initial ranking, starting anchor, and budget are used for the dense, frozen cross-encoder, learned question, and learned complement arms. Decisions can change which paragraphs are available in the second step. No confidence gate suppresses a recommendation.

| Source scope | Search and input |
| --- | --- |
| Known | Search only the designated paper with the original question |
| All papers, raw | Search every paper in the official split with the original question |
| All papers, cued | Search the same collection with a hash-selected contiguous title fragment of up to three words appended to the question |

Many raw questions were written about a designated article. The cued condition adds a realistic form of disambiguation, inspired by [QASPER-MP](https://github.com/debobanerjee/LC-EBG/tree/main/datasets/QASPER-MP). This experiment uses all papers in each split as the collection and replays our own deterministic cue, distinct from the published QASPER-MP construction. Neither open condition requires all facts to come from distinct papers. Extractive references are scoped to the designated paper.

The learned arms compute features over the union of papers represented in the anchors or current four candidates. Section identifiers are qualified by paper. Position distance is computed in the resulting concatenated paragraph sequence. At each round the frozen reranker sees the same history format as it did in the previous experiment, with a cap of 128 tokens per selected paragraph.

**Completion** requires the three outputs to contain all paragraphs of any extractable human reference. **Seed ceiling** counts questions for which a complete reference can fit within three outputs while keeping the prescribed first seed. **Window ceiling** additionally requires that reference to lie in the first six dense-ranked paragraphs, all that can become reachable under the fixed candidate rule. These are post-hoc bounds using human labels. The three-output cap cannot cover any annotated reference in 20 of 112 dev questions and 27 of 137 test questions, even with a freely chosen seed. Component P95 adds observed question encoding and similarity search to each route's own time for selecting two further paragraphs, excluding retrieval index build, waiting, service overhead and expert attribution.

## Results

Slurm job 42748 ran on one RTX 5090 on ws5. All 281 development papers and 416 test papers were indexed. The test index contains 19,710 passages. The following cells give completed questions and component P95 in milliseconds.

| Split and scope | Right source at seed | Annotated seed | Seed ceiling | Window ceiling | Dense | Frozen | Learned question | Learned complement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev, known (112) | 112 | 41 | 76 | 33 | 16 / 7.46 | 23 / 62.07 | 18 / 9.45 | 25 / 9.21 |
| Dev, all raw (112) | 34 | 11 | 75 | 7 | 2 / 8.86 | 4 / 54.68 | 4 / 16.78 | 5 / 16.54 |
| Dev, all cued (112) | 63 | 13 | 77 | 10 | 5 / 8.87 | 7 / 58.09 | 5 / 15.72 | 8 / 15.51 |
| Test, known (137) | 137 | 60 | 92 | 36 | 16 / 7.49 | 20 / 56.38 | 17 / 9.22 | 21 / 9.13 |
| Test, all raw (137) | 38 | 15 | 83 | 5 | 1 / 10.01 | 1 / 56.47 | 1 / 17.53 | 1 / 17.21 |
| Test, all cued (137) | 73 | 16 | 81 | 7 | 3 / 10.08 | 5 / 59.72 | 3 / 16.68 | 2 / 16.47 |

With the correct paper supplied, predicted anchors preserve the measured efficiency advantage of the learned selector. Against the frozen model, its test difference in completion rate is +0.73 points, with a paired 95% interval of −2.92 to +4.38. Removing the complement features costs four test completions, a difference of +2.92 points [ +0.73, +5.84 ] in favor of the full model.

In all-paper test search, a title cue raises correct-paper seeds from 38 to 73 but annotated seeds from only 15 to 16. The capped candidate window reaches a complete reference in five raw and seven cued queries. The learned complement model retains its timing advantage over the frozen reranker, while its cued completions fall from five to two relative to that baseline. The paired interval for this difference is −5.11 to 0.00 points. The corresponding development difference is one completion in favor of the learned model. The known-paper completion result therefore does not establish an open-collection accuracy benefit.

This comparison gives the next experiment a practical target: separate document identification from finding the supporting paragraph, and keep an explicit path out of the first paper when the source signal is weak. Compare flat score order, a title or metadata-gated baseline, and source-balanced passage allocation at matched evidence budgets. The synthetic cue is a literal contiguous title fragment and often identifies a paper by itself; its strong lookup baseline should accompany any cue-based evaluation.

[Aggregate results](results.json)

## Reproduce

Use the v0.3 release, frozen models and the published selector artifact. The runner validates the selector file digest and the exact model revisions.

```sh
PYTHONPATH=src:. python -m experiments.qasper.predicted_completion \
  QASPER_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT \
  benchmarks/qasper-completion-selector-v1/results.json NEW_RESULTS.json
```
