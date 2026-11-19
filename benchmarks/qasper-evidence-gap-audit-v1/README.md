# Distinguishing shortlist and final-selection misses on QASPER

Use one deterministically selected question per paper in the official QASPER test split, selected by hashing its question ID. The runner provides the original paper and uses the source paragraph sets supplied by annotators. It reuses the public dataset from the [section-guided retrieval study](../qasper-section-relation-graph-v1/README.md). Some papers and questions have appeared in earlier explorations. The targeted multi-evidence panel selects at most one question per paper after requiring that the smallest available annotated evidence set contain two to five paragraphs. The route receives only the original question and paper, never the label-based inclusion criterion. Panels overlap and their results are reported separately.

Every route independently embeds the question and reranks its own shortlist with the same frozen open-weight models. The final output always takes the five highest-scoring reranked paragraphs. Scores use the existing fixed dense cosine and sigmoid-margin mixture. Neither shortlist construction nor final selection uses the labels.

| Route | Candidates given to the cross-encoder |
| --- | --- |
| Flat 16 | Best 16 dense paragraphs |
| Section 16 | Published heading-aware section policy |
| Dense anchor 16 | Start with eight dense results and expand relative to the best two by dense score |
| Verified anchor 16 | Start with the same eight results and expand relative to the best two after reranking |
| Flat 32 | Best 32 dense paragraphs |

The anchor routes allocate up to four extra slots to high-scoring paragraphs that cover rare question words absent from the anchors. Coverage is updated after each addition. They consider the top 32 dense paragraphs for that step. Remaining slots read unscored immediate same-section neighbors, then fill by dense rank. The dense-anchor control uses the same expansion rule and score budget. Timing includes query encoding, expansion and the actual reranker calls. It excludes offline indexing and network service overhead.

Each question belongs to exactly one audit category. The evaluator keeps available answerable paragraph evidence and treats complete sets as alternatives. `unscored` means no such set is available, including unanswerable or figure-only cases. `output_capacity` means every available set requires more than five paragraphs. For the remainder, a `shortlist_miss` means none of the feasible sets is contained in the shortlist, a `selection_miss` means a feasible set is present but none is returned, and `complete` means the final five contain a full set. This audit distinguishes where observed paragraph coverage is lost. It does not establish whether a dropped paragraph is semantically irrelevant or assess attribution to an individual.

## RTX 5090 results

Jobs 41631 and 41638 scored 416 and 137 questions, respectively. The all-paper panel has 343 feasible questions, including 304 where an annotator's smallest set contains one paragraph. Sixty-six questions have no available answerable paragraph set, and seven have no set that fits in five slots. The multi-evidence panel contains 137 feasible questions.

| Panel | Route | Shortlist miss | Final-selection miss | Complete output | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| All papers | Flat 16 | 34 | 57 | 252 | 79.67 ms |
| All papers | Section 16 | 33 | 56 | 254 | 78.37 ms |
| All papers | Dense anchor 16 | 30 | 61 | 252 | 79.18 ms |
| All papers | Verified anchor 16 | 31 | 58 | 254 | 79.24 ms |
| All papers | Flat 32 | 7 | 81 | 255 | 136.20 ms |
| Multi-evidence | Flat 16 | 54 | 40 | 43 | 71.71 ms |
| Multi-evidence | Section 16 | 53 | 41 | 43 | 71.51 ms |
| Multi-evidence | Dense anchor 16 | 49 | 45 | 43 | 71.52 ms |
| Multi-evidence | Verified anchor 16 | 47 | 47 | 43 | 71.48 ms |
| Multi-evidence | Flat 32 | 20 | 74 | 43 | 124.68 ms |

On the multi-evidence panel, verified-anchor expansion covers seven more feasible evidence sets in its shortlist than flat 16 with nearly the same P95. It returns no additional complete sets. Doubling flat depth from 16 to 32 reduces shortlist misses by 34, with a P95 increase from 71.71 to 124.68 ms and no additional complete final sets. In that wide condition, final selection loses 74 of 117 questions whose shortlist contains a complete set. The all-paper panel also shows that most additional shortlist coverage fails to reach the output. The two more complete outputs for verified-anchor than flat 16 correspond to a paired query-bootstrap interval of 0.00 to +1.46 percentage points.

These results localize a failure after retrieval in this known-paper benchmark. They do not distinguish verifier misjudgment from independent pointwise ranking that underweights combinations. The full-panel run used the original default-only script before the multi-panel flag was added, with the same ranking and audit functions. [All-paper aggregates](full.json), [Multi-evidence aggregates](multi.json)

## Reproduce

Use the official QASPER files and pinned open checkpoints from the [earlier study](../qasper-section-relation-graph-v1/README.md). Run the following on a Slurm CUDA node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.qasper.evidence_gap_audit \
    QASPER_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS

# Multi-evidence panel
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.qasper.evidence_gap_audit \
    QASPER_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_MULTI_RESULTS --multi
```
