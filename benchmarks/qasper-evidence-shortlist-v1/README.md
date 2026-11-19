# Evidence selection at fixed reranking depth on QASPER

Compare four ways of selecting up to 16 paragraphs from a dense Top-32 within the known paper. Every policy keeps the first eight dense candidates. The dense baseline fills in rank order. The sections control prefers sections absent from the current set. The MMR-style control penalizes maximum embedding similarity to already selected candidates. The term-coverage method favors less-covered query terms. It weights terms by within-paper inverse document frequency and reduces each term's residual weight as selections mention it. [MMR, SIGIR 1998](https://doi.org/10.1145/290941.291025)

Select the MMR and coverage weights on the [development calibration partition](../qasper-adaptive-depth-v1/README.md), using complete evidence in the 16-slot candidate set. Present selected paragraphs in dense order, score them using the same frozen cross-encoder and cosine fusion, and return five. Report complete human evidence before and after reranking. Warm timing includes query encoding, dense scoring, shortlist selection, and reranker tokenization and forwards. The path receives a known paper. Cross-paper search and expert attribution remain outside this test.

## Results

The pilot used 96 previously explored QASPER test papers (83 evaluable). At equal depth the coverage method recovered three additional complete candidate sets and one additional complete returned set. P95 was 78.4 ms for coverage and 76.1 ms for the dense shortlist.

Replicate with the same frozen weights on 144 test papers disjoint from the pilot. Add a fixed-32 route in the same GPU workload. The table shows the 127 questions with alignable evidence. The broader QASPER test split has appeared in prior aggregate experiments.

| Replication route | Mean checks | Complete shortlist / 127 | Complete returned / 127 | Evidence F1 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense 16 | 15.73 | 106 | 91 | 30.79% | 78.6 ms |
| New sections | 15.73 | 104 | 91 | 31.02% | 78.6 ms |
| MMR style | 15.73 | 105 | 91 | 30.79% | 78.1 ms |
| Term coverage | 15.73 | 106 | 91 | 30.79% | 78.9 ms |
| Dense 32 | 29.10 | 119 | 90 | 30.75% | 131.9 ms |

The term-coverage gain did not replicate. Dense 32 found 13 more complete candidate sets than dense 16 but returned one fewer complete set. The same pointwise scorer ranks every route. This places the next test at final evidence-set selection, where individually relevant passages can crowd out a necessary complement. It does not establish the cause of any single miss. Source coverage methods based on explicit aspects also have established precedents. [xQuAD, WWW 2010](https://archives.iw3c2.org/www2010/publications/santos10pdf.pdf)

## Reproduce

On a Slurm GPU compute node with the [official QASPER assets](https://aclanthology.org/2021.naacl-main.365/) and the pinned checkpoints:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.qasper.evidence_shortlist \
  DATA_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON --replication
```

Omit `--replication` for the pilot configuration. [Aggregate results](results.json) include both runs, pinned hashes and the unchanged development grid. The pilot records the script hash from before the replication flag was added. Raw rows and corpus text stay local.
