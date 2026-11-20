# Calibrate topical evidence on independent scientific relevance judgments

The [three-condition proof scheduler](../topic-proof-certification-v1)
returned no complete proofs at its default threshold. This study
calibrates each topic scorer on [TREC-COVID Complete](https://ir.nist.gov/covidSubmit/data.html)
and transfers its frozen operating point to the same PLOS ONE and
eLife proof panels. NIST grades document relevance to a topic as 0
(nonrelevant), 1 (partially relevant), or 2 (fully relevant). Exclude
unjudged pairs. The model reads the official short query and title
plus abstract. NIST grades can rely on fuller documents or the
detailed topic description. See the [judgment guidelines](https://ir.nist.gov/trec-covid/round5.html).

Use a pinned [BEIR corpus export](https://huggingface.co/datasets/BeIR/trec-covid/blob/aa883c476025d4a8620af8425b2f061518351b2d/corpus.jsonl.gz)
for titles and abstracts and official NIST XML and qrels for topics
and grades. Hash-order the 50 topics into 30 calibration and 20 held
topics. For each, hash-sample the predeclared 16 grade-0,
8 grade-1 and 16 grade-2 articles. Require both title and abstract,
and exclude a previously used article ID or normalized exact title
throughout the panel. The manifest pins sample membership and bytes.
No article text is included in this release.

Compare the original Qwen3-Reranker-0.6B prompt, a prompt that asks
whether the article substantively studies the research subject, and
cosine from the same frozen Qwen3-Embedding-0.6B model used for retrieval.
For each scorer select the lowest strict score cutoff giving at most
10% grade-0 acceptance on calibration. This is an empirical calibration
target. It does not set a confidence guarantee for the held rate.
Report grade-0, grade-1 and grade-2 acceptance with topic-cluster
bootstrap intervals. Grade-1 does not enter threshold selection.

For transfer, keep the same text-only ranked depths, feasible
same-person assignments and source separation rules. Apply all three
frozen thresholds to every feasible topic edge; compare the original
prompt at zero and −4 as sensitivity controls. A reported known proof
requires all three publisher topic annotations. Unannotated edges
remain unjudged, so neither their acceptance nor their rejection
defines a true error rate on the transfer task. No publisher label
influences threshold selection. This is an offline quality assay,
without first-request timing or further model training.

## Results

Both replicas of Slurm array `45732` run on the ws3 RTX 5090.
Its 1,200 calibration and 800 held pairs meet the planned grade
counts. No sampled title exactly matches the transfer catalog.
The table reports **held** acceptance at the frozen cutoffs.

| Scorer | Cutoff | Grade 0 | Grade 1 | Grade 2 | Known publisher proofs / 48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original, default | 0 | 47 / 320 | 75 / 160 | 234 / 320 | 0 |
| Original, calibrated | 0.375 | 38 / 320 | 70 / 160 | 216 / 320 | 0 |
| Scope prompt, calibrated | −0.0625 | 37 / 320 | 68 / 160 | 216 / 320 | 0 |
| Cosine, calibrated | 0.54201 | 35 / 320 | 68 / 160 | 225 / 320 | 0 |

All three calibrated grade-0 acceptance rates exceed the 10%
calibration target on held topics. At transfer, the calibrated
original and scope prompts each pass 16 of 138 edges carrying
publisher annotations and 4 of 75 unjudged edges. Cosine passes
none. The −4 sensitivity anchor recovers the previously observed
six complete proofs, four with all required topic annotations.
It remains an uncalibrated target-corpus diagnostic.

The source-domain thresholds provide no usable complete-proof
coverage on this transfer. The experiment compares question-to-article
relevance calibrations against broad publisher topic labels. It cannot
establish that the same calibration target applies to both tasks.
Independent target-style judgments should use the actual conditions
and the exact evidence view before fitting an operating point.

[Aggregate results](results.json) · [Frozen membership](trec-panel-manifest.json).

## Reproduce

Use the [official topics XML](https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml),
[official final judgments](https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt)
and the pinned corpus above. The panel builder excludes two entries
with grade −1 from the final file.

```sh
PYTHONPATH=src:. python -m experiments.credit.trec_topic_panel \
  CORPUS_JSONL_GZ TOPICS_XML QRELS NEW_PANEL NEW_MANIFEST

PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.calibrated_topic_evidence \
  PANEL MANIFEST PLOS_XML PLOS_FIXTURE ELIFE_XML ELIFE_MANIFEST \
  QWEN_EMBEDDING QWEN_RERANKER benchmarks/live-joint-source-request-v1/results.json \
  benchmarks/topic-proof-certification-v1/results.json NEW_RESULTS
```

Run model inference inside a Slurm CUDA allocation.
