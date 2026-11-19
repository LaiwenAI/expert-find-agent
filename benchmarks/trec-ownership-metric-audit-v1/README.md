# Audit person-specific evidence with TREC expert judgments

Use the public [TREC 2006 expert and supporting-document qrels](https://trec.nist.gov/data/t15_enterprise.html). A supporting-document judgment covers a topic, an expert and a document. Following the [official track definition](https://trec.nist.gov/pubs/trec15/papers/ENT06.OVERVIEW.pdf), grade 2 supports the expert, grade 1 provides no support either way, and grade 0 counts against expertise. Only grade 2 is relevant for the expert and evidence metrics. The document audit counts each topic and document once per topic. It uses judged pairs only.

Fix a five-slot expert ranking using the existing leave-one-topic-out expert prior over the current topic's judged candidates. For each grade-2 expert with a supporting document in the pool, attach one such document. Construct two interventions on that same ranked output. Append a second document that supports a different candidate but has a judged grade below 2 for this expert, or replace the original citation with the same borrowed document. Prefer a grade-0 document when available, then order by document ID. Test a separate control using exactly two citations in both conditions whenever a second valid citation is available. The expert rankings, eligible slots and first citation stay fixed in this control.

Expert P@5 scores the expert identity. TREC-style supported P@5 requires a relevant expert and at least one supporting citation anywhere in the returned evidence. Clean-citation P@5 additionally requires every returned citation for that expert to provide support. Report citation precision and the counts of negative and unsupportive-neutral citations independently. This challenge isolates extra unsupported citations. It does not label which substantive constraints each document supports.

## Results

Across 49 topics, 1,741 topic-document pairs have different support outcomes for two candidates both judged relevant. These pairs occur in 44 topics. Among documents that support at least one expert, 186 topic-document pairs include a grade-0 judgment and 3,025 include a grade-1 judgment for a different candidate. The two groups can overlap. Restricting mismatched judgments to target experts judged relevant leaves 111 negative and 4,475 neutral person-document pairs.

| Citation construction | Expert P@5 | TREC-style supported P@5 | Clean-citation P@5 | Citation precision |
| --- | ---: | ---: | ---: | ---: |
| Valid citation | 31.02% | 28.16% | 28.16% | 100.00% |
| Append another person's citation | 31.02% | 28.16% | 5.71% | 55.65% |
| Replace with another person's citation | 31.02% | 5.71% | 5.71% | 20.29% |

Appending a borrowed document affects 55 expert slots across 25 topics, including 3 negative and 52 neutral judgments. The descriptive topic-paired 95% interval for the clean-citation change is −30.61 to −14.69 percentage points. In the matched-length control, 50 expert slots have two citations in both views. The at-least-one criterion credits all 50 in both views. Clean citation credits 50 with two valid citations and zero when one citation is borrowed.

The interventions use labels to construct known contrasts. They demonstrate metric sensitivity on pooled judgments. The candidate set comes from that pool, and the study does not assess the prevalence of these errors in actual system outputs or measure open-corpus retrieval. In a live benchmark, missing judgments should be tracked as unknown and the metrics should be paired with query coverage and expert precision. [Aggregate results](results.json)

## Reproduce

Download both qrels from the official track page. The runner checks their SHA-256 hashes before reading them. No corpus text or researcher identities are redistributed by this audit.

```sh
PYTHONPATH=src:. python -m experiments.trec_enterprise.ownership_metric_audit \
  TREC_QRELS_DIRECTORY NEW_RESULTS
```
