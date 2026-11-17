# Source-Support Status Review Data

This is a local silver diagnostic set, not a complete gold standard for industry-expert matching. The inputs come from real company needs and teacher profiles, but the labels answer the question, “Does the supplied material snapshot support this narrowed research-topic query?” They cannot be used to infer that an expert actually lacks a capability or cannot complete a future project.

## Construction and Review

The same twenty needs were used in two construction routes—legacy lexical candidates and frozen research-domain semantic candidates—for a total of eighty generations. Thirty-seven pairs entered condition-by-condition review. Fourteen pairs passed, thirteen were rejected in review, and ten remained unresolved. The two routes changed both candidates and source strategies, so the difference in acceptance rates is not a single-factor causal effect.

Each sample involves only one teacher. The base query contains two to three separately assessed research topics; the counterfactual replaces only one of them. The two requested review models were gpt-4o and gpt-6-luna, with additional reasoning explicitly disabled for the latter. They do not receive the generator’s expected labels, citations, or rationale. Each condition ID must be complete and unique, a supporting citation must appear exactly once in a single original-text snapshot, and a need association must have an exact citation from the original need. The export rechecks the original model outputs, input-source artifact hashes, and complete model set.

A locatable citation does not mean that its semantic entailment has been verified by a human. The two model families overlap with those used in the previous generation review, and API model aliases do not guarantee that future calls will return exactly the same results. Rejection or unresolved status does not mean a true negative.

## Content and Source Reconstruction

Local artifacts are located at `artifacts/dataset-release/scoped-capabilities-audited/`:

- `pairs.jsonl`: fourteen query pairs, including base queries, replacement queries, conditions, and source citations.
- The `source_snapshots` field: lists all input sources available during review, not only the cited positive evidence. Each item contains an evidence ID, associated expert, source location, length of the original-text prefix actually supplied, and SHA-256.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: eight, three, and three pairs, respectively; both variants of a pair are always kept together.
- `splits.json`: records the shared need, expert, hashes of all inputs and cited sources, connected groups of normalized duplicate queries, and file checksums.

The fourteen pairs cover ten needs and fourteen teachers, forming ten connected components, with a maximum component size of two pairs. The resource groups described above do not overlap across splits, but templates are still shared, and near-duplicate sources and semantic paraphrase leakage have not been ruled out. Three test samples are insufficient for a persuasive significance analysis.

The export does not include the original teacher/need texts, model traces, or model rationales. Reproducing the experiment requires holding the original corpus locally. The loader reconstructs the complete review-time inputs using each item’s prefix length and hash, and verifies expert association; it does not read only the cited supporting passages to artificially simplify the counterfactual.

## Labels and Baselines

The base-query label is `supported`, and the counterfactual label is `unknown`; unknown means insufficient material, not a denial of capability. Other experts were not assessed, so these results cannot be used to calculate matching precision across the full corpus or to treat everyone else as a negative example.

`scoped_silver_benchmark` provides constant-unknown and per-topic literal-matching diagnostic baselines. The latter receives the already-generated condition structure directly, excluding natural-language parsing. On the three test pairs, the literal baseline already achieves 1.0 status accuracy; across all fourteen pairs, it achieves 0.9286. This indicates that the current diagnostic set has limited discriminative power and cannot be used to demonstrate the necessity of a complex agent. The constant-unknown baseline has an accuracy of 0.5; when the set of predicted supports is empty, precision is defined as undefined, not as a perfect score.

```bash
PYTHONPATH=src python3 -m expert_search.scoped_dataset_audit --export
PYTHONPATH=src python3 -m expert_search.grouped_dataset
PYTHONPATH=src python3 -m expert_search.scoped_silver_benchmark --split test
```

The export described above requires saved review results. The original corpus, private data, and third-party materials are not redistributed with the software review package. These derived queries have also not yet been released as public data; the MIT software license does not automatically cover the data. A future primary benchmark needs stronger relation/source controls and independent review, rather than merely expanding the same template.
