# Joint evidence selection with bounded source pointers

Use the same 137 multi-evidence questions from the [evidence-gap audit](../qasper-evidence-gap-audit-v1/README.md). The paper is known. Each route has exactly the same set of up to 16 dense candidates and may return up to five different source paragraphs. The gold label admits any annotator's complete evidence set. No test evidence enters the prompt or the selector.

The pointwise control uses the existing fused dense and Qwen3-Reranker-0.6B scores. The listwise routes use a frozen Qwen3-4B-Instruct-2507. Candidate IDs are deterministically shuffled per question, with the same order for all listwise routes. Every candidate remains an option even when the model receives a truncated excerpt. Section names accompany the excerpts. Returned IDs refer to original full paragraphs.

| Route | Reader input and decision rule |
| --- | --- |
| Pointwise | Return the five highest fused individual scores |
| Listwise first | Show 64-token extractive windows and select the five highest ID logits from one forward pass |
| Joint prefix | Show 64-token prefix windows and choose IDs sequentially with KV caching |
| Joint extract | Show 64-token extractive windows and choose IDs sequentially with KV caching |
| Joint long | Show 224-token prefix windows and choose IDs sequentially with KV caching |

An extractive window selects contiguous paragraph tokens by overlap with rare question terms. The selector sees each returned ID before making its next choice. Generation is constrained to one token per source ID, with selected IDs masked. It uses no answer generation, extra search, or repair calls.

Listwise ranking with candidate identifiers and first-token logits are established techniques. This pilot uses a frozen decoder to compare evidence-set completeness and cost. [ListT5, ACL 2024](https://aclanthology.org/2024.acl-long.125/) · [FIRST, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.491/)

The candidate pool and baseline reranker pass are measured once per query. Route totals are the measured shared prefix plus the independently timed selector, including input preparation. These totals account for adding the selector to the current stack. They do not measure a deployment that bypasses the pointwise reranker, serving transport or model initialization. The study controls candidate count and output slots. The 4B selector has additional compute and is reported as such.

## RTX 5090 results

Slurm job 41648 reproduces the prior panel's 54 shortlist misses, 40 selection misses and 43 complete pointwise outputs. All routes see the same 83 shortlists containing at least one complete annotated set.

| Route | Complete / 137 | Gained / lost vs pointwise | Evidence F1 | Selector P95 | Factored total P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pointwise | 43 | 0 / 0 | 0.377 | 0.00 ms | 75.24 ms |
| Listwise first | 25 | 3 / 21 | 0.341 | 70.25 ms | 143.02 ms |
| Joint prefix | 46 | 18 / 15 | 0.428 | 109.19 ms | 181.82 ms |
| Joint extract | 47 | 14 / 10 | 0.430 | 108.16 ms | 180.59 ms |
| Joint long | 40 | 14 / 17 | 0.406 | 183.89 ms | 254.80 ms |

Sequential source IDs with extractive windows return four more complete outputs than pointwise selection. The paired descriptive 95% interval spans −3.65 to +10.22 percentage points. The sequential route returns 47 complete outputs versus 25 when the same frozen model uses the same input and directly ranks the first ID logits. Longer prefix windows raise cost and return fewer complete sets than compact windows on this panel. The best observed sequential variant adds a 108.16 ms selector P95 and does not surpass the pointwise control on the joint quality–latency criterion. This is one known-paper test panel, with no expert-ranking or personal-attribution judgments. [Aggregate results](results.json)

## Reproduce

Use a Slurm CUDA allocation and the verified frozen checkpoints referenced in the runner.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.qasper.listwise_evidence \
    QASPER_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT SELECTOR_CHECKPOINT NEW_RESULTS
```
