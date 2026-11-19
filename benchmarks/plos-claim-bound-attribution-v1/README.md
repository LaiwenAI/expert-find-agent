# Verify contribution claims bound to a specific researcher

Use the earlier 171-publication projection for calibration and the [independent 55-publication PLOS snapshot](../plos-source-gate-transfer-v1/README.md) for evaluation. Resolve each publication through the exact JATS SHA-256 in the projection, verify the license and parse the contributor role fields. Retain one case per publication with complete author declarations. Use two authors who share one requested duty, while only one is credited with the second requested duty. Keep the required duties, target label and its position fixed. Create the counterfactual by swapping the two role lists between their author labels. Both duty names are still present somewhere in both supplied records. Only the target author's support changes.

Give the reranker the same named-author query under a broad relevance prompt or an explicit two-duty support instruction. A third route exposes only the named author's line and uses the same explicit verifier. For the structured baseline, encode the two descriptions using the existing frozen ridge role mapping and take the weaker of their declared-role membership margins for the target author. The 4B reader uses both lines, the explicit decision rule and a constrained binary output. The variants share model checkpoints where applicable. The field route has access to the parsed registry and the other routes see its equivalent role strings.

Freeze each threshold at a 5% wrong-owner acceptance ceiling using one NISO phrase per role on calibration publications. Evaluate the untouched NISO phrases and the existing independent [boundary wording](../plos-credit-external-v1/boundary-phrases.json). The target label, display order and paired presentation order are counterbalanced by DOI. Report paired discrimination, both cases correct at the frozen threshold, and observed accepted supported and wrong-owner counts. The P95 values time one paired batch and exclude source discovery and model loading. Publisher declarations define the target and may be incomplete descriptions of a person's broader expertise. Cases reuse a small wording bank, and within-split publications can share authors.

## RTX 5090 results

Slurm job 42610 retains 136 calibration and 46 test publications. Every fitted threshold permits six wrong-owner acceptances on the calibration set. Test columns count accepted supported views and accepted wrong-owner views, followed by the number of pairs receiving both correct decisions. Both test slices reuse the same publications with different role wording.

| Route | Held NISO accepted support / wrong owner | Held NISO both correct | Boundary accepted support / wrong owner | Boundary both correct | Boundary paired P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Topic relevance | 6 / 4 | 2 | 4 / 6 | 1 | 7.70 ms |
| Explicit author-bound instruction | 4 / 3 | 2 | 2 / 4 | 2 | 7.73 ms |
| Only the target author's line | 7 / 3 | 6 | 2 / 2 | 2 | 8.83 ms |
| Structured ridge membership | 27 / 5 | 25 | 34 / 3 | 31 | 8.50 ms |
| Frozen 4B reader | 11 / 1 | 11 | 2 / 0 | 2 | 22.63 ms |

On boundary wording, the structured baseline ranks the supported view above the swapped view on 41/46 publications, versus 24/46 for topical relevance and 37/46 for the 4B reader. Its paired difference versus topic is +37.0 percentage points, with a descriptive paper-resampled interval of +21.7 to +52.2. Explicit instruction alone changes the ordering by only one pair. Isolating the target line helps ordering on held NISO wording but has little effect at the frozen operating point. The 4B reader's zero observed boundary errors comes with a loss of 44 supported views.

Use structured owner-specific membership as the strongest control when publisher role metadata is available. The improvement here is relative to the text models on supplied publisher declarations. It does not establish semantic attribution in arbitrary free text or full retrieval accuracy. The held wording banks repeat across the panel and the held-NISO threshold accepts five wrong owners, so the calibration ceiling is not a transferable risk guarantee. [Aggregate results](results.json)

## Reproduce

Use the [original projection](../plos-scoped-search-v1/README.md), the [independent holdout](../plos-source-gate-transfer-v1/README.md), the raw JATS archives listed in both construction protocols, and the [frozen role-training inputs](../plos-shared-embedding-witness-v1/README.md). Use local copies of the same open-weight embedding, reranker and 4B reader. Run on a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.claim_bound_attribution \
    CALIBRATION_PROJECTION HOLDOUT_PROJECTION NISO_EXAMPLES ROLE_TRAINING BOUNDARY_PHRASES \
    QWEN_EMBEDDING QWEN_RERANKER QWEN4B NEW_RESULTS \
    CALIBRATION_XML_DIR [OTHER_XML_DIRS...] HOLDOUT_XML_DIR
```
