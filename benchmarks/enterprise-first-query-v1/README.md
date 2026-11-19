# First-query latency with live retrieval and evidence verification

Compare independent full-context clause scoring, prefix caching, and prefix caching with the frozen 2.0-logit whole-block recheck from the [parser benchmark](../enterprise-guarded-parsing-v1/README.md). Each arm runs the complete search independently, with six rotating arm orders. Three synthetic warm-up searches run before timing.

| Stage | Fixed setting |
| --- | --- |
| Read the request | Qwen3-4B-Instruct-2507 scores anchored clause roles. Each arm uses the same prompt and weights. |
| Retrieve | Qwen3-Embedding-0.6B encodes each live query. Full FP32 vector scoring aggregates by maximum source score per expert and retains five experts. |
| Read evidence | Use the first 800 characters of the two highest-scoring owned sources per expert. Check the opening requirement text and each parsed prior qualification separately. |
| Verify and rank | Qwen3-Reranker-0.6B supplies a provisional support vote at a positive yes-minus-no margin. Move fully supported candidates first, preserving retrieval order within each group. Retain source span offsets and mark missing support as unresolved. |

Opening text means the first nonblank line, clipped at 320 characters. Prior qualifications exceeding 320 characters stay unresolved. Batches containing any verifier input truncation are withheld. The verifier uses frozen weights and a fixed threshold without human calibration.

Queries are the 80 development requests from the frozen enterprise evaluation snapshot. The corpus is that snapshot's development-side source inventory. Timing begins before clause extraction and includes query encoding, full-source scoring, source lookup, verification, and ranking. Weights and the source index are already loaded. Offline indexing time is reported separately. No queueing, networking, or concurrent requests are simulated.

## Results on one RTX 5090

| Reader | Complete P50 | Complete P95 | Changed expert order / 80 | Changed citation or unresolved state / 80 |
| --- | ---: | ---: | ---: | ---: |
| Independent full-context batches | 570 ms | 5,473 ms | 0 | 0 |
| Prefix cache | 191 ms | 642 ms | 0 | 3 |
| Prefix cache with whole-block recheck | 230 ms | 3,426 ms | 0 | 0 |

Whole-block rechecking lowered complete P95 by 37.4% with exact agreement against the independent route on all 80 requests. The paired bootstrap interval for P95 speedup was 1.15–3.57× (point estimate 1.60×). Cache-only was 8.53× faster at P95 and changed the qualification or citation state on three requests despite identical expert order. All arms supplied at least one provisional recommendation on 34 requests and supplied provisional support for 70 of 400 candidate slots. Human semantic correctness was unmeasured.

The index contained 52,667 sources from 4,154 experts and took 296.8 seconds to build. With whole-block rechecking, query encoding, full-source scoring, and verification plus ranking had separate P95 values of 9.12 ms, 0.51 ms, and 180.27 ms. The parsing P95 was 3,284 ms and remained the largest stage. Stage quantiles are marginal and should not be summed. [Full aggregates](results.json)

## Reproduce

Use the authorized frozen enterprise snapshot and verified model checkpoints. Project the two original JSONL files, preserving row order and selecting rows whose `split` is `development`. Emit `id,text,split` from `queries.jsonl` and `id,expert_id,text,split` from `corpus.jsonl`, in those field orders, using UTF-8 and Python's default `json.dumps(row, ensure_ascii=False)` followed by a newline. Compare the projection hashes to the aggregate record. Run within the Slurm GPU allocation:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_first_query \
  --queries DEVELOPMENT_QUERIES --corpus DEVELOPMENT_SOURCES \
  --models CHECKPOINT_DIRECTORY --output NEW_AGGREGATE
```

Report both the ordered expert IDs and the complete citation/unresolved state against the independent arm as counts of changed requests. Paired request bootstrap intervals quantify tail-latency speedup on this cohort. Agreement is with a frozen reference policy, and the study uses no human judgments or sealed test queries.
