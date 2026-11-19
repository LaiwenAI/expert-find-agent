# Selective cross-precision checks for enterprise qualification parsing

Use the same frozen model, clause splits, token IDs, cached batch geometry and public and development panels as the [FP16 cache study](../enterprise-precision-guard-v1/README.md). The previous comparison found that changing precision changed some qualification decisions even with independent full-context scoring. This experiment uses another precision only as a check before substituting an independent BF16 result.

Independent BF16 is the agreement reference. The existing BF16 guard recomputes the entire original block if any cached qualification logit is at most 2.0 away from its best alternative. The narrow BF16 control uses 0.5. The prior FP16 guard starts in FP16 and uses 0.5 with independent BF16 as fallback.

The consensus route starts in BF16 and uses the existing 2.0 trigger. On a triggered block, compute FP16 cached scores. Accept the original BF16 cached output only when all qualification decisions agree and all FP16 qualification gaps exceed 0.5. Otherwise rerun the original BF16 block. Allocate the FP16 prefix cache only when the first cross-check is needed and reuse it within the request. Changing between two nonqualification roles does not trigger the consensus fallback. Keep both thresholds from earlier experiments. Agreement across these precisions is an empirical check, not a guarantee that both agree with independent BF16.

Run each route end to end within the parser, rotating the arm order across requests. Include prompt construction, cache setup, all rechecks and result extraction. Report the qualification sets sent to the downstream verifier. The private development requests lack human qualification labels, and the 72 public target roles are balanced synthetic controls. No source retrieval or ranking runs in this timing window.

## Parser results

| Route on 80 development requests | Requests with changed qualifications | Added / removed qualifications | Whole BF16 rechecks | Parser P95 |
| --- | ---: | ---: | ---: | ---: |
| Independent BF16 | 0 | 0 / 0 | 0 | 5,259 ms |
| BF16 2.0 guard | 0 | 0 / 0 | 25 | 1,849 ms |
| Prior FP16 0.5 guard | 1 | 0 / 1 | 7 | 1,014 ms |
| BF16 0.5 guard | 2 | 2 / 0 | 10 | 943 ms |
| Selective cross-precision | **0** | **0 / 0** | **7** | **999 ms** |

The selective policy checks 25 blocks in FP16 across 15 requests and retains seven full BF16 fallbacks. Four blocks have conflicting qualification decisions and six have a low FP16 margin, with three overlapping. Relative to the BF16 2.0 guard, parser P95 is 46.0% lower. The paired request-bootstrap speedup ratio interval is 1.07–2.96×. The narrower BF16 guard is slightly faster on this panel and introduces two qualifications. The selective policy changes twelve fine-grained clause roles while reproducing the independent qualification set.

All five arms match 60 of 72 public synthetic target roles. This is a repeat on known development inputs. The evidence supports parser performance and model-output agreement on these requests. Human qualification judgment, complete search timing and held-out replication are separate tests.

Job 41459 ran on an RTX 5090 on ws5. [Aggregate results](results.json)

## Reproduce

Use the same pinned development query projection and [public controls](../enterprise-heading-roles-v1/controls.json) as the earlier experiment. The query file contains the 80 development rows in original order, serialized as id, text and split. Use a Slurm CUDA allocation and a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_cross_precision \
  DEVELOPMENT_QUERIES PUBLIC_CONTROLS QWEN_4B_CHECKPOINT NEW_RESULTS
```
