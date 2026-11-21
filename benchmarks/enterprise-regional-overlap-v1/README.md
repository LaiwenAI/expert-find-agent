# Audit regional enterprise demand overlap before locking the holdout

Screen the 110 additions from the [regional frame](../enterprise-regional-frame-v3)
against the 245 earlier representatives, the other
new representatives, and 4,301 historical demands. Check the frozen
source bytes and the text and group metadata of every representative.
The earlier and new sets must recreate the 355-group inventory.

Encode each title with the first and last 640 characters of the body
using the frozen Qwen3-Embedding-0.6B checkpoint, with a 384-token cap.
Score the four pairs of windows and use their maximum cosine.
Apply the existing independent issuer-string similarity channel.
Exclude self-pairs, count scores by destination and save distinct pairs
at 0.90 or higher for review, capped at 24 per channel and destination.
If there are no pairs above 0.90, include eight closest pairs as a
below-threshold diagnostic. Test both encoder view orders. The GPU
projection contains only text windows and positional metadata.
Recognizable email and Chinese mobile-number patterns are masked
before transfer. Keep projections, mappings and review forms private.

## Results

Both Slurm jobs in array 46204 completed on ws3 RTX 5090. Reversing
the encoding order yielded identical counts and selected review pairs.

| Destination | New groups with cosine at least 0.85 | New groups with issuer similarity at least 0.85 | Pairs selected for blind review |
| --- | ---: | ---: | ---: |
| Earlier external frame | 0 | 0 | 16 |
| Other new groups | 0 | 2 | 16 |
| Historical demands | 0 | 0 | 16 |

Neither channel reaches the 0.90 review floor in any stratum.
The two issuer hits are focal groups, which can point to the same
undirected pair. The 48 below-threshold pairs are distinct and have
validated blank reviewer forms, with zero completed judgments.
False negatives for paraphrases, issuer aliases and relationships
absent from issuer names remain unmeasured. Unique organization counts,
civil eligibility and independence for a locked holdout remain unverified.
Retain the provisional count until adjudication.
See [aggregate results](results.json).

## Reproduce

Use the archived source snapshots and the 245- and 355-group private
indexes. Use a fresh output path for each step.

```sh
umask 077
PYTHONPATH=src:. python -m experiments.enterprise_regional_overlap prepare \
  EARLIER_SNAPSHOTS MUNICIPAL_SNAPSHOTS REGIONAL_SNAPSHOTS PRIVATE_WORKBOOK \
  PRIVATE_245_INDEX benchmarks/enterprise-regional-frame-v3/results.json \
  PRIVATE_355_INDEX PRIVATE_PROJECTION

# Run each command inside a GPU Slurm allocation.
PYTHONPATH=src:. python -m experiments.external_enterprise_overlap gpu \
  PRIVATE_PROJECTION QWEN_EMBEDDING PRIVATE_GPU_A --order-offset 0
PYTHONPATH=src:. python -m experiments.external_enterprise_overlap gpu \
  PRIVATE_PROJECTION QWEN_EMBEDDING PRIVATE_GPU_B --order-offset 1

PYTHONPATH=src:. python -m experiments.enterprise_regional_overlap packet \
  EARLIER_SNAPSHOTS MUNICIPAL_SNAPSHOTS REGIONAL_SNAPSHOTS PRIVATE_WORKBOOK \
  PRIVATE_245_INDEX benchmarks/enterprise-regional-frame-v3/results.json \
  PRIVATE_355_INDEX PRIVATE_PROJECTION PRIVATE_GPU_A PRIVATE_GPU_B PRIVATE_REVIEW_DIR
PYTHONPATH=src:. python -m experiments.check_enterprise_panel_review \
  PRIVATE_REVIEW_DIR REVIEWER_A_JSONL REVIEWER_B_JSONL
```
