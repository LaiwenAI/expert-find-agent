# Spend packed-verifier savings on missing evidence

The [shared-weight benchmark](../cof-ragged-sustained-v1) found more verification capacity at unchanged output. This experiment asks whether the saved compute can recover additional cited recommendations at the same candidate ranks. It uses the same frozen models and the full KDD and SciRepEval panels from prior CoF experiments, with no new fitting.

For each query, the full-inventory scorer ranks experts once and retrieves four ordered sources each for the top five. Keep those five experts in their original order in every output. A recommendation receives citations only when two sources earn positive verdicts from the same guarded verifier. Compare four routes:

| Route | Evidence policy |
| --- | --- |
| Padded three | Verify up to the first three sources per expert with four-row padded waves |
| Packed three | Same three-source frontier with eight-row variable-length waves |
| Packed four | Make four sources per expert available to the packed frontier from the start |
| Packed rescue | Run packed three, then make the fourth source available only to uncited experts with one confirmed positive from the initial group |

In rescue, reuse the existing score and fixed-shape recheck caches and resume the frontier over the expanded evidence. Earlier attached citations retain their source IDs and order. All arms tokenize the same four-source candidate universe to hold the canonical recheck shape constant; actual dynamic scoring remains demand driven. The padded baseline includes this shared prompt preparation overhead. Each timed route uses live model inference. The reported complete P95 adds the same per-query exact search and embedding time to the observed verification time. Model initialization, waiting and transport are outside these sequential timings.

Report the number of experts with two model-selected citations and distinguish published positive, negative and unjudged expert judgments among them. Positive-and-cited positions measure the intersection of expert compatibility labels and model attachment. CoF supplies no independent labels for the new citations or for each individual's contribution.

## Results

Slurm job 42835 completed both public panels on one RTX 5090. The packed three-source route reproduces every padded three-source response. Candidate ranks remain fixed across all arms.

| Panel | Route | Cited positions | Cited positives | Cited negatives | Cited unjudged | Mean scored sources | Complete P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KDD (122) | Padded three | 260 | 12 | 40 | 208 | 9.70 | 136.94 ms |
| KDD | Packed three | 260 | 12 | 40 | 208 | 9.63 | 118.94 ms |
| KDD | Packed four | 279 | 12 | 43 | 224 | 13.69 | 145.25 ms |
| KDD | Packed rescue | 276 | 12 | 41 | 223 | 10.52 | 128.38 ms |
| SciRepEval (59) | Padded three | 85 | 21 | 16 | 48 | 11.71 | 122.08 ms |
| SciRepEval | Packed three | 85 | 21 | 16 | 48 | 11.51 | 103.14 ms |
| SciRepEval | Packed four | 99 | 26 | 20 | 53 | 16.63 | 129.45 ms |
| SciRepEval | Packed rescue | 96 | 26 | 19 | 51 | 12.63 | 120.21 ms |

SciRepEval gains five labeled-positive cited experts with rescue, matching fixed depth four at lower observed P95. Positive-and-cited P@5 increases by 1.69 percentage points over packed three (descriptive paired 95% interval +0.34 to +3.39). Three additional cited slots are labeled negative and three are unjudged. On KDD, the 16 additional slots contain one labeled negative and 15 unjudged experts. Rescue therefore increases attachment coverage under the padded-three latency in both panels, while its benefit for labeled positives is specific to SciRepEval in this test. Neither target enforces the requirement that the additional source substantively complements a different hard constraint. An independently judged evidence-pair comparison is needed to validate that step.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the verified public CoF datasets, checkpoints and Slurm CUDA setup from the [prior shared-weight benchmark](../cof-ragged-sustained-v1).

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_packed_evidence_rescue \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
PYTHONPATH=src:. python -m experiments.retrieval.cof_packed_evidence_rescue \
  SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
