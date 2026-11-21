# Serve a resident role head under loopback load

The [fixed-pair training study](../elife-role-contrast-training-v1)
found that a linear head fitted with cross-entropy on 84 role
descriptions approached the full-menu 4B reader on selected sources.
This comparison puts those policies behind the same HTTP endpoint.
Every request runs topic encoding, corpus ranking, the directional
top-three source guard, same-author pair selection, role classification,
attribution and response serialization. The worker sees only topic
clues and role phrases. Quality labels stay on the evaluator side.

| Service arm | Online work |
| --- | --- |
| Full menu | Encode the two topic clues. For a selected source pair, run the two-role Qwen3-4B full-menu decoder in one forward. |
| Ridge | Encode topics and, when there is a pair, encode role phrases in a second 0.6B forward. Apply the frozen ridge head. |
| Resident CE | Same two-stage embedding path as ridge, with the CUDA-trained cross-entropy head. |
| Resident CE fused | Encode both topic clues and both role phrases in one 0.6B forward, then apply CE. This arm may alter source or role results through batch shape. |

The first three arms fix the topic encoding shape and source search.
The fused arm spends the role encoding work even when no usable
source pair is found. All arms use the same in-memory article
projection, with two logical evidence reads when a pair passes.
The quality test requires the exact source pair, both intended roles
and at least one gold shared author. An issued result failing these
checks is adverse. Report evidence once per unique query.

## Load and scope

Replay the confirmation and extended eLife panels, with 204 and 288
unique HTTP requests including the two clue views and role-swap
controls. Use the same Poisson arrival schedule for each arm at each
rate, two repetitions of each request, one worker and 64 persistent
loopback clients. Rotate policy execution order between rates and run
the opposite policy order in a second task. Client P95 includes
queueing and local transport. Each arm has the same eight-second
client deadline.

A calibration run at nominal 35 and 65 requests/s encountered seven
4B request deadlines at 65 on confirmation. The matched comparison
uses the lower nominal rates 18 and 35 for complete runs. These are
finite public-benchmark replays with a prebuilt index and documents
resident in memory. They cover a structured first-query path. They
exclude enterprise parsing, remote document acquisition and indexing.

## Results

Slurm array 46656 ran on ws3 RTX 5090. Every primary run completed
all scheduled requests. Per arm and rate, the confirmation replay
serves 408 requests and the extended replay serves 576.

The first table reports strict / adverse on unique queries. Confirmation
has 51 positives per view and extended has 72. Each has the same number
of role-swap controls.

| Classifier | Confirmation keyword | Extended keyword | Confirmation excerpt | Extended excerpt |
| --- | ---: | ---: | ---: | ---: |
| Full menu | 29 / 4 | 39 / 8 | 38 / 8 | 53 / 12 |
| Ridge | 24 / 10 | 30 / 5 | 30 / 14 | 44 / 8 |
| Resident CE | 30 / 4 | 37 / 1 | 36 / 8 | 52 / 4 |
| Resident CE fused | 30 / 4 | 37 / 1 | 36 / 8 | 52 / 4 |

The three unfused classifiers reproduce the respective archived
quality counts. Fused CE selects the same source pair and returns the
same roles and owners as unfused CE for every distinct request in both
cohorts. Against the full-menu reader, the CE head improves strict
counts in one view and gives up one or two strict proofs in the other
three. It issues fewer adverse results in both extended views. The
grouped uncertainty intervals for all four strict differences against
4B include zero, so this replay does not establish quality noninferiority.

| Panel | Offered requests/s | Classifier | Client P95, ms | Achieved requests/s |
| --- | ---: | --- | ---: | ---: |
| Confirmation | 19.7 | Full menu | 162.2–162.9 | 19.6–19.7 |
| Confirmation | 19.7 | Ridge | 76.8–79.3 | 19.7 |
| Confirmation | 19.7 | Resident CE | 78.4–78.9 | 19.7 |
| Confirmation | 19.7 | Resident CE fused | 38.5–40.6 | 19.7 |
| Confirmation | 38.3 | Full menu | 3,056.5–3,130.8 | 29.3–29.4 |
| Confirmation | 38.3 | Ridge | 346.7–352.1 | 37.9–38.0 |
| Confirmation | 38.3 | Resident CE | 344.1–345.1 | 37.9–38.0 |
| Confirmation | 38.3 | Resident CE fused | 64.1–64.4 | 38.3 |
| Extended | 17.3 | Full menu | 151.6–156.5 | 17.3 |
| Extended | 17.3 | Ridge | 77.6–78.7 | 17.4 |
| Extended | 17.3 | Resident CE | 80.1–81.5 | 17.4 |
| Extended | 17.3 | Resident CE fused | 41.5–42.9 | 17.4 |
| Extended | 33.7 | Full menu | 2,783.2–2,787.1 | 28.6 |
| Extended | 33.7 | Ridge | 240.1–240.5 | 33.4 |
| Extended | 33.7 | Resident CE | 220.7–252.8 | 33.3–33.4 |
| Extended | 33.7 | Resident CE fused | 71.2–72.9 | 33.7 |

At the lower rates, the fused path achieves the offered rate while
bringing client P95 from about 152–163 ms to 38–43 ms. At the higher
rates, the full-menu worker drains after arrivals stop at about 29
completed requests/s. The fused path keeps pace with the offered
33.7–38.3 requests/s and has 64–73 ms client P95. With source pairing
unchanged, it removes 356 or 492 role-model forwards per replay and
uses one embedding forward per request. The sharp high-load tail
difference includes queueing. For example, confirmation has about
40 ms worker compute P95 for 4B, versus 20 ms with fused CE.

### Transfer to different publishers and wording

Reuse the frozen [Frontiers and Sensors linked-source panel](../engineering-cross-source-search-v1)
and two previously authored non-NISO wording sets, one description
per role in each set. The panel has 53 articles and 15 role-swap
families in four connected paper/author groups. Run the same source
guard and classifiers with both papers present, then with each
intended paper withheld. Do not change fitting data or thresholds.

All arms recover 10/15 strict keyword proofs and 15/15 excerpt
proofs for each wording set, with zero observed adverse outputs.
They issue no substitutes in the 60 source-withholding controls
per clue view and wording set. The source stage misses the intended
pair on 10/30 keyword cases and none on the excerpts. This small
transfer shows that fused CE matches both classifier references
on the scoped panel, including the missing-source checks. It cannot
establish an advantage on unfamiliar difficult role wording.
Slurm array 46682 reproduces the counts in both case orders.
[Transfer metrics](transfer-results.json)

## Reproduce

Restore the two archived projections, both Qwen checkpoints, and the
NISO example snapshot from the
[fixed-source comparison](../elife-fixed-pair-role-menu-v1#reproduce).
Use one Slurm GPU and a fresh results path, then repeat each panel
with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m experiments.credit.elife_resident_role_service PROJECTION MANIFEST QWEN_EMBEDDING \
  QWEN_4B NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/paired-role-verification-v1/results.json \
  benchmarks/elife-role-contrast-training-v1/role-contrasts.json NEW_RESULT
```

After collecting the four results, verify both replica orders, the
frozen study parity, and the dispatch accounting. The first argument
is the repository root.

```sh
python benchmarks/elife-resident-role-service-v1/aggregate.py . NEW_AGGREGATE \
  CONFIRMATION_FORWARD CONFIRMATION_REVERSE EXTENDED_FORWARD EXTENDED_REVERSE
```

For the publisher transfer, restore the pinned 53-article panel and
reuse the model and calibration files above.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m experiments.credit.elife_resident_role_transfer ENGINEERING_PANEL QWEN_EMBEDDING \
  QWEN_4B NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/paired-role-verification-v1/results.json \
  benchmarks/elife-role-contrast-training-v1/role-contrasts.json NEW_RESULT
```

[Complete service metrics](results.json)
