# Nominate candidates using local relationship context

The [candidate headroom audit](../redocred-joint-candidate-evidence-v1)
found additional labeled candidates that the lexical proposal scorer
ranked below its first choice. Test whether a small ranker can better
use the words that connect a candidate to a requested role. The
experiment keeps the automatic source route, proposal pool, reader,
reader threshold and number of verification slots fixed across the
two learned variants.

- **Automatic** verifies the unchanged initial shortlist.
- **Geometry** scores the incumbent and guarded challengers with
  the existing name-origin, role-cue proximity and source features.
- **Context** adds 512 bits for hashed left and right context around
  candidate mentions. The feature encoder masks capitalized tokens,
  marks requested role words, and records near-name prefixes and
  intervening words between a mention and a role cue within seven
  tokens. It feeds a linear ranker.

The learned arms optimize pairwise logistic loss on predicted names
matched to an annotated positive and an annotated competing role in
the same source. Unjudged names do not supply negative training
labels. Candidate labels and annotated alias mappings serve only
training, validation, and scoring. Development reference papers are
split deterministically by reference document. Each fitting pair's
source must also fall on its assigned side. Work names from the test
panel are excluded in development. Validation chooses an optimization
step and calibrates each replacement gate. Among gates replacing at
most 30% of validation shortlists and with no net increase in proposed
annotated rivals, the selection prioritizes known-positive gain.
All models preserve earlier slots and submit the same number of
prompts to a frozen Qwen3-4B reader.

The two views reuse the prior Re-DocRED panels. Counts based on
documentary relationship labels are partial, so additional unjudged
outputs are reported separately. Warm replay timings include source
retrieval, nomination, reading and verification. They exclude service
queueing and model load.

## Results

Slurm job 43083 used one RTX 5090 on ws5. Each model trained on 542
within-document pairs. Across 227 development validation pairs,
geometry and context achieved 70.5% and 78.4% pair accuracy, with
pairwise log losses of 0.445 and 0.491. The gate changed 24 and 22
of 80 validation shortlists, respectively.

| Test view | Policy | Swaps | Nominated positives | Verified positives | Competing roles accepted | Unjudged accepted | Routing and nomination P95 (ms) | Complete P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact, 161 queries | Automatic | 0 | 152 | 98 | 3 | 10 | 1.52 | 57.36 |
| | Geometry | 57 | 154 | 101 | 2 | 13 | 2.34 | 58.36 |
| | Context | 55 | 155 | 99 | 3 | 12 | 2.92 | 58.80 |
| Transposition, 145 queries | Automatic | 0 | 130 | 76 | 1 | 6 | 1.65 | 58.12 |
| | Geometry | 42 | 132 | 79 | 1 | 9 | 2.46 | 58.43 |
| | Context | 45 | 133 | 77 | 1 | 9 | 2.97 | 58.69 |

All arms verify 747 and 662 slots in the two panels. Geometry gains
three verified known positives per panel versus the automatic
shortlist. The document-clustered 95% intervals for its accepted
known-positive P@5 change are [0.00, +0.84] and [0.00, +0.92]
percentage points. Context nominates one more known positive per
view than Geometry but accepts two fewer. Its intervals against
Automatic are [−0.57, +0.72] and [−0.65, +0.81] percentage points.
The higher contextual pair accuracy did not carry through to the
frozen reader. The gate's 30% cap applies to calibration, and both
learned arms exceed that frequency in the exact test panel. The
actual cost comparison therefore relies on fixed verification slots
and measured latency. These public relationship judgments do not
measure enterprise expert suitability or independently verify
the exact cited sentences.

A direct follow-up should hold the geometry shortlist fixed and
compare the existing reader windows with windows centered on the
mentions that drove nomination. This would isolate whether retrieving
the specific candidate evidence helps the verified results.

[Aggregate results](results.json)

## Reproduce

Use the pinned public inputs and extraction setup from the
[automatic source-index benchmark](../redocred-automatic-entity-index-v1).
Run inside a Slurm allocation with one CUDA GPU. The Qwen checkpoint
revision and input hashes are recorded in the aggregate results.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.contextual_role_nomination \
    REDOCRED_DIR NER_PROJECTION.json PINNED_QWEN4B NEW_RESULTS.json
```
