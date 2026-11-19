# Learn whether to replace an expert candidate before reading evidence

Use the same frozen automatic source index, name projection and reader
as the [role-local repair study](../redocred-role-local-person-rescue-v1).
Keep every policy's source shortlist and per-request number of reader
inputs equal. For each nonempty candidate list, a replacement may
change only its last slot. Empty lists remain empty.

Generate replacement proposals from predicted PERSON spans, names
near the role cue and candidate subspans within a predicted entity.
Compute features from their origin, mention frequency and length,
source rank, cue distance and direction, short wording bridges such
as *by* or *as*, work mention proximity and requested role. These
features see only original text, automatic annotations and work
metadata already used for retrieval.

Use only development queries whose canonical work does not occur
in the test queries. Pair a known positive with an annotated
competing-role person in the same retrieved document. Unjudged
candidates have no training label. Split pairs by their reference
document before fitting. Train a linear ranker and a ranker with a
32-unit tanh hidden layer by minimizing pairwise logistic loss.
Choose the checkpoint every ten steps on development validation
loss during 120 fixed AdamW steps. No test outcomes are used in
training or model selection. Multiple alias contrasts from the same
source can occur in training and are not independent observations.

| Policy | Last-slot decision |
| --- | --- |
| Automatic | Keep the original NER-based result |
| Fixed guard | Give one slot to the role-local name proposal when it offers a new candidate |
| Linear reserve | Rank the incumbent and available alternatives using the fitted linear model |
| MLP reserve | Rank the incumbent and available alternatives using the fitted MLP |
| Guarded linear / MLP | Consider alternatives from automatic PERSON or role-local proposals within three tokens of a cue. Require the fitted score margin to clear a frozen validation threshold. |

The guarded arms were added after the first held-out replay. On the
same development validation set used for model selection, choose a
margin threshold that maximizes known-positive gains without adding
annotated rivals, within a 30% replacement cap. If thresholds tie,
prefer removing more annotated rivals, then making fewer replacements.
Unjudged outcomes remain unknown. This tests a conservative
deployment policy on a reused test panel.

The learned policies keep the incumbent when it ties or outranks
available alternatives. A policy may replace the last candidate
with an existing NER candidate outside the original list as well
as a newly expanded candidate. The first slots remain unchanged.
Inference uses CPU NumPy for the learned rankers. The same
frozen 4B model and cutoff of 0.25 then read no more than two
sentences per candidate, using the proposed name. Resample
reference documents for paired intervals. Outcomes refer to
document–person relationship labels and are not independently
validated sentence entailment. The exact and transposition test
panels are reused from prior exploratory work.

## Results

RTX 5090 job 43064 reproduces job 43060's original four policies and
evaluates the guarded models. Every method sends 747 reader inputs
for exact requests and 662 for transpositions. The models fit 711
contrasts from 39 reference documents and validate on 58 contrasts
from nine documents. Validation orders 72.4% of contrasts correctly
for both rankers. The linear model has 24 parameters and the MLP has
801. GPU optimization was under one second per model after setup.

| Input | Method | Replacements | Proposed known positives | Accepted positive / rival / unjudged | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| exact (161) | Automatic | 0 | 152 | 98 / 3 / 10 | 57.73 ms |
| exact (161) | Fixed guard | 86 | 154 | 99 / 2 / 15 | 57.98 ms |
| exact (161) | Linear reserve | 161 | 144 | 93 / 2 / 35 | 58.61 ms |
| exact (161) | MLP reserve | 161 | 146 | 96 / 2 / 37 | 58.27 ms |
| exact (161) | Guarded linear | 8 | 152 | 98 / 3 / 10 | 58.54 ms |
| exact (161) | Guarded MLP | 21 | 153 | 99 / 3 / 10 | 58.45 ms |
| transposition (145) | Automatic | 0 | 130 | 76 / 1 / 6 | 58.53 ms |
| transposition (145) | Fixed guard | 68 | 130 | 76 / 1 / 8 | 58.12 ms |
| transposition (145) | Linear reserve | 145 | 118 | 72 / 1 / 29 | 58.72 ms |
| transposition (145) | MLP reserve | 145 | 118 | 71 / 1 / 31 | 58.51 ms |
| transposition (145) | Guarded linear | 5 | 130 | 76 / 1 / 7 | 59.14 ms |
| transposition (145) | Guarded MLP | 15 | 131 | 77 / 1 / 9 | 59.14 ms |

Both unrestricted learned methods replace the last candidate on
every test query, despite the option to keep the incumbent. They
decrease known positives and admit substantially more unjudged
results. The guarded models reduce replacements. Guarded MLP changes
the accepted known-positive count by one on each panel. Its paired
P@5 intervals against automatic are 0.00 to +0.42 percentage points
for exact requests and 0.00 to +0.46 for transpositions. The proposed
action does not establish a repeatable gain at the same read count.
Times include live selection and the common 4B read, under warm
serial execution. The differences of about a millisecond are
descriptive and favor neither learned policy on latency.

An oracle headroom diagnostic counts requests where the last baseline
slot is not a known positive and an outside candidate has a known
positive person–document label that is absent from the baseline list.
The unrestricted proposal space has headroom on 49 exact and 38
transposition requests. Constraining candidates to the guarded pool
reduces the count to five in each panel before thresholding or neural
verification. The calculation uses the labels after search. It is an
upper bound on additional known candidate placements from one
replacement, not an achievable or independently adjudicated quality
result. These experiments locate the remaining problem in ranking
new names against incumbents without treating unjudged proposals as
negative examples.

[Aggregate results](results.json)

## Reproduce

Recreate the snapshots and automatic entity projection using the
[automatic index instructions](../redocred-automatic-entity-index-v1).
Run the fitting and replay in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.learned_role_candidate_rescue \
    REDOCRED_DIRECTORY NER_PROJECTION PINNED_QWEN4B NEW_RESULTS.json
```
