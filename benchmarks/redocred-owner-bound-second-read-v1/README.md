# Test owner binding in a two-stage evidence read

Use both the single-role and mixed-role panels from the
[matched-coverage study](../redocred-mixed-role-coverage-v1).
The source document, two people and two role–work requirements
are supplied. The experiment compares what to read next within
a located document.

Select two initial sentences by the shared lexical name-and-role
coverage rule. For each of the four person–requirement pairs,
send the better lexical anchor of those two sentences to the frozen
Qwen3-4B reader. Ensure that both source sentences are used. Score
each candidate by the lower of their two decisions and choose the
initial leader. Every staged arm receives those same first scores.

The next action chooses one requirement to revisit and one new sentence.
Each action rescores the same two people for that requirement using the
new sentence alongside their original anchor. The other two scores are
reused. The four arms have identical requirements, candidates, source,
reader, pair scorer, unique sentence budget and live forward count.

| Arm | Choose which requirement to check | Choose the new sentence |
| --- | --- | --- |
| Pooled gap and pooled search | Lowest best score across the two people | Highest learned pair score for either person |
| Pooled gap and owner search | Lowest best score across the two people | Highest learned pair score for the initial leader |
| Owner gap and pooled search | Initial leader's lowest score | Highest learned pair score for either person |
| Owner gap and owner search | Initial leader's lowest score | Highest learned pair score for the initial leader |

The pooled control can compare both owners' retrieval scores. It
sends the same explicit person-specific prompts as the owner-bound
route. The ablation isolates binding in action selection and in
sentence retrieval. It is an adaptation to expert retrieval of the
generic evidence-gap comparison motivated by [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/),
whose published method uses a learned controller and generates
queries for an open retrieval system.

A parallel control chooses a third sentence by lexical coverage
before any model check, then verifies four two-sentence windows
in one forward. Each of the three selected sentences appears
in at least one prompt. The parallel and staged methods expose
three unique sentences and eight cumulative sentence placements.
The parallel route makes four atomic evaluations in one forward.
The staged routes make six evaluations across two forwards. The
number of prompt tokens can differ. The staged comparison itself
holds all evaluation counts equal.

Report known candidates and complete annotation coverage at fixed
issuance. At 100% issue one candidate on every request. At 50%
issue one candidate on half of the requests based on margins using
a label-free cutoff on the test score distribution. Separately freeze
the 50% development cutoff to measure how issuance transfers.
Margins tie-break using the same deterministic request hashes as
the matched-coverage study. Incomplete relation judgments remain
unresolved.

The action oracle enumerates just the admissible second-round
requirement and third-sentence actions after the same initial read.
It counts how many fully annotated proofs could be presented
regardless of the verifier. The oracle does not select test-time
actions. Latency sums the measured initial routing and reader
time and the arm's additional selection and reader time.
Service queueing and document discovery remain outside the workload.

## Results

Both tasks of Slurm array `44377` ran on an RTX 5090 on ws5.
Every non-timing aggregate reproduced after rotating execution order.
The table gives known matches at exactly 50% issuance, followed by the
count of candidates for whom the prompts contain both complete
annotations, irrespective of issuance.

| Held-out panel | Route | Known among issued | Complete annotations | P95 (ms) |
| --- | --- | ---: | ---: | ---: |
| Same role (106 issued / 213) | Pooled gap and pooled search | 60 | 9 | 75.42–75.60 |
| | Owner gap and owner search | 57 | 9 | 75.11–75.43 |
| | Parallel three sentences | 77 | 46 | 53.67–53.94 |
| Mixed role (56 issued / 113) | Pooled gap and pooled search | 7 | 2 | 71.73–71.83 |
| | Owner gap and owner search | 6 | 2 | 71.93–71.95 |
| | Parallel three sentences | 7 | 5 | 51.53–51.98 |

On the same-role holdout the parallel route adds 17 known issued
choices over the pooled staged control. The change per request is
+7.98 percentage points (clustered interval +1.84 to +14.08).
Binding both staged decisions changes 75 actions relative to the
pooled control. It loses three known choices at equal issuance,
with a paired interval of −4.25 to +1.38 points per request.
Binding retrieval alone ties the pooled control on known matches.
All staged variants return nine complete annotations. The initial
anchors already give seven and the staged action oracle can reach
only ten. On the mixed-role holdout, the two initial proofs are
also the oracle ceiling. The parallel route can distribute a
third sentence across both requirements, which the staged design
cannot do without revisiting additional claims. The oracle ceiling
applies only to the evaluated action space.

The official same-role test also favors parallel verification at
50% issuance (16 known versus eight for pooled and nine for
owner-bound). The official mixed-role test has seven versus six
for both staged arms. Its 53 requests provide limited power for
a ranking comparison. These comparisons do not establish a
benefit of the tested owner-bound routing policy. They support
prioritizing parallel verification on located documents and
measuring action headroom before paying for a further round.

[Complete results for both execution orders](results.json)

## Reproduce

Run on a Slurm GPU allocation using the data and checkpoint from the
matched-coverage experiment.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.owner_bound_second_read \
  REDOCRED_DIR QWEN4B NEW_RESULTS --order-offset 0
```

Repeat with order offset 1.
