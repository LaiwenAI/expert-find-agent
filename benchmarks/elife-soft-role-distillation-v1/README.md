# Distill masked 4B role preferences into a resident linear head

Train two linear heads from the same 84 task descriptions and frozen
Qwen embedding vectors. For each phrase, include the full glossary,
the dense top three, the gold role with two hard negatives, and a
three-role set with the gold excluded. Deduplicate equivalent masks.
Every mask includes a no-match option. Obtain the teacher logits from
the same 4B role prompt used in the
[teacher diagnosis](../elife-mixed-role-verifier-v1).

Both students use half gold cross-entropy and half teacher supervision.
The hard student uses the teacher's top option. The soft student uses
its distribution at temperature 2. Train only the linear heads on CUDA.
Fit each margin on the 48 frozen PLOS controls, maximizing strict
proofs with zero observed adverse issuance. The PLOS comparison gives
the intended source pair to both arms. It calibrates role evidence
under fixed sources.

Freeze the new heads and margins before replaying both archived eLife
panels. Preserve the original role cascade and the highest-ranked
dense source pair on each fallback. The student may relabel either
role from its three alternatives using the resident activity vectors.
Run a same-person join under the student roles before calling the
0.6B source reranker on both clauses. A student rejection uses no model
forward. Each accepted fallback uses one source forward.

Check original service proof parity and count strict proofs, adverse
issuances, and source work by view. The six fallback requests across
both eLife panels have been examined previously. The test compares
matched training and execution paths and has limited statistical
power.

## Results

All four tasks in Slurm array 46289 train both heads on ws3 RTX 5090.
The teacher matches the masked gold label in 220 of 299 training
instances. Both calibrated students produce 24 strict proofs and
zero adverse issuances in the 48 PLOS controls. Calibration sets
each cutoff to zero. Both request orders reproduce exactly.

| Policy | Development keyword strict / adverse | Held keyword strict / adverse | Additional source forwards, development / held |
| --- | ---: | ---: | ---: |
| Cascade alone | 12 / 0 | 10 / 0 | 0 / 0 |
| Dense joint fallback | 15 / 0 | 11 / 1 | 4 / 2 |
| Hard student | 13 / 0 | 10 / 0 | 2 / 0 |
| Soft student | 13 / 0 | 10 / 0 | 2 / 0 |

The hard and soft heads emit identical evidence in these replays.
On held keywords, each head produces an admissible role prediction
for one of the two dense fallback requests, but neither request
passes the same-person join and receives a source check. The two
students therefore suppress both the good extra proof and the wrong
result. Every excerpt response is unchanged.

The earlier two-forward 4B diagnostic separated the held outcomes.
Soft scores on this small, fixed phrase inventory yield no
additional separation over matched hard targets in the resident
linear model. The comparison holds representations and masks fixed
and leaves open the effect of richer activity phrasing or a
nonlinear student. See [aggregate results](results.json).

## Reproduce

Restore the two eLife panels, frozen initialization, PLOS projection,
and per-panel service reference. Use the same setup as in the
[mixed verifier](../elife-mixed-role-verifier-v1#reproduce), substituting
`--soft-role-reference MATCHED_SERVICE_RUN`. Run inside a Slurm GPU
allocation with each request order and a fresh output path.
