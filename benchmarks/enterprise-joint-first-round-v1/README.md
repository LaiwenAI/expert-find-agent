# Test joint first-round evidence selection on live enterprise development queries

Reuse the 80 original development requests and frozen 52,667-source
inventory from the [live hybrid enterprise baseline](../enterprise-hybrid-rescue-v1).
Use the same parsing, live dense query encoding, keyword retrieval,
expert-level fusion shortlist and reranker for all arms. Run each
arm independently and rotate the order across requests. Assert
equal qualifications, ordered top-five candidate shortlist, evidence
slots and logical verifier pair counts for each request.

Compare three ways to fill up to two evidence slots per candidate:

- **Hybrid prefixes:** the shared hybrid path's selected source
  prefixes, up to 800 characters.
- **Independent windows:** pool original prefixes with indexed
  personal-role excerpts for that expert. Rank each excerpt by
  its weakest then summed lexical criterion coverage. Use the
  two strongest individual excerpts.
- **Joint windows:** use the same bounded pool. Enumerate pairs,
  take the highest coverage available across the two excerpts for
  each criterion, then maximize the weakest and summed coverage.
  Prefer distinct original sources on ties. Preserve original
  evidence slots for single-condition requests or candidates with
  only one source slot.

The upgrade pool keeps the 16 highest-ranked individual excerpts
plus any original prefixes outside that bound. For the experiment
the same verifier evaluates each selected excerpt on every
eligible condition. It accepts an expert if each condition has at
least one positive excerpt. Citations translate the selected
windows back to the original source IDs and exact offsets before
review. Two distinct excerpts can be in one original source,
so excerpt slots, source IDs and verifier calls are reported
separately. Model acceptance is provisional.

Timing spans warmed serial request parsing through evidence selection
and verification. Queueing, response transport and building the
indices are outside the measured service boundary. This evaluation
uses development requests only. The [sealed experiment](../enterprise-sealed-first-query-v1)
and [external sampling frame](../enterprise-external-frame-audit-v1)
remain separate.

## Development outcome

Both Slurm array runs `44805` on `ws5` agree on every
request's shortlist, parsed qualifications, selected excerpts and
returned recommendations. Timing ranges span both runs.

| Arm | Issued requests / 80 | Provisional slots | Changed passage plans | Verification pairs | Serial P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid prefixes | 25 | 51 | 0 | 1,279 | 1095.0–1102.2 |
| Independent windows | 26 | 52 | 24 | 1,279 | 1094.7–1094.7 |
| Joint windows | 25 | 51 | 21 | 1,279 | 1096.7–1096.8 |

The shortlisted corpus contains 52,667 source records and the
index has 5,734 predefined role windows. Each arm verifies 799
excerpts and 1,279 condition–excerpt pairs. The P95 difference
between joint selection and hybrid prefixes spans zero in both
paired bootstrap intervals.

[Aggregate results and packet counts](results.json).

Joint selection changes the passages allocated to 21 requests but
returns exactly the same people and citations as hybrid prefixes.
The 25 multi-condition requests contain no overlength qualification.
Joint selection and the prefix control both abstain on all 25;
independent window selection emits one provisional expert. This
separates a real failure to produce different verified results
from failure to exercise the planner.

The exact-view packet has 52 distinct cards. The 51 shared views
would receive the same judgment for all arms. Before any human
review, the possible change in supported P@5 for joint selection
on these 80 requests is exactly zero. For independent selection
the single additional view bounds it to 0–0.25 percentage points
with five slots per request. These are deterministic bounds for
this development trace, without accuracy estimates. All cards
remain unjudged.

An independent test of the main contribution should not rely on
rearranging these same evidence slots alone. This experiment holds
the five-person shortlist fixed and ranks its span pool lexically,
so it cannot distinguish absent experts from inadequate evidence
for candidates already in the shortlist. An upper-bound diagnostic
can use a wider per-condition pool and audit the weakest verified
condition before selecting a new policy.

## Reproduce

Use the pinned development projections and model checkpoints from
the baseline, plus the published role window index and vectors.
Use new output paths and run on a Slurm GPU. The private trace is
needed to prepare exact-view blind review cards locally.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_joint_first_round \
  DEV_QUERIES DEV_CORPUS ROLE_WINDOWS VECTORS.npy VECTOR_METADATA \
  CHECKPOINTS SCRATCH NEW_AGGREGATE PRIVATE_TRACE
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_joint_first_round_review \
  PRIVATE_DATASET DEV_QUERIES PRIVATE_TRACE NEW_AGGREGATE PRIVATE_REVIEW_PACKET
```
