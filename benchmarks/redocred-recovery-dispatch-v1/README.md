# Dispatch second-stage reads to candidate depth, identity, or source

Test the choice of recovery action with a fixed per-query reader
budget. Use raw text from the pinned [Re-DocRED snapshot](https://aclanthology.org/2022.emnlp-main.580/)
and the [automatic entity index](../redocred-automatic-entity-index-v1).
Queries ask for a credited person given a work and role. Use exact
work names and the existing observed-alias, token-drop and
adjacent-transposition variants. The variants share references with
the exact panel and remain separate in reporting. They have been
used in earlier experiments and are not a fresh benchmark split.

For every query the original source router and shortlist determine
the exact number of person-source verification pairs, at most five.
Use the same two-sentence lexical evidence selector, pinned
Qwen3-4B-Instruct-2507 reader and fixed threshold across arms.
Staged methods score the first three pairs, then use the remaining
slots in a second batch. Their action pools are:

- **Depth:** the remaining pairs in the original shortlist.
- **Identity:** automatically extracted or proposed names missing
  from the original list, from the same articles.
- **Source:** candidates from the initial BM25 source list outside
  the articles returned by the first source route.

| Arm | Recovery decision |
| --- | --- |
| Single batch | Verify the original top five together |
| Staged depth | Continue down the same shortlist after the first three |
| Static mix | Interleave depth, source and identity in that fixed order |
| State gap | After some initial support, favor depth and then identity. With no support, favor a new source when the title router signals ambiguity, and otherwise favor a new identity in the current articles. |

If an action pool is empty, advance to the next pool. Candidate
IDs, human judgments and evidence annotations are never used as
policy inputs. Use them only to construct reference queries,
score accepted identities and check annotation overlap. A strict
evidence-coverage hit requires a known positive identity and a
single returned window containing all nonempty annotated evidence
indices for that relation. Report known competing roles and
unjudged acceptances separately. Some relations have no evidence
annotation. Overlap is a measurement of labeled sentence coverage,
not proof that all attached text entails a real technical skill.

P95 includes the live source route, conditional expansion,
evidence selection and actual GPU forwards, excluding queues
and model load. Paired intervals cluster by reference document.

## GPU results

Slurm job 43173 ran 466 queries with one RTX 5090 on ws5.
The following cells show accepted positives / positives whose
selected window contains all annotated sentences / unjudged
acceptances.

| Query view | Single batch | Staged depth | Static mix | State gap |
| --- | ---: | ---: | ---: | ---: |
| Exact (161) | 98 / 53 / 10 | 99 / 53 / 10 | 97 / 51 / 12 | 97 / 52 / 34 |
| Observed alias (24) | 10 / 6 / 0 | 10 / 6 / 0 | 9 / 4 / 1 | 9 / 4 / 3 |
| Token drop (136) | 46 / 20 / 13 | 47 / 20 / 13 | 45 / 18 / 25 | 45 / 18 / 25 |
| Transposition (145) | 76 / 43 / 6 | 76 / 43 / 6 | 73 / 40 / 5 | 74 / 40 / 26 |

At equal criterion-pair count, the staged depth control has 122
fully covered positive results across the views, versus 114 for
state-gap dispatch. The views are dependent and that sum is
descriptive. The state policy has 88 unjudged acceptances versus
29 for staged depth. Known competing-role acceptances are three
and four, respectively. The additional unjudged outputs should be
reported separately from errors. State-gap transitions consume
419 alternate-identity checks and 27 alternate-source checks.
The paired intervals are in the aggregate results.

Single-batch P95 is 57.52–59.72 ms across views. State-gap P95 is
66.91–68.52 ms. Each arm scores the same number of candidate pairs,
but the second forward makes staged execution slower than the
single-batch baseline in this warmed serial measurement. Staged
depth covers exactly the same number of annotated positive results
as single batch in each view. The two batch layouts need not
produce identical acceptance thresholds, as seen in two counts.
The tested state rule does not translate the identity pool's
oracle headroom into confirmed gains. A stronger comparison
would preserve the top five by default and require calibrated
evidence that a replacement can improve the returned result.

[GPU aggregates](results.json)

## Available headroom before verification

A separate CPU audit compares the announced action pools using labels
only after the lists have been built. It excludes known positive
identities already in the first three, and asks how many distinct
positives an oracle could put into the same remaining slots. It
does not model reader errors, evidence sentence coverage, or the
algorithm's ability to select the best pair.

| Query view | Depth positives | Extra distinct identities in original sources | Extra distinct identities in expanded sources | Oracle maximum gain under the same number of slots |
| --- | ---: | ---: | ---: | ---: |
| Exact | 25 | 16 | 1 | 15 |
| Observed alias | 7 | 4 | 0 | 4 |
| Token drop | 23 | 13 | 1 | 12 |
| Transposition | 23 | 15 | 0 | 13 |

The headroom comes mainly from alternate people in reachable
documents. The source expansion in this experiment is limited to
the initial lexical list, so this observation does not cover
alternative indexes or a deeper search over sources.

[Aggregate audit](headroom.json)

## Reproduce

Use the pinned files and projection from the automatic-index
benchmark, with the listed open-weight reader, inside Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.dispatch_recovery_actions \
    REDOCRED_DIR NER_PROJECTION.json PINNED_QWEN4B NEW_RESULTS.json
PYTHONPATH=src:. python -m experiments.docred.audit_dispatch_headroom \
  REDOCRED_DIR NER_PROJECTION.json NEW_HEADROOM.json
```
