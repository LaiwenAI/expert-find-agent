# Allocate an equal reading budget to same-person proof

Use the [Re-DocRED publication](https://aclanthology.org/2022.emnlp-main.580/)
and its pinned ccfb54f snapshot. Each unedited document supplies
two people and two works with one queried role, such as director.
Both works have at least one annotation in the chosen two-person
pool. Include cases with split support and cases where a single
person has both relations. As in the
[work/owner contrast](../redocred-work-owner-contrast-v1/README.md),
the four atomic questions come from the two people by two works.
An individual qualifies only when both questions about that person
are positive. Keep one case per original label count and document
according to the existing deterministic construction. Discard views
with ambiguous name-to-annotation alignment or no annotated
evidence. Relationship labels and sentence annotations are used
for cohort construction and evaluation, outside the selectors.

Use the original development release for threshold calibration and
the original test release for replication. Also hold out an
additional title-disjoint cohort from the upstream training
release. It contributes at most 120 cases per presence/absence of
a known conjunction, selected by the stable case hashes. These
documents are held out from this method's calibration. Related
experiments have used the same public release, and this design
does not screen foundation model pretraining overlap.

Each arm accesses four distinct original sentences for inference
and assigns exactly two sentences to each atomic question. All
four sentences must reach at least one of the four prompts.
Token counts may differ. The selector can inspect lexical features
of every sentence in the supplied document. The experiment
isolates what text is passed to the reader, within a document
whose discovery and entity extraction are supplied.

| Route | Choose the four shared sentences |
| --- | --- |
| Global names | Greedily cover all four names and role terms. |
| Claim quota | Start with the best lexical sentence for each specific person–work claim, deduplicate, then fill by claim coverage. |
| Claim coverage | Maximize the sum of best lexical evidence scores for the four claims using greedy selection. |
| Conjunction | Seed with the strongest sentence pair under the sum of the two candidates' weakest-claim scores, then add sentences by the same objective. |

All four methods use the same claim-specific assignment after
selection. A claim keeps its highest-scoring sentence and a
second that maximizes coverage of the selected set. The lexical
claim score combines the weaker and stronger name matches and
one role cue. It is a retrieval signal. Witnesses still require
verification. Run the frozen Qwen3-4B-Instruct-2507 once on the
four prompts. The conjunction margin is the smaller of a
candidate's two scores. Each arm uses a frozen development
cutoff that allows at most 5% of unjudged candidate conjunctions
to pass in calibration.

Report full coverage of the supplied annotation sentences for
each claim, for both of a candidate's claims together, and in
conjunction with an accepted known candidate. The released
support sentences may omit or over-specify equivalent proof.
Treat missing relation labels as unjudged. A candidate lacking
two positive labels does not count as a known negative.
Compare known acceptance by document bootstrap. Timed execution
includes lexical features, sentence selection and assignment,
and four live model comparisons, but no open-catalog search or
serving. Earlier [Eider](https://aclanthology.org/2022.findings-acl.23/)
operates at entity-pair evidence extraction and combines local
and full-document inference. The controlled distinction here is
allocating a common fixed text budget across two competing
same-person conjunctions.

## Results

Both Slurm tasks in array `44296` completed on an RTX 5090 on
ws5, with identical non-timing aggregates. The additional
held-out cohort covers 213 cases from 178 documents, with 160
known full conjunctions and 266 unjudged candidates. The release
provides sentence annotations for both required relations in 73 of
the 160 cases. Of those, 56 fit within two sentences per relation.
This is the maximum number of *fully reproduced annotations*
attainable under the window budget before the model sees any text.

| Selector | Known accepted / 160 | Unjudged accepted / 266 | All annotated sentences shown for both links / 73 | Known accepted with both annotations shown | P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global names | 50 | 14 | 30 | 11 | 53.90–54.24 |
| Claim quota | 42 | 11 | 24 | 9 | 54.01–54.28 |
| Claim coverage | 42 | 11 | 25 | 9 | 54.16–54.44 |
| Conjunction | 42 | 11 | 24 | 9 | 54.02–54.53 |

The conjunction policy recovers fewer full annotated proofs than
global name coverage. The two routes have different unjudged
acceptance on held-out data after the same calibration ceiling.
Treat the known-acceptance comparison as the result of those
frozen operating points. The native 38-case test has only five
candidates with both sentence annotations, of which three fit.
All four selectors miss full annotation coverage there.

### Entity mention audit

On an identical 208-case portion of the larger holdout, provide
either distinct released name aliases or the locations of all
annotated entity mentions, including noncanonical names. Neither
variant uses relation labels to score a sentence. This is an offline
diagnostic with supplied entity mapping. The clean-name filter
excludes five ambiguous cases from all three variants.

| Index feature | Global names: full annotation sets | Conjunction: full annotation sets |
| --- | ---: | ---: |
| Canonical names | 29 | 23 |
| Distinct aliases | 29 | 24 |
| Annotated mention locations | 30 | 25 |

The oracle mention locations recover two additional full annotation
sets for the conjunction objective, while the best global control
still finds more. Of 71 full annotation sets in this audit, 54 fit
the window. This points to the relation evidence selector and the
assignment of evidence to individual claims as the constraints that
remain after entity grounding. Only 11 known candidates on the
best measured route receive both full annotations and pass the
model, so evaluating the verifier's use of a supplied proof is also
important. Missing annotation remains distinct from missing support.

[GPU aggregates](results.json) · [Mention audit](mentions.json)

## Reproduce

Use Re-DocRED's original dev, test, and train files at
[ccfb54f](https://github.com/tonytan48/Re-DocRED/tree/ccfb54f),
and the same Qwen checkpoint as in the linked work/owner study.
Run in an allocated Slurm CUDA job.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.conjunctive_proof_windows \
  REDOCRED_DIR QWEN4B NEW_RESULTS --order-offset 0
```

Run the annotation-capacity and entity oracle audit on the same release
with `PYTHONPATH=src:. python -m experiments.docred.conjunctive_mentions_audit REDOCRED_DIR NEW_AUDIT_JSON`.
