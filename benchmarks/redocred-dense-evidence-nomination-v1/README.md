# Nominate people and evidence from the same indexed sentence

Test a fixed-budget alternative to selecting a person by
lexical proximity and then choosing their evidence separately.
Compile a frozen Qwen3-Embedding-0.6B vector for each source
sentence. At query time, score those sentences against a
single embedding of the requested work and role. A person's
nomination score is the best sentence score among their
automatic mentions. The winning mention supplies an optional
evidence anchor for the same person.

Reuse the 1,000-source [automatic entity index](../redocred-automatic-entity-index-v1),
its pinned offline spaCy projection, and the same source route
and 306 queries. The query gives a work and relation, with
161 exact names and 145 transposed names. The baseline keeps
the original proximity-ranked person shortlist. Each reserve
arm keeps all but its last person and uses the same expanded
automatic name pool. It replaces the last candidate whenever
the pool has an alternative.

| Arm | Last candidate | Last candidate's evidence |
| --- | --- | --- |
| Automatic | Original proximity candidate | Two lexical sentences |
| Lexical reserve | Best excluded candidate by proximity | Two lexical sentences |
| Dense reserve | Best excluded candidate by mention-sentence similarity | Two lexical sentences |
| Dense anchor | Same dense candidate | Dense mention plus the lexical sentence that best names the work |

Each arm scores the same number of candidates and sentences
with the same frozen Qwen3-4B reader in a single batch. The
threshold of 0.25 is inherited from the published automatic
entity comparison. The dense arm compares people using
their mention sentences; two people mentioned on the same
sentence can receive the same score.

Evaluate using the aligned public
[Re-DocRED](https://aclanthology.org/2022.emnlp-main.580/) and
[Re²-DocRED](https://aclanthology.org/2026.eacl-long.213/) snapshots.
Use the updated relation labels to count known people nominated
and accepted. Count an accepted known link as covering the
annotated evidence only if the submitted window contains all
available annotated sentence IDs. Reuse the original sentence
IDs when the updated row has none. Unknown links remain
unjudged. These annotations measure open-source relationship
retrieval and coverage of benchmark evidence. They do not
establish who is suitable for an enterprise project.

P95 includes the online source lookup, candidate decisions,
window construction and live reader. Dense arms also include
online query encoding and sentence scoring. Loader startup,
sentence-vector compilation and transport are outside the
timer. Pair the accepted-evidence outcome by query and
resample whole anchor documents.

## Results

Slurm job `43446` completed on one RTX 5090. Encoding
8,067 distinct sentences took 6.10 s offline. The four arms
evaluated 747 pairs with 1,494 sentences on exact names and
662 pairs with 1,324 sentences on transpositions.

| Arm | Exact accepted / with all annotated evidence | Exact unjudged accepted | Transposed accepted / with all annotated evidence | Transposed unjudged accepted | P95 exact / transposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Automatic | 98 / 53 | 13 | 76 / 43 | 7 | 56.5 / 57.1 ms |
| Lexical reserve | 95 / 49 | 43 | 71 / 37 | 31 | 58.2 / 58.6 ms |
| Dense reserve | 92 / 47 | 25 | 69 / 35 | 17 | 64.3 / 64.7 ms |
| Dense anchor | 92 / 47 | 25 | 69 / 35 | 17 | 63.5 / 63.7 ms |

The reserve slot replaced a candidate on every request. In
the exact-name panel, the dense reserve changed the known
nomination total from 152 to 148 and the transposed panel
from 130 to 118. Reusing the dense mention as evidence
produced the same aggregate counts as lexical selection on
the dense nominations. Against the automatic baseline, its
annotated-evidence P@5 difference was −0.75 percentage
points (paired descriptive 95% interval −1.53 to −0.13)
on exact queries and −1.10 points (−2.03 to −0.39) on
transpositions.

All mandatory replacements lost measured evidence coverage.
Dense sentence relevance admits fewer unjudged links than the
lexical replacement, while also recovering fewer known links.
Keep the proximity shortlist and lexical evidence windows as
the low-cost reference. This intervention tests compulsory
replacement and sentence-level similarity. It leaves open
whether a calibrated selective action using person-specific
signals can preserve strong incumbents and recover some
missing candidates. [Aggregate results](results.json)

## Reproduce

Use the original and extended snapshots specified in
[the earlier relation study](../redocred-work-owner-contrast-v1),
the [automatic projection](../redocred-automatic-entity-index-v1),
and the pinned open checkpoints. Run inside Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.dense_evidence_nomination \
  REDOCRED_DIR RE2DOCRED_DIR PROJECTED_MENTIONS QWEN_EMBEDDING QWEN4B NEW_RESULTS
```
