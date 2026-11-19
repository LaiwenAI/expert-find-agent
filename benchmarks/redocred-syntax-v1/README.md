# One-batch candidate and evidence selection with grammatical role links

Test whether an index of explicit person–role links improves
candidate recovery and evidence selection without another model
forward. Build the index offline from spaCy dependency parses of
raw sentences. Match role-specific lemmas to active subjects,
passive agents, and some nominal constructions. Follow simple
coordination and omit directly negated predicates. Treat every
link as a retrieval hint and send the candidate and its evidence
to the same open-weight verifier used by the baseline.

## Evaluation

Index the pinned [Re-DocRED snapshot](https://github.com/tonytan48/Re-DocRED/tree/ccfb54f/data).
Keep the prior dev and test documents in the corpus as distractors.
Use only queries with a reference in the upstream train split
for this new project holdout. Remove train documents with a title
already present in dev or test and exclude works appearing in the
earlier development queries. Report exact names and token-drop
variants separately, since some queries share a reference.
The annotations supply query references and evaluation labels.
Automatic entity spans, work aliases, and grammatical links
determine candidates and windows.

| Method | Candidate shortlist | Evidence window |
| --- | --- | --- |
| Single batch | Original top five | Lexical two-sentence selection |
| Proximity | Replace at most the last candidate with a name that has a stronger nearby role cue | Lexical |
| Syntax candidate | Apply the same replacement rule using an attached role predicate | Lexical |
| Syntax window | Original top five | Include the role-linked sentence when the lexical window misses every such sentence |
| Syntax joint | Use the syntax candidate rule | Use the syntax window rule |

The replacement controls share a source router and name proposal pool.
They keep a slot unchanged unless a replacement has a strictly
stronger score than the incumbent. Each arm verifies the same
number of candidates in one batch with the same checkpoint, fixed
acceptance threshold, and at most two evidence sentences per
candidate. Actual sequence lengths can vary. Record pair counts
and warmed serial P95 from source routing to verifier output.
Offline parsing time is reported separately. There is no model
fine-tuning in this experiment.

A covered positive requires the reader to accept a known correct
person–work–role relation and the selected window to contain all
its annotated evidence sentences. Relations lacking annotated
sentences can contribute only to accepted positives. Report
known competing-role acceptances and unjudged acceptances
separately. Paired intervals cluster by reference document.

## Results

Slurm job 43187 completed on a single RTX 5090 on ws5. Offline
parsing took 60.65 s for 4,052 documents. The test included 754
exact queries and 623 token-drop variants across 237 and 210
reference documents, respectively. Every arm checked 3,634
pairs in the exact panel and 3,097 in the token-drop panel.

| View | Method | Accepted positive | Covered positive | Rival | Unjudged | P95, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Exact | Single batch | 464 | 296 | 6 | 62 | 61.21 |
| Exact | Proximity | 459 | 293 | 5 | 125 | 61.54 |
| Exact | Syntax candidate | 477 | 309 | 5 | 136 | 61.43 |
| Exact | Syntax window | 466 | 298 | 6 | 62 | 61.11 |
| Exact | Syntax joint | 479 | 311 | 5 | 136 | 61.45 |
| Token drop | Single batch | 193 | 110 | 2 | 99 | 61.87 |
| Token drop | Proximity | 179 | 103 | 2 | 96 | 62.06 |
| Token drop | Syntax candidate | 196 | 114 | 2 | 128 | 61.97 |
| Token drop | Syntax window | 195 | 111 | 2 | 99 | 61.81 |
| Token drop | Syntax joint | 198 | 115 | 2 | 128 | 62.09 |

For exact names, the joint method adds 15 covered positives
with a covered P@5 gain of 0.40 percentage points (document-clustered
95% interval 0.16 to 0.67). Candidate selection accounts for 13 of
the 15 additional covered positives in the corresponding ablation.
On token-drop variants the joint method adds five, with an interval
including zero (−0.03 to 0.37 points). Serial P95 rises by about
0.23 ms in both panels. Unjudged acceptances increase by 74 on exact
queries and by 29 on token-drop queries. These require adjudication
before using the results to claim a precision gain. The result
supports explicit grammatical links as a candidate-recall signal
on well-specified public role queries, with cross-domain transfer
and behavior under service load still to be measured.

[Aggregate results and paired intervals](results.json)

## Reproduce

Fetch the three JSON files from the pinned Re-DocRED snapshot.
Install spaCy 3.8.7 and en_core_web_sm 3.8.0, and use the pinned
Qwen3-4B-Instruct-2507 checkpoint. Run inside Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.syntax_role_evidence \
  path/to/pinned/snapshot path/to/train_revised.json \
  path/to/qwen3-4b-instruct-2507 path/to/new-results.json \
  c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2
```
