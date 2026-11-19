# Replace manual entity mentions in source and person search

Search the same 1,000 public Re-DocRED documents as the
[global role search](../redocred-global-role-search-v1).
Use the pinned [spaCy English pipeline](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0)
with only its token-to-vector and entity-recognition stages enabled.
Feed the original document tokens one sentence at a time. Extraction
reads the text only. Its projection contains sentence and token
offsets and predicted types, with no names copied from the labels.

Build the automatic work index from document titles plus predicted
work, organization, product, event, facility, legal and place mentions.
Build person candidates from predicted PERSON spans. Link a single-token
surname to a longer name only if exactly one longer candidate in the
same document ends in that token. The four index combinations vary
work aliases and person mentions independently. A fifth arm uses BM25
with the same automatic person scorer and work-position features.
A sixth uses only document titles for source lookup, leaving automatic
work-position features in the person scorer. This control was added
after the first run to distinguish title metadata from body mentions.

| Policy | Work/source index | Candidate person inventory |
| --- | --- | --- |
| Annotated | Annotated aliases and guarded routing | Annotated |
| Automatic sources | Titles and predicted aliases with guarded routing | Annotated |
| Automatic people | Annotated aliases and guarded routing | Predicted |
| Automatic BM25 | BM25 source retrieval with automatic work-position features | Predicted |
| Titles only | Title aliases with guarded routing and the same automatic work-position features | Predicted |
| Fully automatic | Titles and predicted aliases with guarded routing | Predicted |

The guard includes the one-swap fast path from the
[noisy source study](../redocred-noisy-source-routing-v1). All arms use
the same four-document, five-candidate cap and mention proximity scorer.
Use 161 exact work-name requests and 145 matched synthetic transposition
views. Work/role queries and judgments are derived from the published
annotations for evaluation. Routing never receives those judgments.
Compare strict canonical name matches and matches to unique annotated
alias strings within a document. The latter resolve identities for
scoring only. Neither retrieval nor the reader sees those aliases
unless extracted independently.

Report source hit, candidate availability in retrieved documents
before ranking, and accepted document-level role labels. Use the same
Qwen3-4B reader, two-sentence budget and numeric cutoff of 0.25 as the
preceding study. There is no fitting on this reused panel. A match to a
document-level label need not establish that the selected window itself
fully supports the claim. The timed path includes retrieval,
candidate order, window preparation and inference. It excludes the
offline extraction and index build. Unjudged and competing-role
acceptances are reported separately.

## Results on RTX 5090

Job 43049 reproduces the original five policies' decisions from job
43046 and adds the title-only control. CPU retrieval reproduces the
proposed person–source counts before verification. The automatic index
contains 7,619 distinct alias keys, versus 9,049 from annotations.
On this machine, extracting 1,000 documents took 6.18 seconds as an
offline CPU operation.

| Query | Policy | Known source found | Known person available | Expanded documents | Verified person–source pairs | Accepted known positive / rival / unjudged | P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact (161) | Annotated | 161 | 161 | 236 | 694 | 129 / 3 / 10 | 56.56 ms |
| exact (161) | Automatic sources | 155 | 155 | 392 | 743 | 116 / 2 / 10 | 57.61 ms |
| exact (161) | Automatic people | 161 | 136 | 236 | 710 | 109 / 3 / 11 | 57.30 ms |
| exact (161) | Automatic BM25 | 146 | 126 | 644 | 805 | 87 / 3 / 6 | 59.23 ms |
| exact (161) | Titles only | 154 | 131 | 488 | 771 | 97 / 3 / 10 | 58.01 ms |
| exact (161) | Fully automatic | 155 | 130 | 392 | 747 | 98 / 3 / 10 | 57.99 ms |
| transposition (145) | Annotated | 145 | 145 | 151 | 601 | 111 / 1 / 9 | 57.60 ms |
| transposition (145) | Automatic sources | 129 | 129 | 313 | 658 | 91 / 1 / 8 | 57.93 ms |
| transposition (145) | Automatic people | 145 | 121 | 151 | 619 | 93 / 1 / 9 | 57.72 ms |
| transposition (145) | Automatic BM25 | 91 | 79 | 580 | 725 | 45 / 1 / 2 | 60.19 ms |
| transposition (145) | Titles only | 118 | 98 | 439 | 697 | 70 / 1 / 5 | 58.69 ms |
| transposition (145) | Fully automatic | 129 | 107 | 313 | 662 | 76 / 1 / 6 | 58.89 ms |

At exact names, the automatic index reduces document expansions from
644 to 392 versus automatic BM25 and accepts 11 additional known
positives at the fixed cutoff. The positive P@5 gain is +1.37 points
(document-clustered 95% interval +0.39 to +2.41). At transpositions it
reduces expansions from 580 to 313 and accepts 31 additional known
positives, with a gain of +4.28 points (+2.16 to +6.89). Annotated rival
acceptances are unchanged in both panels. The fully automatic route
accepts four additional unjudged results per panel. The P95 differences
are modest in this warm replay.

Title-only routing accounts for most of the accepted-positive gain.
Adding body aliases accepts one more known positive on exact names
and six more on transpositions while reducing expansions by 96 and
126, respectively. This is a post-hoc control on the same panel.
The automatic source index finds a known source for 155 exact queries,
but automatic names leave a known positive available for only 130.
Using annotated names with the same 392 expanded documents restores
availability on all 155. Identity recognition remains a separate
failure point from source selection.

The synthetic perturbations, reuse of evaluation queries, partial
judgments and public work-name task limit transfer to industrial
expert search. The two-sentence selections themselves were not
independently reviewed.

[Aggregate results](results.json)

## Reproduce

Use the pinned Re-DocRED input files from the
[global search benchmark](../redocred-global-role-search-v1).
Install the CPU extraction dependencies in an isolated environment.
The checked wheel SHA-256 is
`1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85`.
The version pins below pass the dependency check with the tested model.
A second local extraction reproduced all spans. Extraction duration
varies across runs and is recorded separately in the projection.

```sh
python -m pip install 'spacy==3.8.7' 'typer==0.15.4' 'typer-slim==0.15.4' \
  'click==8.1.8' 'numpy==2.4.6' PATH_TO_VERIFIED_NER_WHEEL
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.docred.automatic_role_entity_index extract REDOCRED_DIR NER_PROJECTION.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.docred.automatic_role_entity_index run REDOCRED_DIR NER_PROJECTION.json CPU_RESULTS.json
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.automatic_role_entity_index run REDOCRED_DIR NER_PROJECTION.json GPU_RESULTS.json \
    --gpu-model PINNED_QWEN4B
```

Run the last command in a Slurm GPU allocation.
