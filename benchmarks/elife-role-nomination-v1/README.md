# Transfer same-person nomination from PLOS to eLife

Use a separate publisher with a different format for author contributions.
The primary contrast uses the same four-source verification budget for
topic-first and same-person nomination. Also compare a same-budget
article-wide role union, within-top-eight person nomination, and an
eight-source topic-only control. Transfer the PLOS embedding, reranker
and cutoff unchanged. Both phases use the same topical ranking and
verifier. All recommendations require a source binding both requested
roles to the same credited person in the article.

## Data and protocol

Pin the [publisher's XML repository](https://github.com/elifesciences/elife-article-xml/tree/58250dd64e27afae6bc2700bcddd84dd38776e0d).
Take article IDs in `[45000,65000)`. Sort by the SHA-256 of
`expert-role-transfer-20261120:elife-<id>-v1.xml`, take the first
360 existing IDs, and use the highest available numbered XML version
for each. This selection precedes contributor parsing. Keep CC BY 4.0
and CC0 research articles with title, abstract, author and heading
metadata. Exclude exact normalized author names, valid ORCIDs, or
titles overlapping the 479 unique XML snapshots in the earlier
local PLOS collections. The underlying comparison is deterministic
and only uses public source material. The [manifest](manifest.json)
pins all source files and exclusion outcomes.

In eLife XML, resolve article-specific contribution footnotes to
individual byline authors via their footnote references. Map exact
controlled-role phrases to the PLOS inventory, allowing typographic
variants of the two writing labels. If an author lacks a recognized
list or has an unknown term, leave the article eligible for verification
even when its partial declarations do not match the requested pair.
Only unambiguous ORCID fields bind identities across articles. Other
authors are represented by their article and byline position.

Use each publisher subject heading that occurs in at least eight
articles and each exact case-insensitive author keyword that occurs
in at least four articles. For both panels use the existing selection
of one rarer and one more frequent same-person pair per topic with
partial support. Neither metadata label is passed to retrieval or
verification as an article feature. Topic queries themselves contain
the selected term. Results distinguish publisher heading and
author keyword panels, and retain off-label citations as unresolved.
Group paired intervals by topic. Topic labels can be incomplete,
especially author keywords. The labels certify declared contributions
and topic metadata, not human-assessed expertise or enterprise fit.

## Results

Slurm jobs `43765` and `43767` on an RTX 5090 matched all non-timing
metrics. The catalog retains 322 of 360 articles. It excludes 25 by article type,
12 by exact author or title overlap, and one with missing author or heading
metadata. Of the included articles, 280 have complete parseable role
lists. The sparse declaration mask indexes 144 of these. The remaining
records can still enter the verifier and contribute declared positive roles.

| Topic labels and indexed declarations | Topic four | Role union four | Person four | Top-eight person four | Topic eight |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 heading requests, 280 declarations | 15 / 7 | 15 / 7 | 15 / 7 | 15 / 7 | 15 / 10 |
| 48 keyword requests, 144 declarations | 64 / 30 | 64 / 31 | 69 / 31 | 69 / 31 | 84 / 57 |
| 48 keyword requests, 280 declarations | 64 / 30 | 68 / 43 | 80 / 46 | 80 / 46 | 84 / 57 |

Cells report tagged / off-label recommendation slots. At full available
coverage, person nomination gains 16 tagged slots against topic four
on the keyword panel (+6.67 percentage points of tag-backed P@5,
descriptive paired 95% interval +2.50 to +12.08). The off-label count
also rises by 16 slots. Against role union four, the tagged gain is
12 slots (+5.00 points, interval +1.67 to +8.75). Heading results show
no tagged gain. Publisher subjects and author keywords capture
distinct notions of topical relevance.

Bounded person nomination cuts the keyword panel's verification reads
from 384 to 173 and local P95 from 46.47–46.61 to 26.68–26.77 ms versus
topic eight, with four fewer tagged slots.
The paired tagged P@5 interval is −3.75 to 0.00 percentage points,
which fails the −1 point noninferiority criterion. P95 includes local
query encoding, ranking, verification and output assembly.
It excludes process startup, service queues and external storage.

These results reproduce a same-budget benefit in a second publisher
when the request supplies a fine-grained topic. They support testing
early role binding against both same-budget topic-first and
article-wide role unions. A separate label audit is required to
distinguish inadequate citations from missing keyword tags and to
assess the higher recommendation issuance.

[Aggregate results](results.json)

## Reproduce

Download the included files from the pinned path
`https://raw.githubusercontent.com/elifesciences/elife-article-xml/<revision>/articles/<filename>`.
The preparation command checks every included file and the resulting
projection against the manifest. Then run the original open checkpoints
on an allocated Slurm GPU.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.elife_role_nomination prepare XML_DIRECTORY \
  benchmarks/elife-role-nomination-v1/manifest.json NEW_PROJECTION
python -m experiments.credit.elife_role_nomination run NEW_PROJECTION \
  benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/plos-bounded-topic-v1/results.json NEW_RESULTS
```
