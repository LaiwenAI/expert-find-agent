# Same-person role intersections in single-batch contributor search

Test whether two declared roles should be combined before nominating
people or after reading their evidence. Use publisher declarations
in PLOS ONE JATS to record a role on a particular article for a
particular author ordinal. This is retrieval of declared roles.
It measures neither general research aptitude nor enterprise
suitability. [PLOS authorship policy](https://journals.plos.org/plosone/s/authorship) ·
[CRediT role taxonomy](https://credit.niso.org/)

## Paired queries and missing sources

Build two requests per selected article. One has at least one author
credited with both requested roles and a coauthor credited with
only one. The paired request changes one role. The article declares
both requested roles somewhere, but no single author has both.
Select one eligible pair per article in fixed hash order.

Describe the article with alternating distinct content terms from
its title after filtering stopwords. The method receives neither
a DOI nor a full title. The role names are canonical CRediT labels.
This is a controlled paraphrase proxy with overlapping title terms.
The target is the intended article, so an accepted author from any
other article counts against source precision in this protocol.
Human-written ambiguous descriptions could legitimately permit
other articles.

Repeat each query with its source present and withheld. The
withheld DOI is used only by the experiment harness to simulate
absence. It is excluded from retrieval while the lexical index
statistics stay fixed. The published fixture stores the target DOI,
source digest, two roles, and expected ordinals. It omits author
names. The articles are distributed by PLOS under CC BY 4.0 and
credited by DOI in the fixture. [License](https://creativecommons.org/licenses/by/4.0/)

## Matched verification

All arms retrieve six articles using the same lexical index over
titles and abstracts, with threefold title term frequency. Each
arm nominates up to five article–author pairs. The original
byline orders authors within a source by ordinal. The article
union favors sources where both roles occur anywhere in the
byline. The marginal model adds 0.20 to the reciprocal source rank
for each requested role independently linked to that author.

The proposed one-slot method starts from the marginal five and
can substitute one author whose declaration has both roles for
an unqualified last candidate. Full intersection moves all
authors with both roles before the remaining candidates. Both
methods obtain membership from author-scoped inverted postings.

One frozen Qwen3-4B-Instruct-2507 batch verifies the same number
of pairs per query in every arm, with the same prompt, context
allowance, and cutoff. Each prompt contains a source title, up
to 80 words of abstract, and the exact author ordinal's declared
roles. Acceptance uses a support (1) minus no-support (0) logit margin above 0.25.
Measure confirmed correct author–article links, incorrect
same-article role attribution, other-source acceptances, and
exact accepted-set matches. Report warmed P95 from source retrieval
through GPU verification. Offline construction and queues are
separate. Cluster uncertainty by target article.

## Fixed-budget candidate results

Slurm job 43249 ran 120 requests over a 134-article inventory on
one RTX 5090 on ws5. All 60 intended articles were in the lexical
top six when present. The 60 supported requests have 69 valid
author–article links. Every method verifies exactly five candidate
pairs in every state. A withheld test contains 120 requests across
the two query types.

| Method | Supported links nominated / 69 | Supported links accepted / 69 | Wrong role links on split requests | Other-source accepts, source present / withheld | Supported-query P95, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Byline | 65 | 31 | 3 | 0 / 0 | 84.50 |
| Article union | 65 | 31 | 3 | 0 / 0 | 83.40 |
| Marginal | 69 | 34 | 4 | 0 / 0 | 83.34 |
| Replace one | 69 | 35 | 4 | 0 / 0 | 84.45 |
| Owner intersection | 69 | 34 | 4 | 1 / 1 | 84.83 |

Moving role evidence into candidate ranking recovers all 69 valid
links for verification. The marginal method already reaches that
nomination ceiling. The one-slot method confirms one more link
with an interval that includes zero. The full intersection admits
one other-source candidate both when the original article is
present and when it is withheld. The joint verifier accepts just
34 of the 69 nominated correct links in the marginal arm even
though it sees the correct contributor's complete declaration.
This shifts the next comparison from candidate selection to how
the system verifies source identity and author-scoped roles.

[Candidate experiment aggregates](results.json)

## Separate confirmation of the source and the person

Replay the frozen fixture with the same marginal five author
slots. Add a second clue view from one middle abstract sentence,
deleting every fourth word at a fixed offset. The source verifier
sees the title and the same leading extract of up to 80 words used
by the joint verifier. Keep the same open checkpoint and score
cutoff throughout.

- **Joint.** Confirm article identity and both roles for each
  candidate in one model batch.
- **Factor, same slots.** Ask only about article identity in
  the same five slots. Check the author's two role labels
  directly against the structured publisher declaration.
- **Factor, distinct sources.** Ask the same question once per
  distinct article in the five slots and share the score.
- **Factor, eligible sources.** Ask only about sources with at
  least one candidate satisfying both roles, with no call if
  that set is empty.
- **Literal.** Accept a source only when it is the unique source
  among the first six containing every clue term in its title
  or an abstract sentence. Apply the same exact role check.
  This control benefits from the way both clue types were built,
  and does not require a model or independently written paraphrases.

The rules use roles from the source of the reference labels.
Their role agreement is a structural check. The useful
comparisons are end-to-end recovery of links to the intended
source, remaining other-source acceptances, and online compute.
The two clue views reuse the same articles. Timing includes
source selection, candidate nomination, and the actual GPU
work where applicable. It excludes initialization and queuing.

### Factorization results

Slurm job 43255 ran both clue views on one RTX 5090 on ws5.
Each view contains 60 supported and 60 split requests, both
present and withheld. All five arms start with the same five
marginal candidate slots. The table reports supported requests
with the article present. Other-source accepts instead totals
the two withheld request categories in each view.

| Clue | Method | Correct links / 69 | Wrong person | Model rows / 60 requests | P95, ms | Other-source accepts when withheld |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Title fragment | Joint | 34 | 1 | 300 | 82.82 | 0 |
| Title fragment | Factor, same slots | 65 | 0 | 300 | 61.05 | 12 |
| Title fragment | Factor, distinct sources | 65 | 0 | 104 | 38.69 | 12 |
| Title fragment | Factor, eligible sources | 65 | 0 | 92 | 38.66 | 12 |
| Title fragment | Literal | 69 | 0 | 0 | 0.07 | 0 |
| Abstract fragment | Joint | 8 | 0 | 300 | 91.41 | 0 |
| Abstract fragment | Factor, same slots | 30 | 0 | 300 | 72.72 | 7 |
| Abstract fragment | Factor, distinct sources | 32 | 0 | 99 | 41.70 | 7 |
| Abstract fragment | Factor, eligible sources | 32 | 0 | 88 | 36.55 | 7 |
| Abstract fragment | Literal | 69 | 0 | 0 | 0.16 | 0 |

The eligible-source route reduces model rows from 300 to 92
for title fragments and to 88 for abstract fragments, both on
supported requests. Its covered P@5 gains over the joint
reader are +10.33 percentage points (95% article-clustered
interval +7.67 to +13.33) and +8.00 points (+5.00 to +11.33).
On present split-role requests, its same-paper wrong-person
acceptances are zero in both views, versus four and zero for
the joint reader. Reusing a source score exactly preserves
the title-fragment outputs of the factorized fixed-slot arm.
Two abstract-fragment supported queries change outputs when
the duplicate source rows are removed.

The model-based routes that separate verification accept more
wrong sources when the intended source is withheld. The
eligible-source route issues at least one other-source link
on eight of 120 withheld title requests and six of 120
withheld abstract requests. The literal control wins for both
clue types by looking for the same words in entire source
sentences. This is a strong control for term-derived clues.
The experiment supports cheaper source verification and exact
role binding within this publisher format. Independently
written descriptions, missing or conflicting role declarations,
and broader enterprise suitability remain separate tests.

[Factorization aggregates](factored.json) · [Released query fixture](public-queries.json)

## Reproduce

Restore the eligible snapshots listed in
[public-queries.json](public-queries.json) using the article DOI
and the PLOS ONE XML download endpoint. Match hashes before replay.
One directory containing the same deduplicated eligible XML articles
also works. Run with the pinned Qwen checkpoint on Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.compound_contributor_search \
  QWEN_CHECKPOINT FRESH_OUTPUT_DIRECTORY/results.json XML_DIRECTORY
```

Replay the factorization comparison using the same input XML and
the released query fixture.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false   python -m experiments.credit.factored_compound_search   benchmarks/plos-compound-contributor-search-v1/public-queries.json   QWEN_CHECKPOINT FRESH_OUTPUT_DIRECTORY/factored.json XML_DIRECTORY
```
