# Search for people and cite their declared contributions on new articles

This transfer follows the [factorial role training](../plos-factorial-role-training-v1).
It puts the frozen role decision heads behind actual topic retrieval and
makes errors in selected author and selected activity separately observable.

The catalog comes from three [PLOS Search API](https://api.plos.org/solr/search-fields/)
title searches for higher education, urban transit, and housing prices.
The capture takes at most the 25 most recent full research articles per
search in the stated 2025 to October 2026 range. The manifest pins the
candidate inventory receipts and source checksums. Exclusions use DOI,
normalized byline, and ORCID against available previous PLOS XML
snapshots. The final catalog keeps distinct normalized bylines and
observed ORCIDs. Identifiers and names are used for checks and are omitted
from the released manifest. PLOS JATS records carry CC BY 4.0 credit.

Topic paraphrases were written from public titles before model scoring.
Activity requests were written separately from the prior training and
test banks. The wording is benchmark-authored and lacks outside human
review. Each admitted focus article contributes a query for every
phrased activity held by some but not all of its authors. Topic and role
text are supplied separately to isolate verification from query parsing.

Each method retrieves four articles using a frozen dense encoder over
titles and abstracts. It then scores the full CRediT role vocabulary
with the previous checkpoints and thresholds. Compare the single-clause
and near-role cross-entropy heads with article-level versus
author-level binding. The article-level control broadcasts the article's
highest scoring passing declaration to every coauthor. Author-level
binding cites each person's own highest scoring passing declaration.
The original frozen reranker with author-level binding is a stronger
control. The methods return up to five cited people in topic rank order.
Article-level binding deliberately implements the coauthor leakage
failure mode. It is a diagnostic, not a competitive expert baseline.

The main metric counts slots citing the focus DOI, requested activity
and correct owner. Other DOIs are reported as unjudged for topic fit.
Wrong-role counts compare the cited declaration with the requested role.
Wrong-owner counts compare the recommended author with the owner of
that exact declaration. Both can apply to the same result. Author-level
binding ensures correct citation ownership by construction, so this
counts structural leakage rather than a learned ownership classifier.
It does not establish ability or contribution strength beyond the
publisher's declarations.

## Results

Slurm job `43623` ran on one RTX 5090. Of 42 retrieved publisher
records, 36 entered the catalog after screening against 482 distinct
DOIs in the available prior archives. Fourteen authored topic
paraphrases had admitted focus records. Twelve had at least one
contrastive role and supplied 59 queries. Every focus paper entered the
top-four list. Each row has 295 available recommendation slots.

| Scorer | Binding | Supported | Wrong owner | Wrong role | Other DOI | Emitted | P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single clause | Article | 42 | 168 | 46 | 85 | 254 | 14.76 ms |
| Single clause | Author | 61 | 0 | 47 | 151 | 226 | 14.77 ms |
| Near-role swaps | Article | 34 | 168 | 81 | 85 | 254 | 14.70 ms |
| Near-role swaps | Author | 55 | 0 | 73 | 148 | 226 | 14.71 ms |
| Reranker | Author | 32 | 0 | 64 | 99 | 155 | 21.61 ms |

**The synthesis gain does not transfer at the locked operating point.**
Holding exact author binding fixed, the near-role head gives 55
supported slots against 61 for the single-clause head. The paired
difference is −2.03 percentage points of available slots (descriptive
95% interval −2.95 to −1.11, resampling focus papers). Wrong-role slots
increase by 26 (+8.81 points, +5.00 to +13.08). Both heads fill 226 slots.
The earlier experiment compared the requested activity to a single
nearby alternative. This replay applies the same cutoffs to every
declared role, with different request wording. It does not attribute
the reversal to either change individually.

With the single-clause head, attaching each declaration to its owner
adds 19 supported slots (+6.44 points, +3.24 to +9.82) relative to the
coauthor-broadcast control. The latter attaches the requested role to
the wrong person 138 times. Binding by individual removes such
attachments by construction. More outputs then come from different
papers, whose topical fit is unjudged. The reranker emits fewer
results than either head and still makes 64 wrong-role attachments.
Its P95 includes a batch of 14 role comparisons. Head timings include
two encodings (topic and role), the same local dense search, and
citation lookup in already parsed records. Latency excludes index
construction, independent evidence review, persistent I/O, and service
queueing.

The single-clause head is the stronger starting point for this task.
On a different development corpus, compare matched-coverage
calibration across the full role vocabulary, including a top-one
margin rule and a literal role-name baseline. Freeze the rule and
repeat on a third corpus before changing the default. This snapshot
has one publisher and eight shared activity phrasings. It supports
neither claims about unconstrained topic retrieval nor enterprise
first-query precision.

[Aggregate results](results.json)

## Reproduce

Download the published source records using the acquisition command or
retrieve the pinned DOIs from the [PLOS JATS endpoint](https://api.plos.org/text-and-data-mining.html).
The replay verifies every XML hash. The older snapshots listed in the
manifest are required only to reproduce the exclusions. Use the same
checkpoints as the factorial training study.

```sh
PYTHONPATH=src:. python -m experiments.credit.fresh_role_search download XML_SNAPSHOT

PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.fresh_role_search run XML_SNAPSHOT \
    benchmarks/plos-fresh-role-search-v1/manifest.json \
    benchmarks/plos-fresh-role-search-v1/wording.json \
    benchmarks/plos-credit-edits-v1/role-training.json \
    benchmarks/plos-factorial-role-training-v1/results.json \
    QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS.json
```
