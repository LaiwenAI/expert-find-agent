# Inspect the paper pointers already selected by enterprise search

Follow the [paper-locator audit](../enterprise-paper-identity-audit-v1).
Fix the 55 development requests that withheld a recommendation and use
only the 100 distinct paper records already selected in their original
evidence plans. This experiment measures how far those pointers reach
into a publisher contribution statement.

Limit capture to saved anonymous HTTPS PDF pointers on specified
publisher and journal hosts. The original URL and allowed redirects are
the only outbound inputs. Keep all captures and the request-to-paper map in the
restricted workspace. Byte limits and transport failures count as
unresolved. Report availability within this controlled subset of pointers.

For each PDF, compare the normalized first two pages with the original
catalog title and the saved bibliographic title. Treat either match as
a candidate document. Check local JATS snapshots for exact title
matches separately. A normalized name before the abstract is a byline
hint. Detect contribution headings and examine the following bounded
text window, ending at the next recognized disclosure or references
heading. First test Poppler's layout order. If that yields no
contribution heading, test raw reading order because a neighboring
column can hide the heading. The PLOS ONE positive control checks that
the title, contribution heading, and a named contributor are found.
The [CRediT role definitions](https://credit.niso.org/contributor-roles/)
describe the distinction between contributor roles and authorship.

## Results

The table uses original paper records and requests. Multiple records
or requests can refer to one captured PDF. The 29 attempted record
pointers produce 28 distinct initial URLs, excluding the positive
control.

| Stage | Paper records | Requests |
| --- | ---: | ---: |
| Initial abstention with a selected paper | 100 | 39 |
| Saved PDF pointer | 44 | 30 |
| Anonymous publisher pointer eligible for this replay | 29 | 20 |
| Readable PDF with a matching original or bibliographic title | 11 | 10 |
| Readable PDF matching the original catalog title | 8 | 7 |
| Candidate name found before the abstract | 6 | 6 |
| Candidate name found in a contribution section | 1 | 1 |
| Cached publisher JATS with matching title | 0 | 0 |

For these pointers, the layout-only extractor missed the one detected
contribution statement. Raw extraction recovered it. That PDF also
matches the original catalog title. A name-bearing clause includes
editing and guidance terms. The audit treats this as a candidate
person-role statement. Identity, the scope of the named actions, and
fit to the original requirement remain unjudged. The two extraction
modes detected one section across the 11 captured PDFs.

Eight record pointers redirected outside the configured anonymous
capture scope, six encountered network errors and four returned HTTP
errors. The main observed losses are pointer coverage, transport and
the narrow coverage of named contribution statements. Quality and
end-to-end impact require independent inspection of the evidence and
comparison against the original citations.

## Reproduce the offline audit

Store snapshots in the restricted environment. Use the frozen prior
metadata, raw publication inventory, existing capture cache and local
JATS folders. Two replay runs produced identical aggregates. Poppler
`pdftotext` and pypinyin are required. The public repo contains only
code and aggregate outcomes.

```sh
python -m experiments.retrieval.enterprise_fulltext_reach \
  PRIVATE_DATASET PRIVATE_METADATA PRIVATE_RAW_PAPERS PRIVATE_TRACE \
  PRIVATE_CAPTURE_CACHE NEW_AGGREGATES \
  --jats PRIVATE_JATS_CACHE_A PRIVATE_JATS_CACHE_B --offline
```
