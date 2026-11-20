# Verify public full-text leads in the CoF review packets

Audit the 29 distinct public links attached to 31 cases by
the [work-ID provenance lookup](../cof-review-source-id-provenance-v1).
Keep the original CoF records, blind packets, and registry
snapshots fixed. This audit does not attempt institutional
subscription access or extrapolate from the 31 linked cases
to all 177 packet cases.

Try each recorded HTTPS URL using only public access.
Follow redirects to the declared publisher and repository
hosts. An initial probe was used to add three observed and
verified redirects for IJCAI, AAAI, and the NTU repository
CDN. On a landing page, follow only the page's own
`citation_pdf_url`. Limit response size and discovery
steps. HTTP failures, unmanaged redirects and landing pages
without PDF metadata remain unresolved.

If an unversioned arXiv PDF is readable but the title
does not match the indexed record, try its first two
archived versions and stop after the first title match.
Keep the original access result and count the extra
attempts. This fallback was selected after observing
these mismatches. It is a targeted replay, not evidence
of general version-resolution recall.

Extract layout-preserved text with Poppler. Require at
least 2,000 normalized text characters and an exact
normalized title match in the first three pages before
counting the PDF as matched to a case. Separately flag
standalone author-contribution headings and equal-contribution
phrases and record their page numbers locally. Heading and
phrase scans do not establish author roles or rule out
contribution notes elsewhere in the original. Accessible
HTML and PDFs without extractable text are outside the
matched-PDF count. The audit keeps no copies of full-text
PDFs or extracts in the release.

## Results

The final replay recovered 24 title-matched PDFs directly
and two more through version lookup. The 26 matched PDFs
cover 28 of the 31 linked citation cases. The remaining
three URLs returned HTTP 403, HTTP 502, or a landing page
without PDF metadata. All cases in both title-mismatch
groups matched the first archived version. The two PDF
lookups were the only version attempts in the final replay.

Both mismatch groups are confirmed changes in arXiv
version history. [Graph embedding v1](https://arxiv.org/abs/1710.06879v1)
matches the indexed title. The indexed
[FLO v1](https://arxiv.org/abs/1911.04706v1) has two authors,
while the newer [FLAML v3](https://arxiv.org/abs/1911.04706v3)
has four authors and a different title under the same
unversioned arXiv DOI. The examples support pinning the
exact visible version before comparing authorship or citing
an excerpt.

None of the 28 matched cases showed either of the two
scanned contribution cues. This result is limited to the
specified PDF text checks and OA subset. The audit did not
label individual contributions or assess candidate matches.
[Aggregate results](results.json)

## Reproduce

Use the same registry snapshot and private packet as the
[source-ID provenance benchmark](../cof-review-source-id-provenance-v1).
Run with Poppler available and choose new output paths.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_review_fulltext_access \
  COF_CORPUS_ROOT BLIND_PACKET LOCAL_METADATA_CACHE \
  PRIVATE_ACCESS_JSON NEW_RESULTS
```

Access status depends on the public endpoints and the
network at replay. Keep the URL-to-case results local.
