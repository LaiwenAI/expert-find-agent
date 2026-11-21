# Time resident author and author-role fallback over loopback

The [eLife](../elife-directed-source-rescue-v1) and
[PLOS](../plos-prospective-proof-split-v1) replays found that an
author-role index visits fewer entries than an author-only index.
They reported work counts. This study compares equal outputs
under actual request execution and includes an exact-rank
optimization for the shared retrieval path.

## Design

Use the resident 2,011-paper eLife and 414-paper PLOS catalogs.
Each of their 35 and 12 held groups contributes one supported
request and its paired role contrast. Each can use an excerpt
or keyword clue and either full availability or removal of
one of the two designated papers. This produces 420 distinct
eLife and 144 PLOS requests. The paper withdrawal is a test
intervention, not an input available to external users.

For each request, a single GPU encoding batch covers both
source clues and both role phrases. Execute the frozen-role
membership check, rank the catalog if the role confidence
passes, and perform the same learned fallback verification
only when the shallow linked proof cannot issue. Both index
arms use the recorded eLife calibration. Check full response
equality on every timed request, including sources, roles and
the set of authors. Recompute batch encodings for each arm
and request. No query result or request vector cache is used.

Use a 2 × 2 comparison:

| Retrieval | Fallback index |
| --- | --- |
| Original exact Python BM25/dense rank fusion | Author or author-role postings |
| Vectorized exact BM25/dense rank fusion | Author or author-role postings |

The vectorized path accumulates weights from precomputed term
postings instead of scanning all document dictionaries per
query. It resolves near ties with the original Python sums,
preserves the stable document-order ties, and retains the same
dense scoring and reciprocal-rank fusion. Before timing, check
identical dense and fused ranks and all dense scores on one
complete paper group per publisher. Its reported extra memory
counts the resident numeric buffers and excludes Python
container overhead.

A single warmed loopback HTTP worker serves requests serially.
The client timer covers the request/response exchange and
response JSON parsing. Source ranking and fallback substeps
are also timed. Rotate arm order per request and traversal
order per Slurm job. Each job makes two passes over the
requests. Disk loading, model initialization and index
construction are outside timings. Concurrency, corpus growth,
and serving across a network are outside the experiment.

## Results

| Catalog | Rank implementation | Client P95 (ms), author-role | Rank P95 (ms), author-role | Client P95 on fallback requests (ms), author-role |
| --- | --- | ---: | ---: | ---: |
| eLife | Original | 48.34–49.03 | 31.56–31.66 | 50.11–50.18 |
| eLife | Vectorized | 20.48–20.61 | 3.53–3.61 | 21.87–21.88 |
| PLOS | Original | 21.63–21.72 | 5.33 | 21.74–21.83 |
| PLOS | Vectorized | 17.92–18.14 | 0.66–0.70 | 17.09–17.22 |

Ranges span the two runs. Client P95 drops by about 58% on eLife
and 16–17% on PLOS when the author-role route is held fixed.
The eLife P95 improvement is also visible in fallback requests.
Overall median stays near 16.5–16.8 ms because most requests do
not reach source ranking. Rank percentiles include zero-work
abstentions. The posting arrays add 1.68 MiB on eLife and 0.39
MiB on PLOS before container overhead.

In either rank implementation, conditioning the index on role
cuts total posting visits from 2,624 to 628 on eLife and from
224 to 86 on PLOS per run. Relative to author-only fallback,
this yields no consistent client P95 reduction in either catalog.
The role-aware fallback work itself takes less than 0.2 ms at
P95. The large measured end-to-end savings come from exact
ranking, which is shared infrastructure for comparing search
policies.

The four paths return identical source, author and role payloads
on all 564 requests, across both passes and both GPU jobs. Per
job each arm observes 48 correct and four wrong outputs in
840 eLife timings and eight correct and four wrong outputs in
288 PLOS timings. Counts include both passes, which are repeated
observations. The timed workload samples one binding and its
contrast per group and does not estimate population quality.

Both RTX 4090 jobs in array 47213 on ws6 verify the same proof
signature and work counts. Keep latency ranges rather than
pooling across hardware runs. The results concern warmed, local
serial service and do not establish throughput or a 52,000-source
enterprise speedup. [Aggregate results](results.json).

## Reproduce

Restore the data and models from the
[prospective PLOS split](../plos-prospective-proof-split-v1).
On a Slurm GPU allocation, use fresh output paths:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m experiments.credit.resident_directed_source_timing \
 ELIFE_CONFIRMATION ELIFE_EXTENDED \
 benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
 benchmarks/elife-held-source-acceptance-v1/panel.json \
 benchmarks/role-first-proof-decoding-v1/results.json \
 benchmarks/elife-directed-source-rescue-v1/results.json \
 benchmarks/plos-open-directed-transfer-v1/manifest.json BASE_PLOS_PROJECTION \
 benchmarks/plos-prospective-proof-split-v1/manifest.json NEW_PLOS_XML_DIRECTORY \
 NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json QWEN_EMBEDDING NEW_RESULTS
```

Repeat with `--reverse` and a separate output path.
