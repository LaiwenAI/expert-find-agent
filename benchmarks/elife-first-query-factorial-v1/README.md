# Cross early role nomination with global evidence rescue

Use the pinned [eLife publisher projection](../elife-role-nomination-v1)
and the same construction of two-role queries from headings and author
keywords. The benchmark holds the document universe and relevance
reader constant and varies two policy decisions independently.

| Arm | First read | If fewer than five people have passing evidence |
| --- | --- | --- |
| Topic, local | Top topical sources | Read unseen sources within the original topical pool |
| Person, local | First role-compatible sources in the topical order | Read unseen sources within the original topical pool |
| Topic, global | Top topical sources | Read unseen sources across the directory |
| Person, global | First role-compatible sources in the topical order | Read unseen sources across the directory |

An article is role-compatible when one author has both roles or its
declaration is unavailable in the index. Both rescue paths use this
compatibility check and the same two-source cap. The final reader uses
the source's actual declaration to verify both roles for the same
person. Keep the five best unique people by topical verifier score
after combining both stages. Each source cited in the result comes from
one of the counted reads. The initial read cap and local-pool size
are varied jointly as 2/4, 4/8 and 6/12. Actual reads can differ.
Run both the sparse and full available declaration masks.

The shared ranking combines dense BGE-small-en-v1.5 with IDF-weighted
term overlap using the repository's reciprocal rank fusion. The
shared quantized MiniLM L12 cross-encoder uses titles and abstracts.
Assign exact topic strings to development using a fixed hash. Choose
a single strict acceptance cutoff at the 90th percentile of off-label
scores in the development top eight. Test on the remaining topics.
Publisher labels stay outside candidate retrieval and the reader.
Overlapping documents and author identities may appear across topics.

Report first-stage and final tag-backed slots, off-label and total
issuance, and article reads. Use a paired topic-cluster bootstrap for
the fixed-budget contrasts and the factorial interaction. Labels can
be incomplete and off-label citations are unresolved by this benchmark.
Cross-encoder scores are cached across arms for the paired replay.
Read counts represent each arm's requested work; the replay does
not measure service latency.

## Results

The eLife projection has 322 articles. The declaration index exposes
144 complete declarations under the sparse mask and 280 under the
full mask. Ten exact topic strings calibrate the cutoff on 50
off-label pairs. The test contains 36 keyword requests on 18 topics
and 24 heading requests on 12 topics.

The main diagnostic below uses four initial reads, a local pool
of eight and the full available declaration mask. The test keyword
panel has 180 fixed recommendation slots.

| Arm | Keyword first → final tag-backed slots | Keyword off-label slots | Keyword article reads | Heading first → final tag-backed slots | Heading article reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Topic, local | 46 → 49 | 21 | 195 | 4 → 5 | 130 |
| Person, local | 51 → 51 | 22 | 166 | 5 → 5 | 122 |
| Topic, global | 46 → 51 | 21 | 202 | 4 → 5 | 144 |
| Person, global | 51 → 52 | 23 | 200 | 5 → 5 | 144 |

Early nomination with local rescue uses 14.9% fewer source reads on
keyword requests. Its end-of-request gain is two tag-backed slots,
with a topic-paired interval of 0 to 3.33 percentage points of tag-backed
P@5. Expanding rescue globally after early nomination adds one
tag-backed slot and one off-label slot for 34 additional reads.
The estimated interaction is −0.56 points, with a 95% interval of
−1.67 to 0.00. The heading panel shows no final gain. Results across
the three budgets and two masks do not establish a positive
interaction between early nomination and global rescue.

The smaller realized local workload may reflect exhaustion of
compatible sources as well as the common stopping rule. The comparison
holds the maximum reading budget fixed; it provides a reason to
compare quality against both the cap and realized work. Qualification
truth, out-of-index expert identities, reviewable citation support
and production latency require separate evaluation.

[Complete aggregates and paired intervals](results.json)

## Reproduce

Rebuild or restore the projection using
[the publisher benchmark](../elife-role-nomination-v1).
Supply the local BGE-small-en-v1.5 optimized quantized ONNX model
directory and the FlashRank cache containing MiniLM L12. Results pin
the model file hashes and dependency versions.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.role_first_query_factorial PROJECTION \
  benchmarks/elife-role-nomination-v1/manifest.json BGE_MODEL_DIRECTORY \
  FLASHRANK_CACHE_DIRECTORY NEW_RESULTS
```
