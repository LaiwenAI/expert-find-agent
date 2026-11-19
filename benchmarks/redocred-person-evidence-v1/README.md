# Verify person-specific roles within a shared source

Use the human-revised [Re-DocRED](https://aclanthology.org/2022.emnlp-main.580/)
relation inventory to contrast two people credited on the same work in
the same document. For each document, work and target role, keep one
deterministically sampled pair. One person has the target role in the
inventory. The other appears with a different labeled role on that work
and lacks the target label. Keep the work, role, source and prompt template
fixed. This is a constructed control against misattribution within the
given document. Missing labels can persist even after reannotation.

Use the official development and test splits and verify that document
titles do not overlap. Query names use the longest annotated mention by
token count, character count and text. The selectors use raw sentences,
the query names and a fixed role cue list. Entity mention positions and
human evidence locations are withheld from selection. Use available
sentence evidence annotations to measure how often a window contains
all annotated support for the positive relation.

| Window | Selection policy |
| --- | --- |
| Lexical two | Rank sentences by summed coverage of person, work and role cues |
| Local two | Use the highest lexical scorer plus its strongest nearest neighbor |
| Joint two | Maximize coverage across both sentences with small bonuses for adjacency and a shared non-query capitalized token |
| Shared two | Use the same two sentences for both people, choosing the window by summed coverage of both names, the work and the role, and breaking ties in favor of the less-covered person |
| Full | Supply the entire document to both people |

All arms use the same frozen Qwen3-Reranker-0.6B. Name and work matching
favor a full contiguous mention and otherwise use inverse-frequency term
coverage. Role cues use the same term weighting. The shared window is an
additional control designed after the first run. Its selector is symmetric
with respect to the two names, although it assumes both names are known.
Rotate the arm order and counterbalance the order of names in each GPU
batch. Exclude a pair if any prompt is truncated. At each arm's threshold,
calibrate to at most eight rival acceptances among the 172 development
pairs, using strict score exceedance. Report test rates without adjusting
the thresholds.

## Results

Slurm job 42966 used an RTX 5090 and reproduced the original four arms'
decisions from job 42963. Development contains 172 pairs from 49 documents.
Test contains 163 pairs from 41 documents. Eighty test positives have
nonempty sentence evidence annotations. No prompts were truncated.

| Window | Correct ordering / 163 | Both decisions correct / 163 | Accepted supported / 163 | Accepted rival / 163 | All annotated evidence retained / 80 | Paired P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical two | 118 | 28 | 28 | 6 | 66 | 9.06 ms |
| Local two | 113 | 19 | 20 | 5 | 38 | 9.07 ms |
| Joint two | 112 | 26 | 26 | 7 | 61 | 9.13 ms |
| Shared two | 113 | 17 | 18 | 6 | 50 | 8.98 ms |
| Full | 113 | 27 | 35 | 10 | 80 | 12.35 ms |

Correct ordering requires the credited person to score strictly higher.
Both decisions correct requires accepting the credited person and
rejecting the rival at the frozen threshold. The lexical window reduces
paired P95 by 26.6% relative to full text, while the paired 95% interval for
full text minus lexical window on both-correct decisions spans −5.0 to
+3.0 percentage points. The shared two-sentence arm has 11 fewer
both-correct pairs than the lexical window. The paired difference is
−6.75 points with a descriptive interval of −12.42 to −1.81 points.
The shared window also retains fewer of the provided evidence spans.
This comparison cannot distinguish information loss from any effect of
showing the same context to both people. Neither pair selector improves
upon lexical selection on this panel.

P95 covers query-dependent selection, prompt preparation and scoring two
people together. Static sentence profiles, loading and source discovery
are outside this timing. Intervals resample whole source documents.
Only 28 of 163 pairs pass both decisions even for the strongest held-out
operating point. These role labels from a supplied Wikipedia document
cannot measure expert suitability or full-corpus search. Use the short
lexical window as an inexpensive comparison and continue to distinguish
missing evidence from errors of attribution when evidence is available.

[Aggregate results](results.json)

## Reproduce

Download `data/dev_revised.json` and `data/test_revised.json` from the
[pinned official snapshot](https://github.com/tonytan48/Re-DocRED/tree/ccfb54f5ddf5836027c87badda10f6dfc56efaac/data)
into `REDOCRED_DIRECTORY`. The runner checks both file digests before
scoring. Use `Qwen/Qwen3-Reranker-0.6B` at revision
`e61197ed45024b0ed8a2d74b80b4d909f1255473` in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.person_bound_evidence \
    REDOCRED_DIRECTORY QWEN_RERANKER NEW_RESULTS.json
```
