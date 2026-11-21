# Transfer source–activity assignment to new papers and wording

Replay the [joint offset and agreement methods](../joint-offset-budgeted-source-role-v1)
on the earlier [eLife publisher projection](../elife-role-nomination-v1)
and the [engineering panel](../engineering-cross-source-search-v1).
Neither contributed training requests to these taggers. Drop earlier
eLife articles whose paper ID or recorded author key occurs in the
training catalog. Verify the engineering panel against its own frozen
manifest. Engineering identity keys are independently salted, so
cross-publisher author disjointness is unverified. Both cohorts were
used in prior work on the underlying search stack.

Choose one positive and swapped negative per article pair by fixed
family hash. Re-render all cases in three new English templates
that place activities first, use explicit second–first references, or
use indirect last–first references. Keyword clues and held NISO
activity fragments follow the previous benchmark. Both target papers
remain available. No template, threshold or weight is refit to these
cases. Refit the two contextual heads from the previous exact training
recipe, sharing a single frozen embedding pass at inference.

Compare supplied spans, joint decoding, and exact agreement between
the two heads. A separate ordinal route takes the joint decoder's
original substrings and reconnects activities if two separate
clauses each contain one recognized antecedent cue before a unique
activity. It abstains when these checks fail. An optional composition
uses exact agreement where ordinal repair abstains. The ordinal
rule leaves the topic and activity span boundaries intact. It
covers direct first, second, former and latter markers. It does
not interpret the indirect last–first paraphrase. The marker forms
are controlled and the two direct cues are deliberate.

Use the same frozen one-call search path and count exact acquisition
of all bound substrings, pure role swaps, strict proofs, adverse results,
parser time and warm serial P95. The supplied-span route isolates the
search headroom. The timed scope omits HTTP, queueing and index build.

## Results

Both jobs of Slurm array 46119 completed in 27 s on ws3 RTX 5090.
Earlier eLife admits 280 papers and ten pairs, with no overlap
removed by the recorded-key screen. Engineering has 53 papers and
four pairs. Entries report strict successes / adverse issuances
across both seeds. Request volume is twice the positive denominator.

| Panel | Route | Activity first | Explicit reverse | Indirect reverse |
| --- | --- | ---: | ---: | ---: |
| Earlier eLife (10) | Supplied spans | 2 / 0 | 2 / 0 | 2 / 0 |
| Earlier eLife (10) | Joint | 2 / 0 | 0–1 / 2 | 0–1 / 0–2 |
| Earlier eLife (10) | Agreement | 0 / 0 | 0 / 0 | 0 / 0 |
| Earlier eLife (10) | Ordinal then agreement | 0 / 0 | 2–3 / 0 | 0 / 0 |
| Engineering (4) | Supplied spans | 2 / 0 | 2 / 0 | 2 / 0 |
| Engineering (4) | Joint | 2 / 0 | 0 / 1 | 0 / 1–2 |
| Engineering (4) | Agreement | 1 / 0 | 0 / 0 | 0 / 0 |
| Engineering (4) | Ordinal then agreement | 1 / 0 | 2 / 0 | 0 / 0 |

For explicit reverse, ordinal repair activated on 16–18 of 20
earlier eLife requests and on seven of eight engineering requests.
It changed 13–18 and five to seven bindings, respectively. Across
the two panels, joint decoding issued three adverse results in
both runs. The ordinal route returned four to five strict proofs
with zero adverse issuances, and the supplied spans returned four.
Neither parser recovered an exact four-fragment assignment on that
form. Search sometimes tolerates fragment-boundary errors. Supplied
spans are a controlled reference rather than a mathematical upper
bound on the number returned by this search stack.

The ordinal check never fired on the indirect reverse form. Its
agreement fallback returned no proofs there. Ungated joint decoding
produced two to three adverse issuances across the panels. Across
seeds, P95 of the ordinal-with-agreement route on explicit reverse
was 31.4–32.7 ms on earlier eLife and 27.5–28.2 ms on engineering.

Even with supplied spans, only four of fourteen distinct positive
requests yielded strict proofs. Further evaluation should separate
source nomination from role verification under supplied input before
attributing the shortfall to parsing. This panel contains published
declarations and synthetic requests. Its limited issuance counts
offer weak evidence about adverse risk in deployment.
[Full results](results.json)

## Reproduce

Restore both eLife projections and their respective manifests and
the engineering panel and manifest. Use the artifacts pinned by
[the offset tagger benchmark](../offset-budgeted-source-role-v1#reproduce)
for the frozen search models and thresholds. Run inside a Slurm
CUDA allocation and repeat with `--seed 29 --reverse` and a new output.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.transfer_source_role_resolution \
  ELIFE_PARSER_DEVELOPMENT ELIFE_DEVELOPMENT_MANIFEST \
  ELIFE_EARLIER_PROJECTION ELIFE_EARLIER_MANIFEST ENGINEERING_PANEL ENGINEERING_MANIFEST \
  NISO_EXAMPLES ROLE_TRAINING SCORE_BANDS SOURCE_FROZEN ROLE_FROZEN \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_RESULTS --seed 17
```
