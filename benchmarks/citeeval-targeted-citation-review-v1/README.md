# Review misleading citations at a fixed target-check budget

The prior [CiteEval addition benchmark](../citeeval-evidence-dispatch-v1)
measured whether to attach another passage. This counterpart asks whether
to remove one misleading citation while preserving other citations attached
to the same claim. It uses independent human edits from the
[official CiteEval release](https://aclanthology.org/2025.acl-long.1574/).

Retain statements with two to four cited passages. A positive has at least
two retrieval-context annotators voting to delete the same cited entry as
misleading. A no-delete control has three retrieval-context annotators and
no misleading-deletion vote. Other edit categories can appear in this
control. Cross-check all claim text and citation IDs against the public
metric inputs. As in the addition experiment, exclude held-out responses
with any normalized query in development. Select at most one citation to
delete. The target label refers to the consensus edit, not to whether
another citation might also deserve removal.

All methods use the same prefix of up to 170 tokenizer tokens per citation.
The cap applies before comparison, including to the lexical control. Exclude
any case where a complete composed prompt would still be clipped. Candidate
snippets use the original title and retrieved text. The benchmark does not
allocate additional reads beyond the delivered citation package.

| Policy | Decision |
| --- | --- |
| Lexical | Nominate the citation with the weakest claim-term overlap |
| Standalone two | Score the two weakest citations individually for useful, nonmisleading support |
| Context two | Check the same two citations, with the highest-overlap other citation shown as context in each prompt |
| Omit one | Compare the package's support score before and after removing the weakest lexical citation |
| Standalone all | Score every existing citation individually |
| Selective check | Learn lexical abstention and acceptance bounds, and use the standalone two-citation check in between |

The neural methods use the same frozen Qwen3-Reranker-0.6B. Standalone and
contextual policies nominate the lowest-support citation. The omission
policy nominates the weakest lexical citation and scores the change in
package support after its removal. Freeze per-method thresholds on
development at at most 10% of observed no-delete cases. Equal target counts
do not imply equal text volume: contextual and omission prompts include
additional excerpts. For the selective check, choose lexical bounds and a
fallback score on development only, with the same no-delete ceiling and
a cap of 49 neural calls per 99 statements. Break development ties by
fewer incorrect removals and then fewer calls. The selected rule never
automatically removes a citation and checks only statements whose lexical
risk exceeds 0.545. The test run executes only the selected fallback
branches and checks their scores against the matched standalone replay.
The cascade was designed after inspecting the first baseline run and is
exploratory on this reused test panel.

## Results

Slurm jobs 42955 and 42959 ran on an RTX 5090 and reproduced all original
method decisions, with the second run also evaluating the selective check.
The test has 73 positive and 232 unanimous no-delete cases in 115 query
groups. No cases exceed the final prompt cap after static snippet clipping.

| Policy | Correct removals / 73 | Other removals on positives | Removals / 232 no-delete | Model calls / 305 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical | 24 | 4 | 15 | 0 | 0.16 ms |
| Standalone two | 25 | 4 | 15 | 305 | 9.16 ms |
| Context two | 17 | 7 | 27 | 305 | 12.73 ms |
| Omit one | 6 | 1 | 9 | 305 | 15.59 ms |
| Standalone all | 25 | 4 | 16 | 305 | 11.95 ms |
| Selective check | 23 | 3 | 13 | 155 | 9.04 ms |

Using individual scores from two priority citations reproduces the correct
removal count of scoring every citation, with a lower P95. The context-pair
route reads the same two target citations but exposes about twice as many
document characters to the model, removes eight fewer consensus citations
and makes twelve more deletions on no-delete cases. Omission-based scores
yield fewer correct removals at their calibrated threshold. The selective
check removes two fewer consensus citations than always reviewing two.
It also makes three fewer removals outside consensus, and cuts neural
calls by 49.2%. Tail latency is almost unchanged because the neural branch
still runs in about half of all cases.

The lexical control already selects 24 consensus deletions with 15 no-delete
removals and zero neural calls. The paired 95% interval for its correct
removal-rate difference versus two neural checks is −12.82 to +10.39
percentage points. This comparison supports retaining lexical signals as
a strong baseline. It supplies limited evidence for spending neural
verification on this curated edit action, even with targeted dispatch.

Timing covers lexical selection, query-dependent prompt preparation and
neural checking. Static excerpt preparation is outside the timed interval.
Intervals resample normalized queries. This evaluates citation editing
inside a provided package, with no source attribution to an expert and no
full-corpus retrieval.

[Aggregate results](results.json)

## Reproduce

Use the four frozen release files from the
[addition benchmark](../citeeval-evidence-dispatch-v1) and the pinned open
reranker. Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.citeeval.targeted_citation_review \
  CITEEVAL_DIRECTORY QWEN_RERANKER NEW_RESULTS.json
```
