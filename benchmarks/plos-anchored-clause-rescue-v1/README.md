# Attribute a missing contribution after a shared first witness

This live-model experiment complements the [TREC rescue audit](../trec-citation-rescue-audit-v1), where target-person judgments were supplied by an oracle. Use original PLOS ONE publisher contribution notes. On each paper, two candidate authors already have a verified initial duty in common. Only one has the second requested duty. Remove the first duty's clause from the candidate evidence, leaving the original text of all other parsed clauses. The shared initial duty controls for author and topical relevance; it is supplied by design, not predicted. A clean completion requires returning an original clause that both states the second duty and names the correct author, while withholding evidence for the partial-match author.

Use the frozen Qwen3-Reranker-0.6B checkpoint, one author pair per paper, and two existing disjoint evaluation cohorts. The [publisher transfer panel](../plos-independent-cascade-v1) additionally excludes normalized byline names found in the earlier capture. The snapshot parser handles the rigid `Activity: INITIALS.` format and retains selected cases only if both arms of the contrast can be recovered. Use the earlier phrasing for calibration and a different fixed paraphrase for evaluation. The same wording banks recur across papers.

| Method | Selection and verification |
| --- | --- |
| Top two, topical | Rank two clauses by action-head character n-grams, run a live role-only cross-encoder, and use its best clause for both authors |
| Top two, personal | Score those two clauses for each author with explicit person-specific instructions |
| All, personal | Score every remaining clause for each author using the same person-specific instructions |
| Top one, then scope | Use the first lexical candidate and role-only verdict, returning it only when that exact clause names the author |
| Top two, scope first | Retrieve the same two topical candidates, use the original local author field to limit each author's choices, and apply the role-only cross-encoder to each unique remaining clause |
| Scope and characters | Use the same scoped candidate set and the character n-gram score without a cross-encoder |

Calibration chooses the maximum number of valid attachments under a cap of 10% of calibration papers with at least one invalid attachment. Among tied calibrations, prefer fewer invalid attachments and then the higher threshold. No threshold is changed on either evaluation cohort. Count accepted partial-match authors separately from wrong-duty evidence for the author who meets both duties. Forced candidate misses count cases where the best proposed clause is wrong before thresholding. P95 includes lexical matching, binding, model scoring, and selection. It excludes acquisition of the first witness, source retrieval, publishing notes and service overhead. Report paired descriptive intervals clustered by normalized author names on the transfer cohort and by paper on the earlier test.

## Results

Slurm jobs 42845 and 42849 ran on one RTX 5090. The second run adds the no-model control and reproduces all original decision counts. Use 30 calibration papers, 20 held-out papers and 45 additional transfer papers. Each table cell for outcomes is valid clause attachments / invalid attachments.

| Method | Earlier test outcomes | Transfer outcomes | Transfer scorer pairs | Transfer component P95 |
| --- | ---: | ---: | ---: | ---: |
| Top two, topical | 0 / 0 | 0 / 0 | 90 | 8.64 ms |
| Top two, personal | 5 / 0 | 7 / 0 | 180 | 8.84 ms |
| All, personal | 3 / 0 | 7 / 0 | 360 | 12.59 ms |
| Top one, then scope | 13 / 2 | 30 / 9 | 45 | 7.46 ms |
| Top two, scope first | 11 / 1 | 22 / 5 | 82 | 8.65 ms |
| Scope and characters | 0 / 0 | 0 / 0 | 0 | 0.08 ms |

The all-clause personal scorer often proposes the correct clause but abstains after frozen calibration. It has only two forced-choice misses on the transfer panel, but seven accepted valid clauses at the calibrated threshold. The direct scope-first route returns 22 valid clauses, 15 more than scoring all clause–author pairs, with a descriptive author-cluster interval of +20.0 to +46.8 percentage points. It also returns five unsuitable clauses, two for partial-match authors and three for the correct author in the wrong duty. The five invalid clauses occur on three of 45 transfer papers, within the 10% calibration target when counted per paper. Top-one checking retrieves more valid clauses but returns nine invalid clauses on six papers, exceeding that target. These are descriptive counts, and the calibration target is not a population guarantee. The role-only and character-only routes abstain throughout under their frozen thresholds.

These results identify two separable decisions in a live scorer: bind the evidence span to the intended person, then decide that the span describes the requested duty. Exact structural credit in these notes prevents another author's clause from serving as direct evidence. It does not validate role semantics. Report both outcomes and abstentions when transferring the route to less structured text.

[Aggregate results](results.json)

## Reproduce

Restore the old and transfer publisher snapshots using their pinned receipts as documented in the [legacy transfer experiment](../plos-independent-cascade-v1), and use the verified open reranker checkpoint in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.credit.anchored_clause_rescue \
  OLD_PUBLISHER_SNAPSHOT TRANSFER_PUBLISHER_SNAPSHOT RERANKER_CHECKPOINT NEW_RESULTS
```
