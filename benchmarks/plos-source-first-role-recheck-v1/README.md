# Recheck uncertain roles after independent source nomination

The [anchor audit](../plos-role-anchor-audit-v1) found that 26
of 36 natural positives stopped at the requested-role margin.
Among the remaining ten, the original keyword pipeline found
all ten exact proofs. This study targets the missing decision:
whether to run one stronger role check when the first query
has already found independently plausible papers.

## Protocol

Use the same 414-paper PLOS corpus with 12 matching-rule
calibration groups and 12 held groups. Run both excerpt and
keyword clues on the intact catalog and after withdrawing
each specified paper separately. The original shallow and
author-role fallback outputs are preserved. Revisit only
requests that are still unissued and have a ridge confidence
gap at or below the frozen cutoff in at least one role.

Propose the top fused paper for each clue, independently
of the requested roles and author lists. Require two distinct
papers. Compare the routes below.

| Route | Source condition | Role decision |
| --- | --- | --- |
| Baseline | Original shallow and directed search | Original ridge cutoff |
| Fused ridge | Each paper leads the fused rank for its clue | Retain ridge top labels regardless of margin |
| Fused 4B | Same | Recheck uncertain clauses only |
| Source score 4B | Fused leaders pass the frozen eLife pair-source score | Same |
| Independent 4B | Each fused leader is also the dense leader for its clue, with higher cosine for its own clue than the crossed clue | Same |

For the 4B routes, put the uncertain phrase or phrases in
one GPU forward. Show the full CRediT taxonomy and an explicit
no-match choice. Use the same open Qwen3-4B-Instruct-2507
checkpoint as the [paired role reader](../paired-role-verification-v1).
Do not show candidate author identities or restrict options
to their recorded roles. Retain any confident first-pass
role. Then check whether the two papers actually declare
the selected roles for a single shared author.

Compare positive 4B margin against a cutoff chosen as the
largest margin of any additional wrong proof in PLOS
calibration, using both availability conditions. Keep the
baseline's existing errors separate. On held requests report
additional exact proofs, newly issued wrong-source and
wrong-role proofs, and how many cases open the 4B stage.
Both task forms reuse publication groups. The held groups
are previously inspected, and the source guards are
exploratory.

## Results

Both tasks of Slurm array 47336 on ws6 RTX 4090s reproduced
the decision counts and cutoffs with opposite traversal
orders. Each held entrance has 72 intact requests and 144
withdrawal requests. Strict proofs also require the requested
roles and the designated pair. The table uses the calibrated
margins for all 4B routes.

| Keyword route | Correct proofs | Wrong roles on intact catalog | Wrong proofs on withdrawal | 4B forwards intact / withdrawn |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 10 | 0 | 2 | 0 / 0 |
| Fused ridge | 16 | 5 | 2 | 0 / 0 |
| Fused 4B | 13 | 0 | 2 | 53 / 106 |
| Source score 4B | 13 | 0 | 2 | 47 / 83 |
| Independent 4B | 15 | 3 | 2 | 17 / 8 |

The source-score check saves 29 keyword forwards relative to
running the same 4B recheck on every eligible fused pair. It
preserves all three additional correct proofs, with zero added
wrong outputs on these keyword groups. The two withdrawal
errors come from the fixed baseline. A stricter source rank
agreement cuts more work but the resulting calibration
threshold allows wrong roles on two held groups. On the
intact keyword catalog, without role-margin calibration,
the source-score arm returns 22 correct and six wrong proofs.
All added wrong keyword outputs are role errors on the
designated sources.

| Excerpt route | Correct proofs | Wrong roles on intact catalog | Wrong proofs on withdrawal | 4B forwards intact / withdrawn |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 10 | 0 | 0 | 0 / 0 |
| Fused ridge | 17 | 7 | 0 | 0 / 0 |
| Fused 4B | 23 | 4 | 0 | 53 / 88 |
| Source score 4B | 21 | 4 | 0 | 47 / 47 |
| Independent 4B | 23 | 4 | 0 | 53 / 14 |

At the source-score operating point, role recheck P95 is
66.0–66.2 ms on issued or unissued intact keyword requests
that reach the recheck. This measures the synchronized
single forward with prompt handling, excluding ranking,
serving, and baseline latency. Source gating saves forwards
but does not resolve attribution errors on the excerpt
entrance. The results support a prospective keyword test
with a frozen policy. They do not establish a risk bound,
a latency improvement on the full service, or faithful
expert matching in enterprise demand.

[Aggregate runs and output digests](results.json)

## Reproduce

Use the input sequence from
[the source-identity replay](../plos-contrastive-rescue-identity-v1#reproduce)
on a Slurm CUDA allocation. Change the module to
`experiments.credit.plos_source_first_role_recheck` and add
`--reader QWEN3_4B_INSTRUCT_2507` with the pinned local
checkpoint. Repeat with `--reverse` and a fresh output path.
