# Locate natural proof losses before expanding anchor lookup

The [source-identity trial](../plos-contrastive-rescue-identity-v1)
showed that filtering an already narrow selection can suppress
wrong proofs while leaving no extra exact proofs on natural
keyword requests. This study traces the missed proofs through
the frozen role, shallow, nomination and scoring stages. It
also tests two bounded changes to source nomination.

## Protocol

Reuse the 414-paper catalog, 12 calibration groups, and 12
test groups from the [prospective PLOS split](../plos-prospective-proof-split-v1).
Use the indexed abstract slice, excerpt or keyword clues, and
natural ranks. Score withdrawal separately. Keep the learned
role decoder, shallow issuance rule, posting type, and source
score fixed. The unmodified arm must reproduce the archived
natural results per entrance. For each positive example with
both sources present, assign the first applicable stage:

1. Role confidence falls below the frozen cutoff, or the
   confident decoded roles differ from the requested roles.
2. Shallow search issues either the exact proof or a substitute.
3. Shallow search abstains, and author-role lookup either
   proposes the exact pair or misses it. Distinguish when
   neither intended source led its lane and when one did.
4. If lookup proposes the exact pair, report whether the
   frozen scorer recovers it or it is rejected or outscored.

Also report both-source reachability in the common top six
and top 18 independently of the role and issuance stages.
Reference source and role labels are used only for evaluation.
The stages count examples; repeated clues and roles share
paper groups.

| Fallback | Source anchors per clue lane | Opposite-lane candidates |
| --- | --- | --- |
| Baseline | Top fused paper | First 12 author-role neighbors after hybrid reranking |
| Role first | First of the top three papers with a remaining author-role neighbor | Same |
| Three anchors | Each of the top three papers with remaining author-role neighbors | At most 12 pairs per direction. Pool by anchor position plus within-neighbor rank before scoring. |

Both candidate directions share the same two-source scoring
function. Keep the original eLife threshold, then calibrate each
arm on the maximum adverse option score on natural PLOS
calibration requests. Use the eLife threshold when no adverse
option supplies a PLOS cutoff. Report eligible options,
provisional outcomes, and postings visits. The comparison
reuses an examined test panel. It estimates neither independent
deployment risk nor client-side latency.

## Results

Both RTX 4090 tasks in Slurm array 47329 reproduced every
count and cutoff with reversed group order. The archived
natural first-query outputs match for both input forms.

| First applicable stage in the 36 held positives | Excerpt | Keywords |
| --- | ---: | ---: |
| Below the role confidence cutoff | 26 | 26 |
| Confident wrong role | 0 | 0 |
| Shallow exact proof | 10 | 9 |
| Shallow substitute | 0 | 0 |
| Missing exact pair in fallback | 0 | 0 |
| Exact pair rejected or outscored | 0 | 0 |
| Exact pair recovered | 0 | 1 |

Both intended papers occur in the top six for all 36 excerpt
positives and 30 keyword positives. Increasing depth to
18 leaves those counts unchanged. Every confident keyword
positive already produces the exact result under the frozen
baseline. There is no currently reached ranking or source
acceptance failure among held positives to fix by adding anchors.

| Keyword fallback with frozen eLife cutoff | Exact options | Wrong options | Natural exact / wrong proofs | Wrong proofs on withdrawal | Posting entries, present / withdrawn |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1 | 6 | 10 / 0 | 2 | 10 / 80 |
| Role first | 1 | 28 | 10 / 2 | 5 | 31 / 142 |
| Three anchors | 1 | 29 | 10 / 2 | 5 | 31 / 147 |

Options include both availability states. For excerpt clues
all three arms propose zero exact fallback options. On natural
PLOS calibration, both expanded keyword routes need a higher
cutoff. That removes their held wrong proofs and their only
held exact rescue. The unfiltered baseline with natural PLOS
calibration retains its one rescue and issues four substitutes
on withdrawal. Risk counts are descriptive and refer to the
same examined held groups.

The source shortlist is not the main barrier in this panel.
The next comparison should test whether uncertain roles can be
resolved on independently identified sources without increasing
wrong role attribution or substitute issuance. Current evidence
does not support adding anchor searches to the default route.

[Aggregate outcomes and replication hashes](results.json)

## Reproduce

Run on an allocated Slurm GPU with the
[identity comparison's input list](../plos-contrastive-rescue-identity-v1#reproduce).
Use the module `experiments.credit.plos_role_anchor_audit`,
a fresh output path, then repeat with `--reverse`.
