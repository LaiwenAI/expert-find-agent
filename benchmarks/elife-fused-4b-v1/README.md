# Check source identity and role labels in one 4B forward

The [mixed 0.6B verifier](../elife-mixed-role-verifier-v1) failed to
reject a held swapped-role control. A 4B role reader rejected that
control and preserved the positive after two separate source and role
forwards. Keep the proposed paper pair and role alternatives fixed and
test whether the resident 4B reader can verify both in one batch.

For a source row, present the clue and the paper title with at most
224 abstract words. The two letter options indicate whether the
paper reports the specific study. Assign letters by a stable clue hash.
The score is the confirmation logit minus the rejection logit.
In the frozen PLOS projection, produce an exact source and two topical
rivals per clue. Rank rivals with the target withheld. Set the source
cutoff separately for keyword and excerpt views to the highest
calibration rival score. The article is the unit of source calibration.

Run two source rows plus two existing three-option role rows together.
Both source scores must exceed the source cutoff. Both role choices
must pass the existing 4B margin, and the selected roles must hold for
the same author across the proposed papers. The model sees no author
or declaration data. A relative-score variant adds the highest-ranked
other paper for each clue. Compare the selected source's confirmation
margin with that rival's margin. In calibration, compare the true
source with the stronger of its two rivals and each wrong source with
the other rival. Freeze the largest wrong difference as the cutoff.
Both variants use one forward. Count its rows and nonpadding
tokens. Compare with the archived cascade and the dense source-only
fallback on the 27-pair development and 12-pair held panels. Check
exact archived proof parity for both controls and repeat both orders.
The eLife panels have already been used. The result does not measure
HTTP latency or missing-source risk under live deployment.

## Results

All four tasks in Slurm array 46302 complete on ws3 RTX 5090.
Each panel reproduces in both request orders. The absolute policy
also matches its initial replay in Slurm array 46296.

| Source policy | PLOS keywords passed | PLOS excerpts passed | PLOS rivals accepted per view | Rows per fallback |
| --- | ---: | ---: | ---: | ---: |
| Absolute margin | 1/24 | 4/24 | 0/48 | 4 |
| Relative margin | 0/24 | 4/24 | 0/48 | 6 |

Both fused policies send the two held and four development
keyword fallbacks through the 4B batch. Five of the six proposed
pairs are the intended sources. All six fail the source margin
despite producing valid role choices. Held fallback inputs total
1,118 nonpadding tokens for the absolute path and 1,950 for the
relative path, across two calls.

| Policy | Development keyword strict / adverse | Held keyword strict / adverse |
| --- | ---: | ---: |
| Cascade alone | 12 / 0 | 10 / 0 |
| Dense source-only fallback | 15 / 0 | 11 / 1 |
| 4B absolute or relative fallback | 12 / 0 | 10 / 0 |

Fusing tasks satisfies the dispatch limit and leaves the cascade's
evidence unchanged on both panels. Here the bottleneck is the source
confirmation margin. The relative score loses coverage at the same
calibration risk and reads two more rows. These results favor
testing source-specific scoring with the stronger role reader.
See [aggregate results](results.json).

## Reproduce

Restore both eLife projections, the corresponding individual service
references, and the frozen model and PLOS inputs from the
[mixed verifier experiment](../elife-mixed-role-verifier-v1#reproduce).
Replace the diagnostic flag with `--fused-4b-reference MATCHED_SERVICE_RUN`
and run both request orders in Slurm GPU allocations.
