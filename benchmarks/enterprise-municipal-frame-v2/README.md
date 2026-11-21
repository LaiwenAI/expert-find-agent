# Extend the enterprise sampling frame without opening the sealed trial

Reproduce the 223-group [external frame](../enterprise-external-frame-audit-v1)
and add the first Xi'an publication to its five later batches. Join
only numeric notice entries to matching complete DOCX attachments.
Ignore Word lock files, and require an exact one-to-one join between
the 30 official entries and the data files. Re-run the same normalized
issuer, exact text and four-gram screen against the 4,301 complete
historical workbook records. Screen the first batch both within the
expanded frame and against the existing frame as earlier exposure.

| Source | Listed entries | Title and body available | Issuer field available | Use in the frame |
| --- | ---: | ---: | --- | --- |
| [Xi'an first batch](https://xakj.xa.gov.cn/kjdt/tzgg/2049397081081192449.html) | 30 | 30 | Yes | Issuer and text screening |
| [Anhui release via Wuhu](https://www.whit.edu.cn/info/1151/28557.htm) | 147 | 147 | No | Text-only inventory |
| [Weifang list via Donghua](https://research.dhu.edu.cn/2026/0526/c19612a379622/page.htm) | 79 | 78 | No | Text-only inventory |

The two workbooks supply technical text and no dedicated issuer
field. The audit uses neither contact details nor an inferred
organization identity. Their rows contribute zero issuer-grouped
candidates.

Five first-batch rows link to historical or earlier external records.
The remaining 25 rows add 22 provisional groups after grouping.
The expanded frame has **245 provisional groups**, including
241 from Xi'an and four from Nantong. No lexical threshold edge
was needed for the 22-group increment. This is a source and
duplication audit before eligibility review. The 245 are not a
verified independent or in-scope test cohort.

Project the title and first/last windows with the frozen 0.6B encoder.
Restrict candidate endpoints to the 22 new groups and compare them
against the expanded external frame and the historical workbook.
Apply the same separate issuer-string and semantic screens as the
[prior overlap audit](../enterprise-external-overlap-v1). When a
screen finds no above-threshold edges, keep a small nearest-neighbor
sample for blind review. The private copies and mappings stay
outside Git.

## Sample size sensitivity

Reuse the original 60% coverage floor and 5% adverse-or-unresolved
risk ceiling with simultaneous 95% confidence. Under the existing
single-arm independent-group planning scenario with 75% true coverage
and 2% conditional risk, moving from 223 to 245 groups raises the
probability of passing both absolute thresholds from **39.3% to
48.7%**. At 500 groups it is 85.9% under the same assumptions.
A 3% risk scenario gives 19.1% at 245 and 42.2% at 500.
Under the separate paired-slot planning model with a true three-point
supported P@5 improvement and 20% discordance, the probability of
passing the two-point lower-bound superiority criterion is 47.2%
at 245 groups and 75.4% at 500. At 40% discordance these fall to
24.9% and 43.1%. Cross-arm dependence and the joint power across
all endpoints remain unspecified. The first-query trial, judgment
counts and sealed cutoffs are unchanged.

## Semantic and issuer overlap

Both jobs in Slurm array 46192 on ws3 RTX 5090 produced identical
counts and private pair selections under reversed view order.
Semantic nearest-neighbor counts at 0.85 are
expanded candidate frame 0, historical workbook 0. The issuer-string counts at
0.85 are expanded candidate frame 0, historical workbook 0. Neither channel reaches the 0.90
review floor. The four strata contribute eight exploratory neighbors
each, yielding 32 distinct private pairs after deduplication.
Both template forms pass validation and have zero completed judgments.
This is an overlap signal with unmeasured false negatives.

## Reproduce

Restore the earlier public snapshots from the
[original frame](../enterprise-external-frame-audit-v1#reproduce).
Save the Xi'an first-batch notice and first linked attachment as
`xian-first.html` and `xian-first.rar` in a separate directory.
Save the two public workbook notices and their attachments as
`wuhu.html`, `wuhu.xlsx`, `weifang.html`, and `weifang.xlsx`.
Verify the hashes in [results.json](results.json) and keep the
historical workbook and private outputs outside Git.

```sh
umask 077
PYTHONPATH=src:. python -m experiments.enterprise_municipal_frame \
  EARLIER_SNAPSHOTS NEW_SNAPSHOTS PRIVATE_WORKBOOK \
  benchmarks/enterprise-sealed-first-query-v1/protocol.json \
  benchmarks/enterprise-external-frame-audit-v1/results.json \
  NEW_AGGREGATE PRIVATE_INDEX
PYTHONPATH=src:. python -m experiments.external_enterprise_overlap prepare \
  EARLIER_SNAPSHOTS PRIVATE_WORKBOOK PRIVATE_INDEX PRIVATE_GPU_PROJECTION \
  --first-batch NEW_SNAPSHOTS
# Execute this step inside the GPU Slurm allocation.
PYTHONPATH=src:. python -m experiments.external_enterprise_overlap gpu \
  PRIVATE_GPU_PROJECTION QWEN_EMBEDDING PRIVATE_GPU_RESULTS
PYTHONPATH=src:. python -m experiments.external_enterprise_overlap packet \
  EARLIER_SNAPSHOTS PRIVATE_WORKBOOK PRIVATE_INDEX PRIVATE_GPU_RESULTS PRIVATE_PACKET \
  --first-batch NEW_SNAPSHOTS
```
