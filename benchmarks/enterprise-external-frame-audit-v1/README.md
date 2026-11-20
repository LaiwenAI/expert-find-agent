# Audit a separate enterprise first-query sampling frame

Inventory five 2026 enterprise demand notices from the Xi'an science
bureau and a Nantong technology-transfer posting. Each Xi'an notice
links a details archive. Join each attachment to its listed request
ID, parse the technical description and organization field, and verify
one attachment per listed request. The Nantong posting has four
directly structured request tables. No expert model, recommendation,
or reviewer judgment enters the frame audit.

| Public source | Listed and joined details | Candidate groups after the historical screen |
| --- | ---: | ---: |
| [Xi'an second call](https://xakj.xa.gov.cn/kjdt/tzgg/2070354906494091266.html) | 70 | 56 |
| [Xi'an third call](https://xakj.xa.gov.cn/kjdt/tzgg/2099713781650178049.html) | 50 | 44 |
| [Xi'an fourth call](https://xakj.xa.gov.cn/kjdt/tzgg/2105191117492449281.html) | 50 | 44 |
| [Xi'an fifth call](https://xakj.xa.gov.cn/kjdt/tzgg/2114566489889652738.html) | 50 | 35 |
| [Xi'an sixth call](https://xakj.xa.gov.cn/kjdt/tzgg/2119682252589887489.html) | 50 | 40 |
| [Nantong posting](https://kjcyc.usts.edu.cn/info/1029/6055.htm) | 4 | 4 |
| Total | 274 | 223 |

All 274 records have a technical description and available
organization field under the parser's placeholder checks.
The normalized organization count before historical comparison is
231. Compare against all 4,301 complete request records in the
historical workbook, conservatively treating that full workbook as
potentially exposed. Reuse the existing exact and lexical duplicate
screen. The screen joins on normalized organization, exact text or
body hash, and high four-gram overlap. It flags 15 external records
linked to historical components. No pair newly exceeds the lexical
threshold in this audit. Select one representative per remaining
component in the inherited fixed hash order: 259 eligible records
produce 223 candidate groups.

These are candidate groups. Different normalized organizations can
share a parent institution, and semantically equivalent requests can
survive the lexical screen. This audit covers only the specified
workbook and publication snapshot. A separately frozen evaluation
requires overlap review against the full exposure inventory, and
eligibility review for the intended civilian expert-search task.
Freeze the first-query comparison before evaluating that cohort.
The separate [123-group study](../enterprise-sealed-first-query-v1)
is already assigned to the abstention-rescue comparison.

The 223 candidates also fall short of the earlier study's
illustrative power targets. Under its same assumptions of
independent groups, 75% true coverage and 2% conditional adverse
risk, 223 groups have a 39.3% probability of passing both absolute
gates for one arm. This is a planning scenario. It does not establish
power for paired ranking comparisons or performance on this frame.
[Design sensitivity](../enterprise-gate-design-audit-v1).

## Reproduce

Download the notice and first attachment for each listed Xi'an
call. Save the notice as `xian-ORDER.html` and the attachment
as `xian-ORDER.zip` or `xian-ORDER.rar`, where ORDER is
`second`, `third`, `fourth`, `fifth` or `sixth`.
Save the Nantong page as `nantong.html`. Compare the downloaded
hashes to [the aggregate snapshot](results.json). Use Python 3.11+,
Beautiful Soup, openpyxl and a RAR-capable `bsdtar`. Keep the
workbook and private index outside Git. The private index stores
the selected source identifiers and content hashes. It does not
contain copied source texts, company names or contact fields.

```sh
PYTHONPATH=src:. python -m experiments.external_enterprise_frame \
  SNAPSHOT PRIVATE_HISTORICAL_WORKBOOK NEW_AGGREGATE PRIVATE_INDEX
```
