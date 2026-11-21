# Expand the regional enterprise sampling frame

Replay the previous 245 provisional groups, then ingest four public
enterprise demand lists with an explicit issuer, title and description.
Use the original issuer and exact-text grouping rules, the same
four-gram overlap screen and the same 4,301 complete historical
records. Check the new sources jointly, treating every earlier public
candidate and historical record as prior exposure. Keep snapshots and
group mappings private.

| Published source | Parsed demands | Complete with issuer | Additional candidate groups |
| --- | ---: | ---: | ---: |
| [Zibo, Shandong University of Technology](https://research.sdut.edu.cn/2025/0626/c12136a544836/page.htm) | 37 | 36 | 25 |
| [Liaocheng, Hebei University of Science and Technology](https://jszyzx.web.hebust.edu.cn/sy/xqdj/a0b0c28b11fe4d218f88aa3e7f039cb8.htm) | 55 | 55 | 13 |
| [Jiangyan, Hebei University of Science and Technology](https://jszyzx.web.hebust.edu.cn/sy/xqdj/56cbb3831cb94afaaa3cc8862cd089d1.htm) | 52 | 52 | 33 |
| [Zhangjiagang, municipal science and technology bureau](https://www.zjg.gov.cn/zjgszwz/zjgqyjsxq/202510/9934da0e65f140a0bf6ff0f7fc560fa0.shtml) | 67 | 67 | 39 |

The Liaocheng table has 53 numbered entries and two further distinct
demands under a merged company cell. Zhangjiagang splits the description
over continuation rows. The parser checks sequence and span consistency.
Zibo descriptions retain the separate target and metrics field where
available. Contact columns are ignored. Of the 210 complete demands, 55
fall below the inherited 64-distinct-four-gram eligibility minimum. The
155 remaining rows yield 110 issuer groups.

The combined frame has **355 provisional groups**, with 241 from Xi'an,
76 from Jiangsu and 38 from Shandong. All 245 prior group signatures are
preserved. The overlap screen found no links from the four additions to
historical or earlier public records. Two complete replays with different
hash seeds returned byte-identical aggregate and private indexes.
Eligibility, issuer aliases, and semantic overlap without high surface
similarity still require review. The frame is suitable for planning a
regional holdout, with access conditions and specificity of demand text
also varying by source. No expert retrieval or outcome labeling ran.

## Implications for an independent trial

Apply the [frozen scoring protocol](../enterprise-sealed-first-query-v1)
to hypothetical sample sizes using the existing
[design calculations](../enterprise-gate-design-audit-v1). These are
planning probabilities under independent groups. The sampling frame
does not determine true coverage, risk, discordance, or joint power.

| Groups | Single-arm gate at 75% true coverage and 2% conditional adverse risk | Paired superiority at +3 points and 20% discordance | Paired superiority at +3 points and 40% discordance |
| ---: | ---: | ---: | ---: |
| 245 | 48.7% | 47.3% | 24.9% |
| 355 | 69.4% | 62.3% | 33.8% |
| 500 | 85.9% | 75.4% | 43.1% |

The absolute gate uses the original 60% coverage floor and 5% risk
ceiling. The superiority gate requires the paired lower limit to
exceed +2 points in supported P@5. At 355 groups, the absolute pass
probability falls to 29.4% if true conditional risk is 3%.
The estimates support expanding and verifying the external frame before
locking a new trial. The original 123-group blind review and its
cutoffs remain in force.

## Reproduce

Restore the snapshots and private index from
[the 245-group audit](../enterprise-municipal-frame-v2#reproduce).
Save the four notices as `zibo-notice.html`, `liaocheng.html`,
`jiangyan.html`, and `zhangjiagang.html`. Save the attachment marked
verified in the Zibo notice as `zibo.xls`. Verify the hashes in
[results.json](results.json). The legacy XLS reader uses `xlrd==2.0.2`.

```sh
umask 077
PYTHONPATH=src:. python -m experiments.enterprise_regional_frame \
  EARLIER_SNAPSHOTS MUNICIPAL_SNAPSHOTS REGIONAL_SNAPSHOTS PRIVATE_WORKBOOK \
  benchmarks/enterprise-sealed-first-query-v1/protocol.json \
  benchmarks/enterprise-municipal-frame-v2/results.json PRIVATE_245_INDEX \
  NEW_AGGREGATE PRIVATE_355_INDEX
```
