# Offline Chinese patent registry join

This follow-up uses the same 80 development requests and candidate
roles as the [registry reach audit](README.md). Only aggregate counts
and code are released. The exact clause-derived locator, candidate
name, institution, and query trace remain in the restricted workspace.
The offline join sends no query or locator to an external service.

## Sources and matching

- [CNIPA's publication authority file](https://english.cnipa.gov.cn/art/2023/2/28/art_3189_182373.html)
  through June 30, 2026 supplies publication numbers, original kind
  codes, and dates. A locator with a kind code must match that code.
  An omitted kind can return several editions. The file carries no
  inventor roster or general application-to-publication crosswalk.
- The [University of Minnesota research archive](https://users.econ.umn.edu/~holmes/data/chinapatent/)
  reproduces SIPO catalog and inventor fields for publications
  through 2012. For application-shaped locators, match the original
  application-number stem. The stem join leaves a trailing check
  character unvalidated. For publication-shaped locators, require
  a single current authority edition and an exact publication
  number and publication date match to the archive. Check that
  inventor indices and declared roster length agree.
- Try one deterministic tone-free Pinyin transliteration for Han
  profile names of two to four characters. Compare that form and the
  original normalized name against archival names. Catalog-unique
  transliterations are stronger leads and still require identity review.
  Compare the archived applicant and current profile institution by an
  unambiguous exact alias in the [ROR 2.16 snapshot](https://zenodo.org/records/24378467).
  An applicant can be a different organization from a contributor's
  institution. The comparison is an optional corroborating signal.

Verify both the archived inventor join and the current
authority-kind join on known public positive controls. The 2012 archive
contains English transliterations for the matched rosters. Application
and publication routes can count the same work separately.

## Results

| Checklist for top-five candidates | All development requests | Requests with initial abstention |
|---|---:|---:|
| Official publication number or number-kind match | 27/80 | 17/55 |
| Historical application join with complete roster | 9/80 | 8/55 |
| Publication edition join with complete historical roster | 4/80 | 3/55 |
| Either route with complete historical roster and Pinyin overlap | 13/80 | 11/55 |
| Exact original-script name or Pinyin plus matching ROR institution | 0/80 | 0/55 |

Across the catalog, the current authority file matched 532 of 533
owner–publication locator pairs. Fifty matches retained multiple
editions. The older application archive matched 63 of 2,233
owner–application stems, including 157 with filing years from 1985
through 2012. Filing in that window does not ensure publication by
the archive cutoff. Eight publication-edition matches reached the
archive, and six had complete inventor rosters. All 69 complete
roster pairs contained a Pinyin match. For 15, both organization
strings resolved to ROR. Their IDs never agreed.

A recent native-script inventor roster and an independent
person-identity anchor are the remaining evidence dependencies.
An authority match alone measures document availability, and a
transliteration match supplies a review lead. The test neither scores
technical suitability nor changes search outcomes. Absence from a
historical archive or disagreement between applicant and current
institution cannot rule out inventorship.

## Reproduce

Download the pinned releases above as `cn-authority.zip`,
`cnpub_basedat.zip`, `cnpub_inventor.zip`, and `ror.zip`.
Only the official download requests are needed. Inputs are checked
against exact release checksums. Use Python 3.11+ and, for the
transliteration step, pypinyin 0.55.0.

```sh
python -m experiments.retrieval.enterprise_cn_registry_join transliterate \
  PRIVATE_PROJECTION PRIVATE_TRANSLITERATIONS
python -m experiments.retrieval.enterprise_cn_registry_join institutions \
  PRIVATE_PROJECTION PRIVATE_DATASET PRIVATE_INSTITUTIONS
python -m experiments.retrieval.enterprise_cn_registry_join audit \
  PRIVATE_PROJECTION DEVELOPMENT_TRACE DEVELOPMENT_CORPUS ARCHIVES cn-offline.json \
  --transliterations PRIVATE_TRANSLITERATIONS --institutions PRIVATE_INSTITUTIONS
```
