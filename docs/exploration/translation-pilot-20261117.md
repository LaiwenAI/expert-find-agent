# Complete-original translation ablations, 2026-11-17

Three sequential development pilots use the same first32 construction needs. They preserve complete original Chinese and aligned source segments. They do not generate new independent needs or inherit expert-fit labels. All calls use gpt-6-luna, three workers and no retries.

| Variant | Generated full English | Combined screen passes /32 | Native audit faithful | Structural failures | API timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 31 | 8 | 12 | 0 | 1 |
| Copied numeric placeholders | 23 | 6 | 9 | 5 | 4 |
| Fixed numeric blocks / free gaps | 24 | 3 | 4 | 7 | 1 |

The combined screen requires both native full-completeness approval and all supported numeric checks to match. Unsupported scoped negation is uncertain. Numeric checks can reject scientifically equivalent reformattings, such as changing a large-count unit; native screens can also be wrong. Neither a rejection nor a passing label is human translation truth. Different timeout counts and sequential execution prevent an isolated algorithmic latency comparison.

The placeholder variant requires each numeric token exactly once in its original segment, with deterministic restoration. The fixed-block variant instead asks for translations of the text gaps and assembles protected quantities programmatically. It rejects new Arabic numeric expressions in free gaps. Completed fixed-block segments have no numeric mismatches in this pilot, but unknown negation remains and native completeness approval is low. Anchoring numbers alone does not preserve sentence scope, completeness, or fluent word order.

Placeholder constraint retention and general translation quality are already distinct concerns in [Zhang et al., ACL2023](https://aclanthology.org/2023.acl-long.332/). That paper studies and trains terminology-aware translation models; these API-only numeric experiments do not reproduce its learned method or justify borrowing its performance claims.

No alternative improves the observed combined screening count enough to justify adopting it for all1,000 originals. The pilots are not held-out trials, and there is no human error adjudication yet. The current5,000 bilingual derived-query dataset remains separate from incomplete full-original translation.

Raw responses/protocols remain under `artifacts/large-bilingual-data-v1/original-translations*`; the paired accounting is `translation-comparison-v1/summary.json`. The exact earlier placeholder helper is preserved as a hashed frozen source beside its protocol because the shared helper subsequently gained gap-assembly functions. Failures are retained and not silently retried or relabeled.
