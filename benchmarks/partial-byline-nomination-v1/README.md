# Select sources when author identifiers are incomplete

Reuse the PLOS and eLife open, two-topic two-role panels and their
frozen hybrid ranking. Keep 32 candidates per condition and the same
three-declaration cap. Author roles and topical judgments stay outside
nomination. Read normalized byline names from the pinned publisher
XML. Do not publish names, aliases or individual decisions.

Starting from the available ORCID fields in the snapshots, reveal
each occurrence with a fixed probability of 100%, 75%, 50%, 25%
or 0%. Use independent deterministic hashes over source identifier,
ordinal and seed. Take three nested mask schedules at intermediate
coverage. These are simulated metadata gaps. A document without a
published ORCID cannot gain one through masking. The missingness
pattern does not model a publisher's true acquisition process.

Compare these plans before opening source declarations:

| Route | Source linking used at nomination |
| --- | --- |
| Topic | Choose three sources by their per-condition ranks |
| Visible ORCID | Join only equal visible ORCIDs |
| Name | Add exact normalized names unique within each byline, with at least one missing ID |
| Name with coauthor | Require another compatible shared normalized byline name for a name edge |
| ORCID first, then name | Use the visible ORCID plan when a linked pair exists. Otherwise allow the name plan. |

Known conflicting ORCIDs cannot establish a name link. No plan
merges identities transitively. The preferred plan selects three
documents with the same minimax-rank rule as in the earlier study.

Verify each proposed bundle in two settings with the same actual
publisher role declarations. In the first setting, the original source
ORCID becomes accessible on read. This diagnoses source nomination
when missing index metadata can be recovered downstream. In the
second, the mask persists at verification and a name cannot verify
a contributor across papers. The persistent-mask results measure
known outputs under that explicit identity rule. They do not measure
identity resolution with registries or additional sources.

In the selected first document pair, audit ambiguous shared names by
their original available ORCIDs. A conflict requires two originally
available and distinct IDs. Other uncertain shared names are counted
as unjudged. These are proposal-level diagnostics, independent of
which contributor ultimately appears in the answer. The public
corpora are small, repeated byline pairs are common, and the absence
of observed natural conflicts does not demonstrate a safe name link.

## Results

Both tasks in Slurm array `44451` ran on RTX 5090 on ws5 and
produced identical aggregates. The original-ID arm reproduces
the PLOS and eLife joint-byline counts at depth 32. The table
shows the three 50% mask schedules as ranges of query counts,
rather than treating masks as independent evaluation samples.

| Panel | Plan | Known if IDs can be recovered at verification | Known if the mask persists |
| --- | --- | ---: | ---: |
| PLOS, 160 requests | Visible ORCID | 1–25 | 0–23 |
| | Name | 44 | 0–23 |
| | Name with coauthor | 22–40 | 0–23 |
| | ORCID first, then name | 39–46 | 0–23 |
| eLife headings, 51 requests | Visible ORCID | 0–3 | 0–3 |
| | Name | 12 | 0 |
| | Name with coauthor | 0–3 | 0–3 |
| | ORCID first, then name | 12–15 | 0–3 |

In PLOS, exact-ID priority followed by the name fallback
preserves the verified matches under each mask and increases
the number of cases that are recoverable if the original
identifiers become available at source read. Simple name
matching sometimes changes which exposed identities can be
verified. On eLife headings, one schedule gives three known
recommendations from visible ORCIDs and zero using name alone.
The priority route keeps those three. Requiring a coauthor
for every name-based proposal suppresses that fallback's recall,
especially in eLife.

At zero visibility, name-based nomination can present 44 PLOS
and 12 eLife heading bibliographic bundles with a complete
reference answer once identifiers are restored. The persistent
mask verifiers issue zero under the exact cross-source identity
rule. No selected ambiguous name pair has two originally
available but conflicting ORCIDs in either corpus. The PLOS
name arm still selects 108 unjudged name edges across the 160
query views. Those edges can repeat across queries. The result
supports a two-step fallback for nomination and independent
identity verification. It leaves error rates for homonyms,
abbreviated bylines and truly missing source IDs unverified.

[Full seed-level counts and paired intervals](results.json)

## Reproduce

Restore the same two publisher snapshots and use a Slurm GPU
allocation with the verified embedding checkpoint.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.partial_byline_nomination \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING NEW_RESULTS
```
