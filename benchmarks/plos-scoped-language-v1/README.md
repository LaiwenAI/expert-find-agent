# Natural-language parsing of source-scoped contributor roles

Render each of the 60 [source-scoped publisher pairs](../plos-credit-source-scoped-v1/README.md) in two forms. Number three contribution clauses and interleave their source references. One form names the source in each clause. The other puts the source after the role description using former or latter with an explicit two-report header. The same role uses the same wording across the two versions in a family. In all 48 source-swap pairs, the wording and order of the activities are identical and only two source attachments change. The role phrases and glossary come from the pinned [NISO examples](https://zenodo.org/records/18421449), with repeated phrases across requests.

Compare a joint classifier with 28 source–role choices to two factored routes with 14 role choices. Include a model-only control that predicts source and role separately from the full request. Every route uses one Qwen3-4B forward with three rows, or six for the model-only source-plus-role control. The joint route sees the full request and predicts a source and a role together. It uses an explicit source-and-role instruction and rotates the assignment of sources and roles to answer codes by family. The factored routes resolve the numbered source anchors deterministically inside these two templates, then classify roles with either the full request or just the target phrase. The same contributor intersection over publisher evidence runs after all three routes. An oracle uses the known scoped roles to check the construction.

Score exact scoped plans, exact contributor sets, unsupported selections, and both versions of each pair. A joint success requires the exact scoped plan and contributor set, preventing an accidental correct empty set from counting. Timing covers the warm local path from the rendered request through parsing, tokenization, one GPU forward and contributor intersection. Paper identifiers are supplied. The test has six linked article pairs and authored template boundaries. Results cover supplied-source recommendation with structured scope cues. The local route sees the same phrases in both renderings.

## Results

| Route | Exact scoped plan and contributor set / 240 | Exact contributor set / 240 | Both variants exact / 120 | Unsupported contributor selections | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint source–role codes | 0 | 121 | 0 | 18 | 119.32 ms |
| Separate modeled scope and role | 24 | 139 | 4 | 6 | 200.79 ms |
| Deterministic scope, full-request role | 84 | 163 | 35 | 5 | 94.10 ms |
| Deterministic scope, phrase-only role | 122 | 180 | 56 | 6 | 85.73 ms |
| Known scoped roles | 240 | 240 | 120 | 0 | 0.003 ms |

The modeled scope control binds 447/720 clauses to the correct source. Its source accuracy is 263/360 with named scope and 184/360 with the former/latter render. Both deterministic routes bind all clauses within the supplied grammar. With source binding held fixed, the phrase-only reader improves exact plans from 84 to 122 and reduces P95 by 8.9%. It introduces one additional unsupported selection compared with the full-request role reader. On the 48 source-swap pairs, the phrase-only route gets both versions right 16 times, versus 12 for the factored full-request route and once for the two-head model control. The joint-code baseline has no exact plans and is a weak comparison in this configuration.

Job 41533 used one RTX 5090 on ws5. The expanded comparator set follows exploratory runs with the same fixture. The results support separating source binding from role interpretation when source anchors are reliable. Independent documents, freely written requirements and unresolved anchor handling are needed to assess transfer. [Aggregate results](results.json)

## Reproduce

Use the [released publisher fixture](../plos-credit-source-scoped-v1/fixture.json), pinned NISO JSON generated with the [example parser](../../experiments/credit/parse_niso_examples.py), and the pinned local Qwen3-4B checkpoint. Run on a Slurm CUDA compute node with a fresh output.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.scoped_credit_language \
  FIXTURE_JSON NISO_EXAMPLES_JSON QWEN_4B_CHECKPOINT NEW_RESULTS --balanced-joint
```
