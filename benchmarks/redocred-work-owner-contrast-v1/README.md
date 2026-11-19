# Separate person and work attribution when the role is shared

Use the public [Re-DocRED](https://aclanthology.org/2022.emnlp-main.580/)
and [Re²-DocRED](https://aclanthology.org/2026.eacl-long.213/)
development and test releases. The source text and entity indices
match between releases. This probe retains an unedited document
with two works, two people and a single relation, such as director.
The four claims use the Cartesian product of the people and works.
Require evidence of distinct bindings in the original annotations:
either the first person has the role on the first work and the second
on the second, or the reverse. A document can also have a third or
fourth supporting link. Select one case per original label count
per document by fixed hash. No model output selects cases.

The newer annotations are used for threshold calibration and
evaluation. Original labels define the construction only. Compare
both annotation inventories to count additional known positives.
Re²-DocRED expands the inventory with model-generated, reviewed
and rule-derived relations. An absent link remains unjudged and
can be a genuine omission. Report it separately from confirmed
support. Group test cases by whether the revised inventory has
two known links or includes shared roles.

Both evidence arms ask the same four explicit person–work questions
in one frozen Qwen3-4B batch. The first arm uses the entire document.
The compact arm greedily selects up to four shared sentences using
marginal coverage of the four entity names and role cues. It uses
neither gold evidence sentences nor role labels for sentence ranking.
On each batch's four scores, compare these acceptance signals.

| Signal | Calculation for a claim |
| --- | --- |
| Absolute | Its own support logit margin |
| Other work | Its margin minus the same person's margin for the other work |
| Other person | Its margin minus the other person's margin on the same work |
| Both | The smaller of the two differences |

Additional decision signals use the original verifier outputs and
require no extra GPU calls. Fit each arm and signal in development
at 5% and 10% ceilings on acceptance of unjudged cells. Freeze each
threshold for the test documents. These ceilings align calibration
criteria rather than proving false-acceptance guarantees. Report
accepted known links, accepted unjudged links, and cases retaining
all known links. The four cells and cases from the same document
are dependent. Timing covers shared window selection and one live
batch, with no corpus retrieval, transport or queueing.

## Results

Slurm job 43349 completed on one RTX 5090 on ws5. The
development panel has 64 cases from 47 documents. Test has 62
cases from 49 documents, with 166 known links and 82 unjudged
links. Re²-DocRED adds no labels for these selected role cells.
The updated release therefore does not resolve the remaining
unjudged cells.

| Evidence and signal, 5% calibration ceiling | Known accepted / 166 | Unjudged accepted / 82 | All known retained in disjoint cases / 29 | All known retained in shared cases / 33 | Batch P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole, absolute | 54 | 4 | 1 | 2 | 117.4 ms |
| Whole, other work | 89 | 4 | 13 | 0 | 117.4 ms |
| Four sentences, absolute | 68 | 5 | 2 | 5 | 68.0 ms |
| Four sentences, other work | 76 | 1 | 9 | 0 | 68.0 ms |
| Four sentences, other person | 47 | 1 | 6 | 0 | 68.0 ms |
| Four sentences, both | 88 | 4 | 19 | 0 | 68.0 ms |

The four-sentence route uses the same 248 test model pairs as
full documents and reduces batch P95 by 42%. With its fixed
inputs, the other-work margin raises accepted known links from
19 to 35 in disjoint cases and lowers them from 49 to 41 in
shared cases. At the 10% ceiling, the four-sentence absolute
and other-work signals retain all known links in eight and zero
shared cases, respectively. A positive other-work cutoff
mathematically prevents the same person from passing for both
works. Positive other-person cutoffs likewise suppress jointly
supported people. Treat a rival margin as a signal to inspect
scope, or use an independently verified exclusive condition.
Preserve explicit support on shared links in settings allowing
multiple valid contributions. The results are within-document
diagnostics with supplied identities and works.
[Aggregate results and frozen cutoffs](results.json)

## Reproduce

Use the dev and test JSON files from [Re-DocRED at ccfb54f](https://github.com/tonytan48/Re-DocRED/tree/ccfb54f)
and [Re²-DocRED at 006412e](https://github.com/klassessg/re2-docred/tree/006412ef599a3fd6039b3746cc2d1e22fc600d62/dataset).
Use the same frozen Qwen3-4B-Instruct-2507 checkpoint on a Slurm
CUDA allocation. The runner verifies all four input hashes and
does not redistribute the source documents.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.work_owner_contrast \
  REDOCRED_DIR RE2DOCRED_DIR QWEN4B NEW_RESULTS
```
