# Apply a contributor-role applicability gate before evidence attachment

The [anchored role margin experiment](../plos-anchored-role-margin-v1) found that a large gap between the two best legacy roles can accept tasks outside those roles. This diagnostic uses independently authored activity examples from the [NISO CRediT guide](https://credit.niso.org/) to test the same concern on research-adjacent language.

The protocol compares four methods using a single live frozen Qwen3 embedding per activity:

| Method | Candidate role | Acceptance signal |
| --- | --- | --- |
| Absolute | Best of five publisher headings | Highest cosine |
| Relative | Best of five publisher headings | Gap over next in-catalog heading |
| Outside contrast | Best of five publisher headings | Gap over best of nine excluded CRediT role definitions |
| Definition contrast | Best of five headings extended with mapped CRediT definitions | Gap over best excluded definition |

The definition arm was added after the initial diagnostic showed low unfiltered role recall. It uses guide definitions, not any evaluation example text, so this is an exploratory comparison within the same standard. The first three routes are untouched. Thresholds for all methods are fitted only from five existing positive legacy paraphrases and five manually authored unsupported administrative requests, with zero accepted negatives on that calibration set. A threshold is chosen for valid coverage, then by rejecting more negatives and then by higher threshold.

Before observing model results, select the guide examples that directly support one of the five narrower legacy duties, and examples whose stated duty alone does not support any of the five. This crosswalk is explicit in the script. It does not map the entire CRediT taxonomy to legacy headings. In particular, a separate guide role or a related activity may still coexist with a legacy contribution. The test counts the example as insufficient on its own. Ambiguous examples outside the selected subsets are left unscored.

Two checks use identical model decisions. The atomic check evaluates each distinct guide example once. The binding check attaches an original clause from the public PLOS transfer snapshot only when its indexed role matches and it directly lists the candidate. Forty-five papers receive a matching example for their genuinely missing role, and a paired unsupported request with the same candidate authors and known initial witness. Hash selection reuses examples across papers. Author-cluster intervals describe this panel conditional on the phrases; they are not uncertainty estimates over new wording families. The P95 is shared live encoding and all score computations in the controller. Acquiring documents and the initial witness is outside the measurement.

## Results

Slurm job 42863 evaluated the four routes on one RTX 5090. It reproduced the three original methods' decision counts from job 42860. The four-route controller's component P95 was about 7.25 ms. Valid and invalid decisions are shown as separate counts.

| Panel | Absolute | Relative | Outside contrast | Definition contrast |
| --- | ---: | ---: | ---: | ---: |
| Human-authored supported examples (8) | 0 / 0 | 1 / 0 | 1 / 0 | 3 / 1 |
| Human-authored unsupported examples (16) | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 |
| Supported requests bound to PLOS notes (45) | 0 / 0 | 6 / 0 | 6 / 0 | 14 / 4 |
| Unsupported requests bound to PLOS notes (45) | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 |

Contrasting with excluded roles removes one unsupported acceptance without losing any of the six valid note attachments from the relative-gap route. Adding definitions changes the selected role: the unfiltered correct assignment rises from 5/8 to 7/8 on distinct examples and 25/45 to 39/45 on the note panel. It yields eight further valid attachments but also four wrong attachments on notes whose requests are supported. Applicability and within-inventory role interpretation therefore need separate error reporting. The definition comparison is exploratory, and the 16 unsupported examples and their reuses do not establish a general false-acceptance rate.

[Aggregate results](results.json)

## Reproduce

Restore both publisher snapshots from the [anchored clause experiment](../plos-anchored-clause-rescue-v1). Parse the pinned guide PDF using the [guide parser](../../experiments/credit/parse_niso_examples.py). Run on a Slurm CUDA node with the frozen Qwen3 embedding model.

```sh
PYTHONPATH=src:. python -m experiments.credit.role_applicability_gate \
  OLD_PUBLISHER_SNAPSHOT TRANSFER_PUBLISHER_SNAPSHOT NISO_EXAMPLES \
  EMBEDDING_CHECKPOINT NEW_RESULTS.json
```
