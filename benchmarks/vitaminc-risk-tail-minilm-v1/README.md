# Train against high-margin false supports by failure class

Freeze the source-disjoint 1,200 training, 400 development, and 400 test pairs and the MiniLM checkpoint from the [previous training comparison](../vitaminc-contrastive-minilm-v1/README.md). Retrain three objectives under each of two seeds with the same architecture, trainable layers, two epochs and pair batching. The frozen model is the common reference.

Compare classification loss, within-pair margin sorting, and a class-specific loss for the upper tail of false supports. In the new objective, select the highest-margin quarter of refutations and insufficient-evidence examples separately in each batch. For each class, penalize these negatives when their margins approach or exceed a true support's margin. Average the pairwise softplus terms with a one-unit margin, weight the two classes equally when both occur, and add the result to the classification loss with coefficient 0.4. The group loss uses labels only during training. Every arm retains one inference forward with the same classification head.

Freeze evaluation thresholds on the original development panel. At the matched point the two development caps are 21 and 28 false accepts per 200 samples. At the strict point the caps are 10 and 10. Report test true supports and the two false-accept counts separately. The test set has already appeared in earlier experiments. It remains disjoint from training and calibration on the selected page and exact claim keys. This is a targeted operating-point experiment rather than a new independent domain replication.

## RTX 5090 results

Each cell reports true supports / false accepts on refutations / false accepts on insufficient evidence. The test denominators are 400 / 200 / 200.

| Model | Seed | Matched development caps | Strict development caps | Single-item P95 |
| --- | --- | ---: | ---: | ---: |
| Frozen | — | 139 / 20 / 6 | 100 / 12 / 2 | 1.24 ms |
| Classification | 20261119 | 144 / 26 / 8 | 98 / 13 / 2 | 1.25 ms |
| Within-pair | 20261119 | 144 / 24 / 6 | 102 / 14 / 2 | 1.28 ms |
| Class-specific upper tail | 20261119 | 139 / 21 / 6 | 113 / 15 / 2 | 1.28 ms |
| Classification | 20261120 | 141 / 26 / 8 | 98 / 14 / 2 | 1.29 ms |
| Within-pair | 20261120 | 144 / 24 / 6 | 109 / 14 / 2 | 1.28 ms |
| Class-specific upper tail | 20261120 | 141 / 21 / 6 | 117 / 16 / 3 | 1.27 ms |

Job 42206 reproduces the earlier seed's classification and within-pair weight hashes exactly. Both class-specific runs increased retained support at the strict operating point versus the frozen model (13 and 17 more), with paired 95% intervals of +1.00 to +5.50 and +2.25 to +6.50 percentage points. Both also accepted more refutations. At the matched point, the class-specific runs accepted one more refutation than frozen, and had equal or slightly higher true support. The apparent benefit depends on the selected operating point. Training for the highest-margin negatives does not yet improve quality under a verified held-out risk constraint. The earlier 4B control remains stronger on true-support recovery and is much slower on these supplied evidence inputs. [Aggregate results](results.json)

Rebuild the same projection and use the previous training command with `--compare-tail` and new output and weights paths inside a Slurm CUDA allocation.
