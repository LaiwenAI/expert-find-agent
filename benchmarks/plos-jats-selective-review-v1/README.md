# Check one or two sources only after narrow retrieval abstains

Reuse the [source discovery and role replay](../plos-jats-discovery-v1/README.md). Freeze the original candidate pools and role verifier. Pass the linked top-four result through when available. On an expanded-only proposal, compare three review policies with the baseline that checks every expanded proposal in one batched forward.

- Selective pair checks both sides together and thresholds their minimum score.
- Selective weak checks only the side with worse fused rank, breaking ties toward the first clue.
- Selective early exit checks the weaker side first. A failing margin stops the route. Otherwise the other side is checked in a separate forward and both margins must pass.

For each policy and held-out linked component, set the acceptance threshold to strictly exceed the largest false-proposal score from the other components. The selective threshold uses expanded-only proposals. Both present-but-wrong and missing-source proposals count as false. An empty set of calibration negatives produces abstention on expansions. A top-four pass-through has no reranker check and can accept a wrong source.

## Results

The source totals below count each of the 150 source scenarios once. Recommendation counts reuse the same articles and role clauses. The 89 positive requests in each view have both sources available.

| Clue | Route | Exact sources / 25 | Wrong source / 50 missing | Review forwards | Reviewed text pairs | Correct recommendations / 89 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Result excerpt | Linked top-four | 20 | 2 | 0 | 0 | 41 |
| Result excerpt | Expanded without check | 25 | 6 | 0 | 0 | 51 |
| Result excerpt | Review every expansion | 16 | 1 | 31 | 62 | 25 |
| Result excerpt | Selective pair | 21 | 3 | 9 | 18 | 41 |
| Result excerpt | Selective weak | 21 | 3 | 9 | 9 | 41 |
| Result excerpt | Selective early exit | 21 | 3 | 11 | 11 | 41 |
| Keywords | Linked top-four | 10 | 1 | 0 | 0 | 21 |
| Keywords | Expanded without check | 15 | 2 | 0 | 0 | 37 |
| Keywords | Review every expansion | 4 | 0 | 18 | 36 | 11 |
| Keywords | Selective pair | 10 | 1 | 6 | 12 | 21 |
| Keywords | Selective weak | 10 | 1 | 6 | 6 | 21 |
| Keywords | Selective early exit | 10 | 1 | 6 | 6 | 21 |

On excerpts, all three selective policies gained one exact source pair and added one missing-source error against linked top-four. None gained correct positive recommendations. Selective weak reduced scored inputs by half relative to selective pair at the same forward count and with the same aggregate quality counts. Sequential early exit added two forwards across the excerpt scenarios. The keyword selective policies retained one wrong answer in present-source scenarios, as did linked top-four.

Job 41779 ran on an RTX 5090. Present-source factored P95 was about 18.5 ms for both selective pair and weak, compared with 29.2 ms when checking every expansion on excerpts. The analogous keyword figures were 17.6, 17.0 and 25.8 ms. Timing reuses the retrieval prefix, separately measured reranker calls and role interpretation. It describes the measured components and does not include a serving load. The linked components and phrases have been reused in earlier studies. [Aggregate results](results.json)

Run the original construction and replay command with `--selective-review` after the `run` subcommand and give it a new output filename.
