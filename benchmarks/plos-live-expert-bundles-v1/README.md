# Text-to-expert search with minimum citation bundles

Present an English request as a three-item task list. Build each item from a human-authored NISO role example. An offline hash split per role reserves 14 calibration phrases and keeps 35 phrases for evaluation. Phrases recur within the evaluation panel. The NISO inventory and publisher records have appeared in earlier studies. [NISO examples](https://zenodo.org/records/18421449)

Search the 16 PLOS contributor identities with multiple linked publications. Keep the byline prior and the [minimum citation rule](../plos-minimal-evidence-bundles-v1/README.md) fixed. Hash-sample 40 role combinations in each condition: no matching expert, supported by single-source bundles, and at least one expert requiring multiple sources. Keep the same materialized per-article declarations and author-cache mask for every route. Confirm the whole ranked response against brute-force gold given the intended roles. Count a recommendation as unsupported if the citations attributed to that expert do not establish all three requested roles.

Compare a frozen Qwen3-Embedding-0.6B glossary lookup, a ridge head trained on 84 separate synthetic task phrases, a batched frozen Qwen3-4B glossary reader and the ridge route with reader fallback. Select the fallback threshold from calibration score-gap quantiles through the 25th percentile. The known-role control isolates search and citation. Timing starts at the request string and includes query formatting, inference, warm file loads as needed and JSON serialization. Model loading, training, service transport and visibility checks sit outside the measurement. The input list supplies the clause boundaries.

## Results

| Route, chat-format run | Exact requested roles / 120 | Exact ranked bundles / 120 | Queries with unsupported recommendations | P95 |
| --- | ---: | ---: | ---: | ---: |
| Dense glossary | 9 | 17 | 28 | 8.65 ms |
| Ridge | 38 | 43 | 53 | 8.62 ms |
| Batched reader | 47 | 60 | 44 | 72.45 ms |
| Ridge with reader fallback | 50 | 58 | 48 | 59.60 ms |
| Known roles | 120 | 120 | 0 | 0.09 ms |

The reader's chat template improves the complete response from 16 to 60 out of 120 relative to the plain prompt. The fallback invoked the reader on 51 requests. It traded latency for two fewer exact responses and four more queries with unsupported recommendations than always using the chat reader. It selected the same threshold as the plain-format run, based on just 14 calibration phrases. Treat these figures as a scoped diagnostic, with repeated phrases and controlled role mixes. Known-role accuracy shows consistency with the publisher-backed reference, while the language routes quantify how often incorrect role interpretation changes the cited expertise.

## Reproduce

On a Slurm GPU compute node:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.live_expert_bundles \
  PUBLISHER_FIXTURE NISO_EXAMPLES ROLE_TRAINING EMBEDDING_CHECKPOINT \
  READER_CHECKPOINT SCRATCH_DIRECTORY NEW_RESULT_JSON --chat-reader
```

Extract the NISO example JSON from the pinned reference using [the parser](../../experiments/credit/parse_niso_examples.py). Omit `--chat-reader` for the plain-format run. [Aggregate results](results.json) include both runs and model/input hashes. The plain run records its original script hash. NISO phrase copies and per-query traces stay outside the release.
