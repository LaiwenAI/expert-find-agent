# Multi-hop evidence retrieval and sufficiency readout

Use [MuSiQue-Full](https://aclanthology.org/2022.tacl-1.31/) as a
public proxy for queries whose requirements cannot be checked with one
source. Its answerable and unanswerable views share the same question
and swap out at least one of two to four annotated support paragraphs.
The experiment evaluates retrieval and evidence sufficiency for the annotated answers.

## Design

Hash question IDs to select 464 pairs from the official development
split. The first 80 calibrate each reader. Evaluate the same other 384
once per arm (212 two-hop, 108 three-hop, and 64 four-hop). The public
official test set is unused. The shared retrieval catalog pools positive
contexts for these pairs, deduplicated by exact title and paragraph,
for 6,396 paragraphs. Each positive request searches that shared pool
rather than its own 20 documents. The contrastive negatives are read
only from their official negative contexts. Their paired positive
documents in the catalog are not returned as negative inputs.

Use BM25 (title counted twice), frozen Qwen3-Embedding-0.6B and
reciprocal-rank fusion of the two. All are separately reported as
ranking baselines. Candidate count is four, six, or twelve. The diverse
route prefers new titles within the first 24 hybrid results. The linked
route begins with the first two hybrid results. At every other
subsequent position, try the highest-ranked unused cross-title neighbor
of either of the last two reads, within the initial hybrid top 256.
Use a literal normalized title mention of two to eight words as an edge
in either direction. Fill remaining slots from the shared ranked list.
No answer or oracle decomposition is used in retrieval.

The reader sees either natural retrieval, an oracle replacement that
inserts the missing annotated passages by replacing the lowest-ranked
distractors, or the official unanswerable paired view. The paired
diagnostic keeps its surviving gold paragraphs and fills the remaining
slots with its own distractors. In four-paragraph runs it also sees
dense retrieval and the annotated-only view. Only diagnostic views use
the labels. Oracle replacement holds paragraph count, with variable token
counts and content order. Report clipping explicitly and compute reader
quality from unclipped inputs. Each reader has a separate threshold,
calibrated to at most 10% acceptance on the 80 calibration pairs'
unclipped missing-evidence views. The 0.6B relevance reranker and 4B
instruction reader use different model prompts and output heads.

## Retrieval diagnostic

| Retrieval | Complete at 4 | Complete at 6 | All source titles at 12 | Complete at 12 |
|---|---:|---:|---:|---:|
| Lexical | 41 | 56 | 101 | 89 |
| Dense | 72 | 98 | 140 | 124 |
| Hybrid | 71 | 89 | 131 | 124 |
| Source diversity | 66 | 78 | 142 | 104 |
| Linked reads | 84 | 97 | 148 | 136 |

At twelve reads, 236 of 384 linked cases still miss at least one
annotated source title. Another 12 contain all annotated titles
and miss a particular support paragraph. Compared with hybrid at
that budget, linked reads gain 19 complete sets and lose seven.
The query encoding and shared dense-plus-lexical ranking take about
9.94 ms at P95, with 0.114 ms post-ranking P95 for linked reads. This
timing excludes network, generation and initial index construction.

### Propagation on the same title graph

A follow-up control starts with the first 16 hybrid passages and
runs three personalized propagation steps with 40% restart. It
uses the same undirected title edges, with 15,331 directed entries
including self-loops for isolated passages. The control uses exact
mentions and a small fixed iteration budget. Graph propagation is an
established retrieval baseline. [HippoRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html)

Propagation found 79, 106, and 148 complete sets at depths 4, 6,
and 12. Linked reads found 84, 97, and 136, respectively. Linked
minus dense paired 95% intervals were −0.52 to +6.77 percentage
points at depth four and −1.04 to +7.29 at depth twelve. Against
propagation, the intervals were −1.30 to +3.91 and −6.25 to 0.00.
The exploratory graph control reuses the same evaluation cases and
its intervals are descriptive. Use one-shot propagation as a strong
reference for any sequential action policy under the same read budget.

## Matched four-paragraph reader comparison

All counts below exclude truncated inputs separately for each reader.
An accepted incomplete natural view is an evidence-coverage warning.
Acceptance of the official unanswerable contrast is measured against its
dataset label.

| View or comparison | Qwen3-Reranker-0.6B | Qwen3-4B-Instruct-2507 |
|---|---:|---:|
| Complete linked outputs accepted | 60/78 | 51/78 |
| Oracle replacement accepted | 180/371 | 214/373 |
| Annotated-only accepted | 195/381 | 229/381 |
| Official missing-evidence contrast accepted | 48/378 | 41/378 |
| Matched pair with complete accepted and missing rejected | 142/367 | 173/369 |
| Reject-to-accept after restoring missing evidence to an incomplete hybrid view | 51/296 | 105/297 |
| Accept-to-reject under the same restoration | 15/296 | 10/297 |
| Natural linked-view reader P95 | 12.89 ms | 54.32 ms |

The larger reader gains after support is supplied explicitly, and accepts
fewer of the complete natural linked views. The four-paragraph comparison
also leaves substantial losses when only the gold paragraphs are shown.
This motivates separating coverage from verifier calibration and assessing
any expensive reader only on cases it could improve. These counts
characterize evidence sufficiency for the benchmark questions and the
measured reader stage.

## Reproduce

Use the [publisher's official archive](https://github.com/stonybrooknlp/musique)
and pinned local Qwen checkpoints. The three frozen revisions are
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` (embedding),
`e61197ed45024b0ed8a2d74b80b4d909f1255473` (reranker), and
`cdbee75f17c01a7cc42f958dc650907174af0554` (instruction reader).
Use the repository's existing model provenance files and run on a Slurm
CUDA node.

```sh
study=benchmarks/musique-evidence-gap-v1
PYTHONPATH=src:. python "$study/musique_search.py" ARCHIVE EMBEDDING RERANKER results.json
PYTHONPATH=src:. python "$study/musique_search.py" ARCHIVE EMBEDDING RERANKER reader-06b.json --reader-paragraphs 4
PYTHONPATH=src:. python "$study/musique_search.py" ARCHIVE EMBEDDING CHAT4B reader-4b.json --reader-paragraphs 4 --instruction-reader
PYTHONPATH=src:. python "$study/musique_graph_control.py" ARCHIVE EMBEDDING graph-comparison.json
```

The six-paragraph archive comes from ws3 job 45467. Job 45473 replays
the same retrieval outputs while testing the two four-paragraph readers.
Job 45490 reproduces dense, hybrid, and linked coverage and adds propagation.
