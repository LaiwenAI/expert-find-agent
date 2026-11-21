# Three Reads, One Contributor

`main.tex` is the single editable English manuscript. The draft studies
single-contributor evidence allocation with a fixed declaration-read budget.
The standalone appendix source preserves protocols and development
comparisons for reference.

## Build the anonymous draft

Build with Tectonic from the checkout that contains the source:

```bash
python3 experiments/paper/build_sigir.py --frozen-results
```

The command replaces `output/pdf/expert-search-sigir-anonymous-draft.pdf`
after compilation succeeds. Set `--output` when the shared manuscript
lives outside the source checkout.
Omit `--frozen-results` to refresh the tables from local experiment
artifacts. The build stages temporary inputs and removes them on exit.
The original ACM template and its license are kept under `template/`.

## Writing references

The draft uses Yang et al.'s [SIM-RAG, SIGIR 2025](https://doi.org/10.1145/3726302.3730018)
as its structural reference. The introduction makes an agent decision
failure concrete. The method starts with the inference flow, then
derives the mechanism and its assumptions. The evaluation specifies
tasks, implementation and baselines before reporting matched-budget
results, work and latency ablations, and transfer failures. The roles,
method, experiments, figures, and wording in this draft are our own.

Salemi and Zamani's [eRAG, SIGIR 2024](https://doi.org/10.1145/3626772.3657957)
is the reference for separating a retrieval-stage signal from a
downstream outcome. SIGIR awarded it [Best Short Paper](https://sigir-2024.github.io/best_papers.html).
Crossref's DOI records reported 17 references to SIM-RAG and 182 to
eRAG on 2026-11-21. Counts use the published DOIs and can differ from
metrics that consolidate preprint versions.

The observed-role continuation ablation reports proof availability,
issued expert evidence, and latency as separate endpoints. Its paired
controls and reproduction commands are in the
[benchmark](../../../benchmarks/open-topic-observed-repair-v1).
The figure and equation sources are editable LaTeX.

## Main figures and comparison tables

The main paper has four figures and five tables. The figures cover the
agent architecture, issuance and quality uncertainty, the cost of
query-local filtering, and exact source-read savings. The
[pre-read benchmark](../../../benchmarks/pre-read-proof-bound-v1)
compares against both fixed-two dispatch and post-read certification
on the same ranked plans, using the two frozen eLife projections.

## Scope of the supplement

The supplement contains the protocols and ablations that support this
draft. Earlier development notes remain in the
[archived extended appendix](https://github.com/LaiwenAI/expert-find-agent/blob/af8508704af8983d2b54da28f90ef0d2df37c058/docs/paper/sigir/appendix.tex).
The individual benchmark folders retain executable commands and
aggregate receipts. The public draft records the evaluation status.
The frozen review packets and their labels stay outside the source tree.
