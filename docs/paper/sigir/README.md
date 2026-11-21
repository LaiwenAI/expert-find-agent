# SIGIR manuscript

`main.tex` is the canonical editable English manuscript; `supplement.tex` builds the separate supplementary PDF from `appendix.tex`, containing artifact pointers and earlier development comparisons. The main draft studies contributor-constrained first-round evidence allocation under a fixed reading budget. It reports coverage against fixed and sequential controls, exact execution checks, and a shared-corpus feasibility ablation for 4B role rechecks. The supplement identifies the reproducible runs and separates label-supported coverage from human evidence judgments. The current paper scope is the structured publication benchmark and fixed-inventory verification. The complete enterprise quality claim still requires independent adjudication and an end-to-end comparison at matched risk and cost. The main file uses ACM `acmart` with `sigconf,natbib=true,anonymous=true,pbalance=true`, following the official SIGIR2027 full-paper guidance:
https://sigir2027.org/pages/submit-full.html

Build from repository root:

```bash
python3 experiments/paper/build_sigir.py
```

Outputs: `output/pdf/expert-search-sigir-draft.pdf` and `output/pdf/expert-search-sigir-supplement.pdf`. Every build refreshes artifact-backed result macros and a hashed synthesis-count snapshot in `generated/`; it does not make API calls or generate labels. Tectonic is required. A public source-only checkout can use `python3 experiments/paper/build_sigir.py --frozen-results` to compile the committed snapshot without private experimental artifacts; default builds refresh from local artifacts and fail if inputs are unavailable. The unmodified current ACM class is generated from the included original CTAN `acmart.dtx`/`acmart.ins`; its LPPL terms are retained separately from repository MIT. The class-supported `pbalance` option balances columns; no fonts/margins were edited. `template-provenance.json` records the source and checksums. This is a research draft, not a submitted or accepted paper.

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
