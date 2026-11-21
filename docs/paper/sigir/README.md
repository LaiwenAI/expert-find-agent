# SIGIR manuscript

`main.tex` is the canonical editable English manuscript; `supplement.tex` builds the separate supplementary PDF from `appendix.tex`, containing artifact pointers and earlier development comparisons. The main draft studies contributor-constrained first-round evidence allocation under a fixed reading budget. It reports coverage against fixed and sequential controls, exact execution checks, and a shared-corpus feasibility ablation for 4B role rechecks. The supplement identifies the reproducible runs and separates label-supported coverage from human evidence judgments. The current paper scope is the structured publication benchmark and fixed-inventory verification. The complete enterprise quality claim still requires independent adjudication and an end-to-end comparison at matched risk and cost. The main file uses ACM `acmart` with `sigconf,natbib=true,anonymous=true,pbalance=true`, following the official SIGIR2027 full-paper guidance:
https://sigir2027.org/pages/submit-full.html

Build from repository root:

```bash
python3 experiments/paper/build_sigir.py
```

Outputs: `output/pdf/expert-search-sigir-draft.pdf` and `output/pdf/expert-search-sigir-supplement.pdf`. Every build refreshes artifact-backed result macros and a hashed synthesis-count snapshot in `generated/`; it does not make API calls or generate labels. Tectonic is required. A public source-only checkout can use `python3 experiments/paper/build_sigir.py --frozen-results` to compile the committed snapshot without private experimental artifacts; default builds refresh from local artifacts and fail if inputs are unavailable. The unmodified current ACM class is generated from the included original CTAN `acmart.dtx`/`acmart.ins`; its LPPL terms are retained separately from repository MIT. The class-supported `pbalance` option balances columns; no fonts/margins were edited. `template-provenance.json` records the source and checksums. This is a research draft, not a submitted or accepted paper.
