# SIGIR manuscript

`main.tex` is the canonical editable English manuscript; `supplement.tex` builds the separate supplementary PDF from `appendix.tex`, containing English protocols and limitations. Obsolete Markdown manuscripts have been removed. The main file uses ACM `acmart` with `sigconf,natbib=true,anonymous=true`, following the official SIGIR2027 full-paper guidance:
https://sigir2027.org/pages/submit-full.html

Build from repository root:

```bash
python3 experiments/paper/build_sigir.py
```

Outputs: `output/pdf/expert-search-sigir-draft.pdf` and `output/pdf/expert-search-sigir-supplement.pdf`. Every build refreshes artifact-backed result macros and a hashed synthesis-count snapshot in `generated/`; it does not make API calls or generate labels. Tectonic is required. The unmodified current ACM class is generated from the included original CTAN `acmart.dtx`/`acmart.ins`; its LPPL terms are retained separately from repository MIT. No fonts/margins were edited. `template-provenance.json` records the source and checksums. This is a research draft, not a submitted or accepted paper.
