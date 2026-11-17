# SIGIR manuscript

`main.tex` is the canonical editable English manuscript; `appendix.tex` contains English protocols and limitations. Obsolete Markdown manuscripts have been removed. The main file uses ACM `acmart` with `sigconf,natbib=true,anonymous=true`, following the official SIGIR2027 full-paper guidance:
https://sigir2027.org/pages/submit-full.html

Build from repository root:

```bash
python3 experiments/paper/build_sigir.py
```

Output: `output/pdf/expert-search-sigir-draft.pdf`. Tectonic is required. The unmodified current ACM class is generated from the included original CTAN `acmart.dtx`/`acmart.ins`; its LPPL terms are retained separately from repository MIT. No fonts/margins were edited. `template-provenance.json` records the source and checksums. This is a research draft, not a submitted or accepted paper.
