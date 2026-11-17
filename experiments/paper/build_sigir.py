"""Compile the canonical SIGIR LaTeX draft, retaining one PDF and build report."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def build():
    root=Path(__file__).resolve().parents[2];paper=root/'docs/paper/sigir';out=root/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    with (out/'build.log').open('w') as log:result=subprocess.run(['tectonic','-Z','search-path='+str(paper/'template'),'--outdir',str(out),'--keep-logs','main.tex'],cwd=paper,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('LaTeX failed; inspect output/pdf/build.log')
    pdf=out/'expert-search-sigir-draft.pdf';(out/'main.pdf').replace(pdf)
    log=(out/'build.log').read_text();warnings=[line for line in log.splitlines() if 'Overfull' in line or 'undefined' in line.lower() or 'Missing character' in line]
    files=[paper/'main.tex',paper/'appendix.tex',paper.parent/'references.bib'];report=dict(pdf=str(pdf.relative_to(root)),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),engine=subprocess.check_output(['tectonic','--version'],text=True).strip(),sources_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},layout_warning_count=len(warnings),layout_warnings=warnings,scope='Compiled local draft, not submitted; visual QA required. Official ACM acmart sigconf/anonymous class resolved through TeX distribution.')
    (out/'build-manifest.json').write_text(json.dumps(report,indent=2))
    for pattern in ['main.blg','main.log','main.aux','main.bbl','main.out']:
        path=out/pattern
        if path.exists():path.unlink()
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources_sha256','layout_warnings']},indent=2))


if __name__=='__main__':build()
