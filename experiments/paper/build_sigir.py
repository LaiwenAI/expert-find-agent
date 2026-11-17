"""Refresh artifact-backed numbers and compile separate SIGIR draft/supplement."""
import hashlib
import json
from pathlib import Path
import subprocess


def build():
    root = Path(__file__).resolve().parents[2]
    paper = root / 'docs/paper/sigir'
    out = root / 'output/pdf'
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(root / '.venv-local/bin/python'), str(Path(__file__).with_name('export_results.py'))], cwd=root, check=True, stdout=subprocess.DEVNULL)
    reports = []
    logs = []
    for stem, name in [('main', 'expert-search-sigir-draft'), ('supplement', 'expert-search-sigir-supplement')]:
        result = subprocess.run(['tectonic', '-Z', 'search-path=' + str(paper / 'template'), '--outdir', str(out), '--keep-logs', stem + '.tex'], cwd=paper, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs.append(result.stdout)
        (out / 'build.log').write_text('\n'.join(logs))
        if result.returncode:
            raise RuntimeError('LaTeX failed; inspect output/pdf/build.log')
        pdf = out / (name + '.pdf')
        (out / (stem + '.pdf')).replace(pdf)
        warnings = [line for line in result.stdout.splitlines() if 'Overfull' in line or 'undefined' in line.lower() or 'Missing character' in line]
        reports.append(dict(pdf=str(pdf.relative_to(root)), pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), layout_warning_count=len(warnings), layout_warnings=warnings))
        for suffix in ['blg', 'log', 'aux', 'bbl', 'out']:
            (out / (stem + '.' + suffix)).unlink(missing_ok=True)
    sources = list(paper.rglob('*.tex')) + [paper.parent / 'references.bib', paper / 'generated/results.json']
    report = dict(documents=reports, engine=subprocess.check_output(['tectonic', '--version'], text=True).strip(), sources_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, scope='Separate English main draft and supplementary material; not submitted. Vendored ACM acmart selected with explicit search path. Visual QA required after each build.')
    (out / 'build-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    build()
