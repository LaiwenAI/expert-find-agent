"""Build the single signed SIGIR manuscript from the canonical LaTeX source."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/paper/sigir"
DEFAULT_AUTHORS = ROOT / "artifacts/sigir-author-copy/authors.tex"
DEFAULT_OUTPUT = ROOT / "output/pdf/expert-search-sigir-author-draft.pdf"


def build(authors, output, frozen_results):
    if not authors.is_file():
        raise FileNotFoundError(f"Local author metadata is missing: {authors}. Pass --authors to select it.")
    if output.suffix.lower() != ".pdf":
        raise ValueError("Manuscript output must be a PDF")
    if not frozen_results:
        subprocess.run([sys.executable, str(Path(__file__).with_name("export_results.py"))],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so replacement is atomic even for a custom output.
    with TemporaryDirectory(prefix=".sigir-", dir=output.parent) as directory:
        build_dir = Path(directory)
        paper = build_dir / "sigir"
        shutil.copytree(PAPER, paper)
        shutil.copy2(PAPER.parent / "references.bib", build_dir)
        shutil.copyfile(authors, paper / "authors.tex")
        result = subprocess.run(
            ["tectonic", "-Z", f"search-path={paper / 'template'}", "--outdir", str(build_dir), "main.tex"],
            cwd=paper, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if result.returncode:
            print(result.stdout, file=sys.stderr, end="")
            result.check_returncode()
        pdf = build_dir / "main.pdf"
        pdf.chmod(0o600)
        report = dict(
            pdf=str(output), pdf_sha256=sha256(pdf.read_bytes()).hexdigest(),
            layout_warnings=[line for line in result.stdout.splitlines()
                             if any(marker in line.lower() for marker in ("overfull", "undefined", "missing character"))],
        )
        pdf.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", type=Path, default=DEFAULT_AUTHORS,
                        help="Local acmart author and affiliation statements.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Stable signed manuscript path.")
    parser.add_argument("--frozen-results", action="store_true",
                        help="Use the committed result snapshot without refreshing local data.")
    args = parser.parse_args()
    build(args.authors.resolve(), args.output.resolve(), args.frozen_results)
