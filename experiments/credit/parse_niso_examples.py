"""Extract the canonical glossary and examples from the pinned NISO table."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from experiments.credit.plos_role_inventory import ROLES


PDF_SHA256 = "69083598cf1625d28a35fe27734a30ee44227ed6ef54b18cbadb061f8d0c15d0"
EXAMPLE_COUNTS = (4, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 2, 2, 3)
DOI = "10.5281/zenodo.18421449"


def crop_text(pdf, *options):
    return subprocess.check_output(["pdftotext", "-layout", *options, str(pdf), "-"], text=True)


def definitions(left_text):
    keys = {name.casefold(): name for name in ROLES}
    glossary = {}
    current = None
    for line in left_text.splitlines():
        value = line.strip()
        if not value:
            continue
        if value.casefold() in keys:
            current = keys[value.casefold()]
            if current in glossary:
                raise ValueError("Repeated taxonomy definition")
            glossary[current] = []
        elif current is not None:
            glossary[current].append(value)
    if tuple(glossary) != ROLES or any(not lines for lines in glossary.values()):
        raise ValueError("Incorrect taxonomy column ordering")
    return {role: " ".join(lines) for role, lines in glossary.items()}


def examples(right_text, exception_counts):
    paragraphs = []
    for line in right_text.splitlines():
        value = line.strip()
        if value.startswith("§ "):
            paragraphs.append([value[2:]])
        elif value and paragraphs:
            paragraphs[-1].append(value)
    if len(paragraphs) != sum(exception_counts):
        raise ValueError("Incorrect example column ordering")
    rows = []
    start = 0
    for role, count in zip(ROLES, exception_counts, strict=True):
        for i, paragraph in enumerate(paragraphs[start:start + count], 1):
            rows.append(dict(role=role, ordinal=i, text=" ".join(paragraph)))
        start += count
    return rows


def main(pdf, output):
    if sha256(pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("Changed NISO reference PDF")
    first = crop_text(pdf, "-x", "38", "-y", "143", "-W", "138", "-H", "670", "-f", "1", "-l", "1")
    second = crop_text(pdf, "-x", "38", "-y", "35", "-W", "138", "-H", "530", "-f", "2", "-l", "2")
    right = crop_text(pdf, "-x", "179", "-y", "0", "-W", "387", "-H", "842")
    if [page.count("§") for page in right.split("\f")] != [30, 19, 0]:
        raise ValueError("Unexpected example bullet layout")
    roles = definitions(first + second)
    rows = examples(right, EXAMPLE_COUNTS)
    if len({row["text"] for row in rows}) != len(rows):
        raise ValueError("Duplicated example text")
    with output.open("x") as handle:
        json.dump(dict(doi=DOI, pdf_sha256=PDF_SHA256, definitions=roles, examples=rows), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.pdf, args.output)
