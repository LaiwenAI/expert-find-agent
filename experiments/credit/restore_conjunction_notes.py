"""Restore original author notes from pinned publisher XML objects."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
from pathlib import Path
from urllib.request import urlopen
import time
from xml.etree import ElementTree as ET

from experiments.credit.elife_role_nomination import REVISION, parse
from experiments.credit.engineering_cross_source_panel import owner_key
from experiments.followir.core17_reranking import write_json


def restore(doi, source):
    name = source["filename"]
    url = f"https://raw.githubusercontent.com/elifesciences/elife-article-xml/{REVISION}/articles/{name}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=25) as response:
                raw = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if sha256(raw).hexdigest() != source["sha256"] or sha1(
        f"blob {len(raw)}\0".encode() + raw
    ).hexdigest() != source["git_blob"]:
        raise ValueError(f"Changed publisher object: {name}")
    article, error = parse(raw, name)
    if error or not article["declaration_complete"] or article["doi"] != doi:
        raise ValueError(f"Source no longer resolves to a complete declaration: {name}")
    root = ET.fromstring(raw)
    byline = root.findall("./front/article-meta/contrib-group/contrib[@contrib-type='author']")
    notes = {note.get("id"): " ".join("".join(note.itertext()).split())
             for note in root.findall("./back//fn[@fn-type='con']")}
    resolved = {}
    for i, (author, reference) in enumerate(zip(byline, article["authors"], strict=True)):
        if not reference["person"]:
            continue
        belongs = {item for link in author.findall("xref[@ref-type='fn']") for item in link.get("rid", "").split()}
        direct = [" ".join("".join(role.itertext()).split()) for role in author.findall("role")]
        text = ", ".join(direct or [notes[key] for key in sorted(belongs & notes.keys())])
        alias = f"Author {i + 1}"
        if not text or reference["person"] in resolved:
            raise ValueError("Missing original declaration or duplicate author identity")
        resolved[reference["person"]] = dict(alias=alias, text=f"{alias}: {text}", roles=reference["roles"],
                                              engineering_key=owner_key(doi, i, author))
    return doi, resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    src = json.loads(args.manifest.read_bytes())
    articles = {}
    for panel in src["panels"].values():
        if articles.keys() & panel["articles"].keys():
            raise ValueError("eLife panels overlap by source")
        articles.update(panel["articles"])
    with ThreadPoolExecutor(max_workers=10) as pool:
        data = dict(pool.map(lambda row: restore(*row), sorted(articles.items())))
    write_json(args.output, data)
    print(f"Verified {len(data)} raw XML objects")


if __name__ == "__main__":
    main()
