"""Freeze a topic- and article-disjoint sample of official relevance judgments."""

import argparse
from collections import defaultdict
from gzip import open as gzip_open
from hashlib import sha256
import json
from pathlib import Path
from xml.etree import ElementTree


BEIR_SHA256 = "1bcbb294282f84a6c38d3881af2c7f392afcf88c1247e7599546cb33f33cb48c"
PER_GRADE = {0: 16, 1: 8, 2: 16}
SEED = "topic-calibration-20261121"


def hash_order(*parts):
    return sha256(":".join((SEED, *map(str, parts))).encode()).digest()


def file_digest(path, algorithm):
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            algorithm.update(chunk)
    return algorithm.hexdigest()


def normalize_title(title):
    return " ".join(title.casefold().split())


def build(archive, topics_path, qrels_path, output, manifest):
    if output.exists() or manifest.exists():
        raise ValueError("Use fresh output and manifest paths")
    if file_digest(archive, sha256()) != BEIR_SHA256:
        raise ValueError("BEIR archive check failed")
    topics = {}
    for item in ElementTree.parse(topics_path).getroot().findall("topic"):
        topic = int(item.attrib["number"])
        topics[topic] = " ".join((item.findtext("query") or "").split())
    if set(topics) != set(range(1, 51)) or not all(topics.values()):
        raise ValueError("Unexpected official topic file")
    qrels = defaultdict(list)
    pairs = set()
    for line in qrels_path.read_text().splitlines():
        topic, _, doc, grade = line.split()
        topic, grade = int(topic), int(grade)
        if grade == -1:
            continue
        if (topic, doc) in pairs or topic not in topics or grade not in PER_GRADE:
            raise ValueError("Unexpected duplicate or invalid official grade")
        pairs.add((topic, doc))
        qrels[topic, grade].append(doc)
    wanted = {doc for _, doc in pairs}
    records = {}
    with gzip_open(archive, "rb") as corpus:
        for line in corpus:
            row = json.loads(line)
            if row["_id"] in wanted and row["title"].strip() and row["text"].strip():
                records[row["_id"]] = (row["title"].strip(), row["text"].strip())

    groups = sorted(topics, key=hash_order)
    used_docs = set()
    used_titles = set()
    rows = {}
    for split, selected in (("calibration", groups[:30]), ("held", groups[30:])):
        choices = []
        for topic in selected:
            for grade, count in PER_GRADE.items():
                kept = []
                for doc in sorted(qrels[topic, grade], key=lambda doc: hash_order(topic, grade, doc)):
                    if doc not in records or doc in used_docs:
                        continue
                    title, text = records[doc]
                    normalized = normalize_title(title)
                    if normalized in used_titles:
                        continue
                    kept.append(dict(topic=topic, doc=doc, grade=grade, title=title, abstract=text))
                    used_docs.add(doc)
                    used_titles.add(normalized)
                    if len(kept) == count:
                        break
                if len(kept) != count:
                    raise ValueError(f"Insufficient distinct sampled articles for topic {topic}, grade {grade}")
                choices.extend(kept)
        rows[split] = choices

    data = dict(topics=topics, splits=rows)
    output.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps(dict(
        panel_sha256=file_digest(output, sha256()),
        archive_sha256=BEIR_SHA256, topics_sha256=file_digest(topics_path, sha256()),
        qrels_sha256=file_digest(qrels_path, sha256()),
        splits={split: [[row["topic"], row["doc"], row["grade"]] for row in choices]
                for split, choices in rows.items()},
    ), separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "topics_path", "qrels_path", "output", "manifest"):
        parser.add_argument(field, type=Path)
    build(**vars(parser.parse_args()))
