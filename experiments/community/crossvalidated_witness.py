"""Frozen community duplicate links as author-and-source retrieval targets."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from time import sleep
from urllib.parse import urlencode
from urllib.request import urlopen

ENDPOINT = "https://api.stackexchange.com/2.3"
FILTER = "!)RL-JogHwoZuazx.6_xtpS5P"
DUPLICATE_PAGES = 4
BACKGROUND_PAGES = 4


def timestamp(year):
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())


def fetch(folder, endpoint, **params):
    url = f"{ENDPOINT}/{endpoint}?" + urlencode(dict(site="stats", pagesize=100, filter=FILTER, **params))
    path = folder / f"{sha256(url.encode()).hexdigest()[:16]}.json"
    if not path.exists():
        with urlopen(url, timeout=25) as response:
            raw = response.read()
        page = json.loads(raw)
        if "error_id" in page:
            raise ValueError(f"Stack Exchange API error {page['error_id']}")
        if "backoff" in page:
            sleep(page["backoff"])
        path.write_bytes(raw)
    return json.loads(path.read_bytes())["items"]


def lookup(folder, endpoint, ids, field):
    ids = sorted(set(ids))
    values = {}
    for offset in range(0, len(ids), 100):
        batch = ";".join(map(str, ids[offset:offset + 100]))
        values.update({row[field]: row for row in fetch(folder, f"{endpoint}/{batch}")})
    return values


class Text(HTMLParser):
    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.feed(markup)
        self.close()

    def handle_data(self, fragment):
        self.parts.append(fragment)

    def plain(self):
        visible = re.sub(r"https?://\S+", " ", " ".join(self.parts))
        return " ".join(visible.split())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.write("\n")


def capture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    q2024 = [
        row
        for i in range(1, DUPLICATE_PAGES + 1)
        for row in fetch(folder, "search/advanced", closed="true", sort="creation", page=i,
                         fromdate=timestamp(2024), todate=timestamp(2025)-1)
        if row.get("closed_reason", "").casefold() == "duplicate"
    ]
    links = {row["question_id"]: {
        x["question_id"] for x in row.get("closed_details", {}).get("original_questions", [])
    } for row in q2024}
    cutoff = timestamp(2023)
    background = {
        row["question_id"]: row
        for i in range(1, BACKGROUND_PAGES + 1)
        for row in fetch(folder, "search/advanced", accepted="true", sort="votes", page=i, todate=cutoff-1)
    }
    canonical = lookup(folder, "questions", set().union(*links.values()), "question_id")
    questions = {**background, **canonical}
    answers = lookup(folder, "answers", (q["accepted_answer_id"] for q in questions.values()
                                        if q.get("accepted_answer_id")), "answer_id")
    documents = {}
    for qid, post in questions.items():
        aid = post.get("accepted_answer_id")
        answer = answers.get(aid, {})
        owner = answer.get("owner", {}).get("user_id")
        if (not owner or answer.get("question_id") != qid or not answer.get("is_accepted")
                or not post.get("body") or not answer.get("body")
                or max(post["creation_date"], answer["creation_date"], answer.get("last_edit_date", 0),
                       post.get("last_edit_date", 0)) >= cutoff):
            continue
        body = Text(answer["body"]).plain()
        if not body:
            continue
        documents[aid] = dict(qid=qid, owner=owner, question=unescape(post["title"]),
                              question_text=Text(post["body"]).plain(), answer=body)
    by_question = {row["qid"]: aid for aid, row in documents.items()}
    queries = []
    for post in sorted(q2024, key=lambda row: row["question_id"]):
        qid = post["question_id"]
        relevant = sorted(by_question[target] for target in links[qid] if target in by_question)
        title = unescape(post["title"])
        if not relevant or not post.get("body") or post["creation_date"] < timestamp(2024):
            continue
        source_link = any(re.search(rf"/(?:questions|q)/{documents[aid]['qid']}(?:[/?#\"']|$)", post["body"])
                          for aid in relevant)
        if source_link:
            continue
        if any(" ".join(title.casefold().split()) == " ".join(documents[aid]["question"].casefold().split())
               for aid in relevant):
            continue
        queries.append(dict(qid=qid, title=title, query=Text(post["body"]).plain(), answers=relevant))
    if not queries:
        raise ValueError("No eligible duplicate questions")
    text = folder / "snapshot.json"
    write(text, dict(queries=queries, documents=documents, background=len(background), duplicates=len(q2024),
                     before=cutoff, metadata_source=ENDPOINT))
    manifest = dict(
        snapshot_sha256=sha256(text.read_bytes()).hexdigest(),
        queries=[dict(question=q["qid"], accepted_answers=q["answers"]) for q in queries],
        candidate_answers=sorted(documents),
        receipts={path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(folder.glob("*.json"))
                  if path != text},
    )
    write(folder / "manifest.json", manifest)
    print(f"{len(queries)} queries, {len(documents)} eligible accepted answers", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    capture(parser.parse_args().folder)
