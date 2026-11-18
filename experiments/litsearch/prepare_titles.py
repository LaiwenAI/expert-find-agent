"""Project LitSearch Parquet onto titles, abstracts, IDs and released judgments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryFile
from urllib.request import Request, urlopen

import pyarrow.parquet as parquet


REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
ROOT = f"https://huggingface.co/datasets/princeton-nlp/LitSearch/resolve/{REVISION}"
QUERY_SHA256 = "38cdbe4a6b7a7f5776d08055bfbf6e4511000aa2a523866592b4bb273388b914"
SIZES = (275244429, 195618564, 194994867, 198835360, 194313426, 198077458)
COLUMNS = ("corpusid", "title", "abstract")


def get_range(url, first, last):
    request = Request(f"{url}?segment={first}-{last}", headers={"Range": f"bytes={first}-{last}"})
    for retry in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                content = response.read()
                if not response.headers.get("Content-Range", "").startswith(f"bytes {first}-{last}/"):
                    raise OSError("Unexpected range response")
            if len(content) != last - first + 1:
                raise OSError("Truncated range response")
            return content
        except OSError:
            if retry == 2:
                raise


def project(spec):
    shard, size = spec
    url = f"{ROOT}/corpus_clean/full-{shard:05d}-of-00006.parquet"
    footer_first = size - 65536
    footer = get_range(url, footer_first, size - 1)
    if footer[-4:] != b"PAR1" or int.from_bytes(footer[-8:-4], "little") + 8 > len(footer):
        raise ValueError("Unexpected Parquet footer")
    with TemporaryFile() as sparse:
        sparse.write(b"PAR1")
        sparse.seek(footer_first)
        sparse.write(footer)
        sparse.seek(0)
        meta = parquet.ParquetFile(sparse).metadata
        for n in range(meta.num_row_groups):
            group = meta.row_group(n)
            spans = []
            for i in range(group.num_columns):
                column = group.column(i)
                if column.path_in_schema in COLUMNS:
                    offsets = (column.dictionary_page_offset, column.data_page_offset)
                    start = min(x for x in offsets if x is not None and x > 0)
                    spans.append((start, start + column.total_compressed_size - 1))
            if len(spans) != len(COLUMNS):
                raise ValueError("Unexpected title/abstract columns")
            a, b = min(x[0] for x in spans), max(x[1] for x in spans)
            if b - a + 1 > 6_000_000:
                raise ValueError("Projection would include an unexpectedly large range")
            content = get_range(url, a, b)
            sparse.seek(a)
            sparse.write(content)
        sparse.seek(0)
        result = parquet.ParquetFile(sparse).read(columns=COLUMNS).to_pylist()
    print(f"Projected shard {shard}: {len(result)} papers", flush=True)
    return result


def prepare(query, output):
    if output.exists():
        raise FileExistsError(output)
    if sha256(query.read_bytes()).hexdigest() != QUERY_SHA256:
        raise ValueError("Unexpected query snapshot")
    with ThreadPoolExecutor(max_workers=6) as work:
        papers = [row for group in work.map(project, enumerate(SIZES)) for row in group]
    questions = parquet.read_table(query).to_pylist()
    doc_ids = {row["corpusid"] for row in papers}
    if len(questions) != 597 or len(doc_ids) != len(papers) or not all(
            row["corpusids"] and set(row["corpusids"]) <= doc_ids for row in questions):
        raise ValueError("Unexpected LitSearch inventory")
    payload = dict(revision=REVISION, query_sha256=QUERY_SHA256,
                   papers=papers, questions=questions)
    with output.open("xb") as target:
        with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as archive:
            archive.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.query, args.output)
