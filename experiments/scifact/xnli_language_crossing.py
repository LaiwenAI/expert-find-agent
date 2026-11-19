"""Measure direct and cross-language NLI on aligned English and Chinese XNLI test pairs."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
import os
from pathlib import Path

PARQUET_SHA = "599e3a0191403f19cbe802afdf69841152000b41eaed725e4f463d432c0ffb49"
PROJECTION_SHA = "766f10336767b2a9da9d72d0273bcc817b339e3311e2ace99dffa8c638fcece1"
LABELS = {"entailment", "contradiction", "neutral"}
LANGUAGES = ("en", "zh")
FORMS = tuple((premise, hypothesis) for premise in LANGUAGES for hypothesis in LANGUAGES)
COUNT = 5010


def prepare(source, output):
    import pyarrow.parquet as parquet

    with source.open("rb") as stream:
        checksum = file_digest(stream, "sha256").hexdigest()
    if output.exists() or checksum != PARQUET_SHA:
        raise ValueError("Require the pinned XNLI all-languages test parquet and a fresh output")
    table = parquet.read_table(source)
    dataset = json.loads(table.schema.metadata[b"huggingface"])
    names = tuple(dataset["info"]["features"]["label"]["names"])
    if len(table) != COUNT or set(names) != LABELS:
        raise ValueError("Changed corpus structure")
    with output.open("x", encoding="utf-8") as stream:
        for row in table.to_pylist():
            hypothesis = dict(zip(row["hypothesis"]["language"], row["hypothesis"]["translation"], strict=True))
            result = dict(label=names[row["label"]],
                          premise={language: row["premise"][language] for language in LANGUAGES},
                          hypothesis={language: hypothesis[language] for language in LANGUAGES})
            if any(not value for field in ("premise", "hypothesis") for value in result[field].values()):
                raise ValueError("Missing aligned translation")
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")


def compare(projected, checkpoints, output):
    import numpy as np
    import torch

    from experiments.followir.core17_reranking import digest, write_json
    from experiments.scifact.small_nli_verifier import REVISIONS, load, measure

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh Slurm GPU run")
    if digest(projected) != PROJECTION_SHA:
        raise ValueError("Changed English and Chinese XNLI projection")
    with projected.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != COUNT or {row["label"] for row in rows} != LABELS:
        raise ValueError("Incorrect projection")
    expected = [row["label"] for row in rows]
    report = {}
    for name in REVISIONS:
        tokenizer, reader, label_index, provenance = load(checkpoints / f"{name}-nli", name)
        cases = {f"{a}_{b}": [dict(source=row["premise"][a], criterion=row["hypothesis"][b])
                              for row in rows] for a, b in FORMS}
        measure(tokenizer, reader, label_index, cases["en_en"][:16])
        results = {}
        outputs = {}
        for arm, data in cases.items():
            values, batch_times = measure(tokenizer, reader, label_index, data)
            predicted = [item[0] if item is not None else "over_limit" for item in values]
            results[arm] = dict(
                correct=sum(a == b for a, b in zip(predicted, expected, strict=True)),
                over_limit=predicted.count("over_limit"),
                false_entailment_on_contradiction=sum(
                    a == "entailment" and b == "contradiction" for a, b in zip(predicted, expected, strict=True)),
                confusion=[dict(gold=a, predicted=b, cases=n)
                           for (a, b), n in sorted(Counter(zip(expected, predicted, strict=True)).items())],
                batch_p50_p95_ms=np.quantile(batch_times, [.5, .95]).tolist(),
            )
            outputs[arm] = predicted
        for arm, table in results.items():
            table["exact_agreement_with_en_en"] = sum(
                a == b and a != "over_limit" for a, b in zip(outputs["en_en"], outputs[arm], strict=True))
        report[name] = results
        del reader
        print(name, provenance["revision"], "finished", flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), cases=COUNT,
        parquet_sha256=PARQUET_SHA, script_sha256=digest(Path(__file__)), projected_sha256=digest(projected),
        counts=dict(Counter(expected)), models=REVISIONS, results=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    inputs = modes.add_parser("prepare")
    inputs.add_argument("source", type=Path)
    inputs.add_argument("output", type=Path)
    inputs.set_defaults(action=prepare)
    run = modes.add_parser("compare")
    run.add_argument("projected", type=Path)
    run.add_argument("checkpoints", type=Path)
    run.add_argument("output", type=Path)
    run.set_defaults(action=compare)
    options = vars(parser.parse_args())
    options.pop("action")(**options)
