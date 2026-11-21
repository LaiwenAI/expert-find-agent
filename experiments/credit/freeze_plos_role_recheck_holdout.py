"""Freeze a previously unevaluated PLOS cohort for keyword role rechecking."""

import argparse
import json
from pathlib import Path

from experiments.credit import plos_prospective_proof_split as collector

QUERY = collector.QUERY.replace("2025-", "2024-")
MIN_GROUPS = 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("elife_confirmation", "elife_extended", "base_manifest", "prior_holdout", "archive", "manifest"):
        parser.add_argument(name, type=Path)
    parser.add_argument("prior_folders", type=Path, nargs="+")
    args = parser.parse_args()
    if args.manifest.exists() or args.archive.exists():
        raise ValueError("Require new output paths")
    base = json.loads(args.base_manifest.read_bytes())
    prior = json.loads(args.prior_holdout.read_bytes())
    old_dois, old_ids, _ = collector.source.prior(args.prior_folders)
    locked = base["source_sha256"] | prior["source_sha256"]
    digests = {}
    for folder in args.prior_folders:
        for file in collector.source.xml_sources(folder):
            data = file.read_bytes()
            doi = collector.source.meta(data).findtext("article-id[@pub-id-type='doi']")
            if doi in locked:
                digests.setdefault(doi, set()).add(collector.digest(data))
    if any(value not in digests.get(doi, ()) for doi, value in locked.items()):
        raise ValueError("Prior source or exact author-exclusion inputs missing")
    elife = collector.load_elife((args.elife_confirmation, args.elife_extended))
    owners, _, metadata = collector.metadata(QUERY)
    picks, proposed = collector.proposals(owners, old_dois, old_ids, elife)
    args.archive.mkdir()
    print(f"{metadata['total']} metadata records, {proposed} pair proposals, screening {len(picks)}", flush=True)
    groups, digests, diagnostics = collector.confirm(picks, args.archive, old_ids, elife)
    if len(groups) < MIN_GROUPS:
        raise ValueError(f"Insufficient fresh publisher groups: {diagnostics}")
    groups.sort(key=lambda row: collector.stable("plos-role-recheck-holdout-v1", row["papers"]))
    report = dict(
        inventory=metadata, proposals=proposed, attempted=len(picks), verified=len(groups),
        source_sha256=digests, old_papers=len(old_dois), old_recorded_authors=len(old_ids),
        test=dict(pairs=groups), calibration=dict(pairs=[]), diagnostics=diagnostics,
        prior_digests={name: collector.digest(path.read_bytes())
                       for name, path in (("base", args.base_manifest), ("holdout", args.prior_holdout))},
        elife_sha256=collector.SHA_ELIFE, script_sha256=collector.digest(Path(__file__).read_bytes()),
    )
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Froze {len(groups)} new groups before model evaluation", flush=True)


if __name__ == "__main__":
    main()
