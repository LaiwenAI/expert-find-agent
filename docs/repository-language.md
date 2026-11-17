# Repository Language Policy

Maintained prose, comments, report labels, diagrams, configuration descriptions, and file names use English. Run the current-tree check before publishing:

```bash
python experiments/maintenance/check_repository_language.py
```

The check scans Git-visible text and paths, following `.gitignore`. It rejects ideographs and CJK/fullwidth punctuation. The source archive builder independently checks the files it packages.

## Bilingual Data and Fixtures

Bilingual search remains supported. Language-specific schema keys, matching expressions, and test fixtures use Python/JSON Unicode escapes instead of literal ideographs in source files. These spellings preserve their decoded values. Original UTF-8 enterprise/researcher records remain in ignored local data/artifact directories.

Generated repository reports use English labels and escape original-language source excerpts. English typography, scientific symbols, and Latin names remain readable. The original local artifact is authoritative; an escaped excerpt is not an English translation.

## Frozen Experiment Provenance

Most Python localization changes preserve the parsed runtime AST exactly. Report-template prose was translated, and legacy generated manuscript reports now default to local `artifacts/reports/` paths.

Byte-level source hashes still change when literal spellings change. Earlier protocols remain bound to their original source snapshots and historical commits. Do not replace old protocol hashes with current hashes or silently resume under a changed protocol. Read completed results without regenerating the original protocol, reproduce the pinned version, or freeze a new version for a new run.

The local migration ledger retains original source identities, AST-equivalence checks, and documentation-localization records. Original data, prior results, and historical Git commits are not rewritten by the language migration.
