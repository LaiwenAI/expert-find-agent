"""Check service parity against the fixed-pair archive and collect both orders."""

from hashlib import sha256
import json
from pathlib import Path
import sys


repo, output, *paths = map(Path, sys.argv[1:])
frozen = json.loads((repo / "benchmarks/elife-role-contrast-training-v1/results.json").read_bytes())["replicas"]
runner = repo / "experiments/credit/elife_resident_role_service.py"
runs = [json.loads(p.read_bytes()) for p in paths]
assert len(runs) == len(frozen) == 4
for run, reference in zip(runs, frozen, strict=True):
    assert run["runner"] == sha256(runner.read_bytes()).hexdigest()
    for current, old in (("head", "student_training"), ("encoder", "encoder"), ("reader", "role_checkpoint"),
                         ("reverse", "reverse"), ("manifest", "manifest_sha256"), ("projection", "projection_sha256")):
        assert run[current] == reference[old], (current, run[current], reference[old])
    for view, arms in run["quality"].items():
        for arm, previous in (("full_menu", "teacher_all_roles"), ("ridge", "ridge_top_one"), ("resident_ce", "original_ce")):
            assert arms[arm]["source_changes"] == 0
            for metric in ("strict", "adverse"):
                assert arms[arm][metric] == reference["report"]["both_present"][view][previous][metric], (view, arm, metric)
    for by_rate in run["timings"].values():
        total = sum(run["quality"][name]["full_menu"]["requests"] for name in ("keyword", "excerpt")) * 2
        for arm, data in by_rate.items():
            work = data["model_and_reads"]
            assert data["requests"] == total and not data["failures"]
            if arm != "full_menu":
                assert work["role_forwards"] == work["role_tokens"] == 0
            if arm in ("full_menu", "resident_ce_fused"):
                assert work["encoder_forwards"] == total
            else:
                assert work["encoder_forwards"] == total + work["logical_reads"] // 2

for first, second in zip(runs[::2], runs[1::2], strict=True):
    for key in ("quality", "signatures", "head"):
        assert first[key] == second[key], key

assert not output.exists()
output.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
