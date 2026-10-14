"""Coverage gate: turn collected coverage results into a pass/fail verdict.

The collection scripts (network_matrix.py, training_interfaces.py) deliberately
record failures and exit 0. This tool compares their JSON output with a
checked-in manifest of *expected* outcomes and fails on any surprise:

* unexpected failure     - manifest says "pass" (or lists nothing), result failed
* changed failure        - a known failure now fails with a different status/stage
* unexpected pass        - a known failure passes; the manifest must be reviewed
                           so the feature is not silently left labelled unsupported
* unlisted case          - a result the manifest does not mention
* missing result         - a manifest entry produced no result (case dropped)
* NPU routing regression - fewer NPU runs than the manifest's ``min_npu_runs``

It is standard-library only so it can run on any CI runner. Numerical pass/fail
is independent of performance; this tool never looks at timings.

Usage:
    python tests/coverage/gate.py results.json [more.json ...] [--manifest M]
    python tests/coverage/gate.py results.json --init   # print a draft manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")
PASS = "pass"
KNOWN_FAILURE = "known_failure"
# Non-"pass" statuses the interface probes use for "ran, nothing to compare".
OK_NON_PASS = {"runs"}


def load_results(path: Path) -> dict[str, dict]:
    """Return {key: result} where key is ``case/dtype`` (or ``case``)."""
    raw = json.loads(path.read_text())
    rows = raw["results"] if isinstance(raw, dict) else raw
    out: dict[str, dict] = {}
    for row in rows:
        case = row["case"]
        dtype = row.get("dtype")
        key = f"{case}/{dtype}" if dtype else case
        if key in out:
            raise ValueError(f"duplicate result {key!r} in {path}")
        out[key] = row
    return out


def failure_stage(result: dict) -> str | None:
    errors = result.get("errors") or {}
    for device in ("xdna", "cpu"):
        if device in errors:
            return errors[device].get("stage")
    return None


def total_npu_runs(result: dict) -> int:
    return sum(int(v) for v in (result.get("npu_runs") or {}).values())


def evaluate(results: dict[str, dict], manifest: dict) -> dict[str, list[str]]:
    entries: dict[str, dict] = manifest.get("entries", {})
    problems: dict[str, list[str]] = {
        "unexpected_failure": [],
        "changed_failure": [],
        "unexpected_pass": [],
        "unlisted": [],
        "missing": [],
        "npu_regression": [],
    }
    for key, result in sorted(results.items()):
        status = result.get("status")
        entry = entries.get(key)
        if entry is None:
            problems["unlisted"].append(f"{key}: status={status}")
            continue
        expect = entry.get("expect", PASS)
        if expect == PASS:
            if status not in entry.get("ok_statuses", [PASS]):
                stage = failure_stage(result)
                where = f" at {stage}" if stage else ""
                problems["unexpected_failure"].append(f"{key}: {status}{where}")
                continue
            floor = int(entry.get("min_npu_runs", 0))
            got = total_npu_runs(result)
            if got < floor:
                problems["npu_regression"].append(
                    f"{key}: {got} NPU runs, expected at least {floor}"
                )
        elif expect == KNOWN_FAILURE:
            if status == PASS:
                problems["unexpected_pass"].append(
                    f"{key}: now passes (issue {entry.get('issue', '?')}); review manifest"
                )
                continue
            want_status = entry.get("status")
            want_stage = entry.get("stage")
            stage = failure_stage(result)
            if (want_status and status != want_status) or (
                want_stage and stage and stage != want_stage
            ):
                problems["changed_failure"].append(
                    f"{key}: expected {want_status}/{want_stage}, got {status}/{stage}"
                )
        else:
            raise ValueError(f"{key}: unknown expect {expect!r}")
    for key in sorted(set(entries) - set(results)):
        problems["missing"].append(key)
    return problems


def draft_manifest(results: dict[str, dict], baseline: str = "") -> dict:
    entries: dict[str, dict] = {}
    for key, result in sorted(results.items()):
        if result.get("status") == PASS:
            entries[key] = {"expect": PASS, "min_npu_runs": 0}
        elif result.get("status") in OK_NON_PASS:
            entries[key] = {"expect": PASS, "ok_statuses": [result["status"]]}
        else:
            entries[key] = {
                "expect": KNOWN_FAILURE,
                "status": result.get("status"),
                "stage": failure_stage(result),
                "issue": "UNTRIAGED",
            }
    return {"schema": 1, "baseline": baseline, "entries": entries}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--init", action="store_true", help="print a draft manifest and exit 0")
    ap.add_argument("--baseline", default="", help="commit recorded in a drafted manifest")
    ap.add_argument(
        "--scope",
        default="",
        help="only check manifest entries/results whose key starts with this prefix",
    )
    args = ap.parse_args(argv)

    results: dict[str, dict] = {}
    for path in args.results:
        for key, row in load_results(path).items():
            if key in results:
                print(f"duplicate result {key!r} across inputs", file=sys.stderr)
                return 2
            results[key] = row
    if args.scope:
        results = {k: v for k, v in results.items() if k.startswith(args.scope)}

    if args.init:
        json.dump(draft_manifest(results, args.baseline), sys.stdout, indent=2, sort_keys=True)
        print()
        return 0

    manifest = json.loads(args.manifest.read_text())
    if args.scope:
        manifest = {
            **manifest,
            "entries": {
                k: v for k, v in manifest.get("entries", {}).items() if k.startswith(args.scope)
            },
        }
    problems = evaluate(results, manifest)
    bad = sum(len(v) for v in problems.values())
    print(f"{len(results)} results checked against {args.manifest.name}: {bad} problem(s)")
    for kind, items in problems.items():
        for item in items:
            print(f"  [{kind}] {item}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
