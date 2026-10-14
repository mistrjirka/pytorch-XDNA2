"""Unit tests for tests/coverage/gate.py; no torch or hardware required."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "coverage_gate", Path(__file__).with_name("coverage") / "gate.py"
)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def res(status="pass", stage=None, npu=None):
    row = {"status": status, "npu_runs": npu or {}}
    if stage:
        row["errors"] = {"xdna": {"stage": stage}}
    return row


MANIFEST = {
    "entries": {
        "a/bf16": {"expect": "pass", "min_npu_runs": 2},
        "b/bf16": {"expect": "known_failure", "status": "xdna_error", "stage": "forward", "issue": "X-1"},
        "c/fp32": {"expect": "pass"},
    }
}


class GateTest(unittest.TestCase):
    def kinds(self, results):
        return {k for k, v in gate.evaluate(results, MANIFEST).items() if v}

    def good(self):
        return {
            "a/bf16": res(npu={"npu.conv.run": 2}),
            "b/bf16": res("xdna_error", "forward"),
            "c/fp32": res(),
        }

    def test_all_expected_passes(self):
        self.assertEqual(self.kinds(self.good()), set())

    def test_unexpected_failure(self):
        r = self.good()
        r["c/fp32"] = res("numeric_mismatch")
        self.assertEqual(self.kinds(r), {"unexpected_failure"})

    def test_unexpected_pass_requests_review(self):
        r = self.good()
        r["b/bf16"] = res()
        self.assertEqual(self.kinds(r), {"unexpected_pass"})

    def test_changed_failure_stage(self):
        r = self.good()
        r["b/bf16"] = res("xdna_error", "backward")
        self.assertEqual(self.kinds(r), {"changed_failure"})

    def test_unlisted_and_missing(self):
        r = self.good()
        r["d/fp32"] = res()
        del r["c/fp32"]
        self.assertEqual(self.kinds(r), {"unlisted", "missing"})

    def test_npu_routing_regression(self):
        r = self.good()
        r["a/bf16"] = res(npu={"npu.conv.run": 1})
        self.assertEqual(self.kinds(r), {"npu_regression"})

    def test_draft_manifest_roundtrip(self):
        r = self.good()
        draft = gate.draft_manifest(r, "abc")
        self.assertEqual(draft["entries"]["b/bf16"]["issue"], "UNTRIAGED")
        self.assertFalse(any(gate.evaluate(r, draft).values()))

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = [
                {"case": "a", "dtype": "bf16", **res(npu={"npu.conv.run": 2})},
                {"case": "b", "dtype": "bf16", **res("xdna_error", "forward")},
                {"case": "c", "dtype": "fp32", **res()},
            ]
            (d / "r.json").write_text(json.dumps({"results": rows}))
            (d / "m.json").write_text(json.dumps(MANIFEST))
            self.assertEqual(gate.main([str(d / "r.json"), "--manifest", str(d / "m.json")]), 0)
            rows[2]["status"] = "numeric_mismatch"
            (d / "r.json").write_text(json.dumps({"results": rows}))
            self.assertEqual(gate.main([str(d / "r.json"), "--manifest", str(d / "m.json")]), 1)


if __name__ == "__main__":
    unittest.main()
