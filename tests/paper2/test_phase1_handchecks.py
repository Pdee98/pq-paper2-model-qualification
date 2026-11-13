"""Literal hand-derived event sequences, independent trace/resource checks."""

from tests.paper2._provenance import setUpModule, tearDownModule

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.paper2.model import run_model
from src.paper2.model.service_times import ROOT
from tests.paper2._support import check_trace, config, sequence


class Phase1HandChecks(unittest.TestCase):
    def run_case(self, settings):
        temp = tempfile.TemporaryDirectory(prefix="paper2-test-")
        self.addCleanup(temp.cleanup)
        settings = dict(settings, output_dir=temp.name)
        log, manifest = run_model(settings)
        full, _ = run_model(dict(settings, policy_view="full", output_dir=str(Path(temp.name)/"full")))
        self.assertEqual(log.read_bytes(), full.read_bytes())
        self.assertEqual((log.parent/"jobs.csv").read_bytes(), (full.parent/"jobs.csv").read_bytes())
        return check_trace(self, log, manifest, settings["k"])

    def test_1_one_worker_admission_before_block(self):
        rows, _ = self.run_case(config(
            admissions=[dict(tx_id="a", arrival_t=0), dict(tx_id="b", arrival_t=0)],
            block_start=0.5, blocks=[dict(block_id="B", tx_ids=["a", "b"])]))
        # (time, event, job_id, worker, q_admission, q_block, busy), written by hand.
        expected = [
            (0, "tx_arrive", "admission:0", None, 0, 0, 0),
            (0, "enqueue", "admission:0", None, 0, 0, 0),
            (0, "start", "admission:0", 0, 0, 0, 1),
            (0, "tx_arrive", "admission:1", None, 0, 0, 1),
            (0, "enqueue", "admission:1", None, 1, 0, 1),
            (0.5, "block_arrive", "", None, 1, 0, 1),
            (0.5, "enqueue", "block:0:tx:0", None, 1, 1, 1),
            (0.5, "enqueue", "block:0:tx:1", None, 1, 2, 1),
            (0.5, "enqueue", "block:0:sig", None, 1, 3, 1),
            (1, "end", "admission:0", 0, 1, 3, 0),
            (1, "start", "admission:1", 0, 0, 3, 1),
            (2, "end", "admission:1", 0, 0, 3, 0),
            (2, "start", "block:0:tx:0", 0, 0, 2, 1),
            (3, "end", "block:0:tx:0", 0, 0, 2, 0),
            (3, "start", "block:0:tx:1", 0, 0, 1, 1),
            (4, "end", "block:0:tx:1", 0, 0, 1, 0),
            (4, "start", "block:0:sig", 0, 0, 0, 1),
            (5, "end", "block:0:sig", 0, 0, 0, 0),
            (5, "state_check", "", None, 0, 0, 0),
            (5, "block_accept", "", None, 0, 0, 0),
        ]
        self.assertEqual(sequence(rows), expected)

    def test_2_two_workers_simultaneous_arrivals(self):
        rows, _ = self.run_case(config(k=2,
            admissions=[dict(tx_id="a", arrival_t=0), dict(tx_id="b", arrival_t=0),
                        dict(tx_id="c", arrival_t=0)],
            service=dict(mode="constant", seconds=dict(admission=2, block_tx=1, block_sig=1))))
        expected = [
            (0, "tx_arrive", "admission:0", None, 0, 0, 0),
            (0, "enqueue", "admission:0", None, 0, 0, 0),
            (0, "start", "admission:0", 0, 0, 0, 1),
            (0, "tx_arrive", "admission:1", None, 0, 0, 1),
            (0, "enqueue", "admission:1", None, 0, 0, 1),
            (0, "start", "admission:1", 1, 0, 0, 2),
            (0, "tx_arrive", "admission:2", None, 0, 0, 2),
            (0, "enqueue", "admission:2", None, 1, 0, 2),
            (2, "end", "admission:0", 0, 1, 0, 1),
            (2, "start", "admission:2", 0, 0, 0, 2),
            (2, "end", "admission:1", 1, 0, 0, 1),
            (4, "end", "admission:2", 0, 0, 0, 0),
        ]
        self.assertEqual(sequence(rows), expected)

    def test_3_block_cannot_preempt_long_job(self):
        rows, _ = self.run_case(config(admissions=[dict(tx_id="a", arrival_t=0)],
            block_start=1, blocks=[dict(block_id="B", tx_ids=["x"])],
            service=dict(mode="constant", seconds=dict(admission=5, block_tx=1, block_sig=1))))
        expected = [
            (0, "tx_arrive", "admission:0", None, 0, 0, 0),
            (0, "enqueue", "admission:0", None, 0, 0, 0),
            (0, "start", "admission:0", 0, 0, 0, 1),
            (1, "block_arrive", "", None, 0, 0, 1),
            (1, "enqueue", "block:0:tx:0", None, 0, 1, 1),
            (1, "enqueue", "block:0:sig", None, 0, 2, 1),
            (5, "end", "admission:0", 0, 0, 2, 0),
            (5, "start", "block:0:tx:0", 0, 0, 1, 1),
            (6, "end", "block:0:tx:0", 0, 0, 1, 0),
            (6, "start", "block:0:sig", 0, 0, 0, 1),
            (7, "end", "block:0:sig", 0, 0, 0, 0),
            (7, "state_check", "", None, 0, 0, 0),
            (7, "block_accept", "", None, 0, 0, 0),
        ]
        self.assertEqual(sequence(rows), expected)

    def test_4_periodic_blocks_overlap(self):
        rows, data = self.run_case(config(deadline=1.5,
            blocks=[dict(block_id="B0", tx_ids=["a"]), dict(block_id="B1", tx_ids=["b"])]))
        expected = [
            (0, "block_arrive", "", None, 0, 0, 0),
            (0, "enqueue", "block:0:tx:0", None, 0, 0, 0),
            (0, "start", "block:0:tx:0", 0, 0, 0, 1),
            (0, "enqueue", "block:0:sig", None, 0, 1, 1),
            (1, "end", "block:0:tx:0", 0, 0, 1, 0),
            (1, "start", "block:0:sig", 0, 0, 0, 1),
            (1, "block_arrive", "", None, 0, 0, 1),
            (1, "enqueue", "block:1:tx:0", None, 0, 1, 1),
            (1, "enqueue", "block:1:sig", None, 0, 2, 1),
            (2, "end", "block:0:sig", 0, 0, 2, 0),
            (2, "state_check", "", None, 0, 2, 0),
            (2, "block_accept", "", None, 0, 2, 0),
            (2, "start", "block:1:tx:0", 0, 0, 1, 1),
            (3, "end", "block:1:tx:0", 0, 0, 1, 0),
            (3, "start", "block:1:sig", 0, 0, 0, 1),
            (4, "end", "block:1:sig", 0, 0, 0, 0),
            (4, "state_check", "", None, 0, 0, 0),
            (4, "block_accept", "", None, 0, 0, 0),
        ]
        self.assertEqual(sequence(rows), expected)
        self.assertEqual(data["blocks"]["B0"]["deadline_t"], 1.5)
        self.assertEqual(data["blocks"]["B1"]["deadline_t"], 2.5)

    def test_tail_drop_and_zero_waiting_capacity(self):
        for bound, expected_drops in [(0, 2), (1, 1)]:
            rows, _ = self.run_case(config(q_max=bound, admissions=[
                dict(tx_id="a", arrival_t=0), dict(tx_id="b", arrival_t=0),
                dict(tx_id="c", arrival_t=0)]))
            self.assertEqual(sum(r["event"] == "drop" for r in rows), expected_drops)

    def test_serial_structural_checks_do_not_consume_workers(self):
        rows, data = self.run_case(config(structural_cost=0.5, admissions=[
            dict(tx_id="a", arrival_t=0), dict(tx_id="b", arrival_t=0, structurally_valid=False)]))
        self.assertEqual([(float(r["t"]), r["event"]) for r in rows if r["event"] != "dispatch"], [
            (0, "tx_arrive"), (0, "tx_arrive"), (0.5, "enqueue"), (0.5, "start"),
            (1, "struct_reject"), (1.5, "end")])
        rejection = next(r for r in rows if r["event"] == "struct_reject")
        self.assertEqual(rejection["worker_id"], "")
        self.assertEqual(rejection["busy_workers"], "1")
        self.assertFalse(next(j["structurally_valid"] for j in data["jobs"] if j["status"] == "struct_reject"))

    def test_invalid_signature_outcome_is_not_hidden(self):
        settings = config(blocks=[dict(block_id="B", tx_ids=[])])
        settings["blocks"][0]["block_sig_label"] = "bit_flip"
        settings["service"]["seconds"]["block_sig"] = {"valid": 1, "bit_flip": 1}
        rows, data = self.run_case(settings)
        self.assertEqual(next(r["outcome"] for r in rows if r["event"] == "end"), "invalid")
        self.assertFalse(data["blocks"]["B"]["valid"])
        self.assertIsNone(data["blocks"]["B"]["accept_t"])
        self.assertEqual(data["blocks"]["B"]["reject_t"], 1)
        self.assertEqual(data["blocks"]["B"]["reject_reason"], "invalid_signature")

    def test_structurally_rejected_block_has_no_acceptance(self):
        rows, _ = self.run_case(config(blocks=[dict(block_id="B", tx_ids=["x"], structurally_valid=False)]))
        self.assertEqual([r["event"] for r in rows], ["block_arrive", "struct_reject", "block_reject"])

    def test_zero_service_and_artifact_provenance(self):
        settings = config(admissions=[dict(tx_id="a", arrival_t=0), dict(tx_id="b", arrival_t=0)])
        settings["service"]["seconds"] = dict(admission=0, block_tx=0, block_sig=0)
        rows, data = self.run_case(settings)
        self.assertTrue(all(float(r["t"]) == 0 for r in rows))
        self.assertRegex(data["git"]["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(set(data["seeds"]), {"arrivals", "arrivals_invalid", "block_contents", "labels", "cache_evict", "service_admission", "service_block_tx", "service_block_sig"})
        self.assertTrue(data["source_sha256"])

    def test_output_inside_repository_is_refused(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            run_model(config(output_dir=str(ROOT / "results/paper2/phase1-forbidden")))

    def test_cli_writes_only_to_temp(self):
        with tempfile.TemporaryDirectory(prefix="paper2-cli-") as temp:
            directory = Path(temp)
            settings = config(output_dir=str(directory / "run"), admissions=[dict(tx_id="a", arrival_t=0)])
            source = directory / "config.json"
            source.write_text(json.dumps(settings), encoding="utf-8")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPYCACHEPREFIX=str(directory / "pycache"))
            result = subprocess.run([sys.executable, "-B", "-m", "src.paper2.model", str(source)],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifacts = json.loads(result.stdout)
            check_trace(self, Path(artifacts["event_log"]), Path(artifacts["manifest"]), 1)
