"""Part A checks, executed before policy integration; no research campaign."""

from tests.paper2._provenance import setUpModule, tearDownModule

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from collections import OrderedDict

from src.paper2.model import run_model
from src.paper2.state import SignatureCache, TransactionPool
from src.paper2.workload import generate_workload
from src.paper2.workload.generator import tx_draw_integer
from tests.paper2._phase2_support import generator_parameters
from tests.paper2._support import config, check_trace, tx


class Phase3Corrections(unittest.TestCase):
    def test_identity_draw_and_generator_independence(self):
        payload = ["paper2-tx-draw-v1", "paired", 2, "amount", ["valid", 3]]
        expected = int.from_bytes(hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).digest(), "big")
        self.assertEqual(tx_draw_integer("paired", 2, "amount", ["valid", 3]), expected)
        p = generator_parameters(horizon=8, block_size=3)
        a = generate_workload(dict(p, arrivals_invalid=dict(kind="poisson", rate=0)))
        b = generate_workload(dict(p, arrivals_invalid=dict(kind="poisson", rate=8)))
        c = generate_workload(dict(p, arrivals_invalid=dict(kind="poisson", rate=8), p_seen=0))
        valid = lambda w: {k: v for k, v in w["transactions"].items() if k.startswith("v:")}
        self.assertEqual(valid(a), valid(b))
        self.assertEqual(b["transactions"], c["transactions"])
        self.assertEqual(a["blocks"], b["blocks"])
        self.assertEqual(b["blocks"], c["blocks"])
        self.assertNotEqual(b["admissions"], c["admissions"])

    def run_case(self, settings):
        with tempfile.TemporaryDirectory(prefix="paper2-part-a-") as directory:
            settings = dict(settings, output_dir=directory)
            paths = run_model(settings)
            full = run_model(dict(settings, policy_view="full", output_dir=str(Path(directory)/"full")))
            self.assertEqual(paths[0].read_bytes(), full[0].read_bytes())
            self.assertEqual((paths[0].parent/"jobs.csv").read_bytes(), (full[0].parent/"jobs.csv").read_bytes())
            return check_trace(self, *paths, settings["k"])

    def test_running_wait_and_off_literal_worker_trace(self):
        base = config(admissions=[dict(tx_id="a", arrival_t=0)], block_start=0.5,
                      blocks=[dict(block_id="B", tx_ids=["a"])])
        off_rows, off = self.run_case(base)
        on_rows, on = self.run_case(dict(base, coalesce_inflight=True))
        projection = lambda rows: [(float(r["t"]), r["event"], r["job_id"]) for r in rows
                                  if r["event"] in ("start", "end", "coalesce_wait")]
        self.assertEqual(projection(off_rows), [(0, "start", "admission:0"), (1, "end", "admission:0"),
            (1, "start", "block:0:tx:0"), (2, "end", "block:0:tx:0"),
            (2, "start", "block:0:sig"), (3, "end", "block:0:sig")])
        self.assertEqual(projection(on_rows), [(0, "start", "admission:0"),
            (0.5, "coalesce_wait", "block:0:tx:0"), (1, "end", "admission:0"),
            (1, "start", "block:0:sig"), (2, "end", "block:0:sig")])
        self.assertLessEqual(sum(j["service_time"] or 0 for j in on["jobs"]),
                             sum(j["service_time"] or 0 for j in off["jobs"]))

    def test_queued_promotion_keeps_admission_draw_and_enqueue(self):
        settings = config(admissions=[dict(tx_id="busy", arrival_t=0), dict(tx_id="a", arrival_t=0.1)],
                          block_start=0.5, blocks=[dict(block_id="B", tx_ids=["a"])],
                          service=dict(mode="constant", seconds=dict(admission=2, block_tx=7, block_sig=1)),
                          coalesce_inflight=True)
        rows, data = self.run_case(settings)
        self.assertEqual([(float(r["t"]), r["event"], r["job_id"]) for r in rows
                          if r["event"] in ("start", "end", "coalesce_promote")],
                         [(0, "start", "admission:0"), (0.5, "coalesce_promote", "admission:1"),
                          (2, "end", "admission:0"), (2, "start", "admission:1"),
                          (4, "end", "admission:1"), (4, "start", "block:0:sig"), (5, "end", "block:0:sig")])
        promoted = next(j for j in data["jobs"] if j["job_id"] == "admission:1")
        self.assertEqual((promoted["enqueue_t"], promoted["service_time"], promoted["job_class"], promoted["service_class"]),
                         (0.1, 2, "block_tx", "admission"))

    def test_already_included_detail(self):
        rows, data = self.run_case(config(blocks=[dict(block_id="B", tx_ids=["a"])],
            admissions=[dict(tx_id="a", arrival_t=3)], cache=dict(capacity=None)))
        self.assertEqual([r["detail"] for r in rows if r["event"] == "state_reject"], ["already_included"])
        self.assertEqual(data["admission_checks"][0]["reason"], "already_included")

    def test_one_promoted_check_serves_multiple_blocks(self):
        rows, data = self.run_case(config(coalesce_inflight=True,
            policy=dict(name="deadline_slack", beta=0, W=5, m=10),
            admissions=[dict(tx_id="busy", arrival_t=0), dict(tx_id="a", arrival_t=0.1)],
            block_start=0.5, block_period=0.1,
            blocks=[dict(block_id="B0", tx_ids=["a"]), dict(block_id="B1", tx_ids=["a"])],
            service=dict(mode="constant", seconds=dict(admission=2, block_tx=7, block_sig=1))))
        self.assertEqual(sum(r["event"] == "coalesce_promote" for r in rows), 1)
        self.assertEqual(sum(r["event"] == "coalesce_wait" for r in rows), 1)
        self.assertEqual(sum(j["status"] == "coalesced" for j in data["jobs"]), 2)
        self.assertTrue(data["blocks"]["B0"]["valid"])
        self.assertEqual(data["blocks"]["B1"]["reject_reason"], "state")

    def test_coalesced_invalid_signature_cannot_accept(self):
        rows, data = self.run_case(config(coalesce_inflight=True,
            transactions={"bad": tx("bad", sig_label="bit_flip")},
            admissions=[dict(tx_id="bad", arrival_t=0)], block_start=0.5,
            blocks=[dict(block_id="B", tx_ids=["bad"])],
            service=dict(mode="constant", seconds={c: dict(valid=1, bit_flip=1)
                for c in ("admission", "block_tx", "block_sig")})))
        self.assertEqual(data["accepted_transactions"], [])
        self.assertEqual(data["blocks"]["B"]["reject_reason"], "invalid_signature")
        self.assertTrue(any(r["event"] == "drop" and r["detail"] == "invalid_signature" for r in rows))

    def test_expiry_versions_and_pool_lazy_deletion(self):
        cache = SignatureCache(capacity=2, ttl=2)
        cache.insert("a", "p1", 0, ("d", "p"))
        cache.insert("a", "p2", 1, ("d", "p"))
        entry, removed = cache.lookup("a", 2, ("d", "p"))
        self.assertEqual(entry["proof_job_id"], "p2")
        self.assertEqual(removed, [])
        self.assertEqual(cache.lookup("a", 3, ("d", "p")), (None, [("a", "ttl")]))
        pool = TransactionPool(enabled=True, capacity_bytes=1)
        pool.insert("a", tx("a", size_bytes=1))
        pool.remove("a")
        pool.insert("a", tx("a", size_bytes=1, fee_rate=3))
        self.assertEqual([k for k, _ in pool.insert("b", tx("b", size_bytes=1, fee_rate=2))], ["b"])

    def test_50000_overloaded_pool_admissions_with_500_prefix_reference(self):
        # Component code-cost smoke: offer already-verified admissions to a full pool.
        pool = TransactionPool(enabled=True, capacity_bytes=100)
        reference = OrderedDict()
        for i in range(100):
            key = f"prefill:{i}"
            value = tx(key, size_bytes=1, fee_rate=0)
            pool.insert(key, value)
            reference[key] = value
        self.assertEqual(pool.bytes, 100)
        start = time.perf_counter()
        for i in range(50000):
            value = tx(str(i), size_bytes=1 + i % 3, fee_rate=(i * 17) % 101)
            removed = pool.insert(str(i), value)
            if i < 500:
                reference[str(i)] = value
                expected = []
                while sum(t["size_bytes"] for t in reference.values()) > 100:
                    victim = min(reference, key=lambda k: reference[k]["fee_rate"])
                    expected.append((victim, reference.pop(victim)))
                self.assertEqual(removed, expected)
                self.assertEqual(list(pool.entries.items()), list(reference.items()))
        self.assertGreater(pool.evicted_before_inclusion, 49000)
        self.assertLess(time.perf_counter() - start, 60)
