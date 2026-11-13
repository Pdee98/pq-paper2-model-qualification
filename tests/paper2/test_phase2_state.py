"""Hand-derived cache/state behavior and exact-key acceptance safety."""

from tests.paper2._provenance import setUpModule, tearDownModule

from pathlib import Path
import tempfile
import unittest

from src.paper2.model import run_model
from src.paper2.model.service_times import WorkloadStream, derive_seed
from src.paper2.state import SignatureCache, cache_key, normalize_tx
from src.paper2.workload import generate_workload
from tests.paper2._support import config, tx, check_trace, sequence
from tests.paper2._phase2_support import generator_parameters


class Phase2State(unittest.TestCase):
    def run_case(self, settings):
        temp = tempfile.TemporaryDirectory(prefix="paper2-state-test-")
        self.addCleanup(temp.cleanup)
        settings = dict(settings, output_dir=temp.name)
        paths = run_model(settings)
        full = run_model(dict(settings, policy_view="full", output_dir=str(Path(temp.name)/"full")))
        self.assertEqual(paths[0].read_bytes(), full[0].read_bytes())
        self.assertEqual((paths[0].parent/"jobs.csv").read_bytes(), (full[0].parent/"jobs.csv").read_bytes())
        return check_trace(self, *paths, settings["k"])

    def test_1_literal_admitted_transaction_hits_in_block(self):
        rows, data = self.run_case(config(transactions={"a": tx("a", sender="alice", amount=10, fee_rate=1)},
            genesis=dict(balances={"alice": 1000}, next_nonces={"alice": 0}),
            admissions=[dict(tx_id="a", arrival_t=0)], block_start=2,
            blocks=[dict(block_id="B", tx_ids=["a"])], cache=dict(capacity=10), pool=dict(enabled=True)))
        expected = [
            (0, "tx_arrive", "admission:0", None, 0, 0, 0),
            (0, "cache_miss", "admission:0", None, 0, 0, 0),
            (0, "enqueue", "admission:0", None, 0, 0, 0),
            (0, "start", "admission:0", 0, 0, 0, 1),
            (1, "end", "admission:0", 0, 0, 0, 0),
            (1, "cache_insert", "admission:0", 0, 0, 0, 0),
            (1, "pool_insert", "admission:0", 0, 0, 0, 0),
            (2, "block_arrive", "", None, 0, 0, 0),
            (2, "cache_hit", "block:0:tx:0", None, 0, 0, 0),
            (2, "enqueue", "block:0:sig", None, 0, 0, 0),
            (2, "start", "block:0:sig", 0, 0, 0, 1),
            (3, "end", "block:0:sig", 0, 0, 0, 0),
            (3, "state_check", "", None, 0, 0, 0),
            (3, "block_accept", "", None, 0, 0, 0),
        ]
        self.assertEqual(sequence(rows), expected)
        self.assertEqual(data["ledger"]["balances"], {"alice": 741, "sink": 10})
        self.assertEqual(data["blocks"]["B"]["state_checks"], [dict(tx_id="a", passed=True, reason=None)])

    def test_2_literal_p_seen_removal_first_seen_in_block(self):
        workload = generate_workload(generator_parameters(horizon=2, block_period=2,
            arrivals_valid=dict(kind="poisson", rate=10), max_transactions=1, senders=["alice"],
            p_seen=0, arrivals_invalid=dict(kind="poisson", rate=0), fee_distribution=dict(name="constant", value=1)))
        self.assertEqual(workload["admissions"], [])
        self.assertEqual(workload["blocks"][0]["tx_ids"], ["v:0"])
        rows, _ = self.run_case(config(**workload, cache=dict(capacity=10), pool=dict(enabled=True)))
        expected = [
            (2, "block_arrive", "", None, 0, 0, 0),
            (2, "cache_miss", "block:0:tx:0", None, 0, 0, 0),
            (2, "enqueue", "block:0:tx:0", None, 0, 0, 0),
            (2, "start", "block:0:tx:0", 0, 0, 0, 1),
            (2, "enqueue", "block:0:sig", None, 0, 1, 1),
            (3, "end", "block:0:tx:0", 0, 0, 1, 0),
            (3, "cache_insert", "block:0:tx:0", 0, 0, 1, 0),
            (3, "start", "block:0:sig", 0, 0, 0, 1),
            (4, "end", "block:0:sig", 0, 0, 0, 0),
            (4, "state_check", "", None, 0, 0, 0),
            (4, "block_accept", "", None, 0, 0, 0),
        ]
        self.assertEqual(sequence(rows), expected)

    def test_each_key_component_binds(self):
        base = normalize_tx(tx("a"))
        for field in ("domain_tag", "chain_id", "alg_id", "param_set_id", "public_key_id", "sender",
                      "receiver", "amount", "timestamp", "transaction_id", "nonce", "signature_id"):
            with self.subTest(field=field):
                changed = dict(base)
                changed[field] = base[field] + 1 if isinstance(base[field], (int, float)) else base[field] + ":changed"
                cache = SignatureCache(3)
                cache.insert(cache_key(base), "proof", 0, (base["domain_tag"], base["param_set_id"]))
                self.assertNotEqual(cache_key(base), cache_key(changed))
                entry, _ = cache.lookup(cache_key(changed), 0, (changed["domain_tag"], changed["param_set_id"]))
                self.assertIsNone(entry)

    def test_cached_stale_nonce_rejected(self):
        rows, data = self.run_case(config(admissions=[dict(tx_id="a", arrival_t=0)], block_start=2,
            block_period=2, blocks=[dict(block_id="B0", tx_ids=["a"]), dict(block_id="B1", tx_ids=["a"])],
            cache=dict(capacity=10), pool=dict(enabled=True)))
        self.assertEqual(sum(r["event"] == "cache_hit" for r in rows), 2)
        self.assertTrue(data["blocks"]["B0"]["valid"])
        self.assertEqual(data["blocks"]["B1"]["reject_reason"], "state")
        self.assertEqual(len(data["accepted_transactions"]), 1)

    def test_same_sender_nonce_one_cached_at_most_one_accepted(self):
        transactions = {name: tx(name, sender="alice") for name in ("a", "b")}
        rows, data = self.run_case(config(transactions=transactions,
            admissions=[dict(tx_id="a", arrival_t=0), dict(tx_id="b", arrival_t=2)], block_start=4,
            block_period=2, blocks=[dict(block_id="B0", tx_ids=["a"]), dict(block_id="B1", tx_ids=["b"])],
            cache=dict(capacity=10), pool=dict(enabled=True)))
        self.assertTrue(any(r["event"] == "state_reject" and r["detail"] == "pool_nonce_duplicate" for r in rows))
        self.assertEqual(len(data["accepted_transactions"]), 1)
        self.assertEqual(data["blocks"]["B1"]["reject_reason"], "state")

    def test_key_rotation_cannot_hit_old_key(self):
        old = tx("a", public_key_id="old", signature_id="old-signature")
        new = tx("a", public_key_id="new", signature_id="new-signature")
        rows, data = self.run_case(config(transactions={"a": new},
            admissions=[dict(old, arrival_t=0), dict(new, arrival_t=3)], cache=dict(capacity=10),
            blocks=[dict(block_id="B", tx_ids=["a"])], block_start=5))
        admission_hits = [r for r in rows if r["event"] == "cache_hit" and r["job_class"] == "admission"]
        self.assertEqual(admission_hits, [])
        self.assertEqual([j["status"] for j in data["jobs"] if j["job_class"] == "admission"], ["end", "end"])

    def test_negative_repeat_and_later_valid_victim_nonce(self):
        bad = tx("bad", sender="alice", nonce=0, sig_label="bit_flip", signature_id="forgery")
        good = tx("good", sender="alice", nonce=0, signature_id="authentic")
        settings = config(transactions={"bad": bad, "good": good},
            admissions=[dict(tx_id="bad", arrival_t=0), dict(tx_id="bad", arrival_t=1.5), dict(tx_id="good", arrival_t=2)],
            block_start=4, blocks=[dict(block_id="B", tx_ids=["good"])],
            cache=dict(capacity=10, negative_capacity=10, negative_ttl=10), pool=dict(enabled=True))
        settings["service"]["seconds"]["admission"] = {"valid": 1, "bit_flip": 1}
        rows, data = self.run_case(settings)
        # Hand outcome: forgery ends invalid at 1; exact repeat hits negative at 1.5;
        # authentic nonce 0 verifies at 3 and is accepted via block signature at 5.
        self.assertEqual([(float(r["t"]), r["tx_id"]) for r in rows if r["event"] == "neg_hit"], [(1.5, "bad")])
        self.assertEqual(data["blocks"]["B"]["accept_t"], 5)
        self.assertEqual([a["tx"]["tx_id"] for a in data["accepted_transactions"]], ["good"])

    def test_lru_three_entries_and_random_reproducibility(self):
        cache = SignatureCache(3)
        for key in ("A", "B", "C"):
            cache.insert(key, key, 0, ("domain", "params"))
        cache.lookup("A", 0, ("domain", "params"))
        _, removed = cache.insert("D", "D", 0, ("domain", "params"))
        self.assertEqual(removed, [("B", "capacity")])
        self.assertEqual(list(cache.entries), ["C", "A", "D"])
        histories = []
        for _ in range(2):
            cache = SignatureCache(3, "random", rng=WorkloadStream(derive_seed("evict", "cache_evict", 0)))
            histories.append([cache.insert(str(i), str(i), 0, ("d", "p"))[1] for i in range(12)])
        self.assertEqual(histories[0], histories[1])

    def test_ttl_and_domain_parameter_flush(self):
        cache = SignatureCache(3, ttl=2)
        cache.insert("a", "proof", 0, ("d", "p"))
        self.assertIsNotNone(cache.lookup("a", 1, ("d", "p"))[0])
        self.assertEqual(cache.lookup("a", 2, ("d", "p")), (None, [("a", "ttl")]))
        for context in (("new-domain", "p"), ("d", "new-params")):
            cache.insert("a", "proof", 3, ("d", "p"))
            self.assertEqual(cache.lookup("different", 3, context)[1], [("a", "context")])
            self.assertEqual(len(cache.entries), 0)

    def test_mixed_size_pool_eviction_lowest_fee_first(self):
        transactions = {"low": tx("low", size_bytes=100, fee_rate=1),
                        "high": tx("high", size_bytes=250, fee_rate=5),
                        "mid": tx("mid", size_bytes=300, fee_rate=3)}
        settings = config(transactions=transactions, pool=dict(enabled=True, capacity_bytes=500),
            admissions=[dict(tx_id=name, arrival_t=i) for i, name in enumerate(("low", "high", "mid"))])
        settings["service"]["seconds"] = dict.fromkeys(("admission", "block_tx", "block_sig"), 0)
        rows, data = self.run_case(settings)
        self.assertEqual([r["tx_id"] for r in rows if r["event"] == "pool_evict"], ["low", "mid"])
        self.assertEqual(data["pool"]["bytes"], 250)
        self.assertEqual(data["pool"]["evicted_before_inclusion"], 2)

    def test_duplicate_digest_dropped_before_verification(self):
        rows, data = self.run_case(config(admissions=[dict(tx_id="a", arrival_t=0), dict(tx_id="a", arrival_t=0)],
            cache=dict(capacity=10), pool=dict(enabled=True)))
        self.assertEqual(sum(r["event"] == "dup_drop" for r in rows), 1)
        self.assertEqual(sum(r["event"] == "start" for r in rows), 1)

    def test_cached_structural_failure_has_no_signature_jobs(self):
        rows, data = self.run_case(config(admissions=[dict(tx_id="a", arrival_t=0)], block_start=2,
            genesis=dict(balances={"sender:a": 1000000}, next_nonces={"sender:a": 1}),
            blocks=[dict(block_id="B", tx_ids=["a"], structurally_valid=False, block_sig_label="bit_flip")],
            cache=dict(capacity=10)))
        self.assertEqual(data["blocks"]["B"]["reject_reasons"], "structural")
        self.assertEqual([j for j in data["jobs"] if j["block_id"]], [])
        self.assertFalse(any(r["event"] == "state_check" for r in rows))

    def test_two_cause_rejection_runs_all_jobs_and_state_checks(self):
        settings = config(transactions={"bad": tx("bad", nonce=1, sig_label="bit_flip")},
                          blocks=[dict(block_id="B", tx_ids=["bad"])])
        settings["service"]["seconds"]["block_tx"] = {"valid": 1, "bit_flip": 1}
        rows, data = self.run_case(settings)
        block = data["blocks"]["B"]
        self.assertEqual(block["reject_t"], 2)
        self.assertEqual(block["reject_reason"], "invalid_signature")
        self.assertEqual(block["reject_reasons"], "invalid_signature;state")
        self.assertEqual([(float(r["t"]), r["event"]) for r in rows if r["event"] in ("end", "state_check", "state_reject", "block_reject")],
                         [(1, "end"), (2, "end"), (2, "state_check"), (2, "state_reject"), (2, "block_reject")])
        self.assertEqual(data["ledger"], settings["genesis"])

    def test_integer_fees_intrablock_funding_and_atomic_reject(self):
        transactions = {"a": tx("a", sender="alice", receiver="bob", amount=50, fee_rate=1, size_bytes=10),
                        "b": tx("b", sender="bob", receiver="carol", amount=20, fee_rate=1, size_bytes=10),
                        "c": tx("c", sender="bob", nonce=7)}
        for reject in (False, True):
            settings = config(transactions=transactions,
                genesis=dict(balances={"alice": 100, "bob": 0}, next_nonces={"alice": 0, "bob": 0}),
                blocks=[dict(block_id="B", tx_ids=["a", "b", "c"] if reject else ["a", "b"])])
            settings["service"]["seconds"] = dict.fromkeys(("admission", "block_tx", "block_sig"), 0)
            _, data = self.run_case(settings)
            if reject:
                self.assertEqual(data["ledger"], settings["genesis"])
            else:
                self.assertEqual(data["ledger"], dict(balances={"alice": 40, "bob": 20, "carol": 20},
                                                       next_nonces={"alice": 1, "bob": 1, "carol": 0}))
        with self.assertRaises(ValueError):
            normalize_tx(tx("float", fee_rate=0.5))

    def test_state_cost_is_dispatcher_work_and_erase_on_inclusion(self):
        settings = config(admissions=[dict(tx_id="a", arrival_t=0)], state_cost=0.5,
            block_start=3, block_period=3, blocks=[dict(block_id="B0", tx_ids=["a"]), dict(block_id="B1", tx_ids=["a"])],
            cache=dict(capacity=10, erase_on_inclusion=True), pool=dict(enabled=True))
        rows, data = self.run_case(settings)
        self.assertEqual(data["blocks"]["B0"]["accept_t"], 4.5)
        self.assertEqual(sum(r["event"] == "cache_hit" and r["job_class"] == "block_tx" for r in rows), 1)
        self.assertTrue(any(r["event"] == "cache_evict" and r["detail"] == "positive:inclusion" for r in rows))
        self.assertTrue(all(r["busy_workers"] == "0" for r in rows if r["event"] == "state_check"))
