"""0010: the in-RAM durable-key index.

Every test here is CPU-only: a temp-dir DiskStore plus the real
_MetadataRPC, no engine, no weights.
"""
import time
import unittest

NS = "a" * 64

from recipe_persistence.storage import DiskStore, Limits
from recipe_persistence.coordinator_http import _MetadataRPC


def make_store(root, **kw):
    return DiskStore(str(root), Limits(**kw) if kw else None,
                     clock=time.time)


def make_rpc(store, world=1):
    identity = {"group_refs": [(0,)], "cache_fingerprint": "t",
                "tenant_namespace": "t", "layout_fingerprint": "t"}
    return _MetadataRPC(store=store, identity=identity, rank=0,
                        world_size=world, idem_entries=1024)


def reserve_write_full(store, ns, key, size=4096, owner="w"):
    token = store.reserve_write(ns, key, size, owner)
    assert token is not None
    assert store.write(token, [bytes(size)])
    return token


class RamKeyIndexTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_rebuild_contains_exactly_the_committed_rows(self):
        store = make_store(self.root)
        reserve_write_full(store, NS, b"k-live-1")
        reserve_write_full(store, NS, b"k-live-2")
        t = store.reserve_write(NS, b"k-pending", 4096, "w")
        self.assertIsNotNone(t)  # stays 'W': never completed
        # Simulate a pre-0010 restart: close, reopen, the set is rebuilt.
        store.close()
        store2 = make_store(self.root)
        self.assertTrue(store2.durable(NS, b"k-live-1"))
        self.assertTrue(store2.durable(NS, b"k-live-2"))
        self.assertFalse(store2.durable(NS, b"k-pending"))
        self.assertFalse(store2.durable(NS, b"k-never"))
        store2.close()

    def test_absent_reserve_read_does_zero_index_io(self):
        store = make_store(self.root)
        executes = {"n": 0}
        class _CountingConn:
            def __init__(self, conn): self._conn = conn
            def __getattr__(self, name): return getattr(self._conn, name)
            def execute(self, sql, *a):
                if "objects" in sql or "leases" in sql or "totals" in sql:
                    executes["n"] += 1
                return self._conn.execute(sql, *a)
        real_db = store.db
        store.db = _CountingConn(real_db)
        t0 = time.perf_counter()
        for i in range(200):
            self.assertIsNone(store.reserve_read(NS, f"absent-{i}".encode(), "o"))
        elapsed = time.perf_counter() - t0
        # The fast negative must not run the SELECT ... state='C' probe.
        self.assertEqual(executes["n"], 0)
        self.assertLess(elapsed, 1.0)  # 200 x (lock + set lookup)
        store.close()

    def test_complete_makes_key_durable_and_readable(self):
        store = make_store(self.root)
        self.assertFalse(store.durable(NS, b"k"))
        reserve_write_full(store, NS, b"k")
        self.assertTrue(store.durable(NS, b"k"))
        lease = store.reserve_read(NS, b"k", "r")
        self.assertIsNotNone(lease)
        store.close()

    def test_invalidate_forgets_durability(self):
        store = make_store(self.root)
        reserve_write_full(store, NS, b"k")
        store.invalidate(NS, b"k")
        self.assertFalse(store.durable(NS, b"k"))
        self.assertIsNone(store.reserve_read(NS, b"k", "r"))
        store.close()

    def test_reclaim_forgets_durability(self):
        store = make_store(self.root, quota=1 << 28, high=2 << 20,
                           low=1 << 20, lease_seconds=1.0, grace_seconds=0,
                           index_bytes=64 * 1024)
        keys = [f"reclaim-{i}".encode() for i in range(4)]
        for k in keys:
            reserve_write_full(store, NS, k, size=1 << 20)
        # Force expiry eligibility then evict: 'C' rows past expiry, oldest
        # touched first, are tombstoned until usage reaches the low watermark.
        for ident, in store.db.execute("SELECT id FROM objects"):
            store.db.execute("UPDATE objects SET expiry=? WHERE id=?",
                             (store.clock() - 1, ident))
        store.evict()
        forgotten = [k for k in keys if not store.durable(NS, k)]
        self.assertTrue(forgotten)          # at least one key was reclaimed
        for k in forgotten:
            self.assertIsNone(store.reserve_read(NS, k, "r"))
        store.close()

    def test_fallback_none_set_preserves_sql_path(self):
        store = make_store(self.root)
        store._durable = None          # unbuilt-index fallback
        self.assertTrue(store.durable(NS, b"anything"))
        # reserve_read must still refuse a genuinely absent key via SQL.
        self.assertIsNone(store.reserve_read(NS, b"absent", "o"))
        store.close()

    def test_exists_uses_fast_negative(self):
        store = make_store(self.root)
        executes = {"n": 0}
        class _CountingConn:
            def __init__(self, conn): self._conn = conn
            def __getattr__(self, name): return getattr(self._conn, name)
            def execute(self, sql, *a):
                if "objects" in sql or "leases" in sql or "totals" in sql:
                    executes["n"] += 1
                return self._conn.execute(sql, *a)
        store.db = _CountingConn(store.db)
        for i in range(50):
            self.assertFalse(store.exists(NS, f"x-{i}".encode()))
        self.assertEqual(executes["n"], 0)
        reserve_write_full(store, NS, b"live")
        self.assertTrue(store.exists(NS, b"live"))
        store.close()


class CoordinatorFastNegativeTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.store = make_store(self._tmp.name)
        self.rpc = make_rpc(self.store)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _params(self, key, req="ab" * 16):
        gkey = key + (0).to_bytes(4, "big")
        return {"request_id": req, "namespace": NS, "key": gkey.hex(),
                "owner": "o"}

    def test_absent_key_short_circuits_without_lease(self):
        out = self.rpc.dispatch("reserve_read", self._params(b"nope-1"))
        self.assertEqual(out, {"ok": True, "lease": None})
        self.assertEqual(self.rpc.absent_index_skips, 1)

    def test_present_key_takes_the_real_path(self):
        reserve_write_full(self.store, NS, b"live-key" + (0).to_bytes(4, "big"))
        out = self.rpc.dispatch("reserve_read", self._params(b"live-key"))
        self.assertTrue(out.get("ok"))
        self.assertIsNotNone(out.get("lease"))
        self.assertEqual(self.rpc.absent_index_skips, 0)

    def test_idempotent_replay_for_absent_is_consistent(self):
        # Same request_id replayed for an absent key: same fast answer,
        # no lease materialises later.
        p = self._params(b"gone", req="cd" * 16)
        self.assertEqual(self.rpc.dispatch("reserve_read", p),
                         {"ok": True, "lease": None})
        self.assertEqual(self.rpc.dispatch("reserve_read", p),
                         {"ok": True, "lease": None})
        self.assertEqual(self.rpc.absent_index_skips, 2)


if __name__ == "__main__":
    unittest.main()
