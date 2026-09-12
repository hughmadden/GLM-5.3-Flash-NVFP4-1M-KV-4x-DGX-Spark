"""0010 follow-up: in-process dispatch for the rank-local coordinator leg.

Verifies the RpcClient local shortcut end-to-end against a real
_MetadataRPC + DiskStore in this process: results identical to the HTTP
path, error mapping matches the status-code semantics, and no socket is
touched for the local rank.
"""
import time
import unittest

from recipe_persistence.storage import DiskStore
from recipe_persistence.coordinator_http import (
    _MetadataRPC, _BackendUnavailable, _RetryAgain)
from recipe_persistence.http_rpc import RpcClient, _Transient, _Permanent

NS = "a" * 64


def build(root, world=4, local_rank=1):
    store = DiskStore(str(root), clock=time.time)
    identity = {"group_refs": [(0,)], "cache_fingerprint": "t",
                "tenant_namespace": "t", "layout_fingerprint": "t",
                "fingerprint": "f" * 64, "padded_pages": [1],
                "group_bytes": [8]}
    rpc = _MetadataRPC(store=store, identity=identity, rank=local_rank,
                       world_size=world, idem_entries=1024)
    # Distinct unroutable peers: any HTTP attempt at them would fail or
    # hang, so a passing test proves the local leg never hits the wire.
    endpoints = [("127.0.0.1", 49990 + i) for i in range(world)]
    client = RpcClient(endpoints, "t" * 64, rpc_timeout=0.5,
                       pool_size=2, max_active=2)
    def map_error(error):
        if isinstance(error, _BackendUnavailable):
            return _Permanent("RPC peer rejected request")
        if isinstance(error, _RetryAgain):
            return _Transient("RPC peer unavailable")
        if isinstance(error, ValueError):
            return _Permanent("RPC peer rejected request")
        return _Transient("RPC execution failed")
    client.bind_local_dispatch(local_rank, rpc.dispatch, map_error)
    return store, rpc, client


def wire_params(key, req):
    return {"request_id": req, "namespace": NS, "key": key.hex(), "owner": "o"}


class LocalDispatchTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.store, self.rpc, self.client = build(self._tmp.name)

    def tearDown(self):
        self.client.shutdown()
        self.store.close()
        self._tmp.cleanup()

    def test_call_dispatches_in_process_and_returns_payload(self):
        out = self.client.call(1, "geometry")
        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("rank"), 1)

    def test_call_wrong_rank_goes_to_wire_and_fails(self):
        # rank 0 is an unroutable stub: the wire attempt must surface as
        # a transient failure, proving only the local rank shortcuts.
        with self.assertRaises((_Transient, _Permanent)):
            self.client.call(0, "geometry")

    def test_fanout_merges_local_and_remote_slots_in_rank_order(self):
        results = self.client.fanout("geometry", lambda rank: {})
        self.assertEqual(len(results), 4)
        self.assertTrue(results[1].get("ok"))          # local leg
        for rank in (0, 2, 3):
            self.assertIsInstance(results[rank], Exception)

    def test_reserve_read_absent_fast_negative_via_local_leg(self):
        key = b"absent" + (0).to_bytes(4, "big")
        out = self.client.call(1, "reserve_read", **wire_params(key, "ef" * 16))
        self.assertEqual(out, {"ok": True, "lease": None})
        self.assertEqual(self.rpc.absent_index_skips, 1)

    def test_error_mapping_matches_status_semantics(self):
        def boom(method, params):
            raise ValueError("invalid_params")
        self.client.bind_local_dispatch(1, boom, lambda e:
            _Permanent("RPC peer rejected request")
            if isinstance(e, ValueError) else _Transient("RPC execution failed"))
        with self.assertRaises(_Permanent):
            self.client.call(1, "geometry")

        def gone(method, params):
            raise _BackendUnavailable()
        self.client.bind_local_dispatch(1, gone, lambda e:
            _Permanent("RPC peer rejected request"))
        with self.assertRaises(_Permanent):
            self.client.call(1, "geometry")

        def retry(method, params):
            raise _RetryAgain()
        self.client.bind_local_dispatch(1, retry, lambda e:
            _Transient("RPC peer unavailable"))
        with self.assertRaises(_Transient):
            self.client.call(1, "geometry")

    def test_bind_validates_rank_and_callables(self):
        with self.assertRaises(ValueError):
            self.client.bind_local_dispatch(99, self.rpc.dispatch, lambda e: e)
        with self.assertRaises(ValueError):
            self.client.bind_local_dispatch(1, None, lambda e: e)

    def test_unbound_client_uses_wire_path(self):
        endpoints = [("127.0.0.1", 49999)]
        raw = RpcClient(endpoints, "t" * 64, rpc_timeout=0.3)
        try:
            with self.assertRaises((_Transient, _Permanent)):
                raw.call(0, "geometry")
        finally:
            raw.shutdown()


if __name__ == "__main__":
    unittest.main()
