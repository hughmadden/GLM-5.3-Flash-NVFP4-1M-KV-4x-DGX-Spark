# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hugh Madden and contributors
"""CPU-only localhost tests for the authenticated HTTP metadata provider.

Every core flow exercises real HTTP sockets on ephemeral 127.0.0.1 ports with
ephemeral per-rank DiskStore roots; nothing mocks the transport in-process and
no test touches any host outside the loopback interface.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import re
import resource
import secrets
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

from recipe_persistence import DiskStore, Ticket, fingerprint
from recipe_persistence import coordinator_http as ch
from recipe_persistence.geometry import Geometry
from test_storage import NS, KEY

GEOM2 = Geometry((16, 32), (((0, 7), (1, 21)), ((0, 4),)))  # group bytes (28, 4)
BIG = Geometry((2_250_000,), (((0, 2_250_000),),))          # 2.25 MiB logical rows
SMALL_LIMITS = dict(quota=2_000_000, high=1_700_000, low=1_400_000, index_bytes=65536,
                    free_bytes=0, free_inodes=0, max_object_bytes=128_000,
                    grace_seconds=5, lease_seconds=20)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _token_file(directory, token=None):
    path = Path(directory) / "coordinator.token"
    path.write_bytes((token or secrets.token_hex(32)).encode())
    os.chmod(path, 0o600)
    return path


class Deployment:
    """Real HTTP deployment: N worker endpoints plus one scheduler factory."""

    def __init__(self, world_size=2, geometry=GEOM2, *, geometries=None, limits=None,
                 max_pending_keys=8, max_pending_bytes=None, renew_margin=None,
                 rpc_timeout=2.0, startup_timeout=8.0, server_threads=None,
                 close_store=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.token_path = _token_file(self.tmp.name)
        self.token = self.token_path.read_bytes().decode()
        self.root = Path(self.tmp.name) / "root"
        self.world_size = world_size
        self.endpoints = [f"127.0.0.1:{_free_port()}" for _ in range(world_size)]
        self.cfg = {
            "coordinator_endpoints": self.endpoints,
            "coordinator_auth_token_file": str(self.token_path),
            "disk_root": str(self.root),
            "max_pending_keys": max_pending_keys,
            "coordinator_startup_timeout": startup_timeout,
            "coordinator_rpc_timeout": rpc_timeout,
            "coordinator_disk_limits": dict(SMALL_LIMITS, **(limits or {})),
        }
        if max_pending_bytes is not None:
            self.cfg["max_pending_bytes"] = max_pending_bytes
        if renew_margin is not None:
            self.cfg["coordinator_renew_margin"] = renew_margin
        if server_threads is not None:
            self.cfg["coordinator_server_threads"] = server_threads
        if close_store is not None:
            self.cfg["coordinator_close_store_on_shutdown"] = close_store
        shapes = geometries or [geometry] * world_size

        def start(rank):
            return ch.factory(config=self.cfg, role="worker", rank=rank,
                              world_size=world_size, geometry=shapes[rank])

        with ThreadPoolExecutor(max_workers=world_size) as ex:
            self.workers = list(ex.map(start, range(world_size)))
        self.scheduler = ch.factory(config=self.cfg, role="scheduler",
                                    world_size=world_size)

    @property
    def coordinator(self):
        return self.scheduler.coordinator

    def store(self, rank):
        return self.workers[rank].store

    def close(self):
        self.scheduler.close()
        for worker in reversed(self.workers):
            worker.close()
        self.tmp.cleanup()

    # -- helpers ---------------------------------------------------------- #

    def put(self, key=KEY, group=0, payload=None):
        """Full all-rank store: reserve, write every rank, complete durably."""
        sizes = self.scheduler.size_by_group[group]
        payload = payload or [b"x" * size for size in sizes]
        ticket = self.coordinator.reserve_store(NS, key, "writer", sizes)
        assert ticket is not None, "all-rank store reservation failed"
        for rank, chunk in enumerate(payload):
            assert self.store(rank).write(ticket.leases[rank], (chunk,))
        assert self.coordinator.complete_store(ticket, True) is not False
        return ticket, payload

    def leases(self, rank):
        return self.store(rank).db.execute("SELECT COUNT(*) FROM leases").fetchone()[0]

    def writing(self, rank):
        return self.store(rank).db.execute(
            "SELECT COUNT(*) FROM objects WHERE state='W'").fetchone()[0]


class ConfigValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.token_path = _token_file(self.tmp.name)
        self.base = {"coordinator_endpoints": ["127.0.0.1:1"],
                     "coordinator_auth_token_file": str(self.token_path),
                     "disk_root": str(Path(self.tmp.name) / "root"),
                     "coordinator_startup_timeout": 0.2}

    def tearDown(self):
        self.tmp.cleanup()

    def test_bad_role_world_and_endpoint_census(self):
        with self.assertRaises(ValueError):
            ch.factory(config=self.base, role="driver", world_size=1)
        with self.assertRaises(ValueError):
            ch.factory(config=self.base, world_size=0)
        bad = dict(self.base, coordinator_endpoints=["127.0.0.1:1", "127.0.0.1:2"])
        with self.assertRaises(ValueError):
            ch.factory(config=bad, world_size=1)

    def test_scheduler_refuses_rank_and_geometry(self):
        with self.assertRaises(ValueError):
            ch.factory(config=self.base, role="scheduler", world_size=1, rank=0)
        with self.assertRaises(ValueError):
            ch.factory(config=self.base, role="scheduler", world_size=1, geometry=GEOM2)

    def test_token_file_requirements(self):
        for name, mutate in [
            ("missing", lambda p: p.unlink()),
            ("group-readable", lambda p: os.chmod(p, 0o640)),
            ("world-readable", lambda p: os.chmod(p, 0o604)),
            ("symlink", lambda p: (p.unlink(), p.symlink_to("/etc/hostname"))),
        ]:
            path = Path(self.tmp.name) / f"{name}.token"
            path.write_bytes(secrets.token_hex(32).encode())
            os.chmod(path, 0o600)
            mutate(path)
            with self.assertRaises(ValueError, msg=name):
                ch.load_auth_token(str(path))
        weak = Path(self.tmp.name) / "weak.token"
        # Policy: >=32 chars, printable, and >=128 estimated bits
        # (length x log2(alphabet diversity)); "a"*64/"0"*40 collapse to
        # near-zero bits, "x"*31 is too short.
        for token in ("x" * 31, "a" * 64, "0" * 40, "abcdef01" * 2):
            weak.write_bytes(token.encode())
            os.chmod(weak, 0o600)
            with self.assertRaises(ValueError, msg=token[:12]):
                ch.load_auth_token(str(weak))

    def test_endpoint_url_policy(self):
        self.assertEqual(ch.parse_endpoint("127.0.0.1:8080"), ("127.0.0.1", 8080))
        self.assertEqual(ch.parse_endpoint("http://127.0.0.1:8080/"), ("127.0.0.1", 8080))
        self.assertEqual(ch.parse_endpoint("[::1]:0", bind=True), ("::1", 0))
        for bad in ("http://user:pw@127.0.0.1:8080", "http://127.0.0.1:8080/?x=1",
                    "http://127.0.0.1:8080/#f", "http://127.0.0.1:8080/rpc",
                    "ftp://127.0.0.1:8080", "127.0.0.1", "127.0.0.1:0",
                    "127.0.0.1:99999", ":8080", "http://127.0.0.1:8080/junk",
                    " 127.0.0.1:8080", ""):
            with self.assertRaises(ValueError, msg=bad):
                ch.parse_endpoint(bad)

    def test_import_is_cpu_only_metadata(self):
        import subprocess
        import sys
        subprocess.run([sys.executable, "-c",
                        "import sys; import recipe_persistence.coordinator_http; "
                        "assert 'torch' not in sys.modules and 'vllm' not in sys.modules"],
                       check=True)


def raw_http(endpoint, request: bytes, timeout=3.0):
    """Send prebuilt bytes; return (status, body) once Content-Length arrives."""
    host, port = endpoint.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=timeout) as s:
        s.sendall(request)
        s.settimeout(timeout)
        raw = b""
        while True:
            need = None
            head, _, body = raw.partition(b"\r\n\r\n")
            if head:
                match = re.search(rb"Content-Length: *(\d+)", head, re.I)
                if match:
                    need = len(head) + 4 + int(match.group(1))
            if need is not None and len(raw) >= need:
                break
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            raw += chunk
    status = int(raw.split(b" ", 2)[1])
    _, _, body = raw.partition(b"\r\n\r\n")
    return status, body


def json_post(endpoint, token, payload, *, path="/", headers=None, auth=None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    head = {"Content-Type": "application/json",
            "Authorization": auth if auth is not None else "Bearer " + token}
    for key, value in (headers or {}).items():
        if value is None:
            head.pop(key, None)
        else:
            head[key] = value
    head.setdefault("Content-Length", str(len(body)))
    lines = [f"POST {path} HTTP/1.1", f"Host: {endpoint.rsplit(':', 1)[0]}",
             "Connection: close"]
    lines += [f"{k}: {v}" for k, v in head.items()]
    request = ("\r\n".join(lines) + "\r\n\r\n").encode() + body
    return raw_http(endpoint, request)


class RawTransportTests(unittest.TestCase):
    """Wire-level authentication and envelope bounds against a live server."""

    @classmethod
    def setUpClass(cls):
        cls.deployment = Deployment(world_size=1)
        cls.endpoint = cls.deployment.endpoints[0]
        cls.token = cls.deployment.token

    @classmethod
    def tearDownClass(cls):
        cls.deployment.close()

    def test_geometry_rpc_shape(self):
        status, body = json_post(self.endpoint, self.token,
                                 {"method": "geometry", "params": {}})
        self.assertEqual(status, 200)
        info = json.loads(body)
        self.assertTrue(info["ok"])
        self.assertEqual(info["protocol"], ch.PROTOCOL)
        self.assertEqual((info["rank"], info["world_size"]), (0, 1))
        self.assertEqual(info["group_bytes"], [28, 4])
        self.assertEqual(info["group_refs"], [[[0, 7], [1, 21]], [[0, 4]]])
        self.assertRegex(info["fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(info["profile"], ch._profile_from_config(self.deployment.cfg))
        self.assertRegex(info["profile"], r"^[0-9a-f]{64}$")

    def test_authentication_matrix(self):
        geometry = {"method": "geometry", "params": {}}
        cases = [
            ({"Authorization": None}, 401),
            ({"Authorization": "Bearer " + "0" * 64}, 401),
            ({"Authorization": "Basic " + self.token}, 401),
            ({"Authorization": "bearer " + self.token}, 401),
        ]
        for overrides, expected in cases:
            status, body = json_post(self.endpoint, self.token, geometry, headers=overrides)
            self.assertEqual(status, expected, overrides)
            self.assertNotIn(self.token.encode(), body)

    def test_method_path_and_media_type(self):
        geometry = {"method": "geometry", "params": {}}
        self.assertEqual(json_post(self.endpoint, self.token, geometry, path="/x")[0], 404)
        status, _ = json_post(self.endpoint, self.token, geometry,
                              headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_protocol_level_rejections(self):
        host = "127.0.0.1"
        token_line = f"Authorization: Bearer {self.token}"
        no_length = (f"POST / HTTP/1.1\r\nHost: {host}\r\n{token_line}\r\n"
                     "Content-Type: application/json\r\nConnection: close\r\n\r\n").encode()
        self.assertEqual(raw_http(self.endpoint, no_length)[0], 411)
        chunked = (f"POST / HTTP/1.1\r\nHost: {host}\r\n{token_line}\r\n"
                   "Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n"
                   "Connection: close\r\n\r\n0\r\n\r\n").encode()
        self.assertEqual(raw_http(self.endpoint, chunked)[0], 400)
        doubled = (f"POST / HTTP/1.1\r\nHost: {host}\r\n{token_line}\r\n"
                   "Content-Type: application/json\r\nContent-Length: 2\r\n"
                   "Content-Length: 2\r\nConnection: close\r\n\r\n{}").encode()
        self.assertEqual(raw_http(self.endpoint, doubled)[0], 400)

    def test_body_bound_is_enforced(self):
        # Declared bodies beyond the configured bound are refused unread.
        huge = b"x" * 10
        status, _ = json_post(self.endpoint, self.token, huge,
                              headers={"Content-Length": str(10 * 1024 * 1024)})
        self.assertEqual(status, 413)

    def test_envelope_schema_rejections(self):
        good = {"method": "exists", "params": {"namespace": NS, "key": KEY.hex()}}
        cases = [
            (b"not json", 400),
            (json.dumps({"method": "geometry", "params": "notadict"}).encode(), 400),
            (json.dumps({"method": "../os", "params": {}}).encode(), 400),
            (json.dumps({"method": "write", "params": {}}).encode(), 400),
            (json.dumps({"method": "exists", "params": {
                "namespace": NS, "key": KEY.hex(), "path": "/etc/passwd"}}).encode(), 400),
            (json.dumps({"method": "exists", "params": {
                "namespace": "not-a-fingerprint", "key": KEY.hex()}}).encode(), 400),
            (json.dumps({"method": "exists", "params": {
                "namespace": NS, "key": "zz"}}).encode(), 400),
        ]
        for payload, expected in cases:
            status, _ = json_post(self.endpoint, self.token, payload)
            self.assertEqual(status, expected, payload[:40])

    def test_reserve_write_and_release_over_http(self):
        payload = {"method": "reserve_write", "params": {
            "request_id": secrets.token_hex(16), "namespace": NS,
            "key": (b"r" * 8 + bytes(4)).hex(), "owner": "raw", "size": 28}}
        status, body = json_post(self.endpoint, self.token, payload)
        self.assertEqual(status, 200)
        lease = json.loads(body)["lease"]
        self.assertRegex(lease, r"^[0-9a-f]{32}$")
        status, body = json_post(self.endpoint, self.token, {
            "method": "release", "params": {"lease": lease}})
        self.assertEqual((status, json.loads(body)["released"]), (200, True))
        unknown_group = {"method": "reserve_read", "params": {
            "request_id": secrets.token_hex(16), "namespace": NS,
            "key": (b"r" * 8 + b"\x00\x00\x00\x09").hex(), "owner": "raw"}}
        status, body = json_post(self.endpoint, self.token, unknown_group)
        self.assertEqual((status, json.loads(body)["lease"]), (200, None))


class AllRankFlowTests(unittest.TestCase):
    def setUp(self):
        self.deployment = Deployment(world_size=2)
        self.coordinator = self.deployment.coordinator

    def tearDown(self):
        self.deployment.close()

    def test_worker_shares_one_store_object_with_handler(self):
        for worker in self.deployment.workers:
            self.assertIs(worker.server.rpc.store, worker.store)

    def test_full_all_rank_store_then_load_cycle(self):
        ticket, payload = self.deployment.put()
        self.assertEqual(ticket.sizes, (28, 28))
        for rank in range(2):
            self.assertTrue(self.deployment.store(rank).exists(NS, KEY))
        load = self.coordinator.reserve_load(NS, KEY, "reader")
        self.assertIsNotNone(load)
        for rank, chunk in enumerate(payload):
            dest = bytearray(len(chunk))
            self.assertTrue(self.deployment.store(rank).read_into(load.leases[rank], (dest,)))
            self.assertEqual(bytes(dest), chunk)
        self.assertTrue(self.coordinator.renew(load))
        self.assertIsNone(self.coordinator.release(load))
        self.assertEqual(self.coordinator._bytes, 0)

    def test_clean_miss_and_unknown_group_fail_closed(self):
        self.assertIsNone(self.coordinator.reserve_load(NS, b"m" * 32 + bytes(4), "r"))
        self.assertIsNone(self.coordinator.reserve_load(NS, b"m" * 32 + b"\x00\x00\x00\x05", "r"))

    def test_missing_peer_object_rolls_back_every_rank(self):
        self.deployment.put()
        self.deployment.store(1).invalidate(NS, KEY)
        self.assertIsNone(self.coordinator.reserve_load(NS, KEY, "reader"))
        for rank in range(2):
            self.assertEqual(self.deployment.leases(rank), 0)

    def test_corrupt_peer_object_never_advertised(self):
        self.deployment.put()
        store = self.deployment.store(1)
        ident = store.db.execute("SELECT id FROM objects WHERE ns=? AND key=?",
                                 (NS, KEY)).fetchone()[0]
        with open(store._path(ident), "r+b") as f:
            f.truncate(4096 + 4)
        self.assertIsNone(self.coordinator.reserve_load(NS, KEY, "reader"))
        self.assertEqual(self.deployment.leases(0), 0)

    def test_failed_store_invalidates_successful_shards(self):
        sizes = self.deployment.scheduler.size_by_group[0]
        ticket = self.coordinator.reserve_store(NS, KEY, "writer", sizes)
        self.deployment.store(0).write(ticket.leases[0], (b"x" * sizes[0],))
        self.assertIsNone(self.coordinator.complete_store(ticket, False))
        for rank in range(2):
            self.assertFalse(self.deployment.store(rank).exists(NS, KEY))
            self.assertEqual(self.deployment.leases(rank), 0)
        self.assertEqual(len(self.coordinator._tickets), 0)

    def test_success_requires_durability_on_every_rank(self):
        # Write both ranks (each write commits), then drop one rank's bytes
        # before completion: the all-rank durability proof must convert this
        # into an invalidate-everywhere, never an advertised hit.
        sizes = self.deployment.scheduler.size_by_group[0]
        ticket = self.coordinator.reserve_store(NS, KEY, "writer", sizes)
        for rank in range(2):
            self.deployment.store(rank).write(ticket.leases[rank], (b"x" * sizes[rank],))
        store = self.deployment.store(1)
        ident = store.db.execute("SELECT id FROM objects WHERE ns=? AND key=?",
                                 (NS, KEY)).fetchone()[0]
        store._path(ident).unlink()
        self.assertIsNone(self.coordinator.complete_store(ticket, True))
        for rank in range(2):
            self.assertFalse(self.deployment.store(rank).exists(NS, KEY))

    def test_reserve_idempotency_and_owner_isolation(self):
        ticket, _ = self.deployment.put()
        first = self.coordinator.reserve_load(NS, KEY, "reader")
        self.assertIs(self.coordinator.reserve_load(NS, KEY, "reader"), first)
        second = self.coordinator.reserve_load(NS, KEY, "other")
        self.assertIsNot(first, second)
        self.coordinator.release(first)
        self.coordinator.release(second)

    def test_bounded_key_and_byte_credits(self):
        # Tight bounds need their own deployments: the shared setUp fixture
        # keeps up to eight concurrent tickets.
        keyed = Deployment(world_size=2, max_pending_keys=2)
        try:
            keyed.put()
            a = keyed.coordinator.reserve_load(NS, KEY, "a")
            b = keyed.coordinator.reserve_load(NS, KEY, "b")
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            self.assertIsNone(keyed.coordinator.reserve_load(NS, KEY, "c"))  # key bound
            keyed.coordinator.release(a)
            self.assertIsNotNone(keyed.coordinator.reserve_load(NS, KEY, "c"))
            keyed.coordinator.release(b)
            keyed.coordinator.release(keyed.coordinator.reserve_load(NS, KEY, "b"))
        finally:
            keyed.close()
        bytey = Deployment(world_size=2, max_pending_bytes=40)  # one 28B charge fits
        try:
            bytey.put()
            first = bytey.coordinator.reserve_load(NS, KEY, "a")
            self.assertIsNotNone(first)
            self.assertIsNone(bytey.coordinator.reserve_load(NS, KEY, "b"))  # byte bound
            bytey.coordinator.release(first)
        finally:
            bytey.close()

    def test_capacity_deny_on_one_rank_rolls_back_the_other(self):
        # A pre-existing rank-1 object denies its reservation after rank 0
        # already reserved: the partial reservation must be rolled back.
        direct = self.deployment.store(1).reserve_write(NS, KEY, 28, "direct")
        self.assertIsNotNone(direct)
        self.deployment.store(1).write(direct, (b"z" * 28,))
        self.assertIsNone(self.coordinator.reserve_store(
            NS, KEY, "writer", self.deployment.scheduler.size_by_group[0]))
        self.assertEqual(self.deployment.leases(0), 0)
        self.assertEqual(self.deployment.writing(0), 0)
        self.assertEqual(self.coordinator._bytes, 0)

    def test_slow_peer_deadline_returns_none_and_rolls_back(self):
        # ttl 1s, margin 0.25 -> cached window 0.75s; the gated rank stalls
        # 0.9s, so the window is consumed even though both ranks succeed.
        deployment = Deployment(world_size=2, limits={"lease_seconds": 1.0},
                                renew_margin=0.25, rpc_timeout=2.0)
        try:
            deployment.put()
            store = deployment.store(1)
            original = store.reserve_read

            def slow(namespace, key, owner):
                time.sleep(0.9)
                return original(namespace, key, owner)

            store.reserve_read = slow
            try:
                started = time.monotonic()
                self.assertIsNone(deployment.coordinator.reserve_load(NS, KEY, "reader"))
                self.assertGreaterEqual(time.monotonic() - started, 0.9)
                self.assertEqual(deployment.leases(0), 0)
            finally:
                del store.reserve_read
            # With the stall gone the same reservation succeeds.
            self.assertIsNotNone(deployment.coordinator.reserve_load(NS, KEY, "reader"))
        finally:
            deployment.close()

    def test_dead_peer_fails_closed_for_every_operation(self):
        ticket, _ = self.deployment.put()
        self.deployment.workers[1].close()
        self.assertIsNone(self.coordinator.reserve_load(NS, KEY, "reader"))
        self.assertIsNone(self.coordinator.reserve_store(
            NS, KEY, "writer", self.deployment.scheduler.size_by_group[0]))
        self.assertFalse(self.coordinator.renew(ticket))


class CensusTests(unittest.TestCase):
    def test_world_four_census_and_all_rank_reservation(self):
        deployment = Deployment(world_size=4)
        try:
            self.assertEqual(deployment.scheduler.size_by_group, ((28, 28, 28, 28),
                                                                  (4, 4, 4, 4)))
            ticket = deployment.coordinator.reserve_store(
                NS, KEY, "writer", deployment.scheduler.size_by_group[0])
            self.assertIsNotNone(ticket)
            self.assertEqual(len(ticket.leases), 4)
            self.assertIsNone(deployment.coordinator.complete_store(ticket, False))
            for rank in range(4):
                self.assertFalse(deployment.store(rank).exists(NS, KEY))
        finally:
            deployment.close()

    def test_layout_fingerprint_matches_public_formula(self):
        # Independent recomputation of the documented serialization: SHA-256
        # over compact ASCII JSON of the rank-ordered [padded_pages,group_refs].
        import hashlib
        pages = list(GEOM2.padded_pages)
        refs = [[list(r) for r in group] for group in GEOM2.group_refs]
        expected = hashlib.sha256(json.dumps([[pages, refs], [pages, refs]],
                                             separators=(",", ":")).encode("ascii")
                                  ).hexdigest()
        deployment = Deployment(world_size=2)
        try:
            for provider in (*deployment.workers, deployment.scheduler):
                digest = provider.layout_fingerprint
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
                self.assertEqual(digest, expected)
        finally:
            deployment.close()

    def test_layout_fingerprint_distinguishes_reference_order(self):
        # Same per-group byte totals, different canonical order/aliasing.
        straight = [{"padded_pages": [16, 32], "group_refs": [[[0, 7], [1, 21]]]}]
        swapped = [{"padded_pages": [16, 32], "group_refs": [[[1, 21], [0, 7]]]}]
        self.assertNotEqual(ch.layout_fingerprint(straight),
                            ch.layout_fingerprint(swapped))

    def test_profile_mismatch_fails_closed_fast(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            token_path = _token_file(tmp.name)
            endpoints = [f"127.0.0.1:{_free_port()}" for _ in range(2)]
            base = {"coordinator_endpoints": endpoints,
                    "coordinator_auth_token_file": str(token_path),
                    "disk_root": str(Path(tmp.name) / "root"),
                    "max_pending_keys": 4,
                    "coordinator_startup_timeout": 6.0,
                    "coordinator_rpc_timeout": 1.0,
                    "coordinator_disk_limits": SMALL_LIMITS}

            def start(rank):
                cfg = dict(base, cache_fingerprint=f"model-{rank}")
                return ch.factory(config=cfg, role="worker", rank=rank,
                                  world_size=2, geometry=GEOM2)

            started = time.monotonic()
            results = [None, None]
            def guarded(rank):
                try:
                    start(rank)
                    results[rank] = "ok"
                except Exception as e:  # noqa: BLE001
                    results[rank] = e
            with ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(guarded, range(2)))
            elapsed = time.monotonic() - started
            # Bounded: the fast mismatch raises immediately; the peer whose
            # conflicting server died first fail-closes on its own deadline.
            self.assertLess(elapsed, 8.0)
            self.assertTrue(all(isinstance(r, RuntimeError) for r in results), results)
            self.assertTrue(any("rejected" in str(r) for r in results), results)
        finally:
            tmp.cleanup()

    def test_geometry_mismatch_fails_closed_fast(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            token_path = _token_file(tmp.name)
            endpoints = [f"127.0.0.1:{_free_port()}" for _ in range(2)]
            cfg = {"coordinator_endpoints": endpoints,
                   "coordinator_auth_token_file": str(token_path),
                   "disk_root": str(Path(tmp.name) / "root"),
                   "max_pending_keys": 4,
                   "coordinator_startup_timeout": 6.0,
                   "coordinator_rpc_timeout": 1.0,
                   "coordinator_disk_limits": SMALL_LIMITS}
            mismatch = Geometry((16, 32), (((0, 8), (1, 21)), ((0, 4),)))

            def start(rank):
                return ch.factory(config=cfg, role="worker", rank=rank, world_size=2,
                                  geometry=GEOM2 if rank == 0 else mismatch)

            started = time.monotonic()
            results = [None, None]
            def guarded(rank):
                try:
                    start(rank)
                    results[rank] = "ok"
                except Exception as e:  # noqa: BLE001
                    results[rank] = e
            with ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(guarded, range(2)))
            elapsed = time.monotonic() - started
            # At least one rank detects the layout mismatch immediately; the
            # other may legitimately wait out its deadline after the
            # conflicting peer shuts down first. Both must fail closed.
            self.assertLess(elapsed, 8.0)
            self.assertTrue(all(isinstance(r, RuntimeError) for r in results), results)
            self.assertTrue(any("rejected" in str(r) for r in results), results)
        finally:
            tmp.cleanup()

    def test_scheduler_census_deadline_is_bounded(self):
        deployment = Deployment(world_size=2)
        try:
            deployment.workers[1].close()
            cfg = dict(deployment.cfg, coordinator_startup_timeout=1.5)
            started = time.monotonic()
            with self.assertRaises(RuntimeError):
                ch.factory(config=cfg, role="scheduler", world_size=2)
            self.assertLess(time.monotonic() - started, 6.0)
        finally:
            deployment.close()


class RenewalThrottleTests(unittest.TestCase):
    def setUp(self):
        self.deployment = Deployment(world_size=2, limits={"lease_seconds": 1.0},
                                      renew_margin=0.25)
        self.coordinator = self.deployment.coordinator
        self.counts = []
        for worker in self.deployment.workers:
            original = worker.store.renew
            counter = {"n": 0}
            self.counts.append(counter)

            def counted(token, _original=original, _counter=counter):
                _counter["n"] += 1
                return _original(token)

            worker.store.renew = counted

    def tearDown(self):
        self.deployment.close()

    def test_on_schedule_end_bursts_are_throttled_not_replayed(self):
        self.deployment.put()
        load = self.coordinator.reserve_load(NS, KEY, "reader")
        # Fifty decode steps against one ticket stay inside the cached window.
        for _ in range(50):
            self.assertTrue(self.coordinator.renew(load))
        self.assertEqual([c["n"] for c in self.counts], [0, 0])
        time.sleep(0.9)  # window (ttl - margin) expires, lease still alive
        self.assertTrue(self.coordinator.renew(load))
        self.assertEqual([c["n"] for c in self.counts], [1, 1])
        for _ in range(50):
            self.assertTrue(self.coordinator.renew(load))
        self.assertEqual([c["n"] for c in self.counts], [1, 1])

    def test_cached_success_never_outlives_the_real_lease(self):
        self.deployment.put()
        load = self.coordinator.reserve_load(NS, KEY, "reader")
        time.sleep(0.9)
        self.assertTrue(self.coordinator.renew(load))  # real renewal, lease extended
        time.sleep(1.3)  # past the extended lease: renewal must fail closed
        self.assertFalse(self.coordinator.renew(load))
        self.assertIsNone(self.coordinator.release(load))


class SlowRankDeadlineTests(unittest.TestCase):
    """Windows anchor at operation START: slow fan-out latency consumes TTL."""

    def setUp(self):
        # ttl 4s, margin 0.5s -> cached window 3.5s; the gated rank stalls 3.7s.
        self.deployment = Deployment(world_size=2, limits={"lease_seconds": 4.0},
                                      renew_margin=0.5, rpc_timeout=6.0)
        self.coordinator = self.deployment.coordinator

    def tearDown(self):
        self.deployment.close()

    def test_slow_reserve_consumes_window_and_rolls_back(self):
        self.deployment.put()
        store = self.deployment.store(1)
        original = store.reserve_read

        def slow(namespace, key, owner):
            time.sleep(3.7)
            return original(namespace, key, owner)

        store.reserve_read = slow
        try:
            started = time.monotonic()
            self.assertIsNone(self.coordinator.reserve_load(NS, KEY, "reader"))
            self.assertGreaterEqual(time.monotonic() - started, 3.7)
            self.assertEqual(self.deployment.leases(0), 0)  # rank 0 rolled back
            self.assertEqual(self.coordinator._bytes, 0)
        finally:
            del store.reserve_read
        # With the stall gone the same reservation succeeds.
        self.assertIsNotNone(self.coordinator.reserve_load(NS, KEY, "reader"))

    def test_slow_renew_fails_closed_instead_of_caching_stale_success(self):
        self.deployment.put()
        load = self.coordinator.reserve_load(NS, KEY, "reader")
        time.sleep(3.6)  # cached window (3.5s) expires while lease (4s) lives
        store = self.deployment.store(1)
        original = store.renew

        def slow(token):
            time.sleep(3.7)
            return original(token)

        store.renew = slow
        try:
            self.assertFalse(self.coordinator.renew(load))  # window consumed
        finally:
            del store.renew
        self.assertIsNone(self.coordinator.release(load))


class LongPrefixMetadataTests(unittest.TestCase):
    """Descriptor admission: ~4219 keys / ~9GiB logical, few MiB physical."""

    def test_long_prefix_logical_credits_without_huge_allocation(self):
        real_statvfs = os.statvfs

        def inflated(path):
            info = real_statvfs(path)
            fields = list(info)  # struct order: ..., idx4=f_bavail, idx7=f_favail
            fields[4] = max(info.f_bavail, 10 ** 9)  # f_bavail: free blocks
            fields[7] = max(info.f_favail, 10 ** 9)  # f_favail: free inodes
            return os.statvfs_result(fields)

        # index_bytes must cover ~4219 object rows plus index pages; the
        # 64KiB small-fixture budget caps SQLite at 16 pages and would deny
        # reservations after ~100 objects without any real space pressure.
        deployment = Deployment(world_size=2, geometry=BIG, max_pending_keys=32768,
                                limits={"quota": 64_000_000_000, "high": 60_000_000_000,
                                        "low": 50_000_000_000, "index_bytes": 8_388_608,
                                        "max_object_bytes": 64_000_000,
                                         "lease_seconds": 300})
        try:
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            keys = [i.to_bytes(8, "big") + bytes(4) for i in range(4219)]
            tickets = []
            reserve_started = time.monotonic()
            with mock.patch("os.statvfs", inflated):
                # Use the production TTL: this tests descriptor admission,
                # not expiry/GC while deliberately withholding all renewals.
                for index, key in enumerate(keys):
                    ticket = deployment.coordinator.reserve_store(
                        NS, key, "full-prefix", deployment.scheduler.size_by_group[0])
                    self.assertIsNotNone(ticket, f"reservation declined at descriptor {index}")
                    tickets.append(ticket)
            reserve_seconds = time.monotonic() - reserve_started
            logical = sum(max(t.sizes) for t in tickets)
            self.assertGreater(logical, 9_000_000_000)      # ~9GiB logical, no 256MiB cap
            self.assertEqual(len(deployment.coordinator._tickets), 4219)
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.assertLess(rss_after - rss_before, 256 * 1024)  # KB: no payload alloc
            release_started = time.monotonic()
            for ticket in tickets:
                self.assertIsNone(deployment.coordinator.release(ticket))
            release_seconds = time.monotonic() - release_started
            print(json.dumps({"fixture":"metadata_only", "keys":len(tickets),
                              "logical_bytes":logical, "reserve_seconds":reserve_seconds,
                              "release_seconds":release_seconds}, sort_keys=True))
            self.assertEqual(deployment.coordinator._bytes, 0)
            for rank in range(2):
                self.assertEqual(deployment.writing(rank), 0)
        finally:
            deployment.close()


class LifecycleTests(unittest.TestCase):
    def test_close_is_idempotent_and_drains_sockets_threads_and_store(self):
        baseline = threading.active_count()
        deployment = Deployment(world_size=2)
        try:
            self.assertTrue(all(callable(w.close) for w in deployment.workers))
            self.assertGreater(threading.active_count(), baseline)
        finally:
            deployment.scheduler.close()
            for worker in reversed(deployment.workers):
                worker.close()
                worker.close()  # idempotent
        deadline = time.monotonic() + 10.0
        while threading.active_count() > baseline + 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertLessEqual(threading.active_count(), baseline + 1)
        for worker in deployment.workers:
            self.assertTrue(worker.store.closed)
            with self.assertRaises(OSError):
                host, port = worker.endpoint.rsplit(":", 1)
                socket.create_connection((host, int(port)), timeout=1.0).close()
        deployment.tmp.cleanup()

    def test_worker_close_stops_endpoint_before_store_close(self):
        deployment = Deployment(world_size=1)
        worker = deployment.workers[0]
        order = []
        original_close = worker.store.close
        server_stopped = worker.server

        def traced_store_close():
            order.append(("store", original_close is not None,
                          original_server_stopped(server_stopped)))

        def original_server_stopped(server):
            return server._fully_stopped

        worker.store.close = traced_store_close
        try:
            worker.close()
            self.assertEqual(order, [("store", True, True)])
        finally:
            if not worker.store.closed:
                original_close()
            deployment.scheduler.close()
            deployment.tmp.cleanup()

    def test_saturated_server_fails_closed_with_503(self):
        deployment = Deployment(world_size=1, server_threads=1)
        try:
            deployment.put()  # committed object on the single rank
            store = deployment.store(0)
            gate = threading.Event()
            original = store.reserve_read

            def gated(namespace, key, owner):
                gate.wait(5.0)
                return original(namespace, key, owner)

            store.reserve_read = gated
            outcome = {}

            def blocked():
                outcome["ticket"] = deployment.coordinator.reserve_load(NS, KEY, "reader")

            thread = threading.Thread(target=blocked)
            thread.start()
            try:
                time.sleep(0.3)  # the single permit is now held by the gated call
                status, body = json_post(deployment.endpoints[0], deployment.token,
                                         {"method": "geometry", "params": {}})
                self.assertEqual(status, 503)
                self.assertEqual(json.loads(body)["error"], "server_saturated")
            finally:
                gate.set()
                thread.join(10.0)
                del store.reserve_read
            self.assertIsNotNone(outcome["ticket"])
            deployment.coordinator.release(outcome["ticket"])
        finally:
            deployment.close()

    def test_scheduler_close_releases_client_pool_only(self):
        deployment = Deployment(world_size=2)
        scheduler = deployment.scheduler
        scheduler.close()
        self.assertIsNone(scheduler.store)
        # Fail closed, never a transport exception raised into the caller.
        self.assertIsNone(scheduler.coordinator.reserve_load(NS, KEY, "reader"))
        stale = Ticket("x", NS, KEY, ("a" * 32, "b" * 32), (28, 28), "reader", False)
        self.assertFalse(scheduler.coordinator.renew(stale))
        for worker in deployment.workers:  # workers still serve after scheduler close
            self.assertFalse(worker.store.closed)
        deployment.close()


class DiskReviewRegressions(unittest.TestCase):
    """Targeted CPU/HTTP regressions for the six disk-owner review findings."""

    def test_later_rank_success_after_failure_is_rolled_back(self):
        # (1) rank 0 denies; rank 1 succeeds. The early-break bug leaked the
        # rank 1 lease: it must be released, not just the failed rank skipped.
        deployment = Deployment(world_size=2)
        try:
            direct = deployment.store(1).reserve_write(NS, KEY, 28, "direct")
            self.assertTrue(deployment.store(1).write(direct, (b"z" * 28,)))
            deployment.store(1).release(direct)
            self.assertIsNone(deployment.coordinator.reserve_load(NS, KEY, "reader"))
            self.assertEqual(deployment.leases(1), 0)
            self.assertEqual(deployment.leases(0), 0)
        finally:
            deployment.close()

    def test_concurrent_reserves_cannot_exceed_key_bound(self):
        # (2) commit-time recheck: with max_pending_keys=1 exactly one of two
        # simultaneous owners wins; the loser rolls back every rank. Both
        # reserve the SAME stored key under different owners.
        deployment = Deployment(world_size=2, max_pending_keys=1)
        try:
            deployment.put()  # committed object on both ranks
            barrier = threading.Barrier(3)
            outcomes = []

            def reserve(owner):
                barrier.wait()
                outcomes.append(deployment.coordinator.reserve_load(NS, KEY, owner))

            threads = [threading.Thread(target=reserve, args=(c,)) for c in ("a", "b")]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(10.0)
            self.assertEqual(sorted(t is not None for t in outcomes), [False, True])
            self.assertEqual(len(deployment.coordinator._tickets), 1)
            for ticket in list(deployment.coordinator._tickets.values()):
                deployment.coordinator.release(ticket)
            for rank in range(2):
                self.assertEqual(deployment.leases(rank), 0)
        finally:
            deployment.close()

    def test_close_refuses_store_close_until_drain_is_proved(self):
        # (3) a gated handler still holds a permit: close must raise, leave the
        # store open, and succeed (closing the store) once the stall clears.
        deployment = Deployment(world_size=1, server_threads=1)
        worker = deployment.workers[0]
        try:
            deployment.put()
            store = deployment.store(0)
            gate = threading.Event()
            original = store.reserve_read

            def gated(namespace, key, owner):
                gate.wait(10.0)
                return original(namespace, key, owner)

            store.reserve_read = gated
            thread = threading.Thread(
                target=lambda: deployment.coordinator.reserve_load(NS, KEY, "reader"))
            thread.start()
            try:
                time.sleep(0.3)
                with self.assertRaises(RuntimeError):
                    worker.close()
                self.assertFalse(worker.store.closed)  # refused, not closed
            finally:
                gate.set()
                thread.join(10.0)
                del store.reserve_read
            worker.close()  # retryable: now drains and closes the store
            self.assertTrue(worker.store.closed)
        finally:
            deployment.close()

    def test_simultaneous_same_request_id_reserves_exactly_once(self):
        # (4) eight concurrent identical request ids: one rank lease, one
        # store reservation, every response returns the same token.
        deployment = Deployment(world_size=1)
        try:
            deployment.put()
            endpoint, token = deployment.endpoints[0], deployment.token
            store = deployment.store(0)
            calls = {"n": 0}
            original = store.reserve_read

            def counted(namespace, key, owner):
                calls["n"] += 1
                return original(namespace, key, owner)

            store.reserve_read = counted
            request_id = secrets.token_hex(16)
            payload = {"method": "reserve_read", "params": {
                "request_id": request_id, "namespace": NS,
                "key": KEY.hex(), "owner": "same-id"}}
            barrier = threading.Barrier(9)
            leases = []

            def call():
                barrier.wait()
                status, body = json_post(endpoint, token, payload)
                leases.append((status, json.loads(body).get("lease")))

            threads = [threading.Thread(target=call) for _ in range(8)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(10.0)
            self.assertEqual(calls["n"], 1)
            self.assertEqual(len({lease for _, lease in leases}), 1)
            self.assertEqual(deployment.leases(0), 1)
            del store.reserve_read
        finally:
            deployment.close()

    def test_bind_failure_releases_store_root_and_threads(self):
        # (5) an occupied port must fail the worker factory without leaking
        # the exclusive root lock or lingering client threads.
        tmp = tempfile.TemporaryDirectory()
        with socket.socket() as squatter:
            squatter.bind(("127.0.0.1", 0))
            squatter.listen(1)
            port = squatter.getsockname()[1]
            token_path = _token_file(tmp.name)
            cfg = {"coordinator_endpoints": [f"127.0.0.1:{port}"],
                   "coordinator_auth_token_file": str(token_path),
                   "disk_root": str(Path(tmp.name) / "root"),
                   "max_pending_keys": 4, "coordinator_startup_timeout": 2.0,
                   "coordinator_rpc_timeout": 1.0,
                   "coordinator_disk_limits": SMALL_LIMITS}
            baseline = threading.active_count()
            with self.assertRaises(OSError):
                ch.factory(config=cfg, role="worker", rank=0, world_size=1, geometry=GEOM2)
            # The root lock was released: a fresh DiskStore opens immediately.
            from recipe_persistence import Limits
            probe = DiskStore(Path(cfg["disk_root"]) / "rank-0",
                              Limits(**SMALL_LIMITS))
            probe.close()
            deadline = time.monotonic() + 5.0
            while threading.active_count() > baseline + 1 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertLessEqual(threading.active_count(), baseline + 1)
        tmp.cleanup()

    def test_census_failure_shuts_down_client_pool(self):
        # (5b) a census failure (wrong credential) must not leak the client's
        # executor threads.
        deployment = Deployment(world_size=2)
        try:
            other = _token_file(deployment.tmp.name)
            baseline = threading.active_count()
            cfg = dict(deployment.cfg, coordinator_auth_token_file=str(other),
                       coordinator_startup_timeout=1.0)
            with self.assertRaises(RuntimeError):
                ch.factory(config=cfg, role="scheduler", world_size=2)
            deadline = time.monotonic() + 5.0
            while threading.active_count() > baseline + 1 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertLessEqual(threading.active_count(), baseline + 1)
        finally:
            deployment.close()

    def test_overlong_profile_rejected_and_scheduler_profile_checked(self):
        # (6) overlong identity fields are rejected, never truncated; and the
        # scheduler census compares ranks against ITS configured profile.
        tmp = tempfile.TemporaryDirectory()
        try:
            token_path = _token_file(tmp.name)
            base = {"coordinator_endpoints": [f"127.0.0.1:{_free_port()}"],
                    "coordinator_auth_token_file": str(token_path),
                    "disk_root": str(Path(tmp.name) / "root"),
                    "max_pending_keys": 4, "coordinator_startup_timeout": 4.0,
                    "coordinator_rpc_timeout": 1.0,
                    "coordinator_disk_limits": SMALL_LIMITS}
            overlong = dict(base, cache_fingerprint="f" * 2000)
            with self.assertRaises(ValueError):
                ch.factory(config=overlong, role="worker", rank=0, world_size=1,
                           geometry=GEOM2)
            deployment = Deployment(world_size=1)  # empty profile on both roles
            try:
                cfg = dict(deployment.cfg, cache_fingerprint="other-model-profile",
                           coordinator_startup_timeout=2.0)
                with self.assertRaises(RuntimeError):
                    ch.factory(config=cfg, role="scheduler", world_size=1)
            finally:
                deployment.close()
        finally:
            tmp.cleanup()


class HardeningRegressions(unittest.TestCase):
    def test_terminal_http_refusal_disables_native_without_releasing_issued_load(self):
        from types import SimpleNamespace
        from recipe_persistence.native import _Manager
        from recipe_persistence.handlers import DiskLoadStoreSpec
        deployment = Deployment(world_size=1)
        manager = None
        try:
            deployment.put()
            coordinator = deployment.coordinator
            manager = _Manager(coordinator, NS, ((28,), (4,)), 8,
                DiskLoadStoreSpec, SimpleNamespace, SimpleNamespace,
                lookup_keys_per_step=8, metadata_workers=2)
            ctx = SimpleNamespace(req_id="issued")
            self.assertIs(manager.lookup(KEY, ctx), True)
            manager.prepare_load([KEY], ctx)
            manager.on_schedule_end()
            token = coordinator.client._token
            try:
                coordinator.client._token = b"0"*64
                self.assertIs(manager.lookup(b"other"+bytes(4), SimpleNamespace(req_id="next")), False)
            finally:
                coordinator.client._token = token
            self.assertFalse(coordinator.can_store())
            self.assertFalse(manager.can_store())
            self.assertTrue(manager.closed)
            self.assertIn((ctx.req_id, KEY), manager.prepared)
            self.assertEqual(coordinator._bytes, 28)
            with self.assertRaises(RuntimeError):
                manager.on_request_finished(ctx)
            # Only the real core's all-rank physical-drain callback authorizes retirement.
            manager.complete_load([KEY], ctx)
            manager.shutdown()
            self.assertEqual(coordinator._bytes, 0)
        finally:
            if manager is not None and not manager.prepared:
                manager.shutdown()
            deployment.close()

    def test_terminal_rank_store_failure_is_not_disguised_as_cache_miss(self):
        from types import SimpleNamespace
        from recipe_persistence.native import _Manager
        from recipe_persistence.handlers import DiskLoadStoreSpec
        deployment = Deployment(world_size=1)
        manager = _Manager(deployment.coordinator, NS, ((28,), (4,)), 8,
            DiskLoadStoreSpec, SimpleNamespace, SimpleNamespace, metadata_workers=2)
        try:
            deployment.store(0).failed = True
            self.assertIs(manager.lookup(KEY, SimpleNamespace(req_id="reader")), False)
            self.assertFalse(deployment.coordinator.can_store())
            self.assertFalse(manager.can_store())
        finally:
            manager.shutdown()
            deployment.close()

    def test_native_capability_distinguishes_terminal_flags_from_capacity(self):
        from types import SimpleNamespace
        from recipe_persistence.native import _Manager
        from recipe_persistence.handlers import DiskLoadStoreSpec
        deployment = Deployment(world_size=1, max_pending_keys=1)
        coordinator = deployment.coordinator
        def make_manager():
            return _Manager(coordinator, NS, ((28,), (4,)), 8,
                DiskLoadStoreSpec, SimpleNamespace, SimpleNamespace,
                lookup_keys_per_step=8, metadata_workers=2)
        try:
            held = coordinator.reserve_store(NS, KEY, "held", (28,))
            manager = make_manager()
            try:
                self.assertIsNone(manager.prepare_store([b"other"+bytes(4)], SimpleNamespace(req_id="waiting")))
                self.assertTrue(manager.can_store())
                self.assertFalse(manager.closed)
            finally:
                manager.shutdown()
                coordinator.release(held)
            for flag in ("_closed", "_cleanup_lost"):
                manager = make_manager()
                try:
                    setattr(coordinator, flag, True)
                    self.assertFalse(manager.can_store())
                    self.assertTrue(manager.closed)
                finally:
                    setattr(coordinator, flag, False)
                    manager.shutdown()
        finally:
            deployment.close()

    def test_release_keeps_credits_and_identity_until_real_ack(self):
        deployment = Deployment(world_size=2, max_pending_keys=1)
        try:
            deployment.put()
            coordinator = deployment.coordinator
            ticket = coordinator.reserve_load(NS, KEY, "reader")
            original = coordinator.client.call
            def deny(rank, method, **params):
                if rank == 0 and method == "release":
                    return {"ok":True, "released":False}
                return original(rank, method, **params)
            with mock.patch.object(coordinator.client, "call", side_effect=deny):
                self.assertIs(coordinator.release(ticket), False)
            self.assertEqual(coordinator._bytes, 28)
            self.assertEqual(coordinator._tickets, {ticket.token:ticket})
            self.assertIsNone(coordinator.lease_deadline(ticket))
            self.assertIsNone(coordinator.reserve_load(NS, KEY, "reader"))
            self.assertIsNone(coordinator.reserve_load(NS, KEY, "another"))
            self.assertIsNone(coordinator.release(ticket))
            self.assertEqual(coordinator._bytes, 0)
            self.assertFalse(coordinator._retiring)
        finally:
            deployment.close()

    def test_store_completion_retains_credits_until_quarantine_also_acks(self):
        deployment = Deployment(world_size=2)
        try:
            coordinator = deployment.coordinator
            ticket = coordinator.reserve_store(NS, KEY, "writer", (28, 28))
            original = coordinator.client.call
            def deny(rank, method, **params):
                if rank == 0 and method == "invalidate":
                    return {"ok":True, "invalidated":False}
                return original(rank, method, **params)
            with mock.patch.object(coordinator.client, "call", side_effect=deny):
                self.assertIs(coordinator.complete_store(ticket, False), False)
            self.assertEqual(coordinator._bytes, 28)
            self.assertIn(ticket.token, coordinator._tickets)
            self.assertIsNone(coordinator.lease_deadline(ticket))
            self.assertIsNone(coordinator.complete_store(ticket, False))
            self.assertEqual(coordinator._bytes, 0)
        finally:
            deployment.close()

    def test_invalidation_veto_survives_ack_until_stale_tickets_retire(self):
        deployment = Deployment(world_size=1)
        try:
            deployment.put()
            coordinator = deployment.coordinator
            a = coordinator.reserve_load(NS, KEY, "a")
            b = coordinator.reserve_load(NS, KEY, "b")
            self.assertIsNone(coordinator.invalidate(NS, KEY))
            self.assertIn((NS, KEY), coordinator._invalid)
            self.assertIsNone(coordinator.lease_deadline(a))
            self.assertIsNone(coordinator.reserve_load(NS, KEY, "c"))
            self.assertIsNone(coordinator.release(a))
            self.assertIn((NS, KEY), coordinator._invalid)
            self.assertIsNone(coordinator.release(b))
            self.assertNotIn((NS, KEY), coordinator._invalid)
        finally:
            deployment.close()

    def test_validity_snapshot_never_waits_for_io_or_coordinator_lock(self):
        deployment = Deployment(world_size=1)
        entered, release = threading.Event(), threading.Event()
        thread = None
        try:
            deployment.put()
            coordinator = deployment.coordinator
            ticket = coordinator.reserve_load(NS, KEY, "reader")
            def hold():
                with coordinator._lock:
                    entered.set()
                    release.wait(3)
            thread = threading.Thread(target=hold)
            thread.start()
            self.assertTrue(entered.wait(3))
            with mock.patch.object(coordinator.client, "call", side_effect=AssertionError("I/O")), \
                 mock.patch.object(coordinator.client, "fanout", side_effect=AssertionError("I/O")):
                started = time.monotonic()
                self.assertGreater(coordinator.lease_deadline(ticket), started)
                self.assertLess(time.monotonic()-started, 0.2)
        finally:
            release.set()
            if thread is not None:
                thread.join(3)
            deployment.close()

    def test_unexpired_idempotency_receipts_are_never_evicted_or_extended(self):
        deployment = Deployment(world_size=1)
        try:
            deployment.put()
            rpc = deployment.workers[0].server.rpc
            rpc._idem.clear()
            rpc._idem_cap = 2
            store = deployment.store(0)
            original_clock = store.clock
            now = [100.0]
            store.clock = lambda:now[0]
            try:
                with mock.patch.object(ch.time, "monotonic", side_effect=lambda:now[0]):
                    params = dict(namespace=NS, key=KEY.hex(), owner="reader", request_id="a"*32)
                    first = rpc.dispatch("reserve_read", params)["lease"]
                    expiry = store.db.execute("SELECT expiry FROM leases WHERE token=?", (first,)).fetchone()[0]
                    now[0] += 1
                    self.assertEqual(rpc.dispatch("reserve_read", params)["lease"], first)
                    self.assertEqual(store.db.execute("SELECT expiry FROM leases WHERE token=?", (first,)).fetchone()[0], expiry)
                    rpc.dispatch("reserve_read", dict(params, request_id="b"*32))
                    with self.assertRaises(ch._RetryAgain):
                        rpc.dispatch("reserve_read", dict(params, request_id="c"*32))
                    self.assertEqual(deployment.leases(0), 2)
                    now[0] += rpc.lease_seconds + 1
                    self.assertIsNotNone(rpc.dispatch("reserve_read", dict(params, request_id="c"*32))["lease"])
                    self.assertEqual(deployment.leases(0), 1)
            finally:
                store.clock = original_clock
        finally:
            deployment.close()

    def test_numeric_server_startup_does_not_perform_reverse_dns(self):
        with mock.patch.object(socket, "getfqdn", side_effect=AssertionError("DNS forbidden")), \
             mock.patch.object(socket, "gethostbyaddr", side_effect=AssertionError("DNS forbidden")):
            deployment = Deployment(world_size=1)
            deployment.close()

    def test_token_fifo_is_rejected_without_waiting_for_a_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"token-fifo")
            os.mkfifo(path, 0o600)
            started = time.monotonic()
            with self.assertRaises(ValueError):
                ch.load_auth_token(path)
            self.assertLess(time.monotonic()-started, 0.5)

    def test_request_headers_have_a_total_allocation_bound(self):
        deployment = Deployment(world_size=1)
        try:
            status, _ = json_post(deployment.endpoints[0], deployment.token,
                {"method":"geometry", "params":{}}, headers={"X-Oversize":"a"*17000})
            self.assertEqual(status, 431)
        finally:
            deployment.close()

    def test_geometry_claims_are_recomputed_and_ttl_must_be_finite(self):
        deployment = Deployment(world_size=1)
        try:
            info = deployment.workers[0].server.rpc.geometry_payload()
            altered = dict(info, group_bytes=[29, 4])
            with self.assertRaises(ch._Permanent):
                ch._validated_geometry(altered, 0, 1)
            altered = dict(info, padded_pages=[32, 32])
            with self.assertRaises(ch._Permanent):
                ch._validated_geometry(altered, 0, 1)
            with self.assertRaises(ch._Transient):
                ch._validated_geometry(dict(info, lease_seconds=float("inf")), 0, 1)
        finally:
            deployment.close()

    def test_cleanup_receipt_overflow_never_turns_into_acknowledgement(self):
        deployment = Deployment(world_size=1)
        try:
            coordinator = deployment.coordinator
            coordinator._retry = __import__("collections").deque(maxlen=1)
            coordinator._replay("release", 0, lease="a"*32)
            coordinator._replay("release", 0, lease="b"*32)
            self.assertEqual(coordinator.pending_retries(), 1)
            self.assertFalse(coordinator._retry_cleanup())
            self.assertEqual(coordinator.pending_retries(), 0)
            self.assertTrue(coordinator._closed)
            self.assertIsNone(coordinator.reserve_store(NS, KEY, "owner", (28,)))
        finally:
            deployment.close()

    def test_pending_rpc_already_owns_logical_credit(self):
        deployment = Deployment(world_size=1, max_pending_keys=1, max_pending_bytes=28)
        entered, release = threading.Event(), threading.Event()
        thread = None
        try:
            deployment.put()
            coordinator = deployment.coordinator
            original = coordinator.client.fanout
            def gated(method, params_for):
                entered.set()
                self.assertTrue(release.wait(3))
                return original(method, params_for)
            outcomes = []
            with mock.patch.object(coordinator.client, "fanout", side_effect=gated):
                thread = threading.Thread(target=lambda:outcomes.append(
                    coordinator.reserve_load(NS, KEY, "first")))
                thread.start()
                self.assertTrue(entered.wait(3))
                self.assertEqual(len(coordinator._tickets), 0)
                self.assertEqual(len(coordinator._pending), 1)
                self.assertEqual(coordinator._bytes, 28)
                started = time.monotonic()
                self.assertIsNone(coordinator.reserve_load(NS, KEY, "second"))
                self.assertLess(time.monotonic()-started, 0.2)
                release.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(coordinator._pending), 0)
            self.assertIsNotNone(outcomes[0])
            coordinator.release(outcomes[0])
            self.assertEqual(coordinator._bytes, 0)
        finally:
            release.set()
            if thread is not None:
                thread.join(3)
            deployment.close()

    def test_request_id_is_bound_to_method_key_namespace_and_owner(self):
        deployment = Deployment(world_size=1)
        try:
            deployment.put()
            params = dict(request_id=secrets.token_hex(16), namespace=NS,
                          key=KEY.hex(), owner="reader")
            request = {"method":"reserve_read", "params":params}
            status, body = json_post(deployment.endpoints[0], deployment.token, request)
            self.assertEqual(status, 200)
            lease = json.loads(body)["lease"]
            variants = [dict(params, owner="other"), dict(params, key=(b"other"+KEY).hex()),
                        dict(params, namespace=fingerprint(model="other"))]
            for changed in variants:
                status, body = json_post(deployment.endpoints[0], deployment.token,
                                         {"method":"reserve_read", "params":changed})
                self.assertEqual(status, 400)
                self.assertNotIn(deployment.token.encode(), body)
            status, _ = json_post(deployment.endpoints[0], deployment.token,
                {"method":"reserve_write", "params":dict(params, size=28)})
            self.assertEqual(status, 400)
            self.assertEqual(deployment.leases(0), 1)
            deployment.store(0).release(lease)
            # A stale retry is not a fresh reservation or a false hit.
            status, body = json_post(deployment.endpoints[0], deployment.token, request)
            self.assertEqual(status, 200)
            self.assertIsNone(json.loads(body)["lease"])
            self.assertEqual(deployment.leases(0), 0)
        finally:
            deployment.close()

    def test_profile_binds_raw_seed_and_rejects_scheduler_seed_drift(self):
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED":"7"}):
            a = ch._profile_from_config({})
            deployment = Deployment(world_size=1)
        try:
            with mock.patch.dict(os.environ, {"PYTHONHASHSEED":"07"}):
                self.assertNotEqual(a, ch._profile_from_config({}))
                with self.assertRaises(RuntimeError):
                    ch.factory(config=deployment.cfg, role="scheduler", world_size=1)
        finally:
            deployment.close()

    def test_endpoint_dns_and_unspecified_destinations_are_rejected(self):
        for endpoint in ("localhost:8080", "example.invalid:8080", "0.0.0.0:8080",
                         "http://[::]:8080", "224.0.0.1:8080"):
            with self.assertRaises(ValueError):
                ch.parse_endpoint(endpoint)
        self.assertEqual(ch.parse_endpoint("0.0.0.0:0", bind=True), ("0.0.0.0", 0))

    def test_server_shutdown_aborts_unfinished_headers(self):
        deployment = Deployment(world_size=1)
        connection = socket.create_connection(ch.parse_endpoint(deployment.endpoints[0]))
        try:
            connection.sendall(b"POST / HTTP/1.1\r\nHost: localhost\r\nX-Incomplete: ")
            deadline = time.monotonic()+1
            server = deployment.workers[0].server
            while not server._httpd._sockets and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(server._httpd._sockets)
            started = time.monotonic()
            server.shutdown(timeout=0.3)
            self.assertLess(time.monotonic()-started, 1)
            self.assertTrue(server._fully_stopped)
        finally:
            connection.close()
            deployment.close()

    def test_startup_unproved_drain_retains_store(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(coordinator_endpoints=[f"127.0.0.1:{_free_port()}"],
                       coordinator_auth_token_file=str(_token_file(directory)),
                       disk_root=str(Path(directory)/"cache"), max_pending_keys=8,
                       coordinator_disk_limits=SMALL_LIMITS)
            error = None
            try:
                with mock.patch.object(ch, "_census", side_effect=RuntimeError("injected")), \
                     mock.patch.object(ch.MetadataServer, "shutdown", side_effect=RuntimeError("undrained")):
                    try:
                        ch.factory(config=cfg, role="worker", rank=0, world_size=1, geometry=GEOM2)
                    except RuntimeError as caught:
                        error = caught
                self.assertIsNotNone(error)
                client, server, store = error.resources
                self.assertFalse(store.closed)
            finally:
                if error is not None and hasattr(error, "resources"):
                    client, server, store = error.resources
                    client.shutdown()
                    server.shutdown()
                    store.close()


if __name__ == "__main__":
    unittest.main()
