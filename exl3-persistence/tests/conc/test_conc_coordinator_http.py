"""Concurrency / adversarial tests for recipe_persistence.coordinator_http.

STDLIB ONLY. Real loopback sockets on ephemeral 127.0.0.1 ports, ephemeral
disk roots, dummy tokens. Nothing here starts an engine or touches a real rank.

Harness: the validated `conc_stubs` helpers plus a small `Node` wrapper that
builds a real `MetadataServer` around an instrumented DiskStore proxy and a
counting dispatch wrapper, so a test can (a) see when a handler thread is
inside a store method / inside dispatch and (b) make a method slow on purpose
to create real in-flight windows.

NOTE on client admission: `RpcClient` defaults to max_active=min(endpoints, 8),
i.e. ONE in-flight operation for a single-endpoint client. Tests that need a
real server-side race therefore raise max_active explicitly.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import uuid

import conc_stubs as S
from recipe_persistence import coordinator_http as ch
from recipe_persistence.geometry import Geometry

NS = S.dummy_namespace("conc")
PAGES = (65536, 32768)
# NOTE: conc_stubs.dummy_geometry() emits a FLAT group_refs list
# ([[i, s], ...]) while geometry_identity()/layout_fingerprint() require a list
# of GROUPS ([[[i, s], ...], ...]). The real Geometry class is used instead.
GROUPS = (((0, 65536),), ((1, 32768),))
GROUPS_BYTES = [65536, 32768]
TOKEN = S.dummy_token(48)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #

def geom():
    return Geometry(PAGES, GROUPS)


def k(group, tag=0):
    """6-byte key whose last 4 bytes select the cache group."""
    return (tag & 0xFFFF).to_bytes(2, "big") + int(group).to_bytes(4, "big")


def wait_for(pred, timeout, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


class StoreProbe:
    """DiskStore proxy: counts in-method execution and can delay chosen methods."""

    def __init__(self, store, delay=0.0, methods=("reserve_write",)):
        self._store = store
        self._guard = threading.Lock()
        self._delay = float(delay)
        self._methods = frozenset(methods)
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.calls_after_close = 0

    def __getattr__(self, name):
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr

        def call(*a, **kw):
            with self._guard:
                if getattr(self._store, "closed", False):
                    self.calls_after_close += 1
                self.active += 1
                self.calls += 1
                if self.active > self.max_active:
                    self.max_active = self.active
            try:
                if self._delay and name in self._methods:
                    time.sleep(self._delay)
                return attr(*a, **kw)
            finally:
                with self._guard:
                    self.active -= 1

        return call


class Node:
    """One real MetadataServer on loopback with instrumented store + dispatch."""

    def __init__(self, rank=0, world=1, *, token=TOKEN, threads=8, max_body=65536,
                 rpc_timeout=2.0, idem_entries=4096, delay=0.0,
                 delay_methods=("reserve_write",), dispatch_delay=0.0,
                 dispatch_delay_methods=("geometry",), **limits):
        self.rank = rank
        self.token = token
        self.store = S.dummy_store(f"node{rank}", **limits)
        self.probe = StoreProbe(self.store, delay=delay, methods=delay_methods)
        self.identity = ch.geometry_identity(geom())
        self.server = ch.MetadataServer(store=self.probe, identity=self.identity,
                                        rank=rank, world_size=world, host="127.0.0.1",
                                        port=0, token=token, threads=threads,
                                        max_body=max_body, rpc_timeout=rpc_timeout,
                                        idem_entries=idem_entries)
        self.port = self.server._httpd.server_address[1]
        self._dlock = threading.Lock()
        self.dispatch_active = 0
        self.dispatch_max = 0
        self.dispatch_calls = 0
        self._dispatch_delay = float(dispatch_delay)
        self._dispatch_methods = frozenset(dispatch_delay_methods)
        inner = self.server._httpd.dispatch

        def counting(method, params):
            with self._dlock:
                self.dispatch_active += 1
                self.dispatch_calls += 1
                if self.dispatch_active > self.dispatch_max:
                    self.dispatch_max = self.dispatch_active
            try:
                if self._dispatch_delay and method in self._dispatch_methods:
                    time.sleep(self._dispatch_delay)
                return inner(method, params)
            finally:
                with self._dlock:
                    self.dispatch_active -= 1

        self.server._httpd.dispatch = counting
        self.server.start()

    def endpoint(self):
        return ("127.0.0.1", self.port)

    def count(self, sql, *args):
        return self.store.db.execute(sql, args).fetchone()[0]

    def close(self):
        try:
            if not self.server._fully_stopped:
                self.server.shutdown(timeout=3.0)
        except Exception:
            pass
        self.store.close()


def mkclient(nodes, **kw):
    kw.setdefault("rpc_timeout", 2.0)
    kw.setdefault("pool_size", 4)
    return ch.RpcClient([n.endpoint() for n in nodes], TOKEN, **kw)


def mkcoord(nodes, client=None, *, max_pending_keys=64,
            max_pending_bytes=64 * S.MIB, lease_seconds=5.0, renew_margin=1.0):
    client = client or mkclient(nodes)
    sizes = tuple(tuple(s for _ in nodes) for s in GROUPS_BYTES)
    return ch.RemoteCoordinator(client, sizes, max_pending_keys=max_pending_keys,
                                max_pending_bytes=max_pending_bytes,
                                lease_seconds=lease_seconds, renew_margin=renew_margin)


def body(method, **params):
    return json.dumps({"method": method, "params": params}, separators=(",", ":")).encode()


def req(payload=b"", *, path="/", auth="Bearer " + TOKEN, extra=(), method="POST",
        ctype="application/json", length=None, omit_length=False):
    lines = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1", "Connection: close"]
    if auth is not None:
        lines.append("Authorization: " + auth)
    if ctype is not None:
        lines.append("Content-Type: " + ctype)
    if not omit_length:
        lines.append("Content-Length: " + str(len(payload) if length is None else length))
    lines.extend(extra)
    head = ("\r\n".join(lines) + "\r\n\r\n").encode()
    if isinstance(payload, str):
        payload = payload.encode()
    return head + payload


def raw(port, payload, *, timeout=5.0, half_close=False):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.settimeout(timeout)
    data = b""
    t0 = time.perf_counter()
    try:
        s.sendall(payload)
        if half_close:
            s.shutdown(socket.SHUT_WR)
        while True:
            try:
                chunk = s.recv(65536)
            except (socket.timeout, TimeoutError):
                break
            except OSError:
                break
            if not chunk:
                break
            data += chunk
    finally:
        try:
            s.close()
        except OSError:
            pass
    return data, round(time.perf_counter() - t0, 4)


def status_of(data):
    if data.startswith(b"HTTP/"):
        try:
            return int(data.split(b" ", 2)[1])
        except (ValueError, IndexError):
            return None
    return None


def _ticket_ids(coord, rank):
    with coord._lock:
        tickets = list(coord._tickets.values())
    return {t.leases[rank] for t in tickets if t.leases[rank]}


def orphans(coord, nodes):
    """Leases / open write reservations with no live ticket that owns them."""
    bad = []
    for rank, node in enumerate(nodes):
        live = _ticket_ids(coord, rank)
        for (tok,) in node.store.db.execute("SELECT token FROM leases"):
            if tok not in live:
                bad.append((rank, "read_lease", tok))
        for (oid,) in node.store.db.execute("SELECT id FROM objects WHERE state='W'"):
            if oid not in live:
                bad.append((rank, "write_lease", oid))
    return bad


# --------------------------------------------------------------------------- #
# 1. loopback / pure helpers
# --------------------------------------------------------------------------- #

def test_loopback_bind_and_geometry_roundtrip():
    node = Node()
    client = mkclient([node])
    try:
        info = client.call(0, "geometry")
        assert info.get("ok") is True, info
        assert info["rank"] == 0 and info["world_size"] == 1, info
        assert info["fingerprint"] == node.identity["fingerprint"]
        assert info["padded_pages"] == list(PAGES)
        print(f"CONC_LOOPBACK bind 127.0.0.1:{node.port} OK under --network=none")
    finally:
        client.shutdown()
        node.close()


def test_stub_geometry_shapes_are_incompatible_with_the_package():
    """Harness defect: both stubs build a FLAT group_refs list, but
    geometry_identity()/layout_fingerprint() require list-of-groups."""
    try:
        ch.geometry_identity(S.dummy_geometry())
        raise AssertionError("stub geometry unexpectedly accepted")
    except TypeError:
        pass
    try:
        ch.layout_fingerprint(S.dummy_infos(ranks=1))
        raise AssertionError("stub infos unexpectedly accepted")
    except TypeError:
        pass


def test_pure_helpers_stable_under_concurrent_calls():
    ident = ch.geometry_identity(geom())
    infos = [{"padded_pages": list(PAGES),
              "group_refs": [list(group) for group in GROUPS]}]
    out = [None] * 8
    errs = []

    def work(w, _i):
        try:
            g = ch.geometry_identity(geom())
            out[w] = (g["fingerprint"], ch.layout_fingerprint(infos),
                      ch.parse_endpoint("http://127.0.0.1:8080/"),
                      ch.parse_endpoint("127.0.0.1:8080"))
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    r = S.hammer(work, workers=8, iterations=20, timeout=20)
    assert not any(r["errors"]), S.summarize(r, "pure helpers")
    assert not errs, errs
    assert all(x == out[0] for x in out), "non-deterministic pure helper output"
    assert out[0][0] == ident["fingerprint"]
    rejects = 0
    for bad in ("127.0.0.1", "http://127.0.0.1", "127.0.0.1:0", "udp://127.0.0.1:1",
                "http://user@127.0.0.1:1", "http://127.0.0.1:1/q", "0.0.0.0:1"):
        try:
            ch.parse_endpoint(bad)
        except ValueError:
            rejects += 1
    assert rejects == 7, rejects


# --------------------------------------------------------------------------- #
# 2. idempotency races (client admission raised so the server really races)
# --------------------------------------------------------------------------- #

def test_duplicate_request_id_reserves_exactly_once():
    node = Node(threads=8)
    client = mkclient([node], pool_size=8, max_active=8, max_pending=64)
    rid = uuid.uuid4().hex
    params = {"request_id": rid, "namespace": NS, "key": k(0, 7).hex(),
              "owner": "w", "size": 4096}
    got, errors = [], []
    lock = threading.Lock()

    def work(w, _i):
        try:
            r = client.call(0, "reserve_write", **params)
            with lock:
                got.append(r["lease"])
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    r = S.hammer(work, workers=8, iterations=1, timeout=20)
    try:
        assert not any(r["alive"]), S.summarize(r, "duplicate reserve")
        assert not errors, errors
        assert len(set(got)) == 1, f"duplicate request id produced >1 lease: {got}"
        assert node.count("SELECT COUNT(*) FROM objects WHERE state='W'") == 1
        assert node.count("SELECT COUNT(*) FROM objects") == 1
    finally:
        client.shutdown()
        node.close()


def test_duplicate_request_id_conflicting_params_rejected():
    node = Node(threads=8)
    client = mkclient([node], pool_size=8, max_active=8, max_pending=64)
    rid = uuid.uuid4().hex
    got, errors = [], []
    lock = threading.Lock()

    def work(w, _i):
        owner = "a" if w % 2 == 0 else "b"
        try:
            r = client.call(0, "reserve_write", request_id=rid, namespace=NS,
                            key=k(0, 11).hex(), owner=owner, size=4096)
            with lock:
                got.append((owner, r["lease"]))
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(type(e).__name__)

    r = S.hammer(work, workers=8, iterations=1, timeout=20)
    try:
        assert not any(r["alive"]), S.summarize(r, "conflicting duplicate")
        assert got, f"no winner among duplicate racers: {errors}"
        assert len(set(l for _, l in got)) == 1, got
        assert len(errors) == 8 - len(got), (got, errors)
        assert set(errors) <= {"_Permanent"}, errors
        assert node.count("SELECT COUNT(*) FROM objects") == 1
    finally:
        client.shutdown()
        node.close()


def test_duplicate_reserve_read_creates_one_lease():
    node = Node(threads=8)
    key = k(0, 13)
    tok = node.store.reserve_write(NS, key, 4096, "seed")
    assert tok and node.store.write(tok, [S.dummy_blob(4096)])
    client = mkclient([node], pool_size=8, max_active=8, max_pending=64)
    rid = uuid.uuid4().hex
    got, errors = [], []
    lock = threading.Lock()

    def work(w, _i):
        try:
            r = client.call(0, "reserve_read", request_id=rid, namespace=NS,
                            key=key.hex(), owner="r")
            with lock:
                got.append(r["lease"])
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    r = S.hammer(work, workers=8, iterations=1, timeout=20)
    try:
        assert not any(r["alive"]), S.summarize(r, "duplicate read")
        assert not errors, errors
        assert len(set(got)) == 1, f"duplicate read request produced >1 lease: {got}"
        assert node.count("SELECT COUNT(*) FROM leases") == 1
    finally:
        client.shutdown()
        node.close()


# --------------------------------------------------------------------------- #
# 3. auth
# --------------------------------------------------------------------------- #

def test_bad_token_storm_all_401_server_survives():
    node = Node(threads=8)
    statuses, missing = [], []
    lock = threading.Lock()

    def work(w, i):
        data, _dt = raw(node.port, req(body("geometry"), auth="Bearer " + "Z" * 40,
                                      extra=["X-Worker: %d-%d" % (w, i)]))
        with lock:
            statuses.append(status_of(data))
            if status_of(data) is None:
                missing.append(data[:40])

    r = S.hammer(work, workers=8, iterations=10, timeout=25)
    client = mkclient([node])
    try:
        assert not any(r["alive"]), S.summarize(r, "bad token storm")
        assert not missing, f"{len(missing)} requests got no HTTP response"
        assert set(statuses) == {401}, sorted(set(statuses))
        assert client.call(0, "geometry")["ok"] is True, "server died after auth storm"
        t0 = time.perf_counter()
        drained = wait_for(lambda: node.server._permits._value == 8, 5.0)
        print(f"CONC_PERMIT_DRAIN seconds={round(time.perf_counter() - t0, 3)} "
              f"value={node.server._permits._value}")
        assert drained, \
            f"permit leak: _value={node.server._permits._value} expected=8"
    finally:
        client.shutdown()
        node.close()


def test_absent_oversized_and_nonascii_auth_headers():
    node = Node()
    try:
        d, _ = raw(node.port, req(body("geometry"), auth=None))
        assert status_of(d) == 401, (status_of(d), d[:120])

        d, _ = raw(node.port, req(body("geometry"), auth="Basic " + "Q" * 40))
        assert status_of(d) == 401, (status_of(d), d[:120])

        d, _ = raw(node.port, req(body("geometry"), auth="Bearer " + "A" * 3000))
        assert status_of(d) == 401, (status_of(d), d[:120])

        # header allocation bound (16 KiB) must produce a response, never a hang
        d, dt = raw(node.port, req(body("geometry"), auth="Bearer " + "A" * 20000))
        st = status_of(d)
        assert st in (400, 431), (st, d[:120])
        assert dt < 2.0

        d, _ = raw(node.port, req(body("geometry"), auth="Bearer " + "\u00e9" * 40))
        assert status_of(d) == 401, (status_of(d), d[:120])

        client = mkclient([node])
        try:
            assert client.call(0, "geometry")["ok"] is True
        finally:
            client.shutdown()
    finally:
        node.close()


def test_token_file_validation_paths():
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "token")
        tok = S.dummy_token(48)
        with open(good, "w") as fh:
            fh.write(tok + "\n")
        os.chmod(good, 0o600)
        assert ch.load_auth_token(good) == tok

        link = os.path.join(tmp, "link")
        os.symlink(good, link)
        for path, label in ((link, "symlink"),):
            try:
                ch.load_auth_token(path)
                raise AssertionError(f"{label} token file accepted")
            except ValueError:
                pass

        wide = os.path.join(tmp, "wide")
        with open(wide, "w") as fh:
            fh.write(tok)
        os.chmod(wide, 0o644)
        try:
            ch.load_auth_token(wide)
            raise AssertionError("group/other-readable token accepted")
        except ValueError:
            pass

        big = os.path.join(tmp, "big")
        with open(big, "w") as fh:
            fh.write("A" * 5000)
        os.chmod(big, 0o600)
        try:
            ch.load_auth_token(big)
            raise AssertionError("oversized token accepted")
        except ValueError:
            pass

        for name, value in (("short", "a" * 31), ("low_entropy", "ab" * 24),
                            ("spaces", "a b" + "c" * 40)):
            path = os.path.join(tmp, name)
            with open(path, "w") as fh:
                fh.write(value)
            os.chmod(path, 0o600)
            try:
                ch.load_auth_token(path)
                raise AssertionError(f"bad token accepted: {name}")
            except ValueError:
                pass

        binary = os.path.join(tmp, "binary")
        with open(binary, "wb") as fh:
            fh.write(b"\xff" * 48)
        os.chmod(binary, 0o600)
        try:
            ch.load_auth_token(binary)
            raise AssertionError("non-ascii token accepted")
        except ValueError:
            pass

        fifo = os.path.join(tmp, "fifo")
        os.mkfifo(fifo, 0o600)
        try:
            ch.load_auth_token(fifo)   # O_NONBLOCK: must not block on a writer
            raise AssertionError("fifo token accepted")
        except ValueError:
            pass

        out, errs = [None] * 8, []

        def work(w, _i):
            try:
                out[w] = ch.load_auth_token(good)
            except Exception as e:  # noqa: BLE001
                errs.append(e)

        r = S.hammer(work, workers=8, iterations=10, timeout=20)
        assert not any(r["errors"]) and not errs, errs
        assert set(out) == {tok}


# --------------------------------------------------------------------------- #
# 4. body bounds / malformed requests
# --------------------------------------------------------------------------- #

def test_body_bound_and_framing_matrix():
    node = Node(max_body=256)
    base = body("geometry")
    cases = [
        ("oversize", req(base + b" " * (257 - len(base))), 413),
        ("exact_max", req(base + b" " * (256 - len(base))), 200),
        ("empty", req(b"", length=0), 400),
        ("dup_length", req(base, extra=["Content-Length: %d" % len(base)]), 400),
        ("bad_length", req(base, omit_length=True, extra=["Content-Length: 1e3"]), 400),
        ("neg_length", req(base, omit_length=True, extra=["Content-Length: -5"]), 400),
        ("transfer_encoding", req(base, extra=["Transfer-Encoding: chunked"]), 400),
        ("no_length", req(base, omit_length=True), 411),
        ("wrong_type", req(base, ctype="text/plain"), 415),
        ("wrong_path", req(base, path="/nope"), 404),
        ("get", req(b"", method="GET", omit_length=True), 405),
    ]
    observed = {}
    try:
        for name, payload, expect in cases:
            data, dt = raw(node.port, payload)
            observed[name] = (status_of(data), dt)
            assert status_of(data) == expect, (name, observed[name], data[:160])
    finally:
        node.close()


def test_short_body_closes_cleanly():
    node = Node(threads=2, rpc_timeout=0.5)
    try:
        data, dt = raw(node.port, req(b"12345", length=100), half_close=True, timeout=3.0)
        st = status_of(data)
        assert st == 400, (st, data[:160])
        assert dt < 2.0, f"short body took {dt}s"
        assert wait_for(lambda: node.server._permits._value == 2, 5.0), \
            f"permit leak: _value={node.server._permits._value} expected=2"
    finally:
        node.close()


def test_idle_keepalive_connection_holds_a_handler_permit():
    """A pooled client's idle keepalive connection parks a handler thread (and one
    of the `threads` permits) for up to io_timeout = rpc_timeout clamped to <=5s."""
    node = Node(threads=2, rpc_timeout=0.5)
    a = mkclient([node], pool_size=1)
    b = mkclient([node], pool_size=1)
    try:
        assert a.call(0, "geometry")["ok"] is True
        assert wait_for(lambda: node.server._permits._value == 1, 1.0), \
            f"idle keepalive did not park a permit: {node.server._permits._value}"
        assert b.call(0, "geometry")["ok"] is True
        assert wait_for(lambda: node.server._permits._value == 0, 1.0)
        data, _dt = raw(node.port, req(body("geometry")), timeout=3.0)
        assert status_of(data) == 503, (status_of(data), data[:120])
        t0 = time.perf_counter()
        recovered = wait_for(lambda: node.server._permits._value == 2, 3.0)
        dt = round(time.perf_counter() - t0, 3)
        assert recovered, f"parked permits never released: {node.server._permits._value}"
        print(f"CONC_KEEPALIVE idle_permit_recovery_seconds={dt} io_timeout=0.5")
    finally:
        a.shutdown()
        b.shutdown()
        node.close()


def test_malformed_body_storm_then_recovery():
    node = Node(threads=16)
    bad_bodies = [b"{", b"[]", b"null", b'"x"', b"{}",
                  json.dumps({"method": "nope", "params": {}}).encode(),
                  json.dumps({"method": "exists", "params": {}}).encode(),
                  json.dumps({"method": "exists",
                              "params": {"namespace": NS, "key": k(0, 1).hex(),
                                         "extra": 1}}).encode(),
                  json.dumps({"method": "reserve_write",
                              "params": {"request_id": "z" * 32, "namespace": NS,
                                         "key": k(0, 1).hex(), "owner": "o",
                                         "size": -1}}).encode(),
                  json.dumps({"method": "release",
                              "params": {"lease": "not-hex"}}).encode()]
    statuses, missing = [], []
    lock = threading.Lock()

    def work(w, i):
        payload = req(bad_bodies[(w + i) % len(bad_bodies)])
        data, _dt = raw(node.port, payload)
        st = status_of(data)
        with lock:
            statuses.append(st)
            if st is None:
                missing.append(payload[:60])

    r = S.hammer(work, workers=8, iterations=10, timeout=25)
    client = mkclient([node])
    try:
        assert not any(r["alive"]), S.summarize(r, "malformed storm")
        assert not missing, f"{len(missing)} malformed requests got no response"
        assert set(statuses) == {400}, sorted(set(statuses))
        assert client.call(0, "geometry")["ok"] is True
    finally:
        client.shutdown()
        node.close()


def test_deeply_nested_json_does_not_kill_the_handler():
    node = Node()
    try:
        deep = req(b"[" * 20000)
        data, dt = raw(node.port, deep, timeout=5.0)
        st = status_of(data)
        assert st is not None, "no HTTP response for deeply nested JSON"
        client = mkclient([node])
        try:
            assert client.call(0, "geometry")["ok"] is True
        finally:
            client.shutdown()
        print(f"CONC_DEEP_JSON status={st} dt={dt}")
    finally:
        node.close()


def test_partial_request_abort_leaks_no_permit():
    node = Node(threads=2)

    def work(w, i):
        try:
            s = socket.create_connection(("127.0.0.1", node.port), timeout=3)
            s.sendall(b"POST / HTTP/1.1\r\nHost: x\r\n")
            s.close()
        except OSError:
            pass

    r = S.hammer(work, workers=8, iterations=10, timeout=25)
    try:
        assert not any(r["alive"]), S.summarize(r, "partial abort")
        assert wait_for(lambda: node.server._permits._value == 2, 5.0), \
            f"permit leak: _value={node.server._permits._value} expected=2"
        client = mkclient([node])
        try:
            assert client.call(0, "geometry")["ok"] is True
        finally:
            client.shutdown()
    finally:
        node.close()


# --------------------------------------------------------------------------- #
# 5. lease expiry / renewal races
# --------------------------------------------------------------------------- #

def test_expired_lease_not_served_from_receipt():
    node = Node(lease_seconds=0.4, grace_seconds=0.1)
    client = mkclient([node])
    rid = uuid.uuid4().hex
    params = {"request_id": rid, "namespace": NS, "key": k(0, 17).hex(),
              "owner": "w", "size": 4096}
    try:
        first = client.call(0, "reserve_write", **params)
        lease = first["lease"]
        assert lease and len(lease) == 32
        time.sleep(0.6)
        again = client.call(0, "reserve_write", **params)
        assert again == {"ok": True, "lease": None}, again
        rn = client.call(0, "renew", lease=lease)
        assert rn["renewed"] is False, rn
        fresh = client.call(0, "reserve_write", request_id=uuid.uuid4().hex,
                            namespace=NS, key=k(0, 17).hex(), owner="w2", size=4096)
        assert fresh["lease"] is None, fresh
    finally:
        client.shutdown()
        node.close()


def test_concurrent_renew_and_release_same_lease():
    node = Node(lease_seconds=30.0)
    key = k(0, 19)
    tok = node.store.reserve_write(NS, key, 4096, "seed")
    assert tok and node.store.write(tok, [S.dummy_blob(4096)])
    client = mkclient([node], pool_size=4, max_active=4, max_pending=32)
    lease = client.call(0, "reserve_read", request_id=uuid.uuid4().hex,
                        namespace=NS, key=key.hex(), owner="r")["lease"]
    assert lease
    outcome = []
    lock = threading.Lock()

    def work(w, i):
        try:
            if w % 2 == 0:
                r = client.call(0, "renew", lease=lease)
                with lock:
                    outcome.append(("renew", r["renewed"]))
            else:
                r = client.call(0, "release", lease=lease)
                with lock:
                    outcome.append(("release", r["released"]))
        except Exception as e:  # noqa: BLE001
            with lock:
                outcome.append(("error", f"{type(e).__name__}: {e}"))

    r = S.hammer(work, workers=4, iterations=25, timeout=25)
    try:
        assert not any(r["alive"]), S.summarize(r, "renew/release race")
        assert not [o for o in outcome if o[0] == "error"], outcome[:5]
        assert all(o[1] in (True, False) for o in outcome), outcome[:5]
        assert node.count("SELECT COUNT(*) FROM leases") == 0
    finally:
        client.shutdown()
        node.close()


# --------------------------------------------------------------------------- #
# 6. pool saturation / hangs / shutdown
# --------------------------------------------------------------------------- #

def test_saturated_pool_fails_closed_then_recovers():
    node = Node(threads=2, rpc_timeout=0.5)
    stalls = []
    try:
        for _ in range(2):
            s = socket.create_connection(("127.0.0.1", node.port), timeout=3)
            s.sendall(req(b"x" * 10, length=1000))
            stalls.append(s)
        assert wait_for(lambda: node.server._permits._value == 0, 1.5), \
            "stalled bodies did not occupy the pool"
        data, dt = raw(node.port, req(body("geometry")), timeout=3.0)
        assert status_of(data) == 503, (status_of(data), data[:120])
        assert dt < 0.3, f"rejection was not immediate: {dt}s"

        client = mkclient([node], rpc_timeout=3.0)
        try:
            ok = False
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and not ok:
                try:
                    ok = client.call(0, "geometry")["ok"] is True
                except Exception:  # noqa: BLE001
                    time.sleep(0.05)
            assert ok, "pool never recovered after io_timeout"
        finally:
            client.shutdown()
    finally:
        for s in stalls:
            try:
                s.close()
            except OSError:
                pass
        node.close()


def test_duplicate_waiter_occupies_permit_and_503s_unrelated():
    node = Node(threads=2, delay=1.5, delay_methods=("reserve_write",), rpc_timeout=8.0)
    client = mkclient([node], pool_size=2, max_active=4, max_pending=16, rpc_timeout=8.0)
    rid = uuid.uuid4().hex
    params = {"request_id": rid, "namespace": NS, "key": k(0, 23).hex(),
              "owner": "w", "size": 4096}
    got = {}
    try:
        def owner():
            try:
                got["owner"] = client.call(0, "reserve_write", **params)
            except Exception as e:  # noqa: BLE001
                got["owner_err"] = f"{type(e).__name__}: {e}"

        t1 = threading.Thread(target=owner)
        t1.start()
        assert wait_for(lambda: node.probe.active == 1, 3.0), "owner never entered the store"

        def dup():
            try:
                got["dup"] = client.call(0, "reserve_write", **params)
            except Exception as e:  # noqa: BLE001
                got["dup_err"] = f"{type(e).__name__}: {e}"

        t2 = threading.Thread(target=dup)
        t2.start()
        in_flight = wait_for(lambda: node.server._permits._value == 0, 3.0)
        data, dt = raw(node.port, req(body("geometry")), timeout=3.0)
        st = status_of(data)
        store_active_at_probe = node.probe.active
        store_max_at_probe = node.probe.max_active
        t1.join(15)
        t2.join(15)
        assert in_flight, (got, node.probe.calls, node.server._permits._value)
        assert store_active_at_probe == 1 and store_max_at_probe == 1, \
            f"duplicate should not re-enter the store: {store_max_at_probe}"
        assert st == 503 and dt < 0.3, (st, dt)
        assert "owner" in got and got["owner"].get("lease"), got
        assert got.get("dup", {}).get("lease") == got["owner"]["lease"], got
        print(f"CONC_DUP_WAITER duplicate_blocks_pool status={st} dt={dt}s")
    finally:
        client.shutdown()
        node.close()


def test_shutdown_with_request_in_flight_proves_drain():
    node = Node(threads=4, delay=0.4, rpc_timeout=2.0)
    client = mkclient([node], rpc_timeout=3.0)
    res = {}

    def call():
        try:
            res["r"] = client.call(0, "reserve_write", request_id=uuid.uuid4().hex,
                                   namespace=NS, key=k(0, 29).hex(), owner="w", size=4096)
        except Exception as e:  # noqa: BLE001
            res["err"] = f"{type(e).__name__}: {e}"

    try:
        t = threading.Thread(target=call)
        t.start()
        assert wait_for(lambda: node.probe.active == 1, 3.0), "request never reached the store"
        with S.Stopwatch() as sw:
            node.server.shutdown(timeout=5.0)
        t.join(10)
        assert node.server._fully_stopped
        assert not t.is_alive()
        assert node.probe.active == 0, "handler still inside the store after proved drain"
        assert node.probe.calls_after_close == 0, "store method entered after drain proof"
        assert node.server._permits._value == 4, "permit leak after shutdown"
        assert not node.store.closed, "store closed before the owner closed it"
        print(f"CONC_SHUTDOWN drain_seconds={sw.seconds} inflight={res.get('err') or res}")
    finally:
        client.shutdown()
        node.close()


def test_unproved_drain_raises_and_is_retryable():
    node = Node(threads=2, delay=0.6, rpc_timeout=2.0)
    client = mkclient([node], rpc_timeout=3.0)
    res = {}

    def call():
        try:
            res["r"] = client.call(0, "reserve_write", request_id=uuid.uuid4().hex,
                                   namespace=NS, key=k(0, 31).hex(), owner="w", size=4096)
        except Exception as e:  # noqa: BLE001
            res["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=call)
    t.start()
    try:
        assert wait_for(lambda: node.probe.active == 1, 3.0)
        raised = None
        try:
            node.server.shutdown(timeout=0.05)
        except RuntimeError as e:
            raised = str(e)
        assert raised and "drain" in raised, raised
        assert not node.store.closed, "store closed despite unproved drain"
        t.join(10)
        node.server.shutdown(timeout=5.0)
        assert node.server._fully_stopped
        assert node.probe.active == 0 and node.probe.calls_after_close == 0
        assert node.server._permits._value == 2
    finally:
        client.shutdown()
        node.close()


def test_shutdown_flag_rejects_before_dispatch():
    node = Node()
    client = mkclient([node])
    try:
        node.server._flag.set()
        data, _dt = raw(node.port, req(body("geometry")), timeout=3.0)
        assert status_of(data) == 503, (status_of(data), data[:120])
        try:
            client.call(0, "geometry")
            raise AssertionError("client call succeeded after shutdown flag")
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001
            assert type(e).__name__ == "_Transient", type(e).__name__
        assert node.dispatch_calls == 0 and node.probe.calls == 0, \
            (node.dispatch_calls, node.probe.calls)
    finally:
        client.shutdown()
        node.close()


# --------------------------------------------------------------------------- #
# 7. coordinator admission / credit accounting
# --------------------------------------------------------------------------- #

def test_disabled_rank_admission_and_no_half_tickets():
    nodes = [Node(0, world=2), Node(1, world=2)]
    client = mkclient(nodes, pool_size=2, max_active=8, max_pending=64, rpc_timeout=3.0)
    coord = mkcoord(nodes, client=client, max_pending_keys=16)
    nodes[1].store.failed = True
    results, errors = [], []
    lock = threading.Lock()

    def work(w, i):
        try:
            t = coord.reserve_store(NS, k(0, 2000 + w * 20 + i), f"w{w}",
                                    coord.size_by_group[0])
            with lock:
                results.append(t)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    r = S.hammer(work, workers=8, iterations=5, timeout=25)
    try:
        assert not any(r["alive"]), S.summarize(r, "disabled rank")
        assert not errors, errors
        assert all(t is None for t in results), "ticket issued across a disabled rank"
        assert coord._tickets == {} and coord._bytes == 0 and not coord._pending, \
            (len(coord._tickets), coord._bytes, len(coord._pending))
        assert len(coord._identities) == 0
        assert coord.can_store() is False, "permanent rank failure not published"
        assert coord.pending_retries() == 0
        assert nodes[0].count("SELECT COUNT(*) FROM leases") == 0
        assert nodes[0].count("SELECT COUNT(*) FROM objects WHERE state='W'") == 0
        assert not orphans(coord, nodes)
    finally:
        client.shutdown()
        coord.shutdown()
        for n in nodes:
            n.close()


def test_credit_bound_never_exceeded_under_load():
    nodes = [Node(0, world=1)]
    client = mkclient(nodes, pool_size=2, max_active=8, max_pending=64, rpc_timeout=3.0)
    coord = mkcoord(nodes, client=client, max_pending_keys=4, max_pending_bytes=140000)
    peak = {"keys": 0, "bytes": 0}
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            with coord._lock:
                keys = len(coord._tickets) + len(coord._pending)
                by = coord._bytes
            peak["keys"] = max(peak["keys"], keys)
            peak["bytes"] = max(peak["bytes"], by)
            time.sleep(0.001)

    mon = threading.Thread(target=monitor)
    mon.start()

    def work(w, i):
        t = coord.reserve_store(NS, k(0, 3000 + w * 20 + i), f"w{w}",
                                coord.size_by_group[0])
        if t is not None:
            time.sleep(0.005)
            coord.release(t)

    r = S.hammer(work, workers=8, iterations=15, timeout=30)
    stop.set()
    mon.join(5)
    try:
        assert not any(r["alive"]), S.summarize(r, "credit bound")
        assert peak["keys"] <= 4, peak
        assert peak["bytes"] <= 140000, peak
        assert coord._bytes == 0 and not coord._tickets and not coord._pending, \
            (coord._bytes, len(coord._tickets), len(coord._pending))
        assert len(coord._identities) == 0
        assert not orphans(coord, nodes)
        print(f"CONC_CREDIT peak_keys={peak['keys']} peak_bytes={peak['bytes']}")
    finally:
        client.shutdown()
        coord.shutdown()
        for n in nodes:
            n.close()


def test_mixed_workload_leaves_no_credit_or_lease_leak():
    nodes = [Node(0, world=2), Node(1, world=2)]
    client = mkclient(nodes, pool_size=2, max_active=8, max_pending=64, rpc_timeout=4.0)
    coord = mkcoord(nodes, client=client, max_pending_keys=32,
                    max_pending_bytes=64 * S.MIB)
    for g in (0, 1):
        size = GROUPS_BYTES[g]
        for tag in range(4):
            for n in nodes:
                tok = n.store.reserve_write(NS, k(g, 100 + tag), size, "seed")
                assert tok and n.store.write(tok, [S.dummy_blob(size)])

    stats = {"store_ok": 0, "load_ok": 0, "none": 0, "release_false": 0,
             "inval_false": 0}
    lock = threading.Lock()

    def work(w, i):
        mode = (w + i) % 10
        g = (w + i) % 2
        if mode < 6:
            t = coord.reserve_store(NS, k(g, 1000 + w * 30 + i), f"w{w}",
                                    coord.size_by_group[g])
            if t is None:
                with lock:
                    stats["none"] += 1
                return
            with lock:
                stats["store_ok"] += 1
            if coord.release(t) is False:
                with lock:
                    stats["release_false"] += 1
        elif mode < 9:
            t = coord.reserve_load(NS, k(g, 100 + (w + i) % 4), f"r{w}")
            if t is None:
                with lock:
                    stats["none"] += 1
                return
            with lock:
                stats["load_ok"] += 1
            if coord.release(t) is False:
                with lock:
                    stats["release_false"] += 1
        else:
            if coord.invalidate(NS, k(g, 100 + (w + i) % 4)) is False:
                with lock:
                    stats["inval_false"] += 1

    r = S.hammer(work, workers=8, iterations=12, timeout=30)
    try:
        assert not any(r["alive"]), S.summarize(r, "mixed workload")
        assert stats["release_false"] == 0, stats
        assert stats["inval_false"] == 0, stats
        assert stats["store_ok"] > 0 and stats["load_ok"] > 0, stats
        assert coord._bytes == 0, f"credit leak: _bytes={coord._bytes}"
        assert not coord._tickets, f"ticket leak: {len(coord._tickets)}"
        assert not coord._pending and not coord._identities
        assert not coord._invalid, f"invalidation veto leak: {coord._invalid}"
        assert coord.pending_retries() == 0 and not coord._cleanup_lost
        assert not orphans(coord, nodes), orphans(coord, nodes)[:3]
        for n in nodes:
            assert n.count("SELECT COUNT(*) FROM leases") == 0
            assert n.count("SELECT COUNT(*) FROM objects WHERE state='W'") == 0
        print(f"CONC_MIXED {stats}")
    finally:
        client.shutdown()
        coord.shutdown()
        for n in nodes:
            n.close()


def test_concurrent_double_release_does_not_double_subtract_credit():
    nodes = [Node(0, world=2), Node(1, world=2)]
    client = mkclient(nodes, pool_size=2, max_active=8, max_pending=64, rpc_timeout=3.0)
    coord = mkcoord(nodes, client=client)
    errors = []

    def work(w, _i):
        try:
            for _ in range(3):
                t = coord.reserve_store(NS, k(0, 4000 + w), f"w{w}",
                                        coord.size_by_group[0])
                if t is None:
                    continue
                coord.release(t)
                if w % 2:
                    coord.release(t)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    r = S.hammer(work, workers=8, iterations=1, timeout=25)
    try:
        assert not any(r["alive"]), S.summarize(r, "double release")
        assert not errors, errors[:5]
        assert coord._bytes >= 0, f"credit went negative: {coord._bytes}"
        assert coord._bytes == 0 and not coord._tickets, \
            (coord._bytes, len(coord._tickets))
        assert not orphans(coord, nodes)
        assert len(coord._identities) == 0
    finally:
        client.shutdown()
        coord.shutdown()
        for n in nodes:
            n.close()


def test_dead_peer_rolls_back_and_leaks_no_lease():
    live = Node(0, world=2)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()
    client = ch.RpcClient([("127.0.0.1", live.port), ("127.0.0.1", dead_port)],
                          TOKEN, rpc_timeout=1.0, pool_size=2, max_active=4,
                          max_pending=32)
    sizes = ((65536, 65536), (32768, 32768))
    coord = ch.RemoteCoordinator(client, sizes, max_pending_keys=8,
                                 max_pending_bytes=8 * S.MIB, lease_seconds=5.0,
                                 renew_margin=1.0)
    errors = []

    def work(w, i):
        try:
            t = coord.reserve_store(NS, k(0, 5000 + w * 10 + i), f"w{w}", sizes[0])
            if t is not None:
                coord.release(t)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    r = S.hammer(work, workers=4, iterations=5, timeout=25)
    try:
        assert not any(r["alive"]), S.summarize(r, "dead peer")
        assert not errors, errors[:5]
        assert coord._bytes == 0 and not coord._tickets and not coord._pending, \
            (coord._bytes, len(coord._tickets), len(coord._pending))
        assert live.count("SELECT COUNT(*) FROM objects WHERE state='W'") == 0, \
            "write lease leaked on the live rank"
        assert live.count("SELECT COUNT(*) FROM leases") == 0
        assert coord.can_store() is True, "a transient dead peer must not disable the coordinator"
    finally:
        client.shutdown()
        coord.shutdown()
        live.close()


def test_renew_io_under_global_lock_stalls_unrelated_reservations():
    """A slow renew fan-out is executed while the coordinator's global lock is held."""
    node = Node(dispatch_delay=0.25, dispatch_delay_methods=("renew",))
    client = mkclient([node], pool_size=2, max_active=4, max_pending=32, rpc_timeout=5.0)
    coord = mkcoord([node], client=client, lease_seconds=0.5, renew_margin=0.1)
    key_a, key_b = k(0, 77), k(0, 78)
    try:
        first = coord.reserve_store(NS, key_a, "oA", coord.size_by_group[0])
        assert first is not None
        # ttl - margin = 0.4s: after this the cached renew window has lapsed, so a
        # reservation for the SAME identity performs a real renew fan-out.
        time.sleep(0.45)
        holder = {}

        def hold_a():
            holder["t"] = coord.reserve_store(NS, key_a, "oA", coord.size_by_group[0])

        th = threading.Thread(target=hold_a)
        th.start()
        assert wait_for(lambda: node.dispatch_active >= 1, 2.0), "renew fan-out never started"
        t0 = time.perf_counter()
        tb = coord.reserve_store(NS, key_b, "oB", coord.size_by_group[0])
        dt = time.perf_counter() - t0
        th.join(5)
        assert tb is not None
        assert dt < 0.1, (
            f"unrelated reservation on a different key blocked {dt:.3f}s behind a "
            f"renew fan-out to another identity (global lock held across I/O)"
        )
    finally:
        client.shutdown()
        coord.shutdown()
        node.close()


def test_concurrent_complete_store_releases_exactly_once():
    nodes = [Node(0, world=2), Node(1, world=2)]
    client = mkclient(nodes, pool_size=2, max_active=8, max_pending=64, rpc_timeout=4.0)
    coord = mkcoord(nodes, client=client)
    key = k(0, 91)
    ticket = coord.reserve_store(NS, key, "writer", coord.size_by_group[0])
    assert ticket is not None
    blob = S.dummy_blob(coord.size_by_group[0][0])
    for rank, n in enumerate(nodes):
        assert n.store.write(ticket.leases[rank], (blob,))
    outcomes, errors = [], []
    lock = threading.Lock()

    def work(w, _i):
        try:
            r = coord.complete_store(ticket, True)
            with lock:
                outcomes.append(r)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    r = S.hammer(work, workers=4, iterations=1, timeout=25)
    try:
        assert not any(r["alive"]), S.summarize(r, "complete_store race")
        assert not errors, errors
        assert all(o is not False for o in outcomes), outcomes
        assert coord._bytes == 0 and not coord._tickets, \
            (coord._bytes, len(coord._tickets))
        assert len(coord._identities) == 0
        assert not orphans(coord, nodes)
        for n in nodes:
            assert n.count("SELECT COUNT(*) FROM leases") == 0
            assert n.store.exists(NS, key) is True, \
                "durable object invalidated by a failed ack"
    finally:
        client.shutdown()
        coord.shutdown()
        for n in nodes:
            n.close()


# --------------------------------------------------------------------------- #
# 8. client admission bound: healthy calls rejected instead of queued
# --------------------------------------------------------------------------- #

def test_shared_client_rejects_concurrent_callers_at_default_pending_bound():
    """Default max_pending = max_active * pool_size. For a single-endpoint client
    that is 4, so a 5th simultaneous caller is rejected immediately (not queued)
    even though the server has 16 idle handler threads."""
    node = Node(threads=16)
    errors = []
    lock = threading.Lock()

    def race(client):
        barrier = threading.Barrier(8)

        def work(w, _i):
            try:
                barrier.wait(10)
            except threading.BrokenBarrierError:
                return
            try:
                client.call(0, "geometry")
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{type(e).__name__}: {e}")

        r = S.hammer(work, workers=8, iterations=1, timeout=20)
        assert not any(r["alive"]), S.summarize(r, "admission race", )

    default = ch.RpcClient([("127.0.0.1", node.port)], TOKEN, rpc_timeout=5.0,
                           pool_size=4)
    try:
        assert (default.max_active, default.max_pending) == (1, 4)
        race(default)
        default_errors = list(errors)
    finally:
        default.shutdown()

    errors.clear()
    tuned = ch.RpcClient([("127.0.0.1", node.port)], TOKEN, rpc_timeout=5.0,
                         pool_size=4, max_pending=64)
    try:
        race(tuned)
        tuned_errors = list(errors)
    finally:
        tuned.shutdown()
    node.close()

    assert not default_errors, (
        f"{len(default_errors)}/8 simultaneous callers on ONE endpoint were rejected "
        f"immediately (expected 0): {sorted(set(default_errors))}"
    )
    assert not tuned_errors, tuned_errors


def test_coordinator_turns_client_admission_limit_into_spurious_misses():
    """All-rank reservations are reported as a clean MISS when the shared client's
    pending bound is exhausted, although every rank has capacity and no key is
    contended. max_active=2 / max_pending=8 for a 2-endpoint client, so 8
    concurrent reservations (16 in-flight ops) lose half of the work."""
    nodes = [Node(0, world=2), Node(1, world=2)]
    client = mkclient(nodes, pool_size=4, max_active=2, max_pending=8, rpc_timeout=5.0)
    coord = mkcoord(nodes, client=client, max_pending_keys=64,
                    max_pending_bytes=64 * S.MIB)
    misses, hits = [], []
    lock = threading.Lock()

    def work(w, i):
        t = coord.reserve_store(NS, k(0, 7000 + w * 20 + i), f"w{w}",
                                coord.size_by_group[0])
        with lock:
            (misses if t is None else hits).append((w, i))
        if t is not None:
            coord.release(t)

    r = S.hammer(work, workers=8, iterations=8, timeout=30)
    try:
        assert not any(r["alive"]), S.summarize(r, "spurious miss")
        assert not misses, (
            f"{len(misses)}/{len(misses) + len(hits)} all-rank reservations returned None "
            f"with every rank healthy and no key contention (client max_pending="
            f"{client.max_pending}); first={misses[:3]}"
        )
        assert not orphans(coord, nodes)
    finally:
        client.shutdown()
        coord.shutdown()
        for n in nodes:
            n.close()


# --------------------------------------------------------------------------- #
# 9. perf
# --------------------------------------------------------------------------- #

def _measure_http(port, token, conc, iters, *, shared, client_kw=None):
    kw = {"rpc_timeout": 5.0, "pool_size": 4}
    kw.update(client_kw or {})
    pool = ch.RpcClient([("127.0.0.1", port)], token, **kw)
    lat = [[] for _ in range(conc)]
    errs = []
    barrier = threading.Barrier(conc)

    def work(w):
        # one client per thread for the non-shared variant (measured once, not
        # per iteration: building a client spawns worker threads)
        client = pool if shared else ch.RpcClient([("127.0.0.1", port)], token,
                                                  rpc_timeout=5.0, pool_size=2)
        try:
            barrier.wait(10)
            for _ in range(iters):
                t0 = time.perf_counter()
                try:
                    client.call(0, "geometry")
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{type(e).__name__}: {e}")
                lat[w].append(time.perf_counter() - t0)
        finally:
            if not shared:
                client.shutdown()

    threads = [threading.Thread(target=work, args=(w,)) for w in range(conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    wall = time.perf_counter() - t0
    allat = sorted(x for row in lat for x in row)
    pool.shutdown()
    return {"conc": conc, "req_s": round(conc * iters / wall, 1),
            "p50_ms": round(allat[len(allat) // 2] * 1000, 3),
            "p95_ms": round(allat[int(len(allat) * 0.95)] * 1000, 3),
            "errors": len(errs), "max_active": pool.max_active,
            "max_pending": pool.max_pending,
            "error_sample": sorted(set(errs))[:2]}


def _measure_coord(nodes, conc, iters):
    client = mkclient(nodes, pool_size=2, rpc_timeout=5.0)
    coord = mkcoord(nodes, client=client)
    counter = [0]
    lock = threading.Lock()
    barrier = threading.Barrier(conc)

    def work(w):
        try:
            barrier.wait(10)
        except threading.BrokenBarrierError:
            return
        for i in range(iters):
            with lock:
                counter[0] += 1
                tag = counter[0]
            t = coord.reserve_store(NS, k(0, 6000 + tag), f"w{w}",
                                    coord.size_by_group[0])
            if t is not None:
                coord.release(t)

    threads = [threading.Thread(target=work, args=(w,)) for w in range(conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(40)
    wall = time.perf_counter() - t0
    out = {"conc": conc, "req_s": round(conc * iters / wall, 1),
           "max_active": client.max_active, "max_tickets": len(coord._tickets)}
    client.shutdown()
    coord.shutdown()
    return out


def test_perf_http_roundtrip_and_coordinator_scaling():
    node = Node(threads=16)
    shared_rows, tuned_rows, per_rows = [], [], []
    try:
        for conc in (1, 2, 4, 8):
            shared_rows.append(_measure_http(node.port, TOKEN, conc, 40, shared=True))
        for conc in (1, 2, 4, 8):
            tuned_rows.append(_measure_http(node.port, TOKEN, conc, 40, shared=True,
                                            client_kw={"max_pending": 128}))
        for conc in (1, 2, 4, 8):
            per_rows.append(_measure_http(node.port, TOKEN, conc, 40, shared=False))
        print("CONC_PERF_HTTP_SHARED " + json.dumps(shared_rows))
        print("CONC_PERF_HTTP_SHARED_TUNED " + json.dumps(tuned_rows))
        print("CONC_PERF_HTTP_PERCLIENT " + json.dumps(per_rows))
    finally:
        node.close()

    nodes = [Node(0, world=1), Node(1, world=1)]
    try:
        crows = [_measure_coord(nodes, c, 25) for c in (1, 2, 4, 8)]
        print("CONC_PERF_COORD " + json.dumps(crows))
    finally:
        for n in nodes:
            n.close()


def test_rpc_client_active_bound_serializes_single_endpoint():
    node = Node(threads=8, dispatch_delay=0.05, dispatch_delay_methods=("geometry",))
    shared = ch.RpcClient([("127.0.0.1", node.port)], TOKEN, rpc_timeout=5.0, pool_size=8)
    try:
        assert shared.max_active == 1, shared.max_active

        def work(w, _i):
            shared.call(0, "geometry")

        r = S.hammer(work, workers=8, iterations=3, timeout=25)
        assert not any(r["errors"]), S.summarize(r, "active bound")
        assert node.dispatch_max == 1, \
            f"single-endpoint client admitted {node.dispatch_max} concurrent dispatches"
        print(f"CONC_ACTIVE_BOUND max_active={shared.max_active} "
              f"server_dispatch_max={node.dispatch_max}")
    finally:
        shared.shutdown()
        node.close()

    node2 = Node(threads=8, dispatch_delay=0.05, dispatch_delay_methods=("geometry",))
    try:
        def work2(w, _i):
            c = ch.RpcClient([("127.0.0.1", node2.port)], TOKEN, rpc_timeout=5.0,
                             pool_size=2, max_active=4)
            try:
                c.call(0, "geometry")
            finally:
                c.shutdown()

        r = S.hammer(work2, workers=8, iterations=1, timeout=25)
        assert not any(r["errors"]), S.summarize(r, "per-client bound")
        assert node2.dispatch_max > 1, (
            "server cannot overlap even with independent clients: "
            f"{node2.dispatch_max}"
        )
        print(f"CONC_ACTIVE_BOUND per_client_server_dispatch_max={node2.dispatch_max}")
    finally:
        node2.close()
