"""Adversarial concurrency tests for http_rpc / geometry / coordinator.

STDLIB ONLY. Loopback only (127.0.0.1). Dummy data only. No engine, no GPU.
Every stub HTTP server binds an ephemeral loopback port and speaks just enough
HTTP/1.1 to drive RpcClient. Nothing here mutates recipe_persistence.
"""
from __future__ import annotations

import http.server
import json
import random
import socket
import threading
import time
import traceback

import conc_stubs as S

from recipe_persistence.http_rpc import RpcClient, _Permanent, _Transient
from recipe_persistence.geometry import Geometry, key_group
from recipe_persistence.coordinator import LocalCoordinator, Provider, Ticket
from recipe_persistence.coordinator_http import geometry_identity, layout_fingerprint


# --------------------------------------------------------------------------- #
# Loopback HTTP stub
# --------------------------------------------------------------------------- #

def _ok(body=None, **over):
    if body is None:
        payload = {"ok": True}
    elif isinstance(body, (bytes, bytearray)):
        payload = bytes(body)
    else:
        payload = {"ok": True}
        payload.update(body)
    d = {"status": 200, "body": payload}
    d.update(over)
    return d


class _Stub:
    """Configurable loopback HTTP/1.1 server for RpcClient tests."""

    def __init__(self, decide=None):
        self.lock = threading.Lock()
        self.requests = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.connections = 0
        self.decide = decide or (lambda method, params, n: _ok())
        state = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            disable_nagle_algorithm = True  # mirror the package's MetadataServer

            def log_message(self, *a):  # silence
                pass

            def setup(self):
                super().setup()
                with state.lock:
                    state.connections += 1

            def do_POST(self):
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    n = 0
                body = self.rfile.read(n) if n > 0 else b""
                try:
                    payload = json.loads(body.decode("utf-8"))
                except Exception:
                    payload = {}
                with state.lock:
                    state.requests.append((payload.get("method"), payload.get("params")))
                    state.concurrent += 1
                    state.max_concurrent = max(state.max_concurrent, state.concurrent)
                    seq = len(state.requests)
                try:
                    d = state.decide(payload.get("method"), payload.get("params"), seq)
                    delay = d.get("delay") or 0.0
                    if delay:
                        time.sleep(delay)
                    if d.get("abort"):
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        self.close_connection = True
                        return
                    data = d.get("body", {"ok": True})
                    if not isinstance(data, (bytes, bytearray)):
                        data = json.dumps(data).encode()
                    self.send_response(d.get("status", 200))
                    for k, v in (d.get("headers") or []):
                        self.send_header(k, v)
                    if not d.get("omit_length"):
                        self.send_header("Content-Length", str(len(data)))
                    self.send_header("Content-Type", "application/json")
                    if d.get("close"):
                        self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    if d.get("close"):
                        self.close_connection = True
                except (BrokenPipeError, ConnectionResetError, OSError):
                    self.close_connection = True
                finally:
                    with state.lock:
                        state.concurrent -= 1

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.endpoint = ("127.0.0.1", self.port)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       kwargs={"poll_interval": 0.005}, daemon=True)
        self.thread.start()

    def close(self):
        try:
            self.httpd.shutdown()
        except Exception:
            pass
        try:
            self.httpd.server_close()
        except Exception:
            pass

    def stats(self):
        with self.lock:
            return {"requests": len(self.requests), "connections": self.connections,
                    "max_concurrent": self.max_concurrent}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _client(stub, token, **over):
    kw = dict(rpc_timeout=3.0, pool_size=4, max_body=65536)
    kw.update(over)
    return RpcClient([stub.endpoint], token, **kw)


def _spawn(n, fn):
    """Start n threads running fn(i) (no barrier) and return (threads, results, errors)."""
    results, errors = [None] * n, [None] * n

    def wrap(i):
        try:
            results[i] = fn(i)
        except BaseException as exc:  # noqa: BLE001
            errors[i] = exc

    threads = [threading.Thread(target=wrap, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    return threads, results, errors


def _join(threads, timeout=10.0):
    for t in threads:
        t.join(timeout)
    return [t.is_alive() for t in threads]


# --------------------------------------------------------------------------- #
# RPC: loopback plumbing, pool, bounds
# --------------------------------------------------------------------------- #

def test_rpc_loopback_stub_works():
    """Non-negotiable prerequisite: a stdlib loopback server works with --network=none."""
    seen = []
    stub = _Stub(lambda m, p, n: (seen.append((m, p)) or _ok({"rank": 7})))
    try:
        c = _client(stub, S.dummy_token(), max_active=2)
        try:
            got = c.call(0, "geometry")
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert got == {"ok": True, "rank": 7}, got
    assert seen == [("geometry", {})], seen


def test_rpc_keepalive_connection_reuse():
    stub = _Stub()
    try:
        c = _client(stub, S.dummy_token(), pool_size=4, max_active=1)
        try:
            for i in range(6):
                assert c.call(0, "m", i=i)["ok"] is True
        finally:
            c.shutdown()
    finally:
        stub.close()
    st = stub.stats()
    assert st["requests"] == 6, st
    assert st["connections"] <= 2, f"connection not reused: {st}"


def test_rpc_max_active_is_enforced():
    release, entry = threading.Event(), threading.Event()

    def decide(m, p, n):
        entry.set()
        release.wait(8)
        return _ok()

    stub = _Stub(decide)
    try:
        c = _client(stub, S.dummy_token(), pool_size=4, max_active=2, max_pending=32)
        threads, results, errors = [], [None] * 6, [None] * 6
        try:
            def work(i):
                results[i] = c.call(0, "m", i=i)
            threads, results, errors = _spawn(6, work)
            assert entry.wait(5), "no request reached the stub"
            time.sleep(0.4)
            peak = stub.stats()["max_concurrent"]
            assert peak <= 2, f"max_active=2 violated: peak concurrency {peak}"
            release.set()
            alive = _join(threads, 8)
            assert not any(alive), "calls hung after release"
            assert not any(errors), [repr(e) for e in errors if e]
        finally:
            release.set()
            c.shutdown()
    finally:
        stub.close()


def test_rpc_max_pending_capacity_exhausted():
    release, entry = threading.Event(), threading.Event()
    stub = _Stub(lambda m, p, n: (entry.set(), release.wait(8), _ok())[-1])
    try:
        c = _client(stub, S.dummy_token(), pool_size=1, max_active=1, max_pending=1)
        try:
            t, res, err = _spawn(1, lambda i: c.call(0, "m"))
            assert entry.wait(5)
            t0 = time.perf_counter()
            try:
                c.call(0, "m")
                raise AssertionError("second call was admitted past max_pending=1")
            except _Transient as e:
                assert "capacity" in str(e), str(e)
            assert time.perf_counter() - t0 < 0.5, "capacity rejection was not immediate"
            release.set()
            _join(t, 8)
            assert not err[0], repr(err[0])
        finally:
            release.set()
            c.shutdown()
    finally:
        stub.close()


def test_rpc_timeout_bounds_entire_operation_including_queueing():
    """Docstring: timeout bounds the ENTIRE op incl. queueing + the one retry."""
    stub = _Stub(lambda m, p, n: _ok(delay=0.25))
    try:
        c = _client(stub, S.dummy_token(), rpc_timeout=0.35, pool_size=2,
                    max_active=1, max_pending=8)
        try:
            t0 = time.perf_counter()
            threads, results, errors = _spawn(4, lambda i: c.call(0, "m", i=i))
            alive = _join(threads, 6)
            wall = time.perf_counter() - t0
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert not any(alive), "a call hung past the deadline"
    assert wall < 0.9, f"queueing was not charged to the deadline: wall={wall:.3f}s for 4x0.25s of work"
    ok = [r for r in results if r is not None]
    deadline = [e for e in errors if e is not None]
    assert len(ok) + len(deadline) == 4, (results, errors)
    assert len(deadline) >= 2, f"expected queued calls to expire, got ok={len(ok)} err={[repr(e) for e in deadline]}"


def test_rpc_generous_timeout_lets_all_serialize():
    """Control for the previous test: same load, timeout=5 s -> all succeed."""
    stub = _Stub(lambda m, p, n: _ok(delay=0.15))
    try:
        c = _client(stub, S.dummy_token(), rpc_timeout=5.0, pool_size=2,
                    max_active=1, max_pending=8)
        try:
            threads, results, errors = _spawn(3, lambda i: c.call(0, "m", i=i))
            alive = _join(threads, 10)
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert not any(alive) and not any(errors), [repr(e) for e in errors if e]
    assert all(r and r.get("ok") for r in results), results


# --------------------------------------------------------------------------- #
# RPC: retry semantics
# --------------------------------------------------------------------------- #

def test_rpc_retry_exactly_once_on_transient():
    stub = _Stub(lambda m, p, n: {"status": 500, "body": {"ok": False}})
    try:
        c = _client(stub, S.dummy_token(), max_active=1)
        try:
            try:
                c.call(0, "m")
                raise AssertionError("500 was reported as success")
            except _Transient:
                pass
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert stub.stats()["requests"] == 2, f"expected exactly one retry: {stub.stats()}"


def test_rpc_no_retry_on_permanent_rejection():
    stub = _Stub(lambda m, p, n: {"status": 400, "body": {"ok": False}})
    try:
        c = _client(stub, S.dummy_token(), max_active=1)
        try:
            try:
                c.call(0, "m")
                raise AssertionError("400 was reported as success")
            except _Permanent:
                pass
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert stub.stats()["requests"] == 1, f"permanent rejection was retried: {stub.stats()}"


def test_rpc_retry_succeeds_after_one_transient():
    stub = _Stub(lambda m, p, n: ({"status": 500, "body": {"ok": False}} if n == 1
                                  else _ok({"rank": 1})))
    try:
        c = _client(stub, S.dummy_token(), max_active=1)
        try:
            got = c.call(0, "m")
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert got == {"ok": True, "rank": 1}, got
    assert stub.stats()["requests"] == 2, stub.stats()


def test_rpc_retry_after_connection_drop():
    stub = _Stub(lambda m, p, n: ({"abort": True} if n == 1 else _ok({"rank": 2})))
    try:
        c = _client(stub, S.dummy_token(), max_active=1)
        try:
            got = c.call(0, "m")
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert got == {"ok": True, "rank": 2}, got
    assert stub.stats()["requests"] == 2, stub.stats()


# --------------------------------------------------------------------------- #
# RPC: no-hang / deadline behaviour
# --------------------------------------------------------------------------- #

def test_rpc_dead_loopback_port_does_not_hang():
    port = _free_port()
    c = RpcClient([("127.0.0.1", port)], S.dummy_token(), rpc_timeout=1.0, max_active=1)
    try:
        t0 = time.perf_counter()
        try:
            c.call(0, "m")
            raise AssertionError("call to a closed port reported success")
        except _Transient:
            pass
        dt = time.perf_counter() - t0
    finally:
        c.shutdown()
    assert dt < 1.0, f"ECONNREFUSED path took {dt:.3f}s"


def test_rpc_slow_server_respects_deadline():
    stub = _Stub(lambda m, p, n: _ok(delay=5.0))
    try:
        c = _client(stub, S.dummy_token(), rpc_timeout=0.4, max_active=1)
        try:
            t0 = time.perf_counter()
            try:
                c.call(0, "m")
                raise AssertionError("slow server reported success")
            except _Transient:
                pass
            dt = time.perf_counter() - t0
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert dt < 1.5, f"deadline not honoured: {dt:.3f}s for timeout=0.4"


def test_rpc_response_size_bound():
    stub = _Stub(lambda m, p, n: _ok(body={"pad": "x" * 400}))
    observed = None
    try:
        c = _client(stub, S.dummy_token(), max_body=64, max_active=1)
        try:
            try:
                c.call(0, "m")
                raise AssertionError("oversize response accepted")
            except _Transient as e:
                observed = str(e)
        finally:
            c.shutdown()
    finally:
        stub.close()
    print(f"PERF oversize_response_message={observed!r}", flush=True)


def test_rpc_distinct_transient_causes_are_distinguishable():
    """Probe: _execute collapses every transient cause into one static message."""
    observed = {}

    def run(label, stub, **kw):
        try:
            c = _client(stub, S.dummy_token(), **kw)
            try:
                c.call(0, "m")
                observed[label] = "success"
            except BaseException as e:  # noqa: BLE001
                observed[label] = f"{type(e).__name__}: {e}"
            finally:
                c.shutdown()
        finally:
            stub.close()

    run("oversize_response", _Stub(lambda m, p, n: _ok(body={"pad": "x" * 400})),
        max_body=64, max_active=1)
    run("malformed_framing",
        _Stub(lambda m, p, n: _ok(omit_length=True, headers=[("Transfer-Encoding", "gzip")])),
        max_active=1)
    run("non_json_body", _Stub(lambda m, p, n: _ok(body=b"<html>not json</html>")),
        max_active=1)
    run("envelope_missing_ok",
        _Stub(lambda m, p, n: {"status": 200, "body": {"result": 1}}), max_active=1)
    port = _free_port()
    try:
        c = RpcClient([("127.0.0.1", port)], S.dummy_token(), rpc_timeout=1.0, max_active=1)
        try:
            c.call(0, "m")
        except BaseException as e:  # noqa: BLE001
            observed["connection_refused"] = f"{type(e).__name__}: {e}"
        finally:
            c.shutdown()
    except BaseException:
        pass
    print(f"PERF transient_messages={observed}", flush=True)
    assert len(set(observed.values())) > 1, (
        "DEFECT: all distinct transient causes collapse to the same public error, so an "
        f"operator cannot tell oversize/framing/JSON/transport apart: {observed}")


def test_rpc_request_size_bound():
    stub = _Stub()
    observed = None
    try:
        c = _client(stub, S.dummy_token(), max_body=64, max_active=1)
        try:
            try:
                c.call(0, "m", pad="y" * 400)
                raise AssertionError("oversize request accepted")
            except _Permanent as e:
                observed = str(e)
        finally:
            c.shutdown()
    finally:
        stub.close()
    print(f"PERF oversize_request_message={observed!r}", flush=True)
    assert stub.stats()["requests"] == 0, "oversize request reached the peer"


def test_rpc_response_validation_rejects_malformed():
    cases = {
        "non-json": _ok(body=b"not json at all"),
        "envelope-missing-ok": {"status": 200, "body": {"result": 1}},
        "envelope-ok-false": _ok(body={"ok": False}),
        "envelope-list": {"status": 200, "body": [1, 2]},
        "both-cl-and-te": _ok(omit_length=False, headers=[("Transfer-Encoding", "chunked")]),
        "bad-te": _ok(omit_length=True, headers=[("Transfer-Encoding", "gzip")]),
        "non-decimal-cl": _ok(omit_length=True, headers=[("Content-Length", "abc")]),
    }
    for name, decision in cases.items():
        stub = _Stub(lambda m, p, n, d=decision: d)
        try:
            c = _client(stub, S.dummy_token(), max_active=1)
            try:
                try:
                    c.call(0, "m")
                    raise AssertionError(f"{name}: malformed response accepted")
                except _Transient:
                    pass
            finally:
                c.shutdown()
        finally:
            stub.close()


def test_rpc_header_bound_is_enforced():
    stub = _Stub(lambda m, p, n: _ok(headers=[("X-Pad", "z" * 20000)]))
    try:
        c = _client(stub, S.dummy_token(), max_active=1)
        try:
            try:
                c.call(0, "m")
                raise AssertionError("20 KB of headers accepted past the 16 KB bound")
            except _Transient:
                pass
        finally:
            c.shutdown()
    finally:
        stub.close()


def test_rpc_chunked_response_is_usable():
    """The client explicitly allows Transfer-Encoding: chunked. Verify it works."""
    def chunked(body: bytes) -> bytes:
        return (b"%x\r\n" % len(body)) + body + b"\r\n0\r\n\r\n"

    payload = json.dumps({"ok": True}).encode()
    stub = _Stub(lambda m, p, n: {"status": 200, "body": chunked(payload),
                                  "omit_length": True,
                                  "headers": [("Transfer-Encoding", "chunked")],
                                  "close": False})
    try:
        c = _client(stub, S.dummy_token(), rpc_timeout=0.6, max_active=1)
        try:
            t0 = time.perf_counter()
            try:
                got = c.call(0, "m")
                outcome = f"success {got}"
            except BaseException as e:  # noqa: BLE001
                outcome = f"{type(e).__name__}: {e}"
            dt = time.perf_counter() - t0
        finally:
            c.shutdown()
    finally:
        stub.close()
    print(f"PERF chunked_keepalive outcome={outcome!r} seconds={dt:.3f} "
          f"requests={stub.stats()['requests']}", flush=True)
    assert outcome.startswith("success"), (
        f"well-formed chunked keep-alive response rejected after {dt:.3f}s: {outcome}")


def test_rpc_chunked_with_connection_close_control():
    """Same chunked body but Connection: close -- isolates the probe behaviour."""
    def chunked(body: bytes) -> bytes:
        return (b"%x\r\n" % len(body)) + body + b"\r\n0\r\n\r\n"

    payload = json.dumps({"ok": True}).encode()
    stub = _Stub(lambda m, p, n: {"status": 200, "body": chunked(payload),
                                  "omit_length": True,
                                  "headers": [("Transfer-Encoding", "chunked")],
                                  "close": True})
    try:
        c = _client(stub, S.dummy_token(), rpc_timeout=0.6, max_active=1)
        try:
            t0 = time.perf_counter()
            try:
                got = c.call(0, "m")
                outcome = f"success {got}"
            except BaseException as e:  # noqa: BLE001
                outcome = f"{type(e).__name__}: {e}"
            dt = time.perf_counter() - t0
        finally:
            c.shutdown()
    finally:
        stub.close()
    print(f"PERF chunked_close outcome={outcome!r} seconds={dt:.3f} "
          f"requests={stub.stats()['requests']}", flush=True)
    assert outcome.startswith("success"), f"chunked+close rejected: {outcome}"


# --------------------------------------------------------------------------- #
# RPC: leakage, shutdown, fanout, construction
# --------------------------------------------------------------------------- #

def test_rpc_errors_never_leak_endpoint_or_token():
    token = S.dummy_token(48)
    leaks = []

    def check(label, exc):
        text = f"{type(exc).__name__}: {exc} | args={exc.args!r} | ctx={exc.__context__!r}"
        if token in text or "127.0.0.1" in text:
            leaks.append((label, text))

    # transient / permanent / deadline / connect
    for label, decision, kw in [
        ("500", lambda m, p, n: {"status": 500, "body": {"ok": False}}, dict(max_active=1)),
        ("400", lambda m, p, n: {"status": 400, "body": {"ok": False}}, dict(max_active=1)),
        ("slow", lambda m, p, n: _ok(delay=3.0), dict(rpc_timeout=0.3, max_active=1)),
    ]:
        stub = _Stub(decision)
        try:
            c = _client(stub, token, **kw)
            try:
                c.call(0, "m")
            except BaseException as e:  # noqa: BLE001
                check(label, e)
            finally:
                c.shutdown()
        finally:
            stub.close()
    port = _free_port()
    c = RpcClient([("127.0.0.1", port)], token, rpc_timeout=1.0, max_active=1)
    try:
        c.call(0, "m")
    except BaseException as e:  # noqa: BLE001
        check("refused", e)
    finally:
        c.shutdown()
    assert not leaks, f"secret/endpoint leaked in error text: {leaks}"


def test_rpc_shutdown_races_inflight_calls():
    release, entry = threading.Event(), threading.Event()
    stub = _Stub(lambda m, p, n: (entry.set(), release.wait(6), _ok())[-1])
    try:
        c = _client(stub, S.dummy_token(), rpc_timeout=2.0, pool_size=4,
                    max_active=4, max_pending=16)
        threads, results, errors = [], [], []
        try:
            def work(i):
                return c.call(0, "m", i=i)
            threads, results, errors = _spawn(6, work)
            assert entry.wait(5), "no call reached the stub"
            time.sleep(0.15)
            t0 = time.perf_counter()
            c.shutdown()
            dt = time.perf_counter() - t0
            alive = _join(threads, 8)
            assert dt < 3.0, f"shutdown did not drain promptly: {dt:.3f}s"
            assert not any(alive), "calls hung across shutdown"
            t1 = time.perf_counter()
            c.shutdown()
            assert time.perf_counter() - t1 < 0.2, "repeat shutdown was not a no-op"
        finally:
            release.set()
            try:
                c.shutdown()
            except Exception:
                pass
    finally:
        stub.close()


def test_rpc_shutdown_cancels_waiting_callers():
    release, entry = threading.Event(), threading.Event()
    stub = _Stub(lambda m, p, n: (entry.set(), release.wait(6), _ok())[-1])
    try:
        c = _client(stub, S.dummy_token(), rpc_timeout=2.0, pool_size=2,
                    max_active=1, max_pending=16)
        threads, results, errors = [], [], []
        try:
            threads, results, errors = _spawn(5, lambda i: c.call(0, "m", i=i))
            assert entry.wait(5)
            time.sleep(0.1)
            t0 = time.perf_counter()
            c.shutdown()
            dt = time.perf_counter() - t0
            alive = _join(threads, 8)
            assert not any(alive), "waiting callers hung across shutdown"
            assert dt < 3.0, f"shutdown slow: {dt:.3f}s"
            assert all(e is not None for e in errors), \
                f"a call escaped cancellation: {results}"
        finally:
            release.set()
            try:
                c.shutdown()
            except Exception:
                pass
    finally:
        stub.close()


def test_rpc_no_thread_leak_after_shutdown():
    stub = _Stub()
    try:
        baseline = threading.active_count()
        c = _client(stub, S.dummy_token(), max_active=4, pool_size=4)
        try:
            for i in range(40):
                c.call(0, "m", i=i)
        finally:
            c.shutdown()
        time.sleep(0.25)
        after = threading.active_count()
        assert after <= baseline + 1, f"threads leaked: {baseline} -> {after}"
    finally:
        stub.close()


def test_rpc_fanout_covers_all_ranks():
    stubs = [_Stub(lambda m, p, n, r=i: _ok({"rank": r})) for i in range(3)]
    try:
        c = RpcClient([s.endpoint for s in stubs], S.dummy_token(), rpc_timeout=2.0)
        try:
            out = c.fanout("geometry", lambda rank: {"rank": rank})
        finally:
            c.shutdown()
    finally:
        for s in stubs:
            s.close()
    assert len(out) == 3, out
    assert [r.get("rank") for r in out] == [0, 1, 2], out


def test_rpc_fanout_partial_deadline_is_bounded():
    fast = _Stub(lambda m, p, n: _ok({"rank": 0}))
    slow = _Stub(lambda m, p, n: _ok(delay=4.0))
    try:
        c = RpcClient([fast.endpoint, slow.endpoint], S.dummy_token(), rpc_timeout=0.5)
        try:
            t0 = time.perf_counter()
            out = c.fanout("geometry", lambda rank: {})
            dt = time.perf_counter() - t0
        finally:
            c.shutdown()
    finally:
        fast.close()
        slow.close()
    assert len(out) == 2, out
    assert isinstance(out[1], Exception), f"slow rank did not fail: {out[1]!r}"
    assert dt < 2.0, f"fanout exceeded its deadline: {dt:.3f}s"


def test_rpc_concurrent_mixed_ranks_no_crosstalk():
    a = _Stub(lambda m, p, n: _ok({"rank": 0}))
    b = _Stub(lambda m, p, n: _ok({"rank": 1}))
    try:
        c = RpcClient([a.endpoint, b.endpoint], S.dummy_token(), rpc_timeout=5.0,
                      pool_size=4, max_active=4, max_pending=64)
        try:
            bad = []

            def work(i):
                rank = i % 2
                for _ in range(15):
                    got = c.call(rank, "m")
                    if got.get("rank") != rank:
                        bad.append((rank, got))
            threads, results, errors = _spawn(8, work)
            alive = _join(threads, 10)
        finally:
            c.shutdown()
    finally:
        a.close()
        b.close()
    assert not any(alive) and not any(errors), [repr(e) for e in errors if e]
    assert not bad, f"cross-rank response mixups: {bad[:3]}"


def test_rpc_fanout_cancellable_while_queued():
    """The bounded deque must allow queued fanout work to be removed on shutdown."""
    release, entry = threading.Event(), threading.Event()
    stubs = [_Stub(lambda m, p, n: (entry.set(), release.wait(6), _ok())[-1]) for _ in range(3)]
    try:
        c = RpcClient([s.endpoint for s in stubs], S.dummy_token(), rpc_timeout=2.0,
                      pool_size=2, max_active=1, max_pending=3)
        out = {}

        def do_fanout():
            out["results"] = c.fanout("m", lambda rank: {})

        t = threading.Thread(target=do_fanout, daemon=True)
        t.start()
        try:
            assert entry.wait(5), "fanout never dispatched a request"
            time.sleep(0.15)
            t0 = time.perf_counter()
            c.shutdown()
            dt = time.perf_counter() - t0
            t.join(6)
            assert not t.is_alive(), "fanout hung across shutdown"
            assert dt < 3.0, f"shutdown of queued fanout work was slow: {dt:.3f}s"
            res = out.get("results")
            assert res is not None and len(res) == 3, res
            assert all(isinstance(r, Exception) for r in res), res
        finally:
            release.set()
            try:
                c.shutdown()
            except Exception:
                pass
    finally:
        for s in stubs:
            s.close()


def test_rpc_fanout_reports_capacity_per_rank():
    release, entry = threading.Event(), threading.Event()
    a = _Stub(lambda m, p, n: (entry.set(), release.wait(6), _ok())[-1])
    b = _Stub(lambda m, p, n: (entry.set(), release.wait(6), _ok())[-1])
    try:
        c = RpcClient([a.endpoint, b.endpoint], S.dummy_token(), rpc_timeout=0.6,
                      pool_size=1, max_active=1, max_pending=1)
        try:
            out = c.fanout("m", lambda rank: {})
        finally:
            release.set()
            c.shutdown()
    finally:
        a.close()
        b.close()
    assert len(out) == 2, out
    assert sum(1 for r in out if isinstance(r, Exception)) >= 1, out


def test_rpc_construction_validation():
    tok = S.dummy_token()
    bad = [
        ([], tok),
        ([("localhost", 80)], tok),
        ([("127.0.0.1", 0)], tok),
        ([("127.0.0.1", 70000)], tok),
        ([("127.0.0.1", 80)], ""),
        ([("127.0.0.1", 80)], "has space " + "a" * 40),
        ([("127.0.0.1", 80)], tok + "x" * 260),
        ([("127.0.0.1", 80)], "bad\ttab"),
    ]
    for endpoints, token in bad:
        try:
            RpcClient(endpoints, token)
            raise AssertionError(f"construction accepted {endpoints!r}/{token[:8]!r}")
        except ValueError:
            pass
    stub = _Stub()
    try:
        for kw in [dict(pool_size=0), dict(pool_size=129), dict(max_body=0),
                   dict(max_active=4, max_pending=2), dict(rpc_timeout=0),
                   dict(rpc_timeout=float("nan"))]:
            try:
                RpcClient([stub.endpoint], tok, **kw)
                raise AssertionError(f"construction accepted {kw}")
            except ValueError:
                pass
    finally:
        stub.close()


def test_rpc_unknown_rank_and_bad_method_are_permanent():
    stub = _Stub()
    try:
        c = _client(stub, S.dummy_token(), max_active=1)
        try:
            for args in [(5, "m"), (0, ""), (0, "x" * 200), (0, None)]:
                try:
                    c.call(*args)
                    raise AssertionError(f"accepted {args!r}")
                except _Permanent:
                    pass
        finally:
            c.shutdown()
    finally:
        stub.close()
    assert stub.stats()["requests"] == 0, stub.stats()


def test_rpc_throughput_and_latency_scaling():
    stub = _Stub()
    token = S.dummy_token()
    lines = []
    try:
        for conc in (1, 2, 4, 8):
            calls = 60
            c = RpcClient([stub.endpoint], token, rpc_timeout=5.0,
                          pool_size=8, max_active=conc, max_pending=256)
            try:
                def work(i):
                    out = []
                    for _ in range(calls):
                        t = time.perf_counter()
                        c.call(0, "m", i=i)
                        out.append(time.perf_counter() - t)
                    return out
                t0 = time.perf_counter()
                threads, results, errors = _spawn(conc, work)
                alive = _join(threads, 20)
                wall = time.perf_counter() - t0
            finally:
                c.shutdown()
            assert not any(alive), f"conc={conc} hung"
            assert not any(errors), f"conc={conc} errors: {[repr(e) for e in errors if e]}"
            lat = sorted(x for r in results if r for x in r)
            total = len(lat)
            p95 = lat[int(0.95 * (total - 1))] if lat else 0.0
            lines.append(f"conc={conc} calls={total} wall={wall:.3f}s rps={total / wall:.0f} "
                         f"avg_ms={1000 * sum(lat) / total:.2f} p95_ms={1000 * p95:.2f}")
    finally:
        stub.close()
    print("PERF rpc_scaling " + " | ".join(lines), flush=True)


# --------------------------------------------------------------------------- #
# geometry.key_group
# --------------------------------------------------------------------------- #

def test_key_group_determinism_under_threads():
    groups = 8
    seen = []
    lock = threading.Lock()

    def work(w, i):
        g = (w * 37 + i) % groups
        key = (f"hash{w}".encode() + b"!" * 8) + g.to_bytes(4, "big")
        got = key_group(key, groups)
        if got != g:
            raise AssertionError(f"key_group({key!r})={got} expected {g}")
        with lock:
            seen.append(got)

    r = S.hammer(work, workers=8, iterations=200, timeout=15)
    assert not any(r["errors"]), S.summarize(r, "key_group")
    assert not any(r["alive"]), "key_group worker hung"
    assert len(seen) == 1600, len(seen)


def test_key_group_distribution_is_flat():
    groups, n = 8, 8000
    rng = random.Random(1234)
    counts = [0] * groups
    for _ in range(n):
        g = rng.randrange(groups)
        key = rng.randbytes(16) + g.to_bytes(4, "big")
        counts[key_group(key, groups)] += 1
    expect = n / groups
    worst = max(abs(c - expect) for c in counts)
    assert worst < expect * 0.25, f"skewed distribution: {counts}"


def test_key_group_reads_only_last_four_bytes():
    g = 5
    suffix = g.to_bytes(4, "big")
    assert key_group(b"a" * 100 + suffix, 8) == g
    assert key_group(b"z" + suffix, 8) == g
    assert key_group(b"\x00\x01\x02\x03\x04" + suffix, 8) == g


def test_key_group_boundaries_and_adversarial():
    for key, groups in [
        (b"", 4), (b"a", 4), (b"abcd", 4), (b"\x00\x00\x00\x00", 4),
    ]:
        try:
            key_group(key, groups)
            raise AssertionError(f"accepted key={key!r} groups={groups}")
        except ValueError:
            pass
    # exactly 5 bytes is the minimum legal key
    assert key_group(b"a\x00\x00\x00\x02", 4) == 2
    # group index equal to num_groups is out of range
    for groups in (1, 0, -1):
        try:
            key_group(b"aaaa" + (1).to_bytes(4, "big"), groups)
            raise AssertionError(f"accepted out-of-range group with groups={groups}")
        except ValueError:
            pass
    assert key_group(b"aaaa" + (0).to_bytes(4, "big"), 1) == 0
    # max 32-bit index is legal when the census is that large
    assert key_group(b"aaaa" + b"\xff\xff\xff\xff", 2 ** 32) == 2 ** 32 - 1
    # non-bytes keys are rejected
    for bad in (bytearray(b"aaaa\x00\x00\x00\x01"), memoryview(b"aaaa\x00\x00\x00\x01"),
                "aaaa\x00\x00\x00\x01", None, 12345):
        try:
            key_group(bad, 4)
            raise AssertionError(f"accepted non-bytes key {bad!r}")
        except ValueError:
            pass
    # documented normalisation oddities (not crashes)
    assert key_group(b"aaaa\x00\x00\x00\x00", True) == 0
    assert key_group(b"aaaa\x00\x00\x00\x01", 4.0) == 1


# --------------------------------------------------------------------------- #
# Geometry dataclass (pure CPU)
# --------------------------------------------------------------------------- #

def _caches(pages, groups):
    from types import SimpleNamespace
    return SimpleNamespace(
        tensors=[SimpleNamespace(page_size_bytes=p) for p in pages],
        group_data_refs=[[SimpleNamespace(tensor_idx=i, page_size_bytes=s) for i, s in g]
                         for g in groups])


def test_geometry_from_canonical_validation():
    g = Geometry.from_canonical(_caches([100, 200], [[(0, 100)], [(1, 50), (0, 50)]]))
    assert g.padded_pages == (100, 200)
    assert g.group_refs == (((0, 100),), ((1, 50), (0, 50)))
    assert g.group_bytes == (100, 100)
    assert g.row_bytes == 300
    for pages, groups in [
        ([], [[(0, 1)]]),
        ([100], []),
        ([100], [[]]),
        ([100], [[(1, 10)]]),          # tensor index out of range
        ([100], [[(0, 0)]]),           # zero size
        ([100], [[(0, 101)]]),         # size exceeds page
        ([-5], [[(0, 1)]]),
        ([0], [[(0, 1)]]),
    ]:
        try:
            Geometry.from_canonical(_caches(pages, groups))
            raise AssertionError(f"accepted pages={pages} groups={groups}")
        except ValueError:
            pass


def test_geometry_rows_for_budget_and_frozen():
    g = Geometry((100, 200), (((0, 100),),))
    assert g.rows_for_budget(1000, 5) == 3
    assert g.rows_for_budget(300, 5) == 1
    assert g.rows_for_budget(10 ** 9, 5) == 5
    for budget, rows in [(0, 5), (-1, 5), (1000, 0), (299, 5)]:
        try:
            g.rows_for_budget(budget, rows)
            raise AssertionError(f"accepted budget={budget} max_rows={rows}")
        except ValueError:
            pass
    try:
        g.padded_pages = (1,)
        raise AssertionError("Geometry is not frozen")
    except Exception:
        pass


def test_geometry_hash_and_equality_under_threads():
    a = Geometry((100, 200), (((0, 100),),))
    b = Geometry((100, 200), (((0, 100),),))
    c = Geometry((100, 200), (((1, 100),),))
    errs = []

    def work(w, i):
        if hash(a) != hash(b) or a != b or a == c:
            errs.append((w, i, hash(a), hash(b)))

    r = S.hammer(work, workers=8, iterations=200, timeout=15)
    assert not any(r["errors"]), S.summarize(r, "geometry hash")
    assert not errs, errs[:3]


# --------------------------------------------------------------------------- #
# layout_fingerprint / geometry_identity
# --------------------------------------------------------------------------- #

# NOTE: conc_stubs.dummy_infos()/dummy_geometry() emit a flat group_refs list
# (refs, not groups) and are unusable with these two functions; use these
# correctly-nested fixtures instead.


def _refs(groups):
    return [[[int(i), int(s)] for i, s in group] for group in groups]


_GROUPS = (((0, 1024), (1, 64)), ((2, 64),))
_PAGES = (2304, 64, 64)


def _infos(ranks=4, pages=_PAGES, groups=_GROUPS):
    return [{"padded_pages": list(pages), "group_refs": _refs(groups)}
            for _ in range(ranks)]


def _geo(pages=_PAGES, groups=_GROUPS):
    from types import SimpleNamespace
    return SimpleNamespace(
        padded_pages=list(pages),
        group_refs=_refs(groups),
        group_bytes=[sum(s for _, s in group) for group in groups])


def test_stub_dummy_geometry_is_incompatible_with_identity():
    """Probe: the 'validated' harness fixtures cannot feed the package functions."""
    outcome = {}
    try:
        outcome["geometry_identity(dummy_geometry())"] = geometry_identity(S.dummy_geometry())
    except BaseException as e:  # noqa: BLE001
        outcome["geometry_identity(dummy_geometry())"] = f"{type(e).__name__}: {e}"
    try:
        outcome["layout_fingerprint(dummy_infos())"] = layout_fingerprint(S.dummy_infos())
    except BaseException as e:  # noqa: BLE001
        outcome["layout_fingerprint(dummy_infos())"] = f"{type(e).__name__}: {e}"
    print(f"PERF stub_fixture_compat={outcome}", flush=True)
    bad = [k for k, v in outcome.items() if isinstance(v, str)]
    assert not bad, ("HARNESS DEFECT: conc_stubs fixtures are not consumable by the package "
                     f"functions they target: {[(k, outcome[k]) for k in bad]}")


def test_layout_fingerprint_stable_under_threads():
    infos = _infos()
    base = layout_fingerprint(infos)
    assert len(base) == 64 and base == base.lower()
    digests = set()
    lock = threading.Lock()

    def work(w, i):
        d = layout_fingerprint(_infos())
        with lock:
            digests.add(d)
        if d != base:
            raise AssertionError(f"digest drift under concurrency: {d} != {base}")

    r = S.hammer(work, workers=8, iterations=100, timeout=15)
    assert not any(r["errors"]), S.summarize(r, "layout_fingerprint")
    assert not any(r["alive"]), "fingerprint worker hung"
    assert digests == {base}, digests


def test_layout_fingerprint_reorder_changes_digest():
    a = [{"padded_pages": [100, 200],
          "group_refs": [[[0, 50], [1, 50]]]}]
    b = [{"padded_pages": [100, 200],
          "group_refs": [[[1, 50], [0, 50]]]}]
    assert layout_fingerprint(a) != layout_fingerprint(b), \
        "reordered refs produced the same digest"
    c = [{"padded_pages": [200, 100], "group_refs": [[[0, 50], [1, 50]]]}]
    assert layout_fingerprint(a) != layout_fingerprint(c), "page reorder not detected"


def test_layout_fingerprint_alias_changes_digest_at_equal_bytes():
    split = [{"padded_pages": [100], "group_refs": [[[0, 50], [0, 50]]]}]
    whole = [{"padded_pages": [100], "group_refs": [[[0, 100]]]}]
    assert sum(s for _, s in split[0]["group_refs"][0]) == \
        sum(s for _, s in whole[0]["group_refs"][0]) == 100
    assert layout_fingerprint(split) != layout_fingerprint(whole), \
        "different tensor aliasing at equal group bytes produced the same digest"


def test_layout_fingerprint_rank_order_matters():
    row0 = {"padded_pages": [100], "group_refs": [[[0, 100]]]}
    row1 = {"padded_pages": [200], "group_refs": [[[0, 200]]]}
    assert layout_fingerprint([row0, row1]) != layout_fingerprint([row1, row0])


def test_layout_fingerprint_adversarial_inputs():
    # empty census is deterministic (native admission rejects it separately)
    e1, e2 = layout_fingerprint([]), layout_fingerprint([])
    assert e1 == e2 and len(e1) == 64
    # int() normalisation: all of these coerce to the same pages list [1]
    def d(pages):
        return layout_fingerprint([{"padded_pages": pages, "group_refs": [[[0, 7]]]}])
    assert d([1]) == d([True]), "bool/int normalisation"
    assert d([1]) == d([1.0]), "float/int normalisation"
    assert d([1]) == d(["1"]), "str/int normalisation"
    # and non-integral floats are silently truncated before hashing
    assert d([1]) == d([1.9]), "float truncation before hashing"
    # non-numeric values raise instead of silently colliding
    for bad in ([None], [{}], ["x"], [[1]]):
        try:
            layout_fingerprint([{"padded_pages": bad, "group_refs": [[[0, 1]]]}])
            raise AssertionError(f"accepted padded_pages={bad!r}")
        except (ValueError, TypeError):
            pass
    # long-ish but valid census stays stable
    big = [{"padded_pages": list(range(1, 33)),
            "group_refs": [[[i, 1] for i in range(32)]]} for _ in range(8)]
    assert layout_fingerprint(big) == layout_fingerprint(big)


def test_geometry_identity_stable_under_threads():
    g = _geo()
    base = geometry_identity(g)
    out = []
    lock = threading.Lock()

    def work(w, i):
        ident = geometry_identity(_geo())
        with lock:
            out.append(ident)
        if ident != base:
            raise AssertionError(f"geometry_identity drift: {ident} != {base}")

    r = S.hammer(work, workers=8, iterations=100, timeout=15)
    assert not any(r["errors"]), S.summarize(r, "geometry_identity")
    assert not any(r["alive"]), "geometry_identity worker hung"
    assert all(o == base for o in out)
    assert base["padded_pages"] == list(g.padded_pages)
    assert base["group_refs"] == [[list(r) for r in gr] for gr in g.group_refs]
    assert base["group_bytes"] == [int(b) for b in g.group_bytes]
    assert len(base["fingerprint"]) == 64


def test_geometry_identity_is_order_sensitive():
    a = _geo(groups=(((0, 10), (1, 20)),))
    b = _geo(groups=(((1, 20), (0, 10)),))
    assert geometry_identity(a)["fingerprint"] != geometry_identity(b)["fingerprint"]


# --------------------------------------------------------------------------- #
# LocalCoordinator / Ticket / Provider
# --------------------------------------------------------------------------- #

_SIZES = ((10, 10), (20, 20))
_NS = S.dummy_namespace("rank-0")


def _key(name, group):
    return name.encode() + group.to_bytes(4, "big")


def _coord(stores=2, sizes=_SIZES, **over):
    st = [S.dummy_store(f"coord{i}") for i in range(stores)]
    return LocalCoordinator(st, sizes, **over), st


def _write_all(stores, ticket):
    for store, lease, size in zip(stores, ticket.leases, ticket.sizes):
        assert store.write(lease, [S.dummy_blob(size, seed=size)]) is True


def test_ticket_and_provider_shapes():
    t = Ticket("a" * 32, _NS, b"k", ("l1", "l2"), (10, 10), "own", True)
    assert t.token == "a" * 32 and t.writing is True
    assert hash(t) == hash(Ticket("a" * 32, _NS, b"k", ("l1", "l2"), (10, 10), "own", True))
    try:
        t.token = "b"
        raise AssertionError("Ticket is not frozen")
    except Exception:
        pass
    p = Provider(coordinator=None, store=None, size_by_group=_SIZES, max_pending_keys=8)
    assert p.close is None and p.layout_fingerprint == ""


def test_local_coordinator_concurrent_acquire_write_release():
    coord, stores = _coord()
    errors = []
    lock = threading.Lock()

    def work(w, i):
        group = (w + i) % len(_SIZES)
        key = _key(f"w{w}-k{i}", group)
        ticket = coord.reserve_store(_NS, key, f"own{w}", _SIZES[group])
        if ticket is None:
            raise AssertionError("reserve_store returned None under no contention")
        _write_all(stores, ticket)
        if coord.complete_store(ticket, True) is not None:
            raise AssertionError("complete_store did not acknowledge durability")

    r = S.hammer(work, workers=8, iterations=15, timeout=25)
    assert not any(r["errors"]), S.summarize(r, "coordinator acquire")
    assert not any(r["alive"]), "coordinator worker hung"
    assert coord._tickets == {}, f"tickets leaked: {list(coord._tickets)[:3]}"
    assert coord._bytes == 0, f"byte credits leaked: {coord._bytes}"
    for w in range(8):
        for i in range(15):
            group = (w + i) % len(_SIZES)
            assert any(s.exists(_NS, _key(f"w{w}-k{i}", group)) for s in stores)
    for s in stores:
        s.close()


def test_local_coordinator_idempotent_ticket_under_threads():
    coord, stores = _coord()
    key = _key("shared", 0)
    ticket = coord.reserve_store(_NS, key, "own", _SIZES[0])
    assert ticket is not None
    tokens = []
    lock = threading.Lock()

    def work(w, i):
        got = coord.reserve_store(_NS, key, "own", _SIZES[0])
        with lock:
            tokens.append(None if got is None else got.token)

    r = S.hammer(work, workers=8, iterations=10, timeout=15)
    assert not any(r["errors"]), S.summarize(r, "idempotent reserve")
    assert set(tokens) == {ticket.token}, f"idempotency broken: {set(tokens)}"
    # a different owner on the same key must NOT get a second writer ticket:
    # the rank store only has one object slot per (ns,key), so the coordinator
    # must fail closed rather than hand out two independent writes.
    other = coord.reserve_store(_NS, key, "other", _SIZES[0])
    assert other is None, f"two writers were admitted for one key: {other}"
    assert coord.release(ticket) is None
    assert coord._tickets == {} and coord._bytes == 0
    for s in stores:
        s.close()


def test_local_coordinator_pending_key_and_byte_bounds():
    coord, stores = _coord(max_pending_keys=4, max_pending_bytes=1000)
    held = []
    for i in range(4):
        t = coord.reserve_store(_NS, _key(f"k{i}", 0), f"o{i}", _SIZES[0])
        assert t is not None, i
        held.append(t)
    assert coord.reserve_store(_NS, _key("k5", 0), "o5", _SIZES[0]) is None, \
        "max_pending_keys was not enforced"
    coord.release(held[0])
    late = coord.reserve_store(_NS, _key("k5", 0), "o5", _SIZES[0])
    assert late is not None
    coord.release(late)
    for t in held[1:]:
        coord.release(t)
    assert coord._bytes == 0, coord._bytes
    for s in stores:
        s.close()

    coord2, stores2 = _coord(max_pending_keys=100, max_pending_bytes=25)
    a = coord2.reserve_store(_NS, _key("a", 1), "o", _SIZES[1])   # charge 20
    assert a is not None
    assert coord2.reserve_store(_NS, _key("b", 1), "o", _SIZES[1]) is None, \
        "max_pending_bytes was not enforced"
    coord2.release(a)
    assert coord2._bytes == 0
    assert coord2.reserve_store(_NS, _key("b", 1), "o", _SIZES[1]) is not None
    for s in stores2:
        s.close()


def test_local_coordinator_byte_accounting_no_leak_under_threads():
    coord, stores = _coord(max_pending_keys=4096)
    def work(w, i):
        key = _key(f"leak-{w}-{i}", (w + i) % len(_SIZES))
        g = (w + i) % len(_SIZES)
        t = coord.reserve_store(_NS, key, f"o{w}", _SIZES[g])
        if t is None:
            raise AssertionError("reserve failed")
        coord.release(t)
    with S.Stopwatch() as sw:
        r = S.hammer(work, workers=8, iterations=20, timeout=60)
    ops = 8 * 20
    print(f"PERF coordinator_reserve_release conc=8 ops={ops} seconds={sw.seconds} "
          f"ops_per_sec={ops / max(sw.seconds, 1e-9):.0f}", flush=True)
    assert not any(r["errors"]), S.summarize(r, "byte accounting")
    assert not any(r["alive"]), "worker hung"
    # single-thread control: isolates contention from per-op durability cost
    coord_s, stores_s = _coord(max_pending_keys=4096)
    with S.Stopwatch() as sw2:
        for i in range(20):
            t = coord_s.reserve_store(_NS, _key(f"seq-{i}", 0), "o", _SIZES[0])
            assert t is not None
            coord_s.release(t)
    print(f"PERF coordinator_reserve_release conc=1 ops=20 seconds={sw2.seconds} "
          f"ops_per_sec={20 / max(sw2.seconds, 1e-9):.0f}", flush=True)
    assert not any(r["alive"]), "worker hung"
    assert coord._bytes == 0, f"credit leak: {coord._bytes}"
    assert coord._tickets == {}, f"ticket leak: {len(coord._tickets)}"
    assert coord._identities == {}, f"identity leak: {len(coord._identities)}"
    for s in stores + stores_s:
        s.close()


def test_local_coordinator_reserve_load_requires_existing_object():
    coord, stores = _coord()
    key = _key("load", 1)
    assert coord.reserve_load(_NS, key, "o") is None, "reserved a read for a missing object"
    t = coord.reserve_store(_NS, key, "o", _SIZES[1])
    _write_all(stores, t)
    assert coord.complete_store(t, True) is None
    rt = coord.reserve_load(_NS, key, "reader")
    assert rt is not None and rt.writing is False
    assert rt.sizes == _SIZES[1]
    assert coord.release(rt) is None
    # wrong declared sizes are rejected without touching the stores
    assert coord.reserve_store(_NS, _key("bad", 0), "o", (1, 1)) is None
    assert coord.reserve_store(_NS, _key("bad", 9), "o", _SIZES[0]) is None
    for s in stores:
        s.close()


def test_local_coordinator_expiry_and_reuse():
    coord, stores = _coord()
    # shrink the lease TTL via the store limits
    for s in stores:
        s.close()
    stores = [S.dummy_store(f"exp{i}", lease_seconds=0.15, grace_seconds=0.02)
              for i in range(2)]
    coord = LocalCoordinator(stores, _SIZES)
    key = _key("exp", 0)
    t = coord.reserve_store(_NS, key, "o", _SIZES[0])
    assert t is not None
    assert coord.renew(t) is True
    deadline = coord.lease_deadline(t)
    assert isinstance(deadline, float) and deadline > time.monotonic()
    time.sleep(0.30)
    assert coord.renew(t) is False, "renew succeeded after the lease TTL"
    assert coord.lease_deadline(t) is None or coord.lease_deadline(t) <= time.monotonic()
    # after expiry the same identity must be re-reservable (bounded wait for grace)
    t2, waited = None, 0.0
    while waited < 2.0:
        t2 = coord.reserve_store(_NS, key, "o", _SIZES[0])
        if t2 is not None:
            break
        time.sleep(0.02)
        waited += 0.02
    print(f"PERF coordinator_expiry_rerelease_wait={waited:.3f}s token_changed="
          f"{t2 is not None and t2.token != t.token}", flush=True)
    assert t2 is not None, "expired write lease permanently blocked its own key"
    assert coord.release(t2) is None
    for s in stores:
        s.close()


def test_local_coordinator_short_keys_are_misparsed():
    """geometry.key_group demands len(key)>4; reserve_load/store silently accept shorter keys."""
    coord, stores = _coord()
    short = b"\x00\x00\x00\x01"          # exactly 4 bytes, no hash prefix
    try:
        key_group(short, 2)
        key_group_raised = False
    except ValueError:
        key_group_raised = True
    assert key_group_raised
    t = coord.reserve_store(_NS, short, "o", _SIZES[1])
    observed = None if t is None else t.token
    if t is not None:
        coord.release(t)
    print(f"PERF short_key reserve_store(4-byte key, group 1) -> "
          f"{'ticket' if observed else 'None'}", flush=True)
    for s in stores:
        s.close()
    assert observed is None, (
        "DEFECT: LocalCoordinator accepted a 4-byte key and parsed it as group 1 "
        "while geometry.key_group rejects the same key with ValueError")


def test_local_coordinator_non_bytes_key_raises():
    coord, stores = _coord()
    raised = None
    try:
        coord.reserve_load(_NS, "not-bytes-key", "o")
    except BaseException as e:  # noqa: BLE001
        raised = f"{type(e).__name__}: {e}"
    print(f"PERF non_bytes_key reserve_load -> {raised!r}", flush=True)
    for s in stores:
        s.close()
    assert raised is None, (
        "DEFECT: LocalCoordinator.reserve_load leaked a raw exception for a non-bytes key "
        f"instead of returning None: {raised}")


def test_local_coordinator_invalidate_veto_and_recovery():
    coord, stores = _coord()
    key = _key("veto", 0)
    t = coord.reserve_store(_NS, key, "o", _SIZES[0])
    assert t is not None
    assert coord.invalidate(_NS, key) is None
    assert coord.reserve_store(_NS, key, "o2", _SIZES[0]) is None, "veto not enforced"
    coord.release(t)
    assert (_NS, key) not in coord._invalid, \
        "in-memory veto survived the last ticket draining"
    # the rank store keeps a tombstone for grace_seconds, so the key becomes
    # reservable again shortly after, not necessarily immediately
    waited, again = 0.0, None
    while waited < 3.0:
        again = coord.reserve_store(_NS, key, "o2", _SIZES[0])
        if again is not None:
            break
        time.sleep(0.05)
        waited += 0.05
    print(f"PERF invalidate_rewrite_wait={waited:.2f}s", flush=True)
    assert again is not None, "invalidated key never became reservable again"
    coord.release(again)
    for s in stores:
        s.close()


def test_local_coordinator_complete_store_failure_invalidates():
    coord, stores = _coord()
    key = _key("fail", 0)
    t = coord.reserve_store(_NS, key, "o", _SIZES[0])
    assert t is not None
    res = coord.complete_store(t, False)
    assert res is None, res
    assert not any(s.exists(_NS, key) for s in stores)
    assert coord._tickets == {} and coord._bytes == 0
    for s in stores:
        s.close()


def test_local_coordinator_double_release_and_foreign_ticket():
    coord, stores = _coord()
    key = _key("dbl", 0)
    t = coord.reserve_store(_NS, key, "o", _SIZES[0])
    assert coord.release(t) is None
    assert coord.release(t) is None, "second release was not a harmless no-op"
    assert coord._bytes == 0, f"double release corrupted credits: {coord._bytes}"
    forged = Ticket(t.token, _NS, key, t.leases, t.sizes, "attacker", True)
    assert coord.release(forged) is None
    assert coord._bytes == 0
    for s in stores:
        s.close()


def test_local_coordinator_can_store_and_lease_deadline_after_close():
    coord, stores = _coord()
    assert coord.can_store() is True
    t = coord.reserve_store(_NS, _key("cs", 0), "o", _SIZES[0])
    assert t is not None
    assert coord.lease_deadline(t) is not None
    stores[0].close()
    assert coord.can_store() is False, "closed store not reflected"
    assert coord.lease_deadline(t) is None, "lease proof retained after store close"
    coord.release(t)
    for s in stores[1:]:
        s.close()


def test_local_coordinator_closed_rejects_reservations():
    coord, stores = _coord()
    # exhausting the invalid-veto bound closes the coordinator (fail-closed):
    # the veto only persists for pairs with an active ticket, so hold one.
    coord2, stores2 = _coord(max_pending_keys=1)
    held = coord2.reserve_store(_NS, _key("a", 0), "o", _SIZES[0])
    assert held is not None
    assert coord2.invalidate(_NS, _key("a", 0)) is None
    assert coord2.invalidate(_NS, _key("b", 0)) is False
    assert coord2.can_store() is False
    assert coord2.reserve_load(_NS, _key("a", 0), "o") is None
    coord2.release(held)
    for s in stores + stores2:
        s.close()


def test_local_coordinator_complete_store_under_threads():
    coord, stores = _coord(max_pending_keys=4096)
    def work(w, i):
        g = (w + i) % len(_SIZES)
        key = _key(f"cs-{w}-{i}", g)
        t = coord.reserve_store(_NS, key, f"o{w}", _SIZES[g])
        if t is None:
            raise AssertionError("reserve failed")
        _write_all(stores, t)
        if coord.complete_store(t, True) is not None:
            raise AssertionError("complete_store failed")
    r = S.hammer(work, workers=8, iterations=12, timeout=25)
    assert not any(r["errors"]), S.summarize(r, "complete_store")
    assert not any(r["alive"]), "worker hung"
    assert coord._tickets == {} and coord._bytes == 0
    for s in stores:
        s.close()
