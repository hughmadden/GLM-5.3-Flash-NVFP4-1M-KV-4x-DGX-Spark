# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hugh Madden and contributors
"""Real loopback sockets only; deadlines/ownership are tested without CUDA."""
import json
import socket
import threading
import time

import pytest
from recipe_persistence import http_rpc as rpc

TOKEN = "ephemeral-test-only-auth-token-0123456789"


def reply(conn, body=b'{"ok":true}', status=200, *, keepalive=True, length=None):
    length = len(body) if length is None else length
    headers = (f"HTTP/1.1 {status} Test\r\nContent-Length: {length}\r\n"
               f"Connection: {'keep-alive' if keepalive else 'close'}\r\n\r\n").encode()
    conn.sendall(headers + body)


class Peer:
    """Small bounded test peer with explicit thread/socket cleanup."""
    def __init__(self, responder=None, *, ipv6=False):
        self.responder = responder or (lambda conn, request, peer: reply(conn))
        self.stop = threading.Event()
        self.seen = threading.Event()
        self.requests = []
        self.connections = 0
        self.threads = []
        self.sockets = set()
        self.lock = threading.Lock()
        self.listener = socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("::1" if ipv6 else "127.0.0.1", 0))
        self.listener.listen(16)
        self.listener.settimeout(0.1)
        self.endpoint = self.listener.getsockname()[:2]
        self.acceptor = threading.Thread(target=self._accept, daemon=True)
        self.acceptor.start()

    def _accept(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.listener.accept()
            except (OSError, TimeoutError):
                continue
            conn.settimeout(1)
            with self.lock:
                self.sockets.add(conn)
                self.connections += 1
            thread = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            self.threads.append(thread)
            thread.start()

    def _serve(self, conn):
        buffered = b""
        try:
            while not self.stop.is_set():
                while b"\r\n\r\n" not in buffered:
                    part = conn.recv(4096)
                    if not part:
                        return
                    buffered += part
                headers, buffered = buffered.split(b"\r\n\r\n", 1)
                length = int(next(line.split(b":", 1)[1] for line in headers.split(b"\r\n")
                                  if line.lower().startswith(b"content-length:")))
                while len(buffered) < length:
                    part = conn.recv(4096)
                    if not part:
                        return
                    buffered += part
                body, buffered = buffered[:length], buffered[length:]
                request = json.loads(body)
                with self.lock:
                    self.requests.append(request)
                self.seen.set()
                if self.responder(conn, request, self) is False:
                    return
        except (OSError, ValueError, StopIteration):
            pass
        finally:
            with self.lock:
                self.sockets.discard(conn)
            conn.close()

    def close(self):
        self.stop.set()
        self.listener.close()
        with self.lock:
            sockets = tuple(self.sockets)
        for conn in sockets:
            rpc._shutdown_socket(conn)
        self.acceptor.join(1)
        for thread in self.threads:
            thread.join(1)
        assert not self.acceptor.is_alive()
        assert not any(thread.is_alive() for thread in self.threads)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_numeric_socket_connect_avoids_resolver_and_reuses_complete_response(monkeypatch):
    with Peer() as peer:
        def no_dns(*args, **kwargs):
            raise AssertionError("resolver must not be called")
        monkeypatch.setattr(socket, "getaddrinfo", no_dns)
        client = rpc.RpcClient([peer.endpoint], TOKEN)
        try:
            assert client.call(0, "geometry") == {"ok": True}
            assert client.call(0, "geometry") == {"ok": True}
            assert peer.connections == 1
            assert len(peer.requests) == 2
            pooled = next(conn for stack in client._pool.values() for conn in stack)
            assert pooled.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == 1
        finally:
            client.shutdown()
        assert not client._operations and client._active == 0
        assert not any(t.is_alive() for t in client._threads)


@pytest.mark.parametrize("host", ["localhost", "example.invalid", "127.1", "::1%interface", ""])
def test_dns_names_and_ambiguous_literals_are_rejected(host):
    with pytest.raises(ValueError, match="numeric"):
        rpc.RpcClient([(host, 1234)], TOKEN)


def test_ipv6_numeric_socket_connect(monkeypatch):
    if not socket.has_ipv6:
        pytest.skip("IPv6 unavailable")
    try:
        peer = Peer(ipv6=True)
    except OSError:
        pytest.skip("IPv6 loopback unavailable")
    with peer:
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("unexpected DNS"))
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=0.5)
        try:
            assert client.call(0, "geometry")["ok"] is True
        finally:
            client.shutdown()


@pytest.mark.parametrize("part", ["headers", "body"])
def test_dripping_response_cannot_extend_absolute_deadline(part):
    def drip(conn, request, peer):
        if part == "headers":
            conn.sendall(b"HTTP/1.1 200 OK\r\nX-Drip: ")
            payload = b"x" * 200
        else:
            payload = b'{"ok":true,"pad":"' + b"x" * 100 + b'"}'
            conn.sendall(f"HTTP/1.1 200 OK\r\nContent-Length: {len(payload)}\r\n\r\n".encode())
        for byte in payload:
            if peer.stop.wait(0.01):
                return False
            conn.sendall(bytes([byte]))
        return False
    with Peer(drip) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=0.12)
        try:
            started = time.monotonic()
            with pytest.raises(rpc._Transient):
                client.call(0, "geometry")
            assert time.monotonic() - started < 0.5
            assert not client._operations and client._active == 0
            assert sum(map(len, client._pool.values())) == 0
        finally:
            client.shutdown()


def test_retry_shares_original_deadline_and_timeout_is_mutable():
    def response(conn, request, peer):
        if len(peer.requests) == 1:
            reply(conn, status=503)
        else:
            conn.sendall(b"HTTP/1.1 200 OK\r\nX-Drip: ")
            while not peer.stop.wait(0.01):
                conn.sendall(b"x")
    with Peer(response) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=3)
        client.timeout = 0.12
        try:
            started = time.monotonic()
            with pytest.raises(rpc._Transient):
                client.call(0, "geometry")
            assert time.monotonic() - started < 0.5
            assert len(peer.requests) == 2
        finally:
            client.shutdown()


@pytest.mark.parametrize("response_kind", ["oversize", "truncated", "bad_json", "chunked_oversize"])
def test_invalid_or_unread_response_is_never_pooled(response_kind):
    def response(conn, request, peer):
        if response_kind == "oversize":
            reply(conn, b"", length=10000)
        elif response_kind == "truncated":
            reply(conn, b'{"ok":true}', length=100, keepalive=False)
        elif response_kind == "bad_json":
            reply(conn, b"not-json")
        else:
            conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                         b"100\r\n" + b"x" * 256 + b"\r\n0\r\n\r\n")
        return False
    with Peer(response) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=0.5, max_body=128)
        try:
            with pytest.raises(rpc._Transient):
                client.call(0, "geometry")
            assert sum(map(len, client._pool.values())) == 0
            assert peer.connections == 2
        finally:
            client.shutdown()


def test_permanent_rejection_does_not_retry_or_include_secrets():
    def reject(conn, request, peer):
        reply(conn, status=403)
    with Peer(reject) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN)
        try:
            with pytest.raises(rpc._Permanent) as error:
                client.call(0, "secret-method", token="secret-key")
            text = str(error.value)
            assert TOKEN not in text and "secret" not in text and peer.endpoint[0] not in text
            assert len(peer.requests) == 1
        finally:
            client.shutdown()


def test_permanent_error_does_not_wait_for_or_retry_unfinished_body():
    def reject(conn, request, peer):
        conn.sendall(b"HTTP/1.1 403 Rejected\r\nContent-Length: 100\r\n\r\n")
        peer.stop.wait(1)
        return False
    with Peer(reject) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=1)
        try:
            started = time.monotonic()
            with pytest.raises(rpc._Permanent):
                client.call(0, "geometry")
            assert time.monotonic() - started < 0.3
            assert len(peer.requests) == 1 and sum(map(len, client._pool.values())) == 0
        finally:
            client.shutdown()


def test_shutdown_interrupts_owned_header_read_and_drains_watchdog():
    def hang(conn, request, peer):
        conn.sendall(b"HTTP/1.1 200 OK\r\nX-Hang: ")
        peer.stop.wait(2)
        return False
    with Peer(hang) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=2)
        results = []
        thread = threading.Thread(target=lambda: results.extend(client.fanout("geometry", lambda r: {})))
        thread.start()
        assert peer.seen.wait(1)
        started = time.monotonic()
        client.shutdown()
        assert time.monotonic() - started < 0.5
        thread.join(1)
        assert not thread.is_alive() and isinstance(results[0], rpc._Transient)
        assert not client._operations and not client._queue and client._active == 0
        client.shutdown()


def test_cancelled_queued_calls_are_removed_without_releasing_active_credit():
    with Peer() as peer:
        client = rpc.RpcClient([peer.endpoint] * 3, TOKEN, rpc_timeout=0.08,
                               max_active=1, max_pending=3)
        gate, started = threading.Event(), threading.Event()
        executed = []
        def params(rank):
            executed.append(rank)
            started.set()
            gate.wait(2)
            return {}
        try:
            results = client.fanout("geometry", params)
            assert started.is_set() and all(isinstance(r, rpc._Transient) for r in results)
            assert executed == [0]
            assert len(client._operations) == 1 and client._active == 1
            assert not client._queue  # cancelled WorkItems do not accumulate
            unexpected = []
            def queued_params(rank):
                unexpected.append(rank)
                return {}
            for _ in range(3):
                more = client.fanout("geometry", queued_params)
                assert all(isinstance(r, rpc._Transient) for r in more)
                assert len(client._operations) == 1 and not client._queue
                assert not unexpected
            with pytest.raises(RuntimeError, match="not drained"):
                client.shutdown()
            assert not client._fully_stopped and len(client._operations) == 1
        finally:
            gate.set()
            client.timeout = 1
            client.shutdown()
        assert not peer.requests and not client._operations


def test_watchdog_join_is_inside_active_and_submission_bounds(monkeypatch):
    original_timer = threading.Timer
    joining, release = threading.Event(), threading.Event()
    class JoinedTimer(original_timer):
        def join(self, timeout=None):
            joining.set()
            release.wait(2)
            return super().join(timeout)
    monkeypatch.setattr(rpc.threading, "Timer", JoinedTimer)
    with Peer() as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=0.08,
                               max_pending=1, max_active=1)
        try:
            result = client.fanout("geometry", lambda rank: {})
            assert joining.is_set() and isinstance(result[0], rpc._Transient)
            assert client._active == 1 and len(client._operations) == 1
            with pytest.raises(rpc._Transient, match="capacity"):
                client.call(0, "geometry")
            with pytest.raises(RuntimeError, match="not drained"):
                client.shutdown()
            assert not client._fully_stopped
        finally:
            release.set()
            client.timeout = 1
            client.shutdown()
        assert client._active == 0 and not client._operations


@pytest.mark.parametrize("headers", [
    b"Content-Length: 11\r\nContent-Length: 11\r\n",
    b"Content-Length: 11\r\nTransfer-Encoding: chunked\r\n",
    b"Content-Length: -1\r\n",
])
def test_ambiguous_response_framing_is_rejected(headers):
    def malformed(conn, request, peer):
        conn.sendall(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n" + b'{"ok":true}')
        return False
    with Peer(malformed) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=0.5)
        try:
            with pytest.raises(rpc._Transient):
                client.call(0, "geometry")
            assert sum(map(len, client._pool.values())) == 0
        finally:
            client.shutdown()


def test_total_response_headers_are_bounded_before_body():
    def oversized(conn, request, peer):
        conn.sendall(b"HTTP/1.1 200 OK\r\n" + (b"X-Pad: " + b"x" * 400 + b"\r\n") * 50
                     + b"Content-Length: 11\r\n\r\n" + b'{"ok":true}')
        return False
    with Peer(oversized) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=0.5)
        try:
            with pytest.raises(rpc._Transient):
                client.call(0, "geometry")
            assert sum(map(len, client._pool.values())) == 0
        finally:
            client.shutdown()


def test_direct_calls_share_active_and_submission_limits():
    gate, two_active = threading.Event(), threading.Event()
    server_active = [0, 0]
    server_lock = threading.Lock()
    def blocked(conn, request, peer):
        with server_lock:
            server_active[0] += 1
            server_active[1] = max(server_active)
            if server_active[0] == 2:
                two_active.set()
        try:
            gate.wait(2)
            reply(conn)
        finally:
            with server_lock:
                server_active[0] -= 1
    with Peer(blocked) as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=1,
                               max_active=2, max_pending=4)
        barrier = threading.Barrier(9)
        results = []
        def call():
            barrier.wait()
            try:
                results.append(client.call(0, "geometry"))
            except rpc._Transient as error:
                results.append(error)
        threads = [threading.Thread(target=call) for _ in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait()
        try:
            assert two_active.wait(1)
            with client._condition:
                assert client._active == 2
                assert len(client._operations) <= 4
            gate.set()
            for thread in threads:
                thread.join(2)
            assert all(not t.is_alive() for t in threads)
            assert len(results) == 8 and server_active[1] <= 2
        finally:
            gate.set()
            client.shutdown()
            for thread in threads:
                thread.join(2)


def test_shutdown_budget_includes_serialization_wait():
    with Peer() as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=0.06)
        client._shutdown_lock.acquire()
        try:
            started = time.monotonic()
            with pytest.raises(RuntimeError, match="in progress"):
                client.shutdown()
            assert time.monotonic() - started < 0.3
        finally:
            client._shutdown_lock.release()
            client.shutdown()


def test_completed_calls_leave_no_live_deadline_threads(monkeypatch):
    original_timer = threading.Timer
    timers = []
    class TrackedTimer(original_timer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            timers.append(self)
    monkeypatch.setattr(rpc.threading, "Timer", TrackedTimer)
    with Peer() as peer:
        client = rpc.RpcClient([peer.endpoint], TOKEN, rpc_timeout=1)
        try:
            for _ in range(25):
                assert client.call(0, "geometry")["ok"]
                assert all(not timer.is_alive() for timer in timers)
                assert not client._operations and client._active == 0
        finally:
            client.shutdown()
