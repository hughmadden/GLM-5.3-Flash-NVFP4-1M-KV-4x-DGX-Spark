# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hugh Madden and contributors
"""Portable authenticated HTTP metadata provider for all-rank disk persistence.

Implements the coordinator.Provider factory contract in coordinator.py with a
metadata-only transport: no KV payload ever crosses HTTP. Each worker binds a
bounded, authenticated server publishing its actual canonical geometry
(fingerprint, group order, per-group bytes); the scheduler factory performs a
bounded census of every rank, verifies common layout identity and rank census,
and returns a RemoteCoordinator plus the registered per-rank group sizes. The
scheduler never infers sizes from representative layer specs: geometry=None is
the only supported scheduler input, and every published byte comes from a
worker registration.

Fail-closed rules implemented here:
- Any rank missing, slow, corrupt or capacity-denied rolls back every already
  acquired lease and returns None; a Ticket exists only when every rank
  reserved the exact size. There is no local-only hit.
- complete_store reports success only after every rank proves durability
  (exists); otherwise all rank keys are invalidated and the ticket released.
- Cleanup RPC failures are retained in a bounded retry queue or left to remote
  lease expiry; an attempted operation is never reported as proved success.
- Request handling never raises transport exceptions into the caller.

Deployment contract: addresses, ports, auth token file path and disk root are
externally provided configuration; nothing defaults to a contributor's
deployment and no credential is ever logged or passed via argv.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from http.client import LineTooLong
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
import hmac
import hashlib
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import stat
import threading
import time
import uuid
from urllib.parse import urlsplit

from .coordinator import Provider, Ticket
from .storage import DiskStore, Limits, fingerprint

PROTOCOL = "recipe-persistence-coordinator/1"
DEFAULT_MAX_PENDING_KEYS = 32768
DEFAULT_MAX_PENDING_BYTES = 64_000_000_000  # Logical queue credit, not an allocation.
_NS = re.compile(r"[0-9a-f]{64}")
_HEX = re.compile(r"[0-9a-f]+")
_REQUEST_ID = re.compile(r"[0-9a-f]{32,64}")
_LEASE = re.compile(r"[0-9a-f]{32}")
_MAX_JSON_SIZE = 1 << 20
_MAX_SIZE_FIELD = 1 << 40  # JSON sanity bound; the store enforces its own cap.

_LOG = logging.getLogger(__name__)


def _emit_note(message: str, *args) -> None:
    """Diagnostics level gate, same contract as native._emit_trace.

    Static codes only; nothing identifying may reach a log line.
    """
    if os.environ.get("PERSIST_DEBUG_TRACE", "").strip().lower() not in (
            "", "0", "false", "no"):
        _LOG.warning(message, *args)
    else:
        _LOG.info(message, *args)

# Fixed RPC allowlist. Dispatch is a dict lookup, never getattr, so no request
# can execute arbitrary methods or touch file paths.
_RPC_SCHEMA = {
    "geometry": frozenset(),
    "reserve_read": frozenset({"request_id", "namespace", "key", "owner"}),
    "reserve_write": frozenset({"request_id", "namespace", "key", "owner", "size"}),
    "exists": frozenset({"namespace", "key"}),
    "release": frozenset({"lease"}),
    "renew": frozenset({"lease"}),
    "invalidate": frozenset({"namespace", "key"}),
}


from .http_rpc import _Transient, _Permanent


# --------------------------------------------------------------------------- #
# External secrets and addresses
# --------------------------------------------------------------------------- #

def _validate_secret(token: str) -> None:
    if not 32 <= len(token) <= 256:
        raise ValueError("auth token must contain 32..256 characters")
    if any(c.isspace() or not 0x21 <= ord(c) <= 0x7E for c in token):
        raise ValueError("auth token must be printable without whitespace")
    # Conservative alphabet-diversity entropy estimate; operators should
    # generate tokens with a CSPRNG (>= 32 bytes hex or base64).
    if len(set(token)) < 8 or len(token) * math.log2(len(set(token))) < 128:
        raise ValueError("auth token entropy is too low")


def load_auth_token(path) -> str:
    """Read the private shared secret from an owner-only regular file.

    The token never has a literal default, is never logged and never appears in
    argv; it exists only in memory after this call. Symlinks, foreign owners,
    group/other read bits and low-entropy values are rejected.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("coordinator_auth_token_file path is required")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        raise ValueError(f"auth token file cannot be opened: {e.errno}") from e
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("auth token file must be a regular file")
        if info.st_uid != os.geteuid():
            raise ValueError("auth token file must be owned by the service user")
        if info.st_mode & 0o077:
            raise ValueError("auth token file must be owner-only (0600)")
        data = os.read(fd, 4097)
    finally:
        os.close(fd)
    if len(data) > 4096:
        raise ValueError("auth token file is too large")
    try:
        token = data.decode("ascii").strip()
    except UnicodeDecodeError as e:
        raise ValueError("auth token must be ASCII") from e
    _validate_secret(token)
    return token


def parse_endpoint(text, *, bind: bool = False):
    """Validate one externally provided rank endpoint into (host, port).

    Accepted forms: "host:port" and "http://host:port[/]". User info, query
    strings, fragments, non-empty paths and non-http schemes are rejected.
    """
    if not isinstance(text, str) or not text or len(text) > 300:
        raise ValueError("endpoint must be a short non-empty string")
    if "://" in text:
        parts = urlsplit(text)
        if parts.scheme != "http":
            raise ValueError("only plain http coordinator endpoints are accepted")
    else:
        parts = urlsplit("//" + text)
    if parts.username is not None or parts.password is not None:
        raise ValueError("endpoint user info is not allowed")
    if parts.query or parts.fragment:
        raise ValueError("endpoint query/fragment parameters are not allowed")
    if parts.path not in ("", "/"):
        raise ValueError("endpoint path must be empty")
    host = parts.hostname
    if not host or len(host) > 253 or "%" in host or any(c.isspace() for c in host):
        raise ValueError("endpoint host is required")
    try:
        port = parts.port
    except ValueError as e:
        raise ValueError("endpoint port is invalid") from e
    if port is None:
        raise ValueError("explicit endpoint port is required")
    top = 0 if bind else 1
    if not top <= port <= 65535:
        raise ValueError(f"endpoint port must be {top}..65535" + (" for binding" if bind else ""))
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError("coordinator endpoints require numeric IP addresses") from None
    if not bind and (address.is_unspecified or address.is_multicast):
        raise ValueError("coordinator destination must be a unicast IP address")
    return str(address), port


# --------------------------------------------------------------------------- #
# Canonical geometry identity
# --------------------------------------------------------------------------- #

def geometry_identity(geometry) -> dict:
    """Length-delimited fingerprint over the actual canonical registration."""
    pages = tuple(int(p) for p in geometry.padded_pages)
    refs = [[[int(i), int(s)] for i, s in group] for group in geometry.group_refs]
    ident = fingerprint(protocol=PROTOCOL,
                        padded_pages=json.dumps(pages, separators=(",", ":")),
                        group_refs=json.dumps(refs, separators=(",", ":")))
    return {"fingerprint": ident, "group_refs": refs, "padded_pages": list(pages),
            "group_bytes": [int(b) for b in geometry.group_bytes]}


def layout_fingerprint(infos) -> str:
    """Exact public all-rank layout identity used as Provider.layout_fingerprint.

    Formula: SHA-256 (hex) of the ASCII compact JSON — separators
    ``(",", ":")``, ``ensure_ascii=True``, no key sorting — of the
    rank-ordered census list::

        [[padded_pages_r0, group_refs_r0], [padded_pages_r1, ...], ...]

    where ``padded_pages`` is the rank's list of canonical tensor page bytes
    and ``group_refs`` is its list of cache groups, each a list of
    ``[tensor_idx, page_size_bytes]`` pairs in canonical registration order.
    Reordered references or different tensor aliasing change the digest even
    when per-group byte totals match, so objects are never reused across
    layouts. Census currently requires identical rows on every rank (the TP4
    expectation); the rank-ordered formula stays stable if uneven rank layouts
    are ever permitted.
    """
    census = [
        [
            [int(p) for p in info["padded_pages"]],
            [[[int(i), int(s)] for i, s in group] for group in info["group_refs"]],
        ]
        for info in infos
    ]
    blob = json.dumps(census, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def _validated_geometry(info, rank: int, world_size: int) -> dict:
    if not isinstance(info, dict) or info.get("ok") is not True:
        raise _Transient("geometry response malformed")
    if info.get("protocol") != PROTOCOL:
        raise _Permanent("coordinator protocol mismatch")
    if info.get("rank") != rank or info.get("world_size") != world_size:
        raise _Permanent("rank census mismatch")
    ident = info.get("fingerprint")
    if not isinstance(ident, str) or not _NS.fullmatch(ident):
        raise _Transient("geometry fingerprint malformed")
    refs = info.get("group_refs")
    pages = info.get("padded_pages")
    row = info.get("group_bytes")
    if (not isinstance(refs, list) or not refs or not isinstance(pages, list)
            or not isinstance(row, list) or len(row) != len(refs)):
        raise _Transient("geometry shape malformed")
    if not pages or any(type(v) is not int or not 0 < v <= _MAX_SIZE_FIELD for v in pages):
        raise _Transient("geometry page sizes malformed")
    for group in refs:
        if not isinstance(group, list) or not group:
            raise _Transient("geometry group malformed")
        for ref in group:
            if (not isinstance(ref, list) or len(ref) != 2
                    or type(ref[0]) is not int or not 0 <= ref[0] < len(pages)
                    or type(ref[1]) is not int or ref[1] <= 0
                    or ref[1] > pages[ref[0]]):
                raise _Transient("geometry reference malformed")
    if any(type(v) is not int or v <= 0 for v in pages) or any(type(v) is not int or v <= 0 for v in row):
        raise _Transient("geometry sizes malformed")
    expected_bytes = [sum(ref[1] for ref in group) for group in refs]
    expected_fingerprint = fingerprint(
        protocol=PROTOCOL, padded_pages=json.dumps(pages, separators=(",", ":")),
        group_refs=json.dumps(refs, separators=(",", ":")))
    if row != expected_bytes or ident != expected_fingerprint:
        raise _Permanent("registered geometry bytes or fingerprint disagree")
    if (type(info.get("lease_seconds")) not in (int, float)
            or not math.isfinite(info["lease_seconds"]) or info["lease_seconds"] <= 0):
        raise _Transient("lease ttl malformed")
    profile = info.get("profile", "")
    if not isinstance(profile, str) or not _NS.fullmatch(profile):
        raise _Transient("profile identity malformed")
    return info


# --------------------------------------------------------------------------- #
# Bounded authenticated metadata server (worker role)
# --------------------------------------------------------------------------- #

_SATURATED = b'{"ok":false,"error":"server_saturated"}'
_REJECT_503 = (b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
               b"Connection: close\r\nContent-Length: " + str(len(_SATURATED)).encode()
               + b"\r\n\r\n" + _SATURATED)


def _reject(request) -> None:
    # Saturated or shutting down: fail closed immediately, never queue.
    try:
        request.settimeout(0.1)
        request.sendall(_REJECT_503)
    except OSError:
        pass
    finally:
        try:
            request.close()
        except OSError:
            pass


class _BoundedHTTPServer(ThreadingHTTPServer):
    """Threading server whose handler threads are hard-bounded.

    An unacquired permit is answered with an immediate 503, so a stalled store
    (its RLock spans I/O) can wedge at most `max_concurrent` threads and excess
    requests fail closed instead of accumulating.
    """
    daemon_threads = True
    block_on_close = False  # Handler threads are bounded and self-draining.
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address, handler, *, max_concurrent, shutdown_flag):
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        self._sockets = set()
        self._socket_lock = threading.Lock()
        super().__init__(address, handler)
        self._permits = threading.BoundedSemaphore(max_concurrent)
        self._flag = shutdown_flag

    def server_bind(self):
        # HTTPServer.server_bind calls getfqdn even for numeric addresses.
        # Bypass that uncancellable reverse-DNS lookup as well as client DNS.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def process_request(self, request, client_address):
        if self._flag.is_set() or not self._permits.acquire(blocking=False):
            _reject(request)
            return
        with self._socket_lock:
            self._sockets.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self._socket_lock:
                self._sockets.discard(request)
            self._permits.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._socket_lock:
                self._sockets.discard(request)
            self._permits.release()

    def abort_connections(self):
        # Closing a socket alone does not interrupt its makefile readers.
        with self._socket_lock:
            sockets = tuple(self._sockets)
        for connection in sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def handle_error(self, request, client_address):
        # Tracebacks could echo request material; transport stays silent.
        pass


class _RequestHeaderReader:
    """Bound request-line and header allocation before authentication."""
    def __init__(self, stream):
        self.stream = stream
        self.left = 16384

    def readline(self, limit=-1):
        bound = self.left + 1
        line = self.stream.readline(bound if limit < 0 else min(limit, bound))
        self.left -= len(line)
        if self.left < 0:
            raise LineTooLong("metadata request headers")
        return line

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _MetadataHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "rpkv-coordinator"
    sys_version = ""
    disable_nagle_algorithm = True

    def setup(self):
        # Bounded idle keepalive: a parked connection releases its thread.
        self.timeout = self.server.io_timeout
        super().setup()
        self.rfile = _RequestHeaderReader(self.rfile)

    def handle_one_request(self):
        self.rfile.left = 16384
        return super().handle_one_request()

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass  # Never log: headers contain the bearer credential.

    def _send(self, status, payload, *, close=False):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, status, error):
        self._send(status, {"ok": False, "error": error}, close=True)

    def do_GET(self):
        self._fail(405, "method_not_allowed")

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_GET

    def do_POST(self):
        try:
            self._post()
        except Exception:
            self._fail(500, "internal_error")

    def _post(self):
        server = self.server
        if server.flag.is_set():
            return self._fail(503, "unavailable")
        if self.path not in ("/", "/rpc"):
            return self._fail(404, "not_found")
        if self.command != "POST":
            return self._fail(405, "method_not_allowed")
        auth = self.headers.get("Authorization", "")
        if not hmac.compare_digest(auth.encode("utf-8", "replace"), server.expected_auth):
            return self._fail(401, "unauthorized")
        if self.headers.get("Transfer-Encoding"):
            return self._fail(400, "transfer_encoding_not_supported")
        lengths = self.headers.get_all("Content-Length") or []
        if not lengths:
            return self._fail(411, "length_required")
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,10}", lengths[0].strip()):
            return self._fail(400, "bad_content_length")
        length = int(lengths[0])
        if length <= 0:
            return self._fail(400, "empty_body")
        if length > server.max_body:
            return self._fail(413, "body_too_large")
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            return self._fail(415, "unsupported_media_type")
        body = self.rfile.read(length)
        if len(body) != length:
            return self._fail(400, "short_body")
        try:
            request = json.loads(body)
        except ValueError:
            return self._fail(400, "invalid_json")
        if not isinstance(request, dict):
            return self._fail(400, "invalid_envelope")
        method = request.get("method")
        params = request.get("params", {})
        if method not in _RPC_SCHEMA or not isinstance(params, dict):
            return self._fail(400, "unknown_method")
        if set(params) != _RPC_SCHEMA[method]:
            return self._fail(400, "invalid_params")
        try:
            payload = server.dispatch(method, params)
        except ValueError:
            return self._fail(400, "invalid_params")
        except _RetryAgain:
            return self._fail(503, "request_in_flight")
        except _BackendUnavailable:
            return self._fail(410, "rank_backend_disabled")
        self._send(200, payload)


class _RetryAgain(Exception):
    """Another handler is computing this request id; the client must retry."""


class _BackendUnavailable(Exception):
    """This rank's store is permanently disabled for this process lifetime."""


class _MetadataRPC:
    """Allowlisted dispatch onto the worker's own DiskStore object.

    The server shares the exact DiskStore instance used by the native I/O
    handler: no second process, index or root is opened. `profile` publishes
    the configured cache/runtime identity (extra config cache_fingerprint and
    tenant_namespace, which itself encodes the hash seed/policy) alongside the
    actual canonical geometry; the census requires both to agree across ranks.
    """

    def __init__(self, *, store, identity, rank, world_size, idem_entries, profile=""):
        self.store = store
        self.identity = identity
        self.rank = rank
        self.world_size = world_size
        self.profile = profile
        self.group_count = len(identity["group_refs"])
        # 0010: static counter for fast-negative reserve_read short-circuits
        # (observable, never silent -- the design's gate-metric analogue).
        self.absent_index_skips = 0
        self.lease_seconds = float(store.limits.lease_seconds)
        self._idem = OrderedDict()   # request_id -> (payload, expiry); bounded LRU.
        self._pending = OrderedDict()  # request_id -> Event for in-flight computes.
        self._idem_cap = int(idem_entries)
        self._idem_lock = threading.Lock()
        self._wait = max(0.5, min(5.0, self.lease_seconds))
        self._read_refusal_notes = 0  # Bounded why-refusal probe (diagnostics).

    def geometry_payload(self):
        ident = self.identity
        return {"ok": True, "protocol": PROTOCOL, "rank": self.rank,
                "world_size": self.world_size, "fingerprint": ident["fingerprint"],
                "group_refs": ident["group_refs"], "padded_pages": ident["padded_pages"],
                "group_bytes": ident["group_bytes"], "lease_seconds": self.lease_seconds,
                "profile": self.profile}

    def _validated_key(self, value):
        if not isinstance(value, str) or not _HEX.fullmatch(value) or not 2 <= len(value) <= 256:
            raise ValueError("key must be hex encoding 1..128 bytes")
        return bytes.fromhex(value)

    def _validated_ns_key(self, params):
        namespace = params.get("namespace")
        if not isinstance(namespace, str) or not _NS.fullmatch(namespace):
            raise ValueError("namespace must be a sha256 fingerprint")
        return namespace, self._validated_key(params.get("key"))

    def _validated_common(self, params):
        namespace, key = self._validated_ns_key(params)
        owner = params.get("owner")
        if not isinstance(owner, str) or len(owner.encode()) > 128:
            raise ValueError("owner too long")
        return namespace, key, owner

    def _idempotent(self, request_id, signature):
        """Return a cached payload, or None when this caller owns the compute.

        Exactly one handler computes per request id: simultaneous duplicates
        wait on a bounded event for the owner's result instead of creating a
        second rank lease. Registration and cache lookup are atomic under one
        lock; the pending map is capped by the same bound as the result map.
        """
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("invalid request id")
        deadline = time.monotonic() + self._wait
        while True:
            with self._idem_lock:
                now = time.monotonic()
                while self._idem:
                    first = next(iter(self._idem.values()))
                    if first[2] > now:
                        break
                    self._idem.popitem(last=False)
                hit = self._idem.get(request_id)
                if hit is not None and hit[2] > time.monotonic():
                    if hit[0] != signature:
                        raise ValueError("request id reused with different parameters")
                    return hit[1]
                self._idem.pop(request_id, None)
                if request_id not in self._pending:
                    if len(self._pending) + len(self._idem) >= self._idem_cap:
                        raise _RetryAgain("idempotency receipt capacity exhausted")
                    self._pending[request_id] = (signature, threading.Event())
                    return None
                pending = self._pending[request_id]
                if pending[0] != signature:
                    raise ValueError("request id reused with different parameters")
                event = pending[1]
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not event.wait(remaining):
                raise _RetryAgain("reservation still in flight")

    def _remember(self, request_id, signature, payload):
        with self._idem_lock:
            self._idem[request_id] = (signature, payload, time.monotonic() + self.lease_seconds)
            # A pending slot was reserved before I/O. Never evict a still-live
            # receipt to admit another operation: saturation is a clean miss.
            pending = self._pending.pop(request_id, None)
        if pending is not None:
            pending[1].set()

    def _forget(self, request_id):
        # Owner failed before producing a result: wake waiters without caching.
        with self._idem_lock:
            pending = self._pending.pop(request_id, None)
        if pending is not None:
            pending[1].set()

    def dispatch(self, method, params):
        # Fixed allowlist dispatch: no request can name a file, an arbitrary
        # attribute or an unregistered method.
        handlers = {"geometry": self._rpc_geometry,
                    "reserve_read": self._rpc_reserve_read,
                    "reserve_write": self._rpc_reserve_write,
                    "exists": self._rpc_exists, "release": self._rpc_release,
                    "renew": self._rpc_renew, "invalidate": self._rpc_invalidate}
        if method not in handlers or set(params) != _RPC_SCHEMA[method]:
            raise ValueError("unsupported metadata operation")
        if (self.store.failed or self.store.closed) and method not in {"release", "invalidate"}:
            raise _BackendUnavailable()
        result = handlers[method](params)
        if self.store.failed or self.store.closed:
            raise _BackendUnavailable()
        return result

    def _rpc_geometry(self, params):
        return self.geometry_payload()

    def _group_of(self, key):
        group = int.from_bytes(key[-4:], "big")
        return group if len(key) >= 5 and group < self.group_count else None

    def _rpc_reserve_read(self, params):
        request_id = params["request_id"]
        # Validate the envelope before registering ownership of the request id.
        namespace, key, owner = self._validated_common(params)
        # 0010: authoritative fast negative from the store's in-RAM durable
        # index. Absent keys skip the idempotent replay lookup, the store's
        # reserve_read (index SELECT + file stat), and the why-refusal probe
        # (which would itself re-read the index). A cached lease for a key
        # that became absent is dead anyway (invalidate semantics), so
        # short-circuiting before the replay cache is correct.
        if not self.store.durable(namespace, key):
            self.absent_index_skips += 1
            return {"ok": True, "lease": None}
        signature = ("read", namespace, key, owner)
        cached = self._idempotent(request_id, signature)
        if cached is not None:
            lease = cached.get("lease")
            if lease is not None and not self.store.lease_valid(lease):
                return {"ok": True, "lease": None}
            return cached
        try:
            lease = None
            if self._group_of(key) is not None:
                lease = self.store.reserve_read(namespace, key, owner)
                if lease is None:
                    # Why-refusal probe (diagnostics only, static codes):
                    # distinguishes "object absent" (store miss) from lease
                    # capacity / failed state. exists() is one indexed read.
                    # Bounded: at most three notes per server instance.
                    if self._read_refusal_notes < 3:
                        self._read_refusal_notes += 1
                        _emit_note("reserve_read refused: exists=%s failed=%s closed=%s",
                                   bool(self.store.exists(namespace, key)),
                                   bool(self.store.failed), bool(self.store.closed))
            payload = {"ok": True, "lease": lease}
            self._remember(request_id, signature, payload)
            return payload
        except BaseException:
            self._forget(request_id)
            raise

    def _rpc_reserve_write(self, params):
        request_id = params["request_id"]
        namespace, key, owner = self._validated_common(params)
        size = params.get("size")
        if type(size) is not int or not 0 < size <= _MAX_SIZE_FIELD:
            raise ValueError("invalid size")
        signature = ("write", namespace, key, owner, size)
        cached = self._idempotent(request_id, signature)
        if cached is not None:
            lease = cached.get("lease")
            if lease is not None and not self.store.lease_valid(lease):
                return {"ok": True, "lease": None}
            return cached
        try:
            lease = None
            if self._group_of(key) is not None:
                lease = self.store.reserve_write(namespace, key, size, owner)
            payload = {"ok": True, "lease": lease}
            self._remember(request_id, signature, payload)
            return payload
        except BaseException:
            self._forget(request_id)
            raise

    def _rpc_exists(self, params):
        namespace, key = self._validated_ns_key(params)
        return {"ok": True, "exists": bool(self.store.exists(namespace, key))}

    def _rpc_release(self, params):
        lease = params.get("lease")
        if not isinstance(lease, str) or not _LEASE.fullmatch(lease):
            raise ValueError("invalid lease token")
        # None/True acknowledge; only an explicit False is a negative ack
        # (store.release became bool; older revisions return None on success).
        return {"ok": True, "released": self.store.release(lease) is not False}

    def _rpc_renew(self, params):
        lease = params.get("lease")
        if not isinstance(lease, str) or not _LEASE.fullmatch(lease):
            raise ValueError("invalid lease token")
        return {"ok": True, "renewed": bool(self.store.renew(lease))}

    def _rpc_invalidate(self, params):
        namespace, key = self._validated_ns_key(params)
        return {"ok": True, "invalidated": self.store.invalidate(namespace, key) is not False}


class MetadataServer:
    """Worker-side metadata endpoint bound to an externally provided address."""

    def __init__(self, *, store, identity, rank, world_size, host, port, token,
                 threads=16, max_body=65536, rpc_timeout=10.0, idem_entries=32768,
                 profile=""):
        if not 1 <= threads <= 128:
            raise ValueError("server thread bound must be 1..128")
        if not 256 <= max_body <= _MAX_JSON_SIZE:
            raise ValueError("body bound must be 256..1MiB")
        self._flag = threading.Event()
        self.rpc = _MetadataRPC(store=store, identity=identity, rank=rank,
                                world_size=world_size, idem_entries=idem_entries,
                                profile=profile)
        self._httpd = _BoundedHTTPServer((host, port), _MetadataHandler,
                                         max_concurrent=threads, shutdown_flag=self._flag)
        self._httpd.store = store
        self._httpd.flag = self._flag
        self._httpd.max_body = max_body
        self._httpd.io_timeout = max(0.5, min(rpc_timeout, 5.0))
        self._httpd.expected_auth = b"Bearer " + token.encode("ascii")
        self._httpd.dispatch = self.rpc.dispatch
        self._thread = None
        self._stopped = False
        self._fully_stopped = False
        self._permits = self._httpd._permits
        self._thread_bound = threads

    @property
    def endpoint(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"

    @property
    def store(self):
        return self.rpc.store

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._httpd.serve_forever,
                                            kwargs={"poll_interval": 0.05},
                                            name="kv-coord-accept", daemon=True)
            self._thread.start()

    def shutdown(self, timeout=5.0):
        """Stop accepting, then prove in-flight handlers drained.

        Raises RuntimeError when handlers are still active at the deadline:
        the caller must NOT close the shared DiskStore in that state. The
        call is retryable — a later shutdown() re-attempts the drain.
        """
        if self._fully_stopped:
            return
        if not self._stopped:
            self._stopped = True
            self._flag.set()
            if self._thread is not None and self._thread.is_alive():
                self._httpd.shutdown()
                self._thread.join(timeout)
        self._httpd.abort_connections()
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("metadata acceptor did not drain before close")
        # Drain proof: once every permit is free, no metadata request is
        # touching the shared store, so the worker may close it.
        held = 0
        deadline = time.monotonic() + timeout
        while held < self._thread_bound:
            if self._permits.acquire(blocking=False):
                held += 1
            elif time.monotonic() >= deadline:
                break
            else:
                time.sleep(0.005)
        if held < self._thread_bound:
            for _ in range(held):
                self._permits.release()
            raise RuntimeError("metadata endpoint did not drain before close")
        for _ in range(held):
            self._permits.release()
        self._httpd.server_close()
        self._fully_stopped = True


# --------------------------------------------------------------------------- #
# Bounded keepalive RPC client (used by both roles for census/fanout)
# --------------------------------------------------------------------------- #

# Implemented separately so transport resource/deadline tests remain focused.
from .http_rpc import RpcClient


# --------------------------------------------------------------------------- #
# Scheduler-side all-rank coordinator
# --------------------------------------------------------------------------- #

class RemoteCoordinator:
    """CoordinatorProtocol over per-rank metadata RPC.

    Reservations are logically transactional: every rank must reserve the exact
    size or all acquired leases roll back and the call returns None. Queue
    credits (keys and logical bytes) are bounded by active tickets, never by
    disk capacity, and mirror the in-process LocalCoordinator semantics.
    """

    def __init__(self, client, size_by_group, *, max_pending_keys=DEFAULT_MAX_PENDING_KEYS,
                 max_pending_bytes=DEFAULT_MAX_PENDING_BYTES, lease_seconds=300.0,
                 renew_margin=None):
        self.client = client
        self.size_by_group = tuple(tuple(int(n) for n in row) for row in size_by_group)
        world = len(client.endpoints)
        if not self.size_by_group or any(len(row) != world or min(row) <= 0
                                         for row in self.size_by_group):
            raise ValueError("geometry must contain every rank")
        if max_pending_keys <= 0 or max_pending_bytes <= 0:
            raise ValueError("positive coordinator bounds required")
        self.world_size = world
        self.max_pending_keys = int(max_pending_keys)
        self.max_pending_bytes = int(max_pending_bytes)
        self._ttl = float(lease_seconds)
        margin = renew_margin if renew_margin is not None else max(1.0, min(30.0, 0.1 * self._ttl))
        if not 0 < margin < self._ttl:
            raise ValueError("renew margin must be inside the lease ttl")
        self._margin = float(margin)
        self._lock = threading.RLock()
        self._tickets = {}
        self._identities = {}
        self._valid = {}  # ticket token -> monotonic instant a cached renew holds until.
        self._pending = set()  # Identities with credits reserved before any RPC.
        self._retiring = set()
        self._invalid = set()
        self._invalidate_acked = set()
        self._bytes = 0
        self._retry = deque(maxlen=8192)  # Bounded cleanup replay log.
        self._cleanup_lost = False
        self._closed = False
        self._reserve_reasons = {}  # Diagnostics only: why a reservation refused.

    def _note_reserve(self, code):
        reasons = self._reserve_reasons
        reasons[code] = reasons.get(code, 0) + 1
        if reasons[code] <= 3:
            _emit_note("reserve reason %s (#%d): %s",
                       code, reasons[code], sorted(reasons.items()))

    # -- validation helpers ------------------------------------------------ #

    @staticmethod
    def _check_identity(namespace, key, owner):
        if not isinstance(namespace, str) or not _NS.fullmatch(namespace):
            return False
        if not isinstance(key, bytes) or not 5 <= len(key) <= 128:
            return False
        if not isinstance(owner, str) or len(owner.encode()) > 128:
            return False
        return True

    def _retry_cleanup(self, limit=32):
        """Replay bounded cleanup entries; True only when the queue emptied."""
        for _ in range(min(limit, len(self._retry))):
            with self._lock:
                if not self._retry:
                    return not self._cleanup_lost
                kind, rank, params = self._retry.popleft()
            try:
                result = self.client.call(rank, kind, **params)
                key = {"release": "released", "invalidate": "invalidated"}[kind]
                if result.get(key) is not True:
                    self._replay(kind, rank, **params)
                    return False
            except Exception as error:
                self._record_failure(error)
                self._replay(kind, rank, **params)
                return False
        return not self._retry and not self._cleanup_lost

    def _replay(self, kind, rank, **params):
        with self._lock:
            if (kind, rank, params) in self._retry:
                return
            if len(self._retry) == self._retry.maxlen:
                # Losing a receipt is not acknowledgement, even if newer
                # retries later drain. Retain the uncertainty in bounded state.
                self._cleanup_lost = True
                self._closed = True
            self._retry.append((kind, rank, params))

    # -- reservation ------------------------------------------------------- #

    def _rollback(self, leases):
        """Release every acquired rank lease; unknown results are never ACKs."""
        acked = True
        for rank, lease in enumerate(leases):
            if lease is None:
                continue
            try:
                result = self.client.call(rank, "release", lease=lease)
                if result.get("released") is not True:
                    acked = False
                    self._replay("release", rank, lease=lease)
            except Exception as error:
                self._record_failure(error)
                acked = False
                self._replay("release", rank, lease=lease)
        return acked

    def _reserve(self, namespace, key, owner, sizes, writing):
        if not self._check_identity(namespace, key, owner):
            return None
        group = int.from_bytes(key[-4:], "big")
        if group >= len(self.size_by_group) or tuple(sizes) != self.size_by_group[group]:
            return None
        identity = (namespace, key, owner, writing)
        charge = max(sizes)
        with self._lock:
            if self._closed:
                self._note_reserve("closed")
                return None
            if (namespace, key) in self._invalid:
                self._note_reserve("veto_invalid")
                return None
            token = self._identities.get(identity)
            ticket = self._tickets.get(token) if token is not None else None
            retiring = token in self._retiring if token is not None else False
        if retiring:
            self._note_reserve("identity_retiring")
            return None
        if ticket is not None:
            # B3: renew OUTSIDE the global lock. A blocking all-rank renew
            # RPC used to run with the lock held, stalling every unrelated
            # lease_deadline check behind it; re-validating under the lock
            # after the RPC keeps the identity semantics exact.
            if self.renew(ticket):
                with self._lock:
                    if (self._identities.get(identity) == token
                            and token not in self._retiring
                            and (namespace, key) not in self._invalid
                            and not self._closed):
                        return ticket
                # Concurrently released or invalidated: fall through to a
                # fresh reservation.
            else:
                self.release(ticket)
                self._note_reserve("identity_renew_dead")
                return None
        with self._lock:
            if self._closed or (namespace, key) in self._invalid:
                return None
            if (identity in self._pending
                    or len(self._tickets) + len(self._pending) >= self.max_pending_keys
                    or self._bytes + charge > self.max_pending_bytes):
                if identity in self._pending:
                    self._note_reserve("identity_pending")
                elif len(self._tickets) + len(self._pending) >= self.max_pending_keys:
                    self._note_reserve("capacity_keys")
                else:
                    self._note_reserve("capacity_bytes")
                return None
            # Pending descriptor/byte credits exist before network work, not
            # merely when the winning response is inserted into the ticket map.
            self._pending.add(identity)
            self._bytes += charge
        request_id = uuid.uuid4().hex
        method = "reserve_write" if writing else "reserve_read"
        started = time.monotonic()  # Conservative earliest possible lease start.
        leases = [None] * self.world_size
        accepted = False
        try:
            def params_for(rank):
                params = {"request_id": request_id, "namespace": namespace,
                          "key": key.hex(), "owner": owner}
                if writing:
                    params["size"] = int(sizes[rank])
                return params

            results = self.client.fanout(method, params_for)
            complete = len(results) == self.world_size
            rank_failures = 0
            # Never break at an unsuccessful rank: later successes own leases.
            for rank, result in enumerate(results[:self.world_size]):
                self._record_failure(result)
                lease = result.get("lease") if isinstance(result, dict) else None
                if not isinstance(lease, str) or not _LEASE.fullmatch(lease):
                    complete = False
                    rank_failures += 1
            if rank_failures:
                self._note_reserve(f"rank_lease_missing_{rank_failures}")
            for rank, result in enumerate(results[:self.world_size]):
                lease = result.get("lease") if isinstance(result, dict) else None
                if isinstance(lease, str) and _LEASE.fullmatch(lease):
                    leases[rank] = lease
            valid_until = started + self._ttl - self._margin
            with self._lock:
                if (complete and not self._closed and (namespace, key) not in self._invalid
                        and time.monotonic() < valid_until):
                    ticket = Ticket(uuid.uuid4().hex, namespace, key, tuple(leases),
                                    tuple(sizes), owner, writing)
                    self._tickets[ticket.token] = ticket
                    self._identities[identity] = ticket.token
                    self._valid[ticket.token] = valid_until
                    self._pending.remove(identity)
                    accepted = True
                    return ticket
                if not complete:
                    self._note_reserve("fanout_incomplete")
                elif self._closed:
                    self._note_reserve("closed")
                elif (namespace, key) in self._invalid:
                    self._note_reserve("veto_invalid")
                else:
                    self._note_reserve("window_expired")
            return None
        finally:
            if not accepted:
                try:
                    self._rollback(leases)
                    self._retry_cleanup()
                finally:
                    with self._lock:
                        self._pending.discard(identity)
                        self._bytes -= charge
                        self._clear_veto_if_drained(namespace, key)

    def reserve_load(self, namespace, key, owner):
        try:
            return self._reserve(namespace, key, owner,
                                 self.size_by_group[int.from_bytes(key[-4:], "big")], False)
        except Exception as error:
            self._record_failure(error)
            return None

    def reserve_store(self, namespace, key, owner, size_by_rank):
        try:
            return self._reserve(namespace, key, owner, tuple(size_by_rank), True)
        except Exception as error:
            self._record_failure(error)
            return None

    # -- lifecycle --------------------------------------------------------- #

    def _clear_veto_if_drained(self, namespace, key):
        pair = (namespace, key)
        if (pair in self._invalidate_acked
                and not any((ticket.namespace, ticket.key) == pair
                            for ticket in self._tickets.values())
                and not any(identity[:2] == pair for identity in self._pending)):
            self._invalid.discard(pair)
            self._invalidate_acked.discard(pair)

    def can_store(self):
        """Permanent capability snapshot, distinct from transient admission pressure."""
        return not (self._closed or self._cleanup_lost or getattr(self.client, "_closed", False))

    def _record_failure(self, error):
        if isinstance(error, _Permanent):
            self._closed = True

    def lease_deadline(self, ticket):
        """Non-I/O monotonic validity snapshot; never waits on a transport lock."""
        if (self._closed or ticket.token in self._retiring
                or (ticket.namespace, ticket.key) in self._invalid
                or self._tickets.get(ticket.token) != ticket):
            return None
        return self._valid.get(ticket.token)

    def release(self, ticket):
        return self._release(ticket)

    def _release(self, ticket, *, drop=True):
        """Retain identity and byte/key credits until real all-rank ACK."""
        try:
            with self._lock:
                actual = self._tickets.get(ticket.token)
                if actual is not None:
                    self._retiring.add(actual.token)
                    self._valid.pop(actual.token, None)
            direct_ack = self._rollback((actual or ticket).leases)
            acked = self._retry_cleanup() and direct_ack
            if acked and drop:
                with self._lock:
                    actual = self._tickets.pop(ticket.token, None)
                    if actual is not None:
                        identity = (actual.namespace, actual.key, actual.owner, actual.writing)
                        self._identities.pop(identity, None)
                        self._retiring.discard(actual.token)
                        self._bytes -= max(actual.sizes)
                        self._clear_veto_if_drained(actual.namespace, actual.key)
            return None if acked else False
        except Exception:
            return False

    def invalidate(self, namespace, key):
        """Invalidate on every rank; explicit False when any rank did not ack."""
        try:
            if not self._check_identity(namespace, key, ""):
                return False
            with self._lock:
                if ((namespace, key) not in self._invalid
                        and len(self._invalid) >= self.max_pending_keys):
                    self._closed = True
                    return False
                self._invalid.add((namespace, key))
            acked = True
            for rank in range(self.world_size):
                try:
                    result = self.client.call(rank, "invalidate",
                                              namespace=namespace, key=key.hex())
                    if result.get("invalidated") is not True:
                        acked = False
                        self._replay("invalidate", rank,
                                     namespace=namespace, key=key.hex())
                except Exception as error:
                    self._record_failure(error)
                    acked = False
                    self._replay("invalidate", rank,
                                 namespace=namespace, key=key.hex())
            acked = self._retry_cleanup() and acked
            if acked:
                with self._lock:
                    self._invalidate_acked.add((namespace, key))
                    self._clear_veto_if_drained(namespace, key)
            return None if acked else False
        except Exception:  # noqa: BLE001
            return False

    def renew(self, ticket):
        """Throttled all-rank renewal; cached success never outlives the lease.

        on_schedule_end hammers renew per ticket per step; a cached True is
        served inside (ttl - margin) anchored at the operation START, and any
        renewal whose fan-out latency consumed the window returns False rather
        than trusting a possibly-expired remote lease.
        """
        try:
            with self._lock:
                if (self._closed or self._tickets.get(ticket.token) != ticket
                        or ticket.token in self._retiring
                        or (ticket.namespace, ticket.key) in self._invalid):
                    return False
                if time.monotonic() < self._valid.get(ticket.token, 0.0):
                    return True
            started = time.monotonic()
            results = self.client.fanout("renew", lambda r: {"lease": ticket.leases[r]})
            renewed = (len(results) == self.world_size
                       and all(isinstance(x, dict) and x.get("renewed") is True
                               for x in results))
            if renewed:
                valid_until = started + self._ttl - self._margin
                with self._lock:
                    renewed = (not self._closed and ticket.token in self._tickets
                               and ticket.token not in self._retiring
                               and (ticket.namespace, ticket.key) not in self._invalid
                               and time.monotonic() < valid_until)
                    if renewed:
                        self._valid[ticket.token] = valid_until
            self._retry_cleanup()
            return renewed
        except Exception:  # noqa: BLE001
            return False

    def complete_store(self, ticket, success):
        """Success requires every rank to prove durability; else invalidate all.

        The native controller calls this only after worker drain. Any missing
        rank object, or success=False, invalidates the key on every rank (safe
        even for ranks that already fsync-committed) and releases the ticket.
        Returns None/True when cleanup fully acknowledged, explicit False when
        a rank cleanup remains in the bounded retry queue.
        """
        try:
            durable = bool(success)
            if durable:
                results = self.client.fanout(
                    "exists", lambda r: {"namespace": ticket.namespace,
                                         "key": ticket.key.hex()})
                durable = (len(results) == self.world_size
                           and all(isinstance(x, dict) and x.get("exists") is True
                                   for x in results))
            acked = True
            if not durable:
                acked = self.invalidate(ticket.namespace, ticket.key) is not False
            release_ack = self._release(ticket, drop=acked)
            if release_ack is False:
                acked = False
            return None if acked else False
        except Exception:  # noqa: BLE001
            return False

    def pending_retries(self):
        with self._lock:
            return len(self._retry)

    def shutdown(self):
        self._closed = True
        self.client.shutdown()


# --------------------------------------------------------------------------- #
# Startup census and factory
# --------------------------------------------------------------------------- #

def _profile_from_config(cfg) -> str:
    """Authenticated census profile: configured cache/runtime identity.

    cache_fingerprint is the storage.fingerprint container for model, draft,
    layout, tenant and raw hash-seed/policy identity, so binding it (plus the
    tenant namespace) binds the whole runtime profile. Overlong values are
    rejected, never silently truncated. Standalone use may leave both empty.
    """
    def field(name):
        value = cfg.get(name, "")
        if not isinstance(value, str) or len(value.encode()) > 1024:
            raise ValueError(f"{name} must be a string of at most 1024 bytes")
        return value
    algorithm = cfg.get("prefix_caching_hash_algo", "sha256")
    if algorithm != "sha256":
        raise ValueError("coordinator requires sha256 prefix hashing")
    return fingerprint(cache_fingerprint=field("cache_fingerprint"),
                       tenant_namespace=field("tenant_namespace"),
                       hash_seed=os.environ.get("PYTHONHASHSEED", ""),
                       hash_algorithm=algorithm)


def _census(client, world_size, *, startup_timeout, own_rank=None, own_row=None,
            expected_profile=None):
    """Bounded all-rank registration check with retry until the deadline.

    Verifies each rank's self-reported rank index and world size, one common
    fingerprint, group order and profile (compared both peer-to-peer and
    against this process's configured expected profile), and collects each
    rank's registered per-group bytes. With own_row given (worker role), the
    rank's registration must equal its actual canonical geometry bytes.
    """
    deadline = time.monotonic() + startup_timeout
    delay = 0.05
    while True:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("coordinator census deadline expired")
            configured_timeout = client.timeout
            try:
                client.timeout = min(configured_timeout, remaining)
                results = client.fanout("geometry", lambda r: {})
            finally:
                client.timeout = configured_timeout
            if len(results) != world_size:
                raise _Transient("incomplete rank census")
            infos = []
            for rank, result in enumerate(results):
                if isinstance(result, Exception):
                    raise _Transient(f"rank {rank}: {result}") from None
                infos.append(_validated_geometry(result, rank, world_size))
            base = infos[0]
            for rank, info in enumerate(infos[1:], start=1):
                if (info["fingerprint"] != base["fingerprint"]
                        or info["group_refs"] != base["group_refs"]
                        or info.get("profile", "") != base.get("profile", "")):
                    raise _Permanent(f"rank {rank} layout/profile identity differs from rank 0")
            if expected_profile is not None:
                for rank, info in enumerate(infos):
                    if info.get("profile", "") != expected_profile:
                        raise _Permanent(
                            f"rank {rank} profile differs from configured identity")
            if own_row is not None and own_rank is not None:
                if [int(n) for n in infos[own_rank]["group_bytes"]] != [int(n) for n in own_row]:
                    raise _Permanent("registered rank bytes differ from canonical geometry")
            return infos
        except _Permanent as e:
            # Configuration/identity disagreement never heals: fail closed now.
            raise RuntimeError(f"coordinator census rejected: {e}") from e
        except Exception as e:  # noqa: BLE001 - retry any transport failure
            if time.monotonic() >= deadline:
                raise RuntimeError(f"coordinator census failed: {e}") from e
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, 0.5)


def _sizes_from_census(infos):
    return tuple(tuple(int(info["group_bytes"][group]) for info in infos)
                 for group in range(len(infos[0]["group_bytes"])))


def _positive(cfg, key, default, *, top=None, integer=False):
    value = cfg.get(key, default)
    if integer:
        if type(value) is not int or value <= 0 or (top is not None and value > top):
            raise ValueError(f"{key} must be a positive bounded integer")
    else:
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 < value or (top is not None and value > top)):
            raise ValueError(f"{key} must be a positive bounded number")
    return value


@dataclass
class HTTPProvider(Provider):
    """Provider plus the transport lifecycle owned by this module.

    `close` (the Provider callback native owns) is idempotent and runs exactly
    the internal drain order: pooled client first (peer keepalive handler
    threads observe EOF), then the metadata server (acceptor off, in-flight
    requests drained), then — for a worker that created it — the shared
    DiskStore. Native calls it once after GPU/CPU/media drain, never reset.
    """
    endpoint: str = ""
    server: object = None
    client: object = None

    def __post_init__(self):
        self._owns_store = False
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        if self.close is None:
            self.close = self.shutdown

    def shutdown(self):
        """Idempotent clean stop, retryable while a drain is unproven.

        Order: pooled client first (peer keepalive handler threads observe
        EOF), then the metadata server — whose drain MUST prove before the
        shared DiskStore closes. If handlers are still active, shutdown()
        raises and the store stays open; retry after the stall clears.
        """
        if not self._close_lock.acquire(timeout=5.0):
            raise RuntimeError("provider close is still in progress")
        try:
            if self._closed.is_set():
                return
            if self.coordinator is not None:
                self.coordinator.shutdown()
            elif self.client is not None:
                self.client.shutdown()
            if self.server is not None:
                self.server.shutdown()  # Unproved drain leaves store untouched.
            store = self.store
            if (self._owns_store and store is not None and hasattr(store, "close")
                    and not getattr(store, "closed", True)):
                store.close()
            self._closed.set()
        finally:
            self._close_lock.release()


def factory(*, config=None, role="scheduler", rank=None, world_size=None, geometry=None):
    """Build the all-rank HTTP metadata provider for one process.

    config keys (externally supplied, generic): coordinator_endpoints (list of
    world_size "host:port"), coordinator_auth_token_file, disk_root (worker),
    max_pending_keys, plus bounded optional coordinator_* tuning documented in
    README-coordinator-http.md. role='worker' additionally requires rank and
    the actual canonical geometry; role='scheduler' never receives geometry and
    must not infer it.
    """
    cfg = dict(config or {})
    if role not in ("scheduler", "worker"):
        raise ValueError("role must be 'scheduler' or 'worker'")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    raw_endpoints = cfg.get("coordinator_endpoints")
    if (not isinstance(raw_endpoints, list) or len(raw_endpoints) != world_size
            or not all(isinstance(e, str) for e in raw_endpoints)):
        raise ValueError("coordinator_endpoints must list one endpoint per rank")
    endpoints = [parse_endpoint(e) for e in raw_endpoints]
    token = load_auth_token(cfg.get("coordinator_auth_token_file"))
    max_pending_keys = _positive(cfg, "max_pending_keys", DEFAULT_MAX_PENDING_KEYS, integer=True)
    max_pending_bytes = _positive(cfg, "max_pending_bytes", DEFAULT_MAX_PENDING_BYTES, integer=True)
    startup_timeout = _positive(cfg, "coordinator_startup_timeout", 60.0, top=3600.0)
    rpc_timeout = _positive(cfg, "coordinator_rpc_timeout", 10.0, top=300.0)
    server_threads = _positive(cfg, "coordinator_server_threads", 16, top=128, integer=True)
    max_body = _positive(cfg, "coordinator_max_body_bytes", 65536, top=_MAX_JSON_SIZE, integer=True)
    pool_size = _positive(cfg, "coordinator_client_pool", 4, top=16, integer=True)
    idem_entries = _positive(cfg, "coordinator_idempotency_entries", 32768,
                             top=65536, integer=True)
    renew_margin = cfg.get("coordinator_renew_margin")
    close_store = bool(cfg.get("coordinator_close_store_on_shutdown", True))

    if role == "worker":
        if geometry is None or type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("worker role requires rank and canonical geometry")
        root = cfg.get("disk_root")
        if not isinstance(root, str) or not root or not os.path.isabs(root):
            raise ValueError("worker role requires an absolute disk_root")
        limits_kwargs = cfg.get("coordinator_disk_limits") or {}
        if not isinstance(limits_kwargs, dict):
            raise ValueError("coordinator_disk_limits must be a Limits field mapping")
        limits = Limits(**limits_kwargs)
        profile = _profile_from_config(cfg)
        identity = geometry_identity(geometry)
        store = server = client = None
        try:
            store = DiskStore(Path(root) / f"rank-{rank}", limits,
                              journal_mode=cfg.get("sqlite_journal_mode", "delete"),
                              synchronous=cfg.get("sqlite_synchronous", "extra"))
            server = MetadataServer(store=store, identity=identity, rank=rank,
                                    world_size=world_size, host=endpoints[rank][0],
                                    port=endpoints[rank][1], token=token,
                                    threads=server_threads, max_body=max_body,
                                    rpc_timeout=rpc_timeout, idem_entries=idem_entries,
                                    profile=profile)
            server.start()
            # The endpoint serves its own registration immediately; peers may
            # still be binding, so the census retries within the bounded
            # startup deadline and fails closed on mismatch.
            client = RpcClient(endpoints, token, rpc_timeout=rpc_timeout,
                               pool_size=pool_size, max_body=max_body)
            # 0010: the rank-local leg dispatches in-process. The error map
            # mirrors the HTTP status mapping exactly so fail-closed
            # semantics are identical on both legs:
            #   ValueError -> 400 -> _Permanent (peer rejected)
            #   _RetryAgain -> 503 -> _Transient (peer unavailable)
            #   _BackendUnavailable -> 410 -> _Permanent (rank disabled)
            #   anything else -> 500 -> _Transient (execution failed)
            def _map_local_error(error):
                if isinstance(error, _BackendUnavailable):
                    return _Permanent("RPC peer rejected request")
                if isinstance(error, _RetryAgain):
                    return _Transient("RPC peer unavailable")
                if isinstance(error, ValueError):
                    return _Permanent("RPC peer rejected request")
                return _Transient("RPC execution failed")
            client.bind_local_dispatch(rank, server.rpc.dispatch,
                                       _map_local_error)
            infos = _census(client, world_size, startup_timeout=startup_timeout,
                            own_rank=rank, own_row=identity["group_bytes"],
                            expected_profile=profile)
            sizes = _sizes_from_census(infos)
            if tuple(row[rank] for row in sizes) != tuple(identity["group_bytes"]):
                raise RuntimeError("census row disagrees with canonical geometry")
        except BaseException:
            # Release everything this factory moment owns: a leaked client
            # keeps executor threads, a leaked server keeps sockets, and a
            # leaked store keeps the exclusive root lock.
            drained = True
            for transport in (client, server):
                if transport is not None:
                    try:
                        transport.shutdown()
                    except Exception:
                        drained = False
            if not drained:
                error = RuntimeError("startup metadata resources did not drain")
                # Retain ownership for diagnosis/manual retry; never close a
                # shared store while a metadata handler may still be using it.
                error.resources = (client, server, store)
                raise error from None
            if store is not None:
                store.close()
            raise
        provider = HTTPProvider(coordinator=None, store=store, size_by_group=sizes,
                                max_pending_keys=max_pending_keys, endpoint=server.endpoint,
                                server=server, client=client)
        provider.layout_fingerprint = layout_fingerprint(infos)
        # When configured, this provider owns the store's final close and the
        # Provider.close callback stops/drains the endpoint before store.close.
        provider._owns_store = close_store
        return provider

    # Scheduler: no geometry input is accepted; sizes come only from census.
    if rank is not None:
        raise ValueError("scheduler role must not receive a rank")
    if geometry is not None:
        raise ValueError("scheduler must not infer geometry; worker census is authoritative")
    profile = _profile_from_config(cfg)
    client = RpcClient(endpoints, token, rpc_timeout=rpc_timeout,
                       pool_size=pool_size, max_body=max_body)
    try:
        infos = _census(client, world_size, startup_timeout=startup_timeout,
                        expected_profile=profile)
        sizes = _sizes_from_census(infos)
        lease_seconds = min(float(info["lease_seconds"]) for info in infos)
        coordinator = RemoteCoordinator(client, sizes, max_pending_keys=max_pending_keys,
                                        max_pending_bytes=max_pending_bytes,
                                        lease_seconds=lease_seconds,
                                        renew_margin=renew_margin)
    except BaseException:
        client.shutdown()
        raise
    provider = HTTPProvider(coordinator=coordinator, store=None, size_by_group=sizes,
                            max_pending_keys=max_pending_keys, endpoint="",
                            server=None, client=client)
    provider.layout_fingerprint = layout_fingerprint(infos)
    return provider


__all__ = ["PROTOCOL", "HTTPProvider", "MetadataServer", "RemoteCoordinator",
           "RpcClient", "factory", "geometry_identity", "layout_fingerprint",
           "load_auth_token", "parse_endpoint", "_Transient", "_Permanent"]
