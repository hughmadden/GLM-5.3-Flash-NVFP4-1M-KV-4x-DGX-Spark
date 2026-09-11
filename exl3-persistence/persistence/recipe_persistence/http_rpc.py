# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hugh Madden and contributors
"""Bounded metadata HTTP RPC, using numeric endpoints and absolute deadlines.

Every admitted operation owns one submission credit until its callable and
watchdog actually finish (or until a queued item is removed before execution).
A bounded deque, rather than ThreadPoolExecutor's unbounded work queue, prevents
cancelled futures from accumulating behind blocked workers. No DNS is used.
"""
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
import http.client
import ipaddress
import json
import math
import socket
import threading
import time


class _Transient(Exception):
    """Retryable transport, deadline or admission failure; contains no secrets."""


class _Permanent(Exception):
    """Non-retryable request or peer rejection; contains no secrets."""


def _shutdown_socket(sock):
    if sock is not None:
        try:
            # close alone does not interrupt HTTPResponse's makefile references.
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            try:
                sock.close()
            except OSError:
                pass


def _discard(conn):
    _shutdown_socket(getattr(conn, "owned_socket", None))
    conn.close()


class _Operation:
    def __init__(self, rank, method, params_for, deadline):
        self.rank, self.method = rank, method
        self.params_for, self.deadline = params_for, deadline
        self.future = None
        self.timer = None
        self.socket = None
        self.cancelled = threading.Event()
        self.lock = threading.Lock()

    def remaining(self):
        left = self.deadline - time.monotonic()
        if self.cancelled.is_set() or left <= 0:
            raise _Transient("RPC deadline or cancellation")
        return left

    def attach(self, sock):
        with self.lock:
            if self.cancelled.is_set() or time.monotonic() >= self.deadline:
                _shutdown_socket(sock)
                raise _Transient("RPC deadline or cancellation")
            self.socket = sock

    def detach(self):
        with self.lock:
            self.socket = None

    def abort(self):
        with self.lock:
            self.cancelled.set()
            sock, self.socket = self.socket, None
        _shutdown_socket(sock)


class _HeaderReader:
    """Bound total status/header bytes, including interim HTTP responses."""
    def __init__(self, stream):
        self.stream = stream
        self.header_mode = True
        self.left = 16384

    def readline(self, limit=-1):
        if not self.header_mode:
            return self.stream.readline(limit)
        bound = self.left + 1
        line = self.stream.readline(bound if limit < 0 else min(limit, bound))
        self.left -= len(line)
        if self.left < 0:
            raise http.client.HTTPException("RPC headers exceed metadata bound")
        return line

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _BoundedResponse(http.client.HTTPResponse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp = _HeaderReader(self.fp)

    def begin(self):
        reader = self.fp
        try:
            return super().begin()
        finally:
            if reader is not None:
                reader.header_mode = False


class _NumericConnection(http.client.HTTPConnection):
    """HTTPConnection without socket.create_connection/getaddrinfo."""
    response_class = _BoundedResponse

    def __init__(self, endpoint, operation):
        host, port, family = endpoint
        super().__init__(host, port, timeout=operation.remaining())
        self.family = family
        self.operation = operation
        self.owned_socket = None

    def connect(self):
        operation = self.operation
        sock = socket.socket(self.family, socket.SOCK_STREAM)
        self.sock = self.owned_socket = sock
        operation.attach(sock)  # Own it BEFORE connect can block.
        sock.settimeout(operation.remaining())
        # HTTPConnection normally enables this; preserve it when replacing its
        # DNS-based connect path, or headers/body writes incur delayed-ACK stalls.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        address = ((self.host, self.port, 0, 0) if self.family == socket.AF_INET6
                   else (self.host, self.port))
        try:
            sock.connect(address)  # Validated numeric literal: no resolver call.
        except BaseException:
            _discard(self)
            raise


class RpcClient:
    """Compatible call/fanout API with bounded work and proved shutdown.

    ``timeout`` remains mutable and is sampled at admission. It bounds the
    ENTIRE operation, including queueing and the one permitted idempotent retry.
    Optional max_pending/max_active allow tighter embedding/test bounds. Errors
    are static: no endpoint, method, credential, body, key or token is included.
    """
    def __init__(self, endpoints, token, *, rpc_timeout=10.0, pool_size=4,
                 max_body=65536, max_pending=None, max_active=None):
        endpoints = tuple(endpoints)
        if not endpoints or len(endpoints) > 1024:
            raise ValueError("bounded nonempty endpoint census required")
        parsed = []
        for endpoint in endpoints:
            if not isinstance(endpoint, (tuple, list)) or len(endpoint) != 2:
                raise ValueError("endpoint must be a numeric host and port pair")
            host, port = endpoint
            if not isinstance(host, str) or "%" in host:
                raise ValueError("numeric IP literal required")
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                raise ValueError("numeric IP literal required") from None
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError("explicit valid endpoint port required")
            parsed.append((str(address), port, socket.AF_INET6 if address.version == 6 else socket.AF_INET))
        if (not isinstance(token, str) or not 1 <= len(token) <= 256
                or any(not 0x21 <= ord(c) <= 0x7e for c in token)):
            raise ValueError("printable ASCII auth token required")
        if type(pool_size) is not int or not 1 <= pool_size <= 128:
            raise ValueError("positive bounded connection pool required")
        if type(max_body) is not int or not 1 <= max_body <= 1024 * 1024:
            raise ValueError("positive bounded metadata body required")
        active = min(len(endpoints), 8) if max_active is None else max_active
        pending = active * pool_size if max_pending is None else max_pending
        if type(active) is not int or not 1 <= active <= 128:
            raise ValueError("positive bounded active RPC count required")
        if type(pending) is not int or not active <= pending <= 16384:
            raise ValueError("submission bound must cover active RPC count")
        self.endpoints = tuple((host, port) for host, port, _ in parsed)
        self._numeric = tuple(parsed)
        self._token = token.encode("ascii")
        self.timeout = rpc_timeout
        self._pool_size, self._max_body = pool_size, max_body
        self.max_active, self.max_pending = active, pending
        self._condition = threading.Condition(threading.RLock())
        self._shutdown_lock = threading.Lock()
        self._pool = {}
        self._queue = deque()
        self._operations = set()
        self._active = 0
        self._closed = False
        self._fully_stopped = False
        self._threads = []
        try:
            for _ in range(active):
                thread = threading.Thread(target=self._worker, name="kv-coord-cli", daemon=True)
                thread.start()
                self._threads.append(thread)
        except BaseException:
            self.shutdown()
            raise

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError("finite positive RPC timeout required") from None
        if not math.isfinite(value) or not 0 < value <= 3600:
            raise ValueError("finite positive RPC timeout required")
        self._timeout = value

    def _admit(self, rank, method, params_for, deadline):
        if type(rank) is not int or not 0 <= rank < len(self.endpoints):
            raise _Permanent("unknown RPC rank")
        if not isinstance(method, str) or not method or len(method) > 128:
            raise _Permanent("invalid RPC method")
        with self._condition:
            if self._closed:
                raise _Transient("RPC client is shut down")
            if len(self._operations) >= self.max_pending:
                raise _Transient("RPC submission capacity exhausted")
            operation = _Operation(rank, method, params_for, deadline)
            self._operations.add(operation)
            return operation

    def _complete(self, operation):
        with self._condition:
            self._operations.discard(operation)
            self._condition.notify_all()

    def _worker(self):
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue:
                    return
                operation = self._queue.popleft()
                future = operation.future
                if not future.set_running_or_notify_cancel():
                    self._complete(operation)
                    continue
            try:
                result = self._execute(operation)
            except BaseException:
                # Never attach a traceback with credentials/params to a Future.
                future.set_exception(_Transient("RPC execution failed"))
            else:
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
            finally:
                self._complete(operation)

    def _take(self, operation):
        endpoint = self.endpoints[operation.rank]
        with self._condition:
            stack = self._pool.get(endpoint)
            conn = stack.pop() if stack else None
        if conn is None:
            return _NumericConnection(self._numeric[operation.rank], operation)
        conn.operation = operation
        try:
            operation.attach(conn.owned_socket)
            conn.sock.settimeout(operation.remaining())
        except BaseException:
            _discard(conn)
            raise
        return conn

    def _give(self, rank, conn):
        conn.operation = None
        with self._condition:
            stack = self._pool.setdefault(self.endpoints[rank], [])
            if not self._closed and len(stack) < self._pool_size:
                stack.append(conn)
                return
        _discard(conn)

    def _call_once(self, operation, body):
        conn = self._take(operation)
        try:
            operation.remaining()
            conn.request("POST", "/", body=body, headers={
                "Authorization": "Bearer " + self._token.decode("ascii"),
                "Content-Type": "application/json", "Accept": "application/json"})
            with conn.getresponse() as response:
                status = response.status
                if status != 200:
                    # No reason to consume an error body; discard the owned
                    # connection, and never retry a known permanent rejection.
                    if 400 <= status < 500 and status not in (408, 429):
                        raise _Permanent("RPC peer rejected request")
                    raise _Transient("RPC peer unavailable")
                lengths = response.headers.get_all("Content-Length", [])
                encodings = response.headers.get_all("Transfer-Encoding", [])
                if (len(lengths) > 1 or len(encodings) > 1 or (lengths and encodings)
                        or (lengths and (not lengths[0].isascii() or not lengths[0].isdecimal()))
                        or (encodings and encodings[0].lower() != "chunked")):
                    raise _Transient("RPC response framing malformed")
                # Reject declared oversize before reading. Unknown/chunked length
                # is bounded by read(max_body+1); either case discards the socket.
                if response.length is not None and response.length > self._max_body:
                    raise _Transient("RPC response exceeds metadata bound")
                data = response.read(self._max_body + 1)
                if len(data) > self._max_body:
                    raise _Transient("RPC response exceeds metadata bound")
                if response.length is not None and response.length != 0:
                    raise _Transient("RPC response framing incomplete")
                if not response.isclosed():
                    # EOF-delimited responses need an explicit EOF probe; a
                    # short read alone does not prove framing completion.
                    if response.read(1) or not response.isclosed():
                        raise _Transient("RPC response framing incomplete")
                reusable = not response.will_close and response.isclosed()
            operation.remaining()
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeError):
                raise _Transient("RPC response is not JSON") from None
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                raise _Transient("RPC response envelope malformed")
            if not reusable:
                _discard(conn)
                operation.detach()
                conn = None
            return payload, conn
        except BaseException:
            _discard(conn)
            operation.detach()
            raise

    def _execute(self, operation):
        reusable = None
        result = None
        with self._condition:
            while self._active >= self.max_active and not self._closed:
                try:
                    self._condition.wait(operation.remaining())
                except _Transient:
                    return _Transient("RPC deadline or cancellation")
            if self._closed or operation.cancelled.is_set():
                return _Transient("RPC client is shut down")
            self._active += 1
        try:
            operation.remaining()
            timer = threading.Timer(operation.remaining(), operation.abort)
            timer.daemon = True
            operation.timer = timer
            timer.start()
            try:
                params = operation.params_for()
                body = json.dumps({"method": operation.method, "params": params},
                                  separators=(",", ":"), allow_nan=False).encode()
            except Exception:
                raise _Permanent("RPC parameters are not serializable") from None
            if len(body) > self._max_body:
                raise _Permanent("RPC request exceeds metadata bound")
            for attempt in (0, 1):
                operation.remaining()
                try:
                    result, reusable = self._call_once(operation, body)
                    break
                except _Permanent:
                    result = _Permanent("RPC peer rejected request")
                    break
                except (OSError, http.client.HTTPException, _Transient):
                    result = _Transient("RPC transport failed")
                    if attempt == 0:
                        operation.cancelled.wait(min(0.02, operation.remaining()))
        except _Permanent:
            result = _Permanent("RPC request rejected")
        except Exception:
            result = _Transient("RPC deadline or transport failure")
        finally:
            timer = operation.timer
            if timer is not None:
                timer.cancel()
                # Credits cover watchdogs too: cancel alone leaves Timer threads
                # alive and can otherwise accumulate at high request rates.
                if timer.ident is not None:
                    timer.join()
            expired = operation.cancelled.is_set() or time.monotonic() >= operation.deadline
            operation.detach()
            if reusable is not None:
                if expired:
                    _discard(reusable)
                else:
                    self._give(operation.rank, reusable)
            if expired:
                result = _Transient("RPC deadline or cancellation")
            with self._condition:
                self._active -= 1
                self._condition.notify_all()
        return result if result is not None else _Transient("RPC execution failed")

    def call(self, rank, method, **params):
        operation = self._admit(rank, method, lambda: params, time.monotonic() + self.timeout)
        try:
            result = self._execute(operation)
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            self._complete(operation)

    def _cancel(self, operation):
        operation.abort()
        with self._condition:
            future = operation.future
            if future is not None and future.cancel():
                # Remove the actual queued work, not only its Future state.
                # This is what permits safe credit release before workers drain.
                try:
                    self._queue.remove(operation)
                except ValueError:
                    pass
                else:
                    self._complete(operation)
            self._condition.notify_all()

    def fanout(self, method, params_for):
        """Return one result-or-Exception per rank within one absolute deadline."""
        deadline = time.monotonic() + self.timeout
        operations = []
        for rank in range(len(self.endpoints)):
            try:
                operation = self._admit(rank, method, lambda rank=rank: params_for(rank), deadline)
                with self._condition:
                    operation.future = Future()
                    if self._closed:
                        operation.future.cancel()
                        self._complete(operation)
                    else:
                        self._queue.append(operation)
                        self._condition.notify()
                operations.append(operation)
            except Exception as error:
                operations.append(error)
        results = []
        for operation in operations:
            if isinstance(operation, Exception):
                results.append(operation)
                continue
            try:
                results.append(operation.future.result(timeout=max(0, deadline - time.monotonic())))
            except FutureTimeout:
                self._cancel(operation)
                results.append(_Transient("RPC fanout deadline"))
            except Exception as error:
                results.append(error if isinstance(error, (_Transient, _Permanent)) else
                               _Transient("RPC fanout cancelled"))
        return results

    def shutdown(self):
        """Abort owned sockets and prove all work/watchdogs/workers drained.

        A blocked non-socket callback cannot be killed safely. On deadline this
        raises and retains ownership; subsequent shutdown may retry after it
        unblocks. Never closes a caller-owned DiskStore or reports a false drain.
        """
        # The deadline includes waiting behind another shutdown caller.
        deadline = time.monotonic() + self.timeout
        if not self._shutdown_lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise RuntimeError("RPC client shutdown is still in progress")
        try:
            if self._fully_stopped:
                return
            with self._condition:
                self._closed = True
                operations = tuple(self._operations)
                pools, self._pool = self._pool, {}
                self._condition.notify_all()
            for operation in operations:
                self._cancel(operation)
            for stack in pools.values():
                for conn in stack:
                    _discard(conn)
            with self._condition:
                while self._operations:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise RuntimeError("RPC client has not drained")
                    self._condition.wait(left)
            for thread in self._threads:
                thread.join(max(0, deadline - time.monotonic()))
            if any(thread.is_alive() for thread in self._threads):
                raise RuntimeError("RPC client workers have not drained")
            self._fully_stopped = True
        finally:
            self._shutdown_lock.release()
