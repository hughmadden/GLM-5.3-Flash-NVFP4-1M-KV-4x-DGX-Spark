# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hugh Madden and contributors
"""Explicit all-rank metadata contract; no KV payload travels via this API.

A provider supplies bounded authenticated transport and deadlines. It MUST return
None on timeout/unknown rank, rolling back all partial reservations. The scheduler
may not infer remote visibility from its own directory. Metadata only imports.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol
import threading
import time
import uuid


@dataclass(frozen=True)
class Ticket:
    token: str
    namespace: str
    key: bytes
    leases: tuple[str, ...]  # Ordered by global rank, opaque rank-local tokens.
    sizes: tuple[int, ...]  # Exact payload bytes by global rank.
    owner: str = ""
    writing: bool = False


class CoordinatorProtocol(Protocol):
    def reserve_load(self, namespace: str, key: bytes, owner: str) -> Ticket | None: ...
    def reserve_store(self, namespace: str, key: bytes, owner: str,
                      size_by_rank: tuple[int, ...]) -> Ticket | None: ...
    # None is successful acknowledgement; explicit False or exception means
    # cleanup/quarantine was not confirmed and native admission must close.
    def complete_store(self, ticket: Ticket, success: bool) -> bool | None: ...
    def release(self, ticket: Ticket) -> bool | None: ...
    def invalidate(self, namespace: str, key: bytes) -> bool | None: ...
    def renew(self, ticket: Ticket) -> bool: ...
    # Non-I/O monotonic snapshot. None means no current proof; async native
    # callers must defer/renew instead of trusting an indefinitely cached True.
    def lease_deadline(self, ticket: Ticket) -> float | None: ...
    # Non-I/O permanent capability; temporary quota/admission pressure stays True.
    def can_store(self) -> bool: ...


@dataclass
class Provider:
    """Factory return value (duck typing also accepted).

    factory(config=extra_config, role='scheduler'|'worker', rank=None|int,
            world_size=int, geometry=None|Geometry) -> Provider

    Workers register actual canonical geometry BEFORE the scheduler is ready.
    size_by_group[group][rank] must come from those registrations, not scheduler
    UniformType representative specs. Factory MUST check common fingerprint,
    group order and rank census, with bounded waiting/fail-closed startup.
    """
    coordinator: CoordinatorProtocol | None
    store: object | None
    size_by_group: tuple[tuple[int, ...], ...]
    max_pending_keys: int
    # Native owns this callback AFTER GPU, CPU and media drain, never reset.
    # A successful ACK is consumed once; a failed attempt retains retry ownership.
    # A worker provider closes its metadata server before its shared DiskStore.
    close: Callable[[], None] | None = None
    # SHA256 of the stable ordered all-rank (padded_pages, group_refs) census.
    # Empty preserves construction compatibility but native admission rejects it.
    layout_fingerprint: str = ""


class LocalCoordinator:
    """In-process reference/provider conformance fixture, NOT a remote TP bridge.

    Every ticket reserves all-rank file leases plus bounded logical queue byte/key
    credits. Physical staging rows are a separate shared worker ring. Large jobs
    stream through that ring; queue credits are not payload allocations.
    """
    def __init__(self, stores, size_by_group, *, max_pending_keys=32768,
                 max_pending_bytes=64_000_000_000):
        self.stores = tuple(stores)
        self.size_by_group = tuple(tuple(s) for s in size_by_group)
        if not self.stores or max_pending_keys <= 0 or max_pending_bytes <= 0:
            raise ValueError("positive coordinator bounds required")
        if any(len(s) != len(self.stores) or min(s) <= 0 for s in self.size_by_group):
            raise ValueError("geometry must contain every rank")
        self.max_pending_keys = max_pending_keys
        self.max_pending_bytes = max_pending_bytes
        self._tickets = {}
        self._identities = {}  # Bounded by active ticket count, not disk capacity.
        self._bytes = 0
        self._valid = {}
        self._retiring = set()
        self._invalid = set()
        self._invalidate_acked = set()
        self._closed = False
        self._ttl = min(float(store.limits.lease_seconds) for store in self.stores)
        self._margin = min(30.0, self._ttl * 0.1)
        self._lock = threading.RLock()

    def _reserve(self, namespace, key, owner, sizes, writing):
        with self._lock:
            # Idempotency is bounded by active tickets, never by cache capacity.
            identity = (namespace,key,owner,writing)
            if self._closed or (namespace, key) in self._invalid:
                return None
            token = self._identities.get(identity)
            if token is not None:
                if token in self._retiring:
                    return None
                ticket = self._tickets[token]
                if self.renew(ticket):
                    return ticket
                self.release(ticket)
                return None
            charge = max(sizes)
            if len(self._tickets) >= self.max_pending_keys or self._bytes + charge > self.max_pending_bytes:
                return None
            leases = []
            valid_until = time.monotonic() + self._ttl - self._margin
            try:
                for store, size in zip(self.stores,sizes):
                    token = (store.reserve_write(namespace,key,size,owner) if writing
                             else store.reserve_read(namespace,key,owner))
                    if token is None:
                        raise RuntimeError("rank did not reserve")
                    leases.append(token)
                if time.monotonic() >= valid_until:
                    raise RuntimeError("rank reservation validity window expired")
            except Exception:
                for store, token in zip(self.stores,leases):
                    try:
                        store.release(token)
                    except Exception:
                        pass  # Unknown peer remains leased until bounded expiry.
                return None
            ticket = Ticket(uuid.uuid4().hex,namespace,key,tuple(leases),tuple(sizes),owner,writing)
            self._tickets[ticket.token] = ticket
            self._valid[ticket.token] = valid_until
            self._identities[identity] = ticket.token
            self._bytes += charge
            return ticket

    def reserve_load(self, namespace, key, owner):
        group = int.from_bytes(key[-4:], "big")
        if not 0 <= group < len(self.size_by_group):
            return None
        return self._reserve(namespace,key,owner,self.size_by_group[group],False)

    def reserve_store(self, namespace, key, owner, size_by_rank):
        group = int.from_bytes(key[-4:], "big")
        if not 0 <= group < len(self.size_by_group) or tuple(size_by_rank) != self.size_by_group[group]:
            return None
        return self._reserve(namespace,key,owner,tuple(size_by_rank),True)

    def complete_store(self, ticket, success):
        # Native calls only after every rank has drained. Any failure vetoes all
        # shards, including those already fsync-committed. Unknown durability is
        # not acknowledgement: reject globally, but still release after drain.
        accepted = True
        try:
            durable = success and all(s.exists(ticket.namespace,ticket.key) for s in self.stores)
        except Exception:
            durable = False
        if not durable:
            accepted = self.invalidate(ticket.namespace,ticket.key) is not False
        released = self._release(ticket, drop=accepted)
        return None if accepted and released is not False else False

    def _clear_veto_if_drained(self, namespace, key):
        pair = (namespace, key)
        if (pair in self._invalidate_acked
                and not any((t.namespace, t.key) == pair for t in self._tickets.values())):
            self._invalid.discard(pair)
            self._invalidate_acked.discard(pair)

    def can_store(self):
        return not self._closed and all(not s.failed and not s.closed for s in self.stores)

    def lease_deadline(self, ticket):
        # No lock shared with disk operations: this is an in-memory snapshot.
        if (self._closed or ticket.token in self._retiring
                or (ticket.namespace, ticket.key) in self._invalid
                or self._tickets.get(ticket.token) != ticket
                or any(store.failed or store.closed for store in self.stores)):
            return None
        return self._valid.get(ticket.token)

    def release(self, ticket):
        return self._release(ticket)

    def _release(self, ticket, *, drop=True):
        with self._lock:
            actual = self._tickets.get(ticket.token)
            accepted = True
            if actual:
                self._retiring.add(actual.token)
                self._valid.pop(actual.token, None)
                for store, lease in zip(self.stores,actual.leases):
                    try:
                        if store.release(lease) is False:
                            accepted = False
                    except Exception:
                        accepted = False
                if accepted and drop:
                    self._tickets.pop(actual.token, None)
                    self._identities.pop((actual.namespace,actual.key,actual.owner,actual.writing),None)
                    self._retiring.discard(actual.token)
                    self._bytes -= max(actual.sizes)
                    self._clear_veto_if_drained(actual.namespace, actual.key)
            return None if accepted else False

    def invalidate(self, namespace, key):
        with self._lock:
            if ((namespace, key) not in self._invalid
                    and len(self._invalid) >= self.max_pending_keys):
                self._closed = True
                return False
            self._invalid.add((namespace, key))
        accepted = True
        for store in self.stores:
            try:
                if store.invalidate(namespace,key) is False:
                    accepted = False
            except Exception:
                accepted = False
        if accepted:
            with self._lock:
                self._invalidate_acked.add((namespace, key))
                self._clear_veto_if_drained(namespace, key)
        return None if accepted else False

    def renew(self, ticket):
        with self._lock:
            if (self._closed or self._tickets.get(ticket.token) != ticket
                    or ticket.token in self._retiring
                    or (ticket.namespace, ticket.key) in self._invalid):
                return False
            valid_until = time.monotonic() + self._ttl - self._margin
            renewed = all(s.renew(t) for s,t in zip(self.stores,ticket.leases))
            if renewed and time.monotonic() < valid_until:
                self._valid[ticket.token] = valid_until
                return True
            self._valid.pop(ticket.token, None)
            return False
