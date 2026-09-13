# SPDX-License-Identifier: Apache-2.0
# Architecture derived from nodelocal_disk_spec.py (revision 6729433),
# Apache-2.0; native offloading API copyright vLLM contributors.
"""External offloading spec for the pinned vLLM native API.

Importing this module is metadata-only. Accessing NodeLocalDiskOffloadingSpec
loads optional vLLM classes; allocating worker handlers is the only CUDA path.

Required extra config: disk_root (or RECIPE_PERSISTENCE_ROOT), cache_fingerprint,
tenant_namespace, trusted_single_tenant=True, coordinator_module_path,
coordinator_factory, staging_bytes, staging_rows, max_pending_keys. PYTHONHASHSEED must already be fixed at process
startup and prefix_caching_hash_algo must be sha256. The provider
factory contract is defined in coordinator.Provider. It must perform an explicit
bounded all-rank canonical registration handshake and publish a SHA256
layout_fingerprint over the ordered all-rank padded pages and group refs. There
is no local-only TP fallback or payload sizing from representative layer specs.
lookup_keys_per_step is REQUIRED (the old default of 8 hung every
restore past ~900 tokens -- FIX-LOOKUP-BUDGET-20260911); fresh reservations
yield via None at the step budget, round-robin by request owner. Cached
scans are not charged. min_disk_lookup_tokens (default 4096) is the R3
read gate: smaller requests never consult the disk tier.
Production metadata cleanup uses metadata_workers=2, metadata_max_submitted=8,
and metadata_shutdown_timeout=10 seconds. Tickets stay owned until background
callbacks return an actual ACK and the scheduler polls it. The coordinator must
provide lease_deadline(ticket), a non-I/O conservative monotonic snapshot.
"""
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import math
import threading
import time
import hashlib
import importlib
import json
import logging
import os

from .geometry import Geometry, key_group
from .handlers import DiskLoadStoreSpec as _DiskMetadata, DiskTransferPump, _CloseOnce

_LOG = logging.getLogger(__name__)

# Bounded, static-code path tracing for the load path. The disk tier stores but
# has never been observed to serve, and the distance between "the lookup found
# the chunk" and "the transfer ran" spans the scheduler, this manager and the
# worker-side pump. These one-shot notices say which of those boundaries is
# actually reached. Codes are static; no key, path, token or exception message
# is ever logged.
def _trace_enabled() -> bool:
    """Opt-in path tracing.

    Diagnostics must be visible in an engine container, whose effective level
    drops this module's INFO records, without polluting the WARNING stream that
    the test suite asserts on and that is reserved for genuine degradation.

    So the level is the switch: with PERSIST_DEBUG_TRACE set the traces are
    WARNING (visible at the engine's default level); unset they are INFO
    (invisible in production, harmless in tests). Default off.
    """
    return os.environ.get("PERSIST_DEBUG_TRACE", "").strip().lower() not in (
        "", "0", "false", "no")


def _enable_offload_scheduler_debug() -> None:
    """Raise vLLM's offloading scheduler to DEBUG when tracing is on.

    The connector's ``_lookup_complete_chunks`` can return 0 even when the
    backend found chunks, and which of its several ``return 0`` paths fired is
    only visible in its own debug output -- specifically the line
    "Request %s hit %s offloaded tokens after %s GPU hit tokens". Enabling that
    one logger avoids whole-process DEBUG logging, which on this engine is
    unusably verbose.
    """
    if not _trace_enabled():
        return
    for name in (
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading",
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)


def _emit_trace(message: str, *args) -> None:
    if _trace_enabled():
        _LOG.warning(message, *args)
    else:
        _LOG.info(message, *args)


class _TraceState:
    """Per-manager one-shot path tracing.

    Deliberately *not* module-global. A process-global "have I logged this
    code yet" flag means only the first manager ever logs, which makes the
    traces sparse in exactly the long-lived engine runs they exist for, and it
    leaks state between instances so one manager's activity suppresses
    another's diagnostics. Both are bugs in a diagnostic.

    The log level is INFO, not WARNING: the WARNING stream is reserved for the
    static degradation reasons the test suite asserts on.
    """

    __slots__ = ("seen", "counts")

    def __init__(self):
        self.seen: set[str] = set()
        self.counts: dict[str, int] = {}

    def note(self, code: str, detail: str = "") -> None:
        self.counts[code] = self.counts.get(code, 0) + 1
        if code not in self.seen:
            self.seen.add(code)
            _emit_trace("Persistence trace: %s%s", code,
                        (" " + detail) if detail else "")

    def counts_snapshot(self) -> dict[str, int]:
        return dict(self.counts)

# Disk-tier lookup observability. The names are upstream vLLM's documented
# tiering metric names (vllm/v1/kv_offload/tiering/base.py
# TieringOffloadingMetrics); they are reused verbatim because our single
# node-local disk tier is exactly a secondary offload tier, and
# OffloadingSpec.build_metric_definitions is the sanctioned out-of-tree
# registration point (OffloadPromMetrics.__init__ merges it and asserts every
# emitted stats name is present). DISK_TIER_LABEL names this component's tier.
# vLLM-free strings so importing recipe_persistence stays torch/vLLM-free.
DISK_TIER_LABEL = "disk"
TIER_CHUNK_QUERIES_METRIC = "vllm:kv_offload_tiering_chunk_queries"
TIER_CHUNK_HITS_METRIC = "vllm:kv_offload_tiering_chunk_hits"


class _Manager:
    # Wall-clock lifetime of one owner's turn at the lookup gate. Short enough
    # that a deferred request is re-admitted promptly, long enough that one
    # owner still gets its whole per-step key budget in a single scan.
    _LOOKUP_TURN_SECONDS = 0.05
    # How many times one lookup may answer RETRY for a reservation whose
    # lease cannot be vouched for. Past this (or immediately, when the lease
    # is provably expired and renew cannot resurrect it) the reservation is
    # released and the chunk answers a clean MISS: vLLM's scan terminates and
    # the caller acts on the prefix it did find. An unbounded RETRY is a
    # livelock -- prepare_load is never reached, so the reservation is never
    # released, so every later lookup defers again (BUG-LOAD-PATH-STALE-LEASE).
    _STALE_LEASE_MAX_RETRY = 4
    # How many times an un-ACKed background cleanup (release/complete_store/
    # invalidate) is resubmitted before the sustained rejection becomes
    # evidence and the tier fails closed. A single transient RPC failure must
    # not disable the tier; an unbounded retry must not hide a dead one.
    _METADATA_MAX_ATTEMPTS = 6
    # Wall-clock bound for processing metadata ACKs outside on_schedule_end.
    # The scheduler hook normally drives _poll_metadata, but a request deferred
    # BY this manager leaves the engine with no batch -- and then the hook
    # never runs. That is the lookup-gate deadlock one level down: the gate was
    # made wall-clock driven, the ACK pump was not. Every manager entry point
    # the scheduler can reach while idle polls when due, so renewal/release ACK
    # processing, has_pending_work() and the invalidation safe-reset cannot be
    # held hostage by a stalled engine. Single-writer by construction: only
    # the scheduler thread calls manager methods.
    _METADATA_POLL_INTERVAL = 0.05

    def __init__(self, coordinator, namespace, size_by_group, max_pending_keys,
                 disk_spec_cls, output_cls, context_cls, close_provider=None,
                 lookup_keys_per_step=None, metadata_workers=0,
                 metadata_max_submitted=8, metadata_shutdown_timeout=10.0):
        self.coordinator = coordinator
        self.namespace = namespace
        self.sizes = tuple(tuple(s) for s in size_by_group)
        self.capacity = int(max_pending_keys)
        if self.capacity <= 0 or not self.sizes:
            raise ValueError("positive bounded manager geometry required")
        self.disk_cls, self.output_cls, self.context_cls = disk_spec_cls, output_cls, context_cls
        self.loads = {}  # (owner,key) -> ticket; lookup promises are real leases
        self.stores = {}
        self.prepared = set()
        self.closed = False
        self.degraded_reason = None
        self._close_provider = _CloseOnce(close_provider)
        self._shutdown_complete = False
        if (lookup_keys_per_step is not None
                and (type(lookup_keys_per_step) is not int or lookup_keys_per_step <= 0)):
            raise ValueError("lookup_keys_per_step must be a positive integer or None")
        self.lookup_keys_per_step = lookup_keys_per_step
        self._lookup_keys_left = lookup_keys_per_step
        self._lookup_turn_started = None
        self._trace_state = _TraceState()
        self._outcome = {"hit": 0, "miss": 0, "retry": 0}
        self._lookup_owners = {}  # Ordered, bounded by manager capacity.
        self._lookup_turn_owner = None
        self._store_keys_left = lookup_keys_per_step
        if type(metadata_workers) is not int or not 0 <= metadata_workers <= 32:
            raise ValueError("metadata_workers must be bounded")
        if (type(metadata_max_submitted) is not int
                or not max(1, metadata_workers) <= metadata_max_submitted <= 4096):
            raise ValueError("metadata submission window must cover workers")
        if (type(metadata_shutdown_timeout) not in (int, float)
                or not math.isfinite(metadata_shutdown_timeout)
                or not 0 < metadata_shutdown_timeout <= 300):
            raise ValueError("metadata shutdown timeout must be finite and bounded")
        self._async_metadata = bool(metadata_workers)
        if self._async_metadata and (not callable(getattr(coordinator, "lease_deadline", None))
                                     or not callable(getattr(coordinator, "renew", None))):
            raise ValueError("background metadata requires non-I/O lease_deadline and renew")
        self._metadata_limit = min(metadata_max_submitted, max(1, 2 * self.capacity))
        self._metadata_timeout = float(metadata_shutdown_timeout)
        self._metadata_futures = {}
        self._metadata_retry = {}
        self._metadata_attempts = {}
        self._metadata_turn = 0
        self._metadata_stopping = False
        self._retiring_loads = {}  # FIFO intent order; new completions cannot jump older work.
        self._retiring_stores = {}
        self._invalidations = {}  # key -> acknowledged; veto persists until safe reset.
        self._namespace_quarantined = False  # Unrecordable failure: explicit fail-stop.
        self._renew_requested = {}
        self._stale_retries = {}  # identity -> consecutive stale-lease RETRYs; bounded.
        self._reset_veto_requested = False
        self._metadata_executor = (ThreadPoolExecutor(max_workers=metadata_workers,
                                  thread_name_prefix="kv-metadata") if metadata_workers else None)
        self._last_metadata_poll = time.monotonic()
        # Disk-tier lookup observability (monotonic totals, read+reset by
        # take_tier_stats). Never read by admission, eviction or the lookup
        # decision itself: counters only.
        self.tier_chunk_queries = 0
        self.tier_chunk_hits = 0
        # Why-did-the-lookup-answer-miss reasons (diagnostics only, static
        # codes). The outcome sampler says WHAT was answered; this says WHY,
        # separating vetoes, capacity, gate and coordinator-level refusals.
        self._lookup_reasons = {}

    @staticmethod
    def _invoke_metadata(callback, args, renewal):
        # Worker threads return ACK only; they never mutate manager maps.
        try:
            result = callback(*args)
            return result is True if renewal else (result is None or result is True)
        except Exception:
            return False

    def _lease_deadline(self, ticket):
        """Non-I/O monotonic deadline snapshot; None when unvouchable."""
        try:
            deadline = self.coordinator.lease_deadline(ticket)  # Contract: no I/O.
        except Exception:
            self._degrade("coordinator_callback_failed")
            return None
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            return None
        return deadline

    def _lease_valid(self, ticket):
        deadline = self._lease_deadline(ticket)
        if deadline is None:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= min(30.0, self._metadata_timeout):
            self._renew_requested[ticket.token] = ticket
        return remaining > 0

    def _release_stale_reservation(self, identity, ticket):
        """Drop a reservation whose lease cannot be vouched for.

        Ownership is never discarded unacknowledged: async mode retires the
        identity and lets the background worker release the lease; sync mode
        releases inline. The data stays on disk -- a later lookup (or the
        same scan, via the "reserve" outcome) re-reserves it.
        """
        self._stale_retries.pop(identity, None)
        self._renew_requested.pop(ticket.token, None)
        if self._async_metadata:
            self._retiring_loads.setdefault(identity, None)
            self._dispatch_metadata()
        elif identity not in self.prepared:
            self._notify("release", ticket)
            self.loads.pop(identity, None)

    def _stale_lease_lookup(self, identity, ticket):
        """Decide the honest answer for an un-vouchable reservation.

        Returns "reserve" when the lease is provably expired (a past deadline
        renew cannot resurrect -- storage renew requires unexpired leases):
        drop it and let the caller make a fresh reservation in the same scan.
        Returns "miss" when bounded retries are exhausted or the manager is
        closed: drop it and terminate the scan cleanly. Returns "retry" only
        for genuine renewal uncertainty, where another pass can plausibly
        help, and only while the bound lasts.
        """
        deadline = self._lease_deadline(ticket)
        expired = deadline is not None and time.monotonic() >= deadline
        retries = self._stale_retries.get(identity, 0) + 1
        self._stale_retries[identity] = retries
        if expired:
            self._note_lookup_reason("stale_expired")
            self._trace("lookup_stale_lease_released")
            self._release_stale_reservation(identity, ticket)
            return "miss" if self.closed else "reserve"
        if self.closed or retries > self._STALE_LEASE_MAX_RETRY:
            self._note_lookup_reason("stale_bounded")
            self._trace("lookup_stale_lease_released")
            self._release_stale_reservation(identity, ticket)
            return "miss"
        self._renew_requested[ticket.token] = ticket
        self._dispatch_metadata()
        return "retry"

    def _metadata_pending(self):
        return bool(self._namespace_quarantined or self._metadata_futures or self._retiring_loads or self._retiring_stores
                    or self._renew_requested or any(not ack for ack in self._invalidations.values()))

    def _poll_metadata_if_due(self):
        if (self._async_metadata and not self._shutdown_complete
                and time.monotonic() - self._last_metadata_poll >= self._METADATA_POLL_INTERVAL):
            self._poll_metadata()

    def _poll_metadata(self):
        self._last_metadata_poll = time.monotonic()
        for future, work in list(self._metadata_futures.items()):
            if not future.done():
                continue
            kind, identity, ticket = work
            try:
                acknowledged = future.result() is True
            except Exception:
                acknowledged = False
            del self._metadata_futures[future]
            work_id = (kind, identity)
            if not acknowledged:
                if kind == "renew":
                    # A lost renewal is expected under renewal lag -- the lease
                    # may simply have expired before the renew landed, and
                    # renew of an expired lease cannot succeed. Retire the
                    # affected reservations so lookups re-reserve; this is not
                    # evidence against data safety and must not disable the
                    # tier. (Sustained failure surfaces through the releases
                    # that follow, which are retried and escalate below.)
                    self._renew_requested.pop(identity, None)
                    for owner_key, held in self.loads.items():
                        if held.token == identity:
                            self._retiring_loads.setdefault(owner_key, None)
                    continue
                attempts = self._metadata_attempts.get(work_id, 0) + 1
                self._metadata_attempts[work_id] = attempts
                if attempts > self._METADATA_MAX_ATTEMPTS:
                    # Sustained rejection: bounded retries did not absorb it.
                    # Now it is evidence of coordinator trouble and the tier
                    # fails closed -- one transient RPC must not kill it, an
                    # unbounded retry must not hide a dead one.
                    self._degrade("coordinator_callback_rejected")
                else:
                    self._metadata_retry[work_id] = time.monotonic() + 0.05
                continue
            self._metadata_attempts.pop(work_id, None)
            self._metadata_retry.pop(work_id, None)
            if kind == "load":
                if self.loads.get(identity) is ticket:
                    self.loads.pop(identity)
                self._retiring_loads.pop(identity, None)
                self._renew_requested.pop(ticket.token, None)
            elif kind == "store":
                if self.stores.get(identity) is ticket:
                    self.stores.pop(identity)
                self._retiring_stores.pop(identity, None)
                self._renew_requested.pop(ticket.token, None)
            elif kind == "invalidate":
                self._invalidations[identity] = True
            else:
                self._renew_requested.pop(identity, None)
        if (self._reset_veto_requested and not self.loads and not self.stores
                and not self._metadata_pending()):
            self._invalidations.clear()
            self._reset_veto_requested = False

    def _dispatch_metadata(self):
        if (not self._async_metadata or self._metadata_stopping or self._shutdown_complete or self._namespace_quarantined
                or len(self._metadata_futures) >= self._metadata_limit):
            return
        # Intent storage is the bounded ownership maps, not an unbounded queue.
        # Submitted Futures (including ThreadPoolExecutor's queued work) have a
        # separate small cap and stay counted until real completion is polled.
        active = {(kind, identity) for kind, identity, _ in self._metadata_futures.values()}
        tokens = {ticket.token for _, _, ticket in self._metadata_futures.values() if ticket is not None}
        now = time.monotonic()
        submission_failed = False
        def submit(kind, identity, ticket, method, args):
            nonlocal submission_failed
            work_id = (kind, identity)
            if (len(self._metadata_futures) >= self._metadata_limit or work_id in active
                    or self._metadata_retry.get(work_id, 0) > now
                    or (ticket is not None and ticket.token in tokens)):
                return
            try:
                future = self._metadata_executor.submit(self._invoke_metadata,
                            getattr(self.coordinator, method), args, kind == "renew")
            except Exception:
                self._degrade("coordinator_callback_failed")
                submission_failed = True
                return  # Ownership/intents remain in the maps for retry.
            self._metadata_futures[future] = (kind, identity, ticket)
            active.add(work_id)
            if ticket is not None:
                tokens.add(ticket.token)
        def jobs(kind):
            if kind == 0:
                for key, ack in self._invalidations.items():
                    if not ack:
                        yield "invalidate", key, None, "invalidate", (self.namespace, key)
            elif kind == 1:
                for identity in self._retiring_loads:
                    ticket = self.loads.get(identity)
                    if ticket is not None and identity not in self.prepared:
                        yield "load", identity, ticket, "release", (ticket,)
            elif kind == 2:
                for identity, success in self._retiring_stores.items():
                    ticket = self.stores.get(identity)
                    if ticket is not None:
                        yield "store", identity, ticket, "complete_store", (ticket, success)
            else:
                eligible = {t.token for i, t in self.loads.items() if i not in self._retiring_loads}
                eligible.update(t.token for i, t in self.stores.items() if i not in self._retiring_stores)
                for token, ticket in self._renew_requested.items():
                    if token in eligible:
                        yield "renew", token, ticket, "renew", (ticket,)
        # Quarantine gets the next available slot. Rotate the remaining classes
        # so continuous releases cannot starve store finalization or renewal.
        for job in jobs(0):
            submit(*job)
            if submission_failed or len(self._metadata_futures) >= self._metadata_limit:
                return
        turn = self._metadata_turn
        self._metadata_turn = (turn + 1) % 3
        for offset in range(3):
            for job in jobs(1 + (turn + offset) % 3):
                submit(*job)
                if submission_failed or len(self._metadata_futures) >= self._metadata_limit:
                    return

    def _metadata_drain(self, timeout):
        if self._namespace_quarantined:
            raise RuntimeError("namespace quarantine overflow requires external recovery")
        deadline = time.monotonic() + timeout
        while self._metadata_pending():
            self._poll_metadata()
            self._dispatch_metadata()
            if not self._metadata_pending():
                break
            left = deadline - time.monotonic()
            if left <= 0:
                raise RuntimeError("metadata cleanup has not drained")
            futures = tuple(self._metadata_futures)
            if futures:
                wait(futures, timeout=min(left, 0.05), return_when=FIRST_COMPLETED)
            else:
                threading.Event().wait(min(left, 0.01))
        self._poll_metadata()

    def _stop_metadata_workers(self, deadline):
        if not self._metadata_stopping:
            # Do not enqueue another sentinel on every provider-close retry.
            self._metadata_executor.shutdown(wait=False)
            self._metadata_stopping = True
        # ThreadPoolExecutor exposes no public timed shutdown. Its owned thread
        # set is used narrowly here to prove exit without an unbounded join.
        threads = tuple(self._metadata_executor._threads)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("metadata workers have not drained")

    def _forget_lookup_owner(self, owner):
        self._lookup_owners.pop(owner, None)
        # Keep the current turn marker until schedule end: completing/missing
        # midway through a step must not grant a second owner's fresh quantum.

    def _trace(self, code: str, detail: str = "") -> None:
        """Record a reached load-path boundary (per-instance, diagnostics only)."""
        self._trace_state.note(code, detail)

    def trace_counts(self) -> dict[str, int]:
        return self._trace_state.counts_snapshot()

    def _rotate_lookup_turn(self):
        owner = self._lookup_turn_owner
        if owner in self._lookup_owners:
            self._lookup_owners.pop(owner)
            self._lookup_owners[owner] = None
        self._lookup_turn_owner = next(iter(self._lookup_owners), None)
        self._lookup_keys_left = self.lookup_keys_per_step
        self._lookup_turn_started = time.monotonic()

    def _clear_lookup_gate(self):
        self._lookup_owners.clear()
        self._lookup_turn_owner = None
        self._lookup_keys_left = self.lookup_keys_per_step
        self._lookup_turn_started = None

    def _admit_lookup(self, owner):
        if self.lookup_keys_per_step is None:
            return True  # Legacy direct metadata fixtures may remain unbudgeted.
        # Liveness guard. The gate is normally advanced by on_schedule_end, but
        # that hook only runs when the scheduler makes progress -- and a request
        # deferred BY this gate produces no batch, so it never gets a turn,
        # never advances the hook, and waits forever. Observed as a hard
        # deadlock: Running=0 with the request parked in deferred. Advancing the
        # turn on wall-clock as well keeps the same one-owner-at-a-time
        # fairness while making it impossible for the gate to strand a request.
        now = time.monotonic()
        if (self._lookup_turn_started is None
                or now - self._lookup_turn_started > self._LOOKUP_TURN_SECONDS):
            self._rotate_lookup_turn()
        if owner not in self._lookup_owners:
            if len(self._lookup_owners) >= self.capacity:
                return False  # Bounded clean miss instead of untracked waiting.
            self._lookup_owners[owner] = None
        if self._lookup_turn_owner is None:
            self._lookup_turn_owner = next(iter(self._lookup_owners))
        if owner != self._lookup_turn_owner or self._lookup_keys_left <= 0:
            return None
        self._lookup_keys_left -= 1  # Only a new reserve_load attempt is charged.
        return True

    def _degrade(self, code):
        # Static bounded codes only: exception messages, paths, keys and tokens
        # must never reach logs, including errors received from remote providers.
        if code not in {"coordinator_callback_failed", "coordinator_callback_rejected",
                        "lease_renewal_lost", "provider_close_failed"}:
            code = "coordinator_callback_failed"
        self.closed = True
        self._clear_lookup_gate()  # Disabled lookups must not keep an idle engine spinning.
        if self.degraded_reason is None:
            self.degraded_reason = code
            _LOG.warning("Persistence disabled: %s", code)

    def _identity(self, key, ctx):
        key_group(key, len(self.sizes))
        if not isinstance(ctx.req_id, str) or not ctx.req_id:
            raise ValueError("nonempty request owner required")
        return ctx.req_id, key

    def _valid_ticket(self, ticket, key):
        group = key_group(key, len(self.sizes))
        try:
            return (ticket.namespace == self.namespace and ticket.key == key
                    and isinstance(ticket.token, str) and bool(ticket.token)
                    and tuple(ticket.sizes) == self.sizes[group]
                    and len(ticket.leases) == len(self.sizes[group])
                    and all(isinstance(t, str) and t for t in ticket.leases))
        except (AttributeError, TypeError):
            return False

    def _notify(self, method, *args):
        """Transport failure disables new promises, never kills the scheduler.

        Callers may discard ownership only after the core has drained all ranks.
        Unreachable peers reclaim leaked leases using their bounded expiry policy.
        """
        try:
            result = getattr(self.coordinator, method)(*args)
            if method in {"reserve_load", "reserve_store"} and result is None:
                self.can_store()  # Non-I/O terminal state; pressure remains retryable.
            if method in {"invalidate", "complete_store", "release"} and result is False:
                self._degrade("coordinator_callback_rejected")
            elif method == "renew" and not result:
                self._degrade("lease_renewal_lost")
            return result
        except Exception:
            self._degrade("coordinator_callback_failed")
            return False

    def _count_outcome(self, result) -> None:
        """Bounded outcome sampling for the load-path investigation.

        vLLM's scan breaks on the first MISS and discards everything on a RETRY
        without breaking, so which of those the backend returns for the *first*
        chunk decides whether prepare_load is ever reached. A few samples of the
        running totals make that visible from the engine log.
        """
        kind = "hit" if result is True else ("retry" if result is None else "miss")
        self._outcome[kind] += 1
        if self._outcome[kind] <= 3:
            _emit_trace("lookup outcome %s (#%d): hit=%d miss=%d retry=%d",
                        kind, self._outcome[kind], self._outcome["hit"],
                        self._outcome["miss"], self._outcome["retry"])

    def _note_lookup_reason(self, code):
        """Bounded why-was-it-a-miss sampling (diagnostics only, static codes)."""
        reasons = self._lookup_reasons
        reasons[code] = reasons.get(code, 0) + 1
        if reasons[code] <= 3:
            _emit_trace("lookup reason %s (#%d): %s",
                        code, reasons[code], sorted(reasons.items()))

    def lookup(self, key, req_context) -> bool | None:
        self._poll_metadata_if_due()
        result = self._lookup_impl(key, req_context)
        self._count_outcome(result)
        return result

    def _lookup_impl(self, key, req_context) -> bool | None:
        if not self.can_store():
            return False
        identity = self._identity(key, req_context)
        if key in self._invalidations:
            self._note_lookup_reason("veto_invalidated")
            return False
        if identity in self._retiring_loads:
            self._note_lookup_reason("veto_retiring")
            return False
        if identity in self.loads:
            self._trace("lookup_cached_reservation")
            ticket = self.loads[identity]
            if self._async_metadata:
                if self._lease_valid(ticket):
                    return True
                # RETRY on an already-reserved chunk whose lease we cannot
                # vouch for is bounded: see _stale_lease_lookup. An expired
                # lease is released and re-reserved in the same scan; genuine
                # renewal uncertainty may RETRY only _STALE_LEASE_MAX_RETRY
                # times before it too becomes a clean miss.
                self._trace("lookup_cached_lease_stale")
                stale = self._stale_lease_lookup(identity, ticket)
                if stale == "retry":
                    return None if not self.closed else False
                if stale == "miss":
                    return False
                # "reserve": the dead lease is being released; fall through
                # and make a fresh reservation for this chunk now. The data
                # is still on disk -- only the lease died.
            else:
                if getattr(self.coordinator, "renew", None) is not None:
                    if not self._notify("renew", ticket):
                        if identity not in self.prepared:
                            self._notify("release", ticket)
                            self.loads.pop(identity)
                        return False
                return True
        if len(self.loads) + len(self.stores) + len(self._invalidations) >= self.capacity:
            self._forget_lookup_owner(identity[0])
            self._note_lookup_reason("manager_capacity")
            return False  # clean miss, not unbounded scheduler retry
        admitted = self._admit_lookup(identity[0])
        if admitted is not True:
            self._trace("lookup_budget_retry" if admitted is None else "lookup_capacity_miss")
            return admitted
        ticket = self._notify("reserve_load", self.namespace, key, identity[0])
        # Observability only: one fresh reserve_load consultation is one
        # disk-tier chunk query, and a structurally valid all-rank read lease is
        # the disk tier holding the chunk. Counted here (not in prepare_store's
        # durability probe) so the pair reports prefix-lookup behaviour, and
        # never consulted by any decision below.
        self.tier_chunk_queries += 1
        if not ticket:
            self._forget_lookup_owner(identity[0])
            self._note_lookup_reason("reserve_none")
            return False
        if not self._valid_ticket(ticket, key):
            self._notify("release", ticket)
            self._forget_lookup_owner(identity[0])
            self._note_lookup_reason("reserve_invalid_ticket")
            return False
        self.tier_chunk_hits += 1
        self._trace("lookup_disk_hit")
        self.loads[identity] = ticket
        if self._async_metadata and not self._lease_valid(ticket):
            self._trace("lookup_reserved_lease_stale")
            stale = self._stale_lease_lookup(identity, ticket)
            if stale == "retry":
                return None if not self.closed else False
            return False  # expired/bounded-out: clean miss, scan terminates
        return True

    def take_tier_stats(self):
        """Return and reset (queries, hits) for the node-local disk tier.

        vLLM-free so the CPU suite can assert the counters directly; the native
        subclass wraps the pair in OffloadingConnectorStats for /metrics.
        """
        stats = (self.tier_chunk_queries, self.tier_chunk_hits)
        self.tier_chunk_queries = 0
        self.tier_chunk_hits = 0
        return stats

    def prepare_load(self, keys: Collection[bytes], req_context):
        self._trace("prepare_load")
        keys = tuple(keys)
        identities = [self._identity(k, req_context) for k in keys]
        if len(set(keys)) != len(keys) or any(i not in self.loads for i in identities):
            raise ValueError("prepare_load requires unique previously reserved lookup hits")
        # A retiring marker with a live loads entry means an old dead lease is
        # still being released while its replacement reservation (made by the
        # stale-lease fall-through) is already vouched: that is safe to
        # prepare. Only a marker with no live ticket, or a key veto, blocks.
        if any(i in self._retiring_loads and i not in self.loads for i in identities):
            raise ValueError("cannot prepare retiring load tickets")
        if any(i[1] in self._invalidations for i in identities):
            raise ValueError("cannot prepare invalidated load tickets")
        if any(i in self.prepared for i in identities):
            raise ValueError("load already in flight for this request and key")
        self.prepared.update(identities)
        self._forget_lookup_owner(req_context.req_id)
        return self.disk_cls(keys, tuple(tuple(self.loads[i].leases) for i in identities),
                             self.namespace, req_context.req_id)

    def complete_load(self, keys, req_context):
        self._trace("complete_load")
        self._poll_metadata_if_due()
        # Native/core calls this only after all rank-local operations drained.
        for key in keys:
            identity = self._identity(key, req_context)
            ticket = self.loads.get(identity)
            if ticket is not None:
                if self._async_metadata:
                    self._retiring_loads.setdefault(identity, None)
                    self._renew_requested.pop(ticket.token, None)
                else:
                    self._notify("release", ticket)
                    self.loads.pop(identity)
            self._stale_retries.pop(identity, None)
            self.prepared.discard(identity)
        self._dispatch_metadata()

    def on_load_failure(self, keys, req_context):
        self._trace("on_load_failure")
        self._poll_metadata_if_due()
        # Core calls AFTER complete_load. Namespace is immutable for the manager,
        # so key+namespace remain available without retaining unbounded tombstones.
        failed_keys = set()
        for key in keys:
            self._identity(key, req_context)
            if self._async_metadata:
                if key not in self._invalidations and len(self._invalidations) >= self.capacity:
                    self._degrade("coordinator_callback_rejected")
                    acknowledged = next((k for k, ack in self._invalidations.items() if ack), None)
                    if acknowledged is not None:
                        # Global closed state is permanent: it now covers the
                        # evicted, already-ACKed per-key veto. Never evict work.
                        self._invalidations.pop(acknowledged)
                    else:
                        # The new individual key cannot be tracked within the
                        # bound. Retain a global fail-stop, NOT a false claim of
                        # per-key cleanup ownership or successful invalidation.
                        self._namespace_quarantined = True
                        raise RuntimeError("namespace quarantine overflow requires external recovery")
                self._invalidations[key] = False
                failed_keys.add(key)
                # Conservative per-key veto survives invalidation ACK and all
                # stale readers; only a safe reset clears the quarantine.
            else:
                self._notify("invalidate", self.namespace, key)
        if self._async_metadata:
            # One ownership pass, not one full-map pass per failed key.
            for identity, ticket in self.loads.items():
                if identity[1] in failed_keys:
                    self._retiring_loads.setdefault(identity, None)
                    self._renew_requested.pop(ticket.token, None)
                    self._forget_lookup_owner(identity[0])
        self._dispatch_metadata()

    def prepare_store(self, keys, req_context):
        self._poll_metadata_if_due()
        if not self.can_store():
            return None
        selected, skipped, tickets = [], [], []
        seen = set()
        for key in keys:
            if self.closed or len(seen) >= self.capacity:
                break
            identity = self._identity(key, req_context)
            if key in seen or identity in self.stores:
                continue
            seen.add(key)
            if key in self._invalidations:
                continue
            if self._store_keys_left is not None:
                if self._store_keys_left <= 0:
                    self._note_lookup_reason("store_break_budget")
                    break
                self._store_keys_left -= 1
            if len(self.loads) + len(self.stores) + len(self._invalidations) >= self.capacity:
                self._note_lookup_reason("store_break_capacity")
                break
            # None from reserve_store cannot distinguish pressure from existing
            # durable data. A successful all-rank read lease is the only evidence
            # strong enough to advance the core's persistent store cursor.
            existing = self.loads.get(identity)
            durable = existing or self._notify("reserve_load", self.namespace, key, identity[0])
            if durable:
                valid = self._valid_ticket(durable, key)
                if existing is not None:
                    # Do not call an expired cached promise a durable skip, and
                    # never release a lease still backing an active load.
                    if self._async_metadata:
                        valid = valid and identity not in self._retiring_loads and self._lease_valid(durable)
                        if not valid:
                            self._renew_requested[durable.token] = durable
                    else:
                        valid = (valid and getattr(self.coordinator, "renew", None) is not None
                                 and bool(self._notify("renew", durable)))
                else:
                    if self._async_metadata:
                        self.loads[identity] = durable
                        self._retiring_loads.setdefault(identity, None)
                    else:
                        self._notify("release", durable)
                if valid and not self.closed:
                    skipped.append(key)
                    continue
            if self.closed:
                self._note_lookup_reason("store_break_closed")
                break
            ticket = self._notify("reserve_store", self.namespace, key, identity[0],
                                  self.sizes[key_group(key, len(self.sizes))])
            if not ticket:
                self._note_lookup_reason("store_reserve_declined")
                continue
            if not self._valid_ticket(ticket, key):
                self._notify("release", ticket)
                self._note_lookup_reason("store_reserve_invalid")
                continue
            self.stores[identity] = ticket
            selected.append(key)
            tickets.append(tuple(ticket.leases))
        self._dispatch_metadata()
        if not selected and not skipped:
            self._note_lookup_reason("store_empty_output")
            return None
        # Batch-shape probe (diagnostics only): how many keys the core offered
        # vs how many this manager admitted/skipped, and which cache groups
        # the offered keys belong to (small ints only, no key material).
        # Bounded to a small batch of one-shot notes. 24 covers a full
        # request's per-group store rotation plus its finish-drain passes,
        # which is exactly where the load-path investigation needs
        # visibility; it is still a fixed bound, not a log stream.
        self._store_batch_notes = getattr(self, "_store_batch_notes", 0)
        if self._store_batch_notes < 24:
            self._store_batch_notes += 1
            groups = sorted({key_group(k, len(self.sizes)) for k in tuple(keys)})
            _emit_trace("store batch #%d: offered=%d selected=%d skipped=%d groups=%s",
                        self._store_batch_notes, len(tuple(keys)),
                        len(selected), len(skipped), groups)
        # skipped_keys is a required companion-core extension, not an upstream
        # field at 83252ea89 either. Unpatched core is not a safe supported
        # configuration; series commit 0001 (drained-outcome API) on the fork adds it.
        return self.output_cls(keys_to_store=selected,
                               store_spec=self.disk_cls(tuple(selected), tuple(tickets),
                                                        self.namespace, req_context.req_id),
                               evicted_keys=[], skipped_keys=skipped)

    def can_store(self):
        """Permanent capability only; pressure or per-step deferral is not disablement."""
        if not self.closed:
            try:
                capability = getattr(self.coordinator, "can_store", None)
                if capability is not None and capability() is False:
                    self._degrade("coordinator_callback_failed")
            except Exception:
                self._degrade("coordinator_callback_failed")
        return not self.closed

    def complete_store(self, keys, req_context, success=True):
        self._poll_metadata_if_due()
        for key in keys:
            identity = self._identity(key, req_context)
            ticket = self.stores.get(identity)
            if ticket is not None:
                if self._async_metadata:
                    self._retiring_stores[identity] = self._retiring_stores.get(identity, True) and bool(success)
                    self._renew_requested.pop(ticket.token, None)
                else:
                    if self._notify("complete_store", ticket, bool(success)) is False:
                        self._notify("release", ticket)
                    self.stores.pop(identity)
        self._dispatch_metadata()

    def on_new_request(self, req_context):
        return self.context_cls()

    def on_request_finished(self, req_context):
        self._poll_metadata_if_due()
        owner = req_context.req_id
        if (any(i[0] == owner for i in self.prepared)
                or any(i[0] == owner and i not in self._retiring_stores for i in self.stores)):
            raise RuntimeError("request cleanup requires all-rank transfer drain")
        self._forget_lookup_owner(owner)
        for identity, ticket in list(self.loads.items()):
            if identity[0] == owner:
                if self._async_metadata:
                    self._retiring_loads.setdefault(identity, None)
                    self._renew_requested.pop(ticket.token, None)
                else:
                    self._notify("release", ticket)
                    self.loads.pop(identity)
                self._stale_retries.pop(identity, None)
        self._dispatch_metadata()

    def touch(self, keys, req_context):
        return None

    def take_events(self):
        return ()

    def has_pending_work(self):
        self._poll_metadata_if_due()
        return bool(self.prepared or self.stores or self._metadata_pending()
                    or (not self.closed and self._lookup_owners))

    def on_schedule_end(self):
        # Providers must throttle/cache or batch wire renewals; this hook runs
        # every scheduler step. Expired queued leases cause failed transfers,
        # never substituted bytes. Active rows remain held until core drain.
        if self._async_metadata:
            self._poll_metadata()
            for identity, ticket in (*self.loads.items(), *self.stores.items()):
                if identity in self._retiring_loads or identity in self._retiring_stores:
                    continue
                if not self._lease_valid(ticket):
                    self._renew_requested[ticket.token] = ticket
            self._dispatch_metadata()
        elif getattr(self.coordinator, "renew", None) is not None:
            for ticket in (*self.loads.values(), *self.stores.values()):
                self._notify("renew", ticket)
        self._store_keys_left = self.lookup_keys_per_step
        if self.lookup_keys_per_step is not None:
            self._rotate_lookup_turn()

    def reset_cache(self):
        self._poll_metadata_if_due()
        if self._namespace_quarantined:
            raise RuntimeError("namespace quarantine overflow requires external recovery")
        if self.prepared or any(i not in self._retiring_stores for i in self.stores):
            raise RuntimeError("cannot reset persistence while transfers are active")
        if self._async_metadata:
            self._retiring_loads.update(self.loads)
            self._renew_requested.clear()
            self._reset_veto_requested = True
            self._dispatch_metadata()
        else:
            for ticket in list(self.loads.values()):
                self._notify("release", ticket)
            self.loads.clear()
        self._clear_lookup_gate()
        # Async reset retains tickets and quarantine until actual cleanup ACKs.

    def shutdown(self):
        if self._shutdown_complete:
            return
        # This refuses active prepared loads/stores. Their provider must remain
        # alive until the core reports all-rank drain, even during degradation.
        deadline = time.monotonic() + self._metadata_timeout
        self.reset_cache()
        self.closed = True
        if self._async_metadata:
            self._metadata_drain(max(0.0, deadline - time.monotonic()))
            self._stop_metadata_workers(deadline)
        try:
            self._close_provider()
        except Exception:
            self._degrade("provider_close_failed")
            raise RuntimeError("persistence provider shutdown has not drained") from None
        self._shutdown_complete = True


def _prefix_hash_algorithm():
    """Read the pinned prefix-cache hash algorithm from the live vLLM config.

    OffloadingConfig (vLLM a9531edfa6, #48150) no longer carries cache_config,
    so the algorithm is reached through the ambient config instead of the spec
    input. It is part of the namespace fingerprint, so an absent ambient config
    is a hard refusal, never the "sha256" library default: silently assuming it
    would bind objects to a hash policy nobody proved.
    """
    from vllm import config as vllm_config_module

    ambient = getattr(vllm_config_module, "get_current_vllm_config_or_none", None)
    current = ambient() if ambient is not None else vllm_config_module.get_current_vllm_config()
    cache_config = getattr(current, "cache_config", None) if current is not None else None
    if cache_config is None:
        raise ValueError("persistence requires an active vLLM cache configuration")
    return getattr(cache_config, "prefix_caching_hash_algo", None)


def _load_native():
    """Build native subclasses lazily; keep module import CPU-only and optional."""
    from vllm.v1.kv_offload.base import (
        GPULoadStoreSpec, LoadStoreSpec, LookupResult, OffloadingCounterMetadata,
        OffloadingManager, OffloadingSpec, OffloadingWorker, PrepareStoreOutput,
        RequestOffloadingContext, TransferResult,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )

    class NativeDiskLoadStoreSpec(_DiskMetadata, LoadStoreSpec):
        pass

    class NodeLocalDiskManager(_Manager, OffloadingManager):
        """Adapts the vLLM-free _Manager to the current OffloadingManager ABC.

        _Manager keeps its tri-state lookup and no-argument step hook so the CPU
        suite needs neither vLLM nor torch; the two upstream signature changes
        are absorbed here.
        """

        def lookup(self, key, req_context):
            # lookup returns LookupResult since bb61177e49 (#46363).
            # True  -> HIT:   readable now, lease proven valid.
            # None  -> RETRY: location uncertain while a renewal is queued.
            #                 Not HIT_PENDING: that would count as a hit and
            #                 admit a prefix we cannot yet prove we own.
            # False -> MISS:  invalidated, retired, disabled or over capacity.
            result = _Manager.lookup(self, key, req_context)
            if result is True:
                return LookupResult.HIT
            if result is None:
                return LookupResult.RETRY
            return LookupResult.MISS

        def on_schedule_end(self, context=None):
            # ScheduleEndContext arrived in 0fc2512094 (#46450). Our leases and
            # per-step quanta are global to the step, not per new/preempted
            # request, so the context carries nothing we consume.
            return _Manager.on_schedule_end(self)

        def get_stats(self):
            """Publish the disk-tier chunk query/hit deltas for /metrics.

            OffloadingConnectorScheduler.get_stats() calls this once per stats
            interval and the payload is fed to OffloadPromMetrics, which
            requires every emitted name to be registered by
            NodeLocalDiskOffloadingSpec.build_metric_definitions. Deltas only:
            the vLLM-free counters are reset here so intervals cannot
            double-count. None when idle so no empty series is emitted.
            """
            queries, hits = _Manager.take_tier_stats(self)
            if not queries and not hits:
                return None
            stats = OffloadingConnectorStats()
            if queries:
                stats.increase_counter(TIER_CHUNK_QUERIES_METRIC, queries,
                                       (DISK_TIER_LABEL,))
            if hits:
                stats.increase_counter(TIER_CHUNK_HITS_METRIC, hits,
                                       (DISK_TIER_LABEL,))
            return stats

    class DiskOffloadingWorker(DiskTransferPump, OffloadingWorker):
        pass

    class NodeLocalDiskOffloadingSpec(OffloadingSpec):
        @classmethod
        def build_metric_definitions(cls, extra_config):
            """Register the disk-tier lookup counters with OffloadPromMetrics.

            OffloadingSpec.build_metric_definitions is the sanctioned
            out-of-tree extension point: OffloadPromMetrics.__init__ merges
            this mapping over the built-in connector definitions, and observe()
            asserts every emitted stats name is present, so a counter emitted
            without this registration would trip that assert. Names are the
            upstream tiering chunk query/hit counters, labeled by tier.
            """
            metrics = super().build_metric_definitions(extra_config)
            metrics[TIER_CHUNK_QUERIES_METRIC] = OffloadingCounterMetadata(
                documentation=("Number of chunk lookup queries sent to the "
                               "node-local disk tier, labeled by tier."),
                labelnames=("tier",),
            )
            metrics[TIER_CHUNK_HITS_METRIC] = OffloadingCounterMetadata(
                documentation=("Number of chunk lookup hits in the "
                               "node-local disk tier, labeled by tier."),
                labelnames=("tier",),
            )
            return metrics

        def __init__(self, config):
            super().__init__(config)
            _enable_offload_scheduler_debug()
            if self.blocks_per_chunk != 1:
                raise ValueError("disk persistence supports blocks_per_chunk=1 only")
            # The canonical host layout (2c4d348848, #48408) makes one rank the
            # writer of each shared canonical page via CanonicalPageMapping.
            # This lane writes a private per-rank object per key and never reads
            # that mapping, so accepting the flag would drop bytes on the floor.
            if getattr(config, "canonical_layout", False):
                raise ValueError("canonical host layout is unsupported: this lane "
                                 "persists per-rank pages, not one shared canonical page")
            # replicated_layout would make only rank 0 a store writer
            # (offloading/worker.py _is_store_writer); every rank owns its own
            # disk here, so all ranks must write.
            if self.replicated_layout:
                raise ValueError("replicated host layout is unsupported: every rank "
                                 "owns and writes its own node-local objects")
            extra = dict(self.extra_config)
            seed = os.environ.get("PYTHONHASHSEED", "")
            if not seed.isascii() or not seed.isdecimal() or not 0 <= int(seed) <= 4294967295:
                raise ValueError("fixed PYTHONHASHSEED must be set before starting Python")
            # init_none_hash hashes the raw environment string, including
            # leading zeroes; numeric normalization would lose that identity.
            self.hash_seed = seed
            self.hash_algorithm = _prefix_hash_algorithm()
            if self.hash_algorithm != "sha256":
                raise ValueError("persistence requires the pinned sha256 prefix hash algorithm")
            root = extra.get("disk_root") or os.environ.get("RECIPE_PERSISTENCE_ROOT")
            if not isinstance(root, str) or not root or not os.path.isabs(root):
                raise ValueError("explicit absolute disk_root or RECIPE_PERSISTENCE_ROOT required")
            extra["disk_root"] = root
            if extra.get("trusted_single_tenant") is not True:
                raise ValueError("only explicit trusted_single_tenant mode is supported")
            for field in ("tenant_namespace", "cache_fingerprint", "coordinator_module_path",
                          "coordinator_factory"):
                if not isinstance(extra.get(field), str) or not extra[field]:
                    raise ValueError(f"explicit {field} required; no local-only fallback")
            extra.setdefault("lookup_keys_per_step", 8)
            extra.setdefault("metadata_workers", 2)
            extra.setdefault("metadata_max_submitted", 8)
            extra.setdefault("metadata_shutdown_timeout", 10.0)
            for field in ("staging_bytes", "staging_rows", "max_pending_keys", "lookup_keys_per_step",
                          "metadata_workers", "metadata_max_submitted"):
                if type(extra.get(field)) is not int or extra[field] <= 0:
                    raise ValueError(f"positive integer {field} required")
            # R3 read gate (DESIGN-EVICT-ONLY-SPIKE-20260912): requests
            # smaller than this skip the disk lookup entirely -- a small
            # RAM-miss recomputes cheaper than the scan+read would cost.
            # The gate is by REQUEST size (prompt tokens), never by miss
            # size; the write path stays size-agnostic.
            min_lookup = extra.get("min_disk_lookup_tokens", 4096)
            if (type(min_lookup) is not int
                    or not 0 <= min_lookup <= 10_000_000):
                raise ValueError("min_disk_lookup_tokens must be a bounded non-negative integer")
            extra["min_disk_lookup_tokens"] = min_lookup
            if (extra["metadata_workers"] > 32 or not extra["metadata_workers"]
                    <= extra["metadata_max_submitted"] <= 4096):
                raise ValueError("metadata worker/submission bounds are invalid")
            duration = extra["metadata_shutdown_timeout"]
            if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 300:
                raise ValueError("metadata shutdown timeout must be finite and bounded")
            if type(extra.get("disk_io_threads", 1)) is not int or extra.get("disk_io_threads", 1) <= 0:
                raise ValueError("disk_io_threads must be a positive integer")
            # NOTE: OffloadingSpec.__init__ owns self.config (the OffloadingConfig)
            # since a9531edfa6; our validated extra_config lives beside it.
            self.recipe_config = extra
            self.world_size = int(config.parallel.world_size)
            if self.world_size < 1 or config.parallel.data_parallel_size != 1:
                raise ValueError("positive world size and data_parallel_size=1 required")
            self.factory = getattr(importlib.import_module(extra["coordinator_module_path"]),
                                   extra["coordinator_factory"])
            self._manager = None
            self._worker = None

        def _provider(self, role, rank=None, geometry=None):
            if getattr(self, "_startup_cleanup", None) is not None:
                raise RuntimeError("previous native startup cleanup has not drained")
            # A factory that raises before returning owns its partial resources.
            provider = self.factory(config=self.recipe_config, role=role, rank=rank,
                                    world_size=self.world_size, geometry=geometry)
            close = _CloseOnce(getattr(provider, "close", None))
            try:
                sizes = tuple(tuple(s) for s in provider.size_by_group)
                if (len(sizes) != len(self.tokens_per_block)
                        or any(len(s) != self.world_size or any(type(n) is not int or n <= 0 for n in s)
                               for s in sizes)):
                    raise ValueError("provider must publish exact registered all-rank group bytes")
                if type(provider.max_pending_keys) is not int or provider.max_pending_keys <= 0:
                    raise ValueError("provider must publish bounded queue capacity")
                if self.recipe_config["max_pending_keys"] != provider.max_pending_keys:
                    raise ValueError("worker/manager/provider queue bounds must agree")
                if geometry is not None and tuple(s[rank] for s in sizes) != geometry.group_bytes:
                    raise ValueError("provider geometry does not match actual canonical references")
                # Only the provider sees the ordered all-rank canonical census.
                # Byte totals alone cannot distinguish reordered physical refs.
                layout = getattr(provider, "layout_fingerprint", "")
                if role == "scheduler" or layout != "":
                    if (not isinstance(layout, str) or len(layout) != 64
                            or any(c not in "0123456789abcdef" for c in layout)):
                        raise ValueError("provider layout_fingerprint must be a lowercase SHA256 digest")
                return provider, sizes, close
            except BaseException:
                self._cleanup_startup(None, close)
                raise

        def _cleanup_startup(self, native, close):
            # Preserve ownership if native drain fails. Never close SQLite/HTTP
            # out from under a native operation that might still reference data.
            self._startup_cleanup = (native, close)
            if native is not None:
                # CPUOffloadingWorker.shutdown drains both directions and
                # releases any region it owns.
                native.shutdown()
            try:
                close()
            except Exception:
                _LOG.warning("Persistence startup provider cleanup failed")
                raise RuntimeError("startup provider cleanup has not drained") from None
            self._startup_cleanup = None

        def get_manager(self):
            if self._manager is None:
                provider, sizes, close = self._provider("scheduler")
                try:
                    if provider.coordinator is None:
                        raise ValueError("scheduler requires explicit all-rank coordinator")
                    namespace = hashlib.sha256(json.dumps(
                        [self.recipe_config["tenant_namespace"],
                         self.recipe_config["cache_fingerprint"], sizes,
                         self.hash_seed, self.hash_algorithm, provider.layout_fingerprint],
                        separators=(",", ":")).encode()).hexdigest()
                    self._manager = NodeLocalDiskManager(
                        provider.coordinator, namespace, sizes, provider.max_pending_keys,
                        NativeDiskLoadStoreSpec, PrepareStoreOutput, RequestOffloadingContext,
                        close_provider=close,
                        lookup_keys_per_step=self.recipe_config["lookup_keys_per_step"],
                        metadata_workers=self.recipe_config["metadata_workers"],
                        metadata_max_submitted=self.recipe_config["metadata_max_submitted"],
                        metadata_shutdown_timeout=self.recipe_config["metadata_shutdown_timeout"])
                except BaseException:
                    self._cleanup_startup(None, close)
                    raise
            return self._manager

        def create_worker(self, kv_caches):
            from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
            from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
            import torch
            # The worker rank is supplied directly since a9531edfa6 (#48150);
            # get_world_group() is no longer consulted.
            rank = int(self.config.parallel.rank)
            if self.config.parallel.world_size != self.world_size or not 0 <= rank < self.world_size:
                raise ValueError("distributed world rank census differs from provider configuration")
            geometry = Geometry.from_canonical(kv_caches)
            provider, _, close = self._provider("worker", rank, geometry)
            native = None
            try:
                if provider.store is None:
                    raise ValueError("worker provider must supply rank-local DiskStore")
                rows = geometry.rows_for_budget(
                    self.recipe_config["staging_bytes"],
                    min(self.recipe_config["staging_rows"], provider.max_pending_keys))
                # mmap_region=None keeps the staging ring a private per-rank
                # pinned tensor. The /dev/shm SharedOffloadRegion is deliberately
                # not used: this lane is direct GPU<->node-local NVMe, and the
                # ring is only a bounded bounce buffer.
                native = CPUOffloadingWorker(kv_caches=kv_caches, blocks_per_chunk=1,
                                             num_cpu_chunks=rows, mmap_region=None)
                # One shared pump means load+store compete fairly for one
                # bounded ring; direction now comes from submit_store/submit_load.
                worker = DiskOffloadingWorker(
                    geometry=geometry, cpu_gpu=native, store=provider.store, rank=rank,
                    rows=rows, max_pending_keys=provider.max_pending_keys,
                    gpu_spec_cls=GPULoadStoreSpec, cpu_spec_cls=CPULoadStoreSpec,
                    result_cls=TransferResult, drain_cuda=torch.cuda.synchronize,
                    io_threads=self.recipe_config.get("disk_io_threads", 1),
                    close_provider=close)
            except BaseException:
                self._cleanup_startup(native, close)
                raise
            return worker

        def get_worker(self, kv_caches):
            if self._worker is None:
                self._worker = self.create_worker(kv_caches)
            return self._worker

    # Stable module globals/qualnames are necessary for metadata pickling between
    # scheduler and workers. __getattr__ reconstructs these on a fresh process.
    for cls in (NativeDiskLoadStoreSpec, NodeLocalDiskManager, DiskOffloadingWorker,
                NodeLocalDiskOffloadingSpec):
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls


def __getattr__(name):
    if name in {"NodeLocalDiskOffloadingSpec", "NodeLocalDiskManager",
                "NativeDiskLoadStoreSpec", "DiskOffloadingWorker"}:
        _load_native()
        return globals()[name]
    raise AttributeError(name)
