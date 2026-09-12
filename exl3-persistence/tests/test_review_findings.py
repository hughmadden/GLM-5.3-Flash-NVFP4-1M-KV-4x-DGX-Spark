# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hugh Madden and contributors
"""Stubbed confirmations for REVIEW-KV-RESTORE-CONCURRENCY-20260911.md.

Engine-free, GPU-free, network-free: a FakeCoordinator implements the
CoordinatorProtocol in-process and _Manager/DiskStore run for real. Two test
shapes are used:

* defect pins — assert the CURRENT (defective) behaviour so any silent
  behaviour change is visible; each pin names the review section it belongs
  to and is flipped to a fixed-behaviour assertion by its fix;
* fixed-behaviour tests — assert the post-fix contract directly (written with
  the fix; they fail on the defective tree).

No key, path, token or exception text is logged, matching the package rule.
"""
import tempfile
import threading
import time
import unittest

from recipe_persistence.coordinator import Ticket
from recipe_persistence.handlers import DiskLoadStoreSpec
from recipe_persistence.native import _Manager
from recipe_persistence.storage import DiskStore, Limits


class _Out:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Ctx:
    def __init__(self, req_id):
        self.req_id = req_id


class FakeCoordinator:
    """In-process CoordinatorProtocol with controllable answers.

    deadline_mode: "valid" (fresh leases), "past" (grants already expired,
    renew cannot resurrect), or "none" (coordinator cannot vouch at all).
    renew_rescues: whether a successful renew restores a vouchable deadline
    (the genuine-renewal-uncertainty case).
    """

    def __init__(self, *, deadline_mode="valid", renew_rescues=False,
                 renew_result=True, reserve_load_result="ticket"):
        self.deadline_mode = deadline_mode
        self.renew_rescues = renew_rescues
        self.renew_result = renew_result
        self.reserve_load_result = reserve_load_result  # "ticket" | None
        self.complete_store_result = None               # None (ack) | False
        self.deadlines = {}
        self.invalidated = []
        self.released = []
        self.completed = []
        self.renew_calls = 0
        self.calls = []
        self.call_threads = set()
        self._counter = 0
        self.sizes = ((16, 16),)

    def _note(self, name):
        self.calls.append(name)
        self.call_threads.add(threading.get_ident())

    def _mint(self):
        self._counter += 1
        return f"tok{self._counter:04d}"

    def _grant_deadline(self, token):
        if self.deadline_mode == "valid":
            self.deadlines[token] = time.monotonic() + 300.0
        elif self.deadline_mode == "past":
            self.deadlines[token] = time.monotonic() - 1.0
        else:
            self.deadlines[token] = None

    def reserve_load(self, namespace, key, owner):
        self._note("reserve_load")
        if self.reserve_load_result is None:
            return None
        token = self._mint()
        self._grant_deadline(token)
        return Ticket(token, namespace, key, ("rl0", "rl1"), self.sizes[0], owner)

    def reserve_store(self, namespace, key, owner, size_by_rank):
        self._note("reserve_store")
        token = self._mint()
        self._grant_deadline(token)
        return Ticket(token, namespace, key, ("wr0", "wr1"), tuple(size_by_rank),
                      owner, writing=True)

    def complete_store(self, ticket, success):
        self._note("complete_store")
        self.completed.append((ticket.key, success))
        return self.complete_store_result

    def release(self, ticket):
        self._note("release")
        self.released.append(ticket.key)
        return None

    def invalidate(self, namespace, key):
        self._note("invalidate")
        self.invalidated.append(key)
        return None

    def renew(self, ticket):
        self._note("renew")
        self.renew_calls += 1
        if self.renew_result and self.renew_rescues:
            self.deadlines[ticket.token] = time.monotonic() + 300.0
        return self.renew_result

    def lease_deadline(self, ticket):
        self._note("lease_deadline")
        return self.deadlines.get(ticket.token)

    def can_store(self):
        return True


def make_manager(coordinator, *, metadata_workers=0, lookup_keys_per_step=8,
                 capacity=64):
    return _Manager(coordinator, "ab" * 32, ((16, 16),), capacity,
                    DiskLoadStoreSpec, _Out, _Ctx,
                    lookup_keys_per_step=lookup_keys_per_step,
                    metadata_workers=metadata_workers)


def key(n):
    return b"prefix-" + n.to_bytes(2, "big") + b"\x00\x00\x00\x00"


NS = "ab" * 32


def small_limits(**over):
    fields = dict(quota=1_000_000, high=900_000, low=800_000,
                  index_bytes=65536, max_objects=100, max_leases=32,
                  max_object_bytes=4096, lease_seconds=300.0,
                  grace_seconds=30.0)
    fields.update(over)
    return Limits(**fields)


def commit_object(store, ns, k, size=100, owner="o"):
    ident = store.reserve_write(ns, k, size, owner)
    assert ident is not None
    assert store.write(ident, [bytes(size)]) is True
    return ident


class TestStaleLeaseLivelock(unittest.TestCase):
    """Review §1.2, FIXED: an un-vouchable lease must not answer RETRY
    forever. Pre-fix, an expired cached lease returned None (RETRY) on every
    lookup and held the reservation, so prepare_load was unreachable and the
    request deferred permanently (BUG-LOAD-PATH-STALE-LEASE)."""

    def test_expired_lease_becomes_clean_miss_and_is_released(self):
        coord = FakeCoordinator(deadline_mode="past")  # renew cannot resurrect
        manager = make_manager(coord, metadata_workers=1)
        ctx = _Ctx("req-1")
        outcomes = [manager.lookup(key(1), ctx) for _ in range(6)]
        # Never an unbounded RETRY stream: bounded, then a clean miss.
        self.assertNotIn(True, outcomes)
        self.assertFalse(outcomes[-1])
        # The dead lease is released (async ACK), not retained forever.
        manager.on_schedule_end()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and manager.loads:
            time.sleep(0.01)
            manager.on_schedule_end()
        self.assertFalse(manager.loads)
        self.assertIn(key(1), coord.released)

    def test_renewal_uncertainty_retries_then_recovers(self):
        coord = FakeCoordinator(deadline_mode="none", renew_rescues=True)
        manager = make_manager(coord, metadata_workers=1)
        ctx = _Ctx("req-1")
        self.assertIsNone(manager.lookup(key(1), ctx))   # unvouchable -> RETRY
        # The renewal lands (background worker); the next scan vouches.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and coord.renew_calls == 0:
            time.sleep(0.01)
        self.assertIs(manager.lookup(key(1), ctx), True)

    def test_renewal_uncertainty_is_bounded_then_misses(self):
        coord = FakeCoordinator(deadline_mode="none", renew_rescues=False)
        manager = make_manager(coord, metadata_workers=1)
        ctx = _Ctx("req-1")
        outcomes = [manager.lookup(key(1), ctx)
                    for _ in range(manager._STALE_LEASE_MAX_RETRY + 2)]
        self.assertEqual(outcomes[:manager._STALE_LEASE_MAX_RETRY],
                         [None] * manager._STALE_LEASE_MAX_RETRY)
        self.assertFalse(outcomes[manager._STALE_LEASE_MAX_RETRY])
        self.assertEqual(manager._stale_retries, {})  # released, counter cleared


class TestTransientReadMassInvalidation(unittest.TestCase):
    """Review §2.1: on_load_failure vetoes every key of the failed job, not
    just the keys whose objects proved corrupt."""

    def test_whole_job_keys_are_invalidated(self):
        coord = FakeCoordinator()
        manager = make_manager(coord)
        ctx = _Ctx("req-1")
        keys = [key(1), key(2), key(3)]
        for k in keys:
            self.assertIs(manager.lookup(k, ctx), True)
        manager.complete_load(keys, ctx)
        manager.on_load_failure(keys, ctx)
        self.assertEqual(sorted(coord.invalidated), sorted(keys))


class TestCleanupRejectionDegradesTier(unittest.TestCase):
    """Review §3.2, FIXED: one un-ACKed cleanup RPC must not flip the whole
    manager to closed (that converted every transient transport hiccup into
    a permanent tier kill). Rejection is retried boundedly; only SUSTAINED
    rejection past _METADATA_MAX_ATTEMPTS fails closed."""

    def test_single_rejection_absorbed_sustained_rejection_degrades(self):
        coord = FakeCoordinator(reserve_load_result=None)
        coord.complete_store_result = False
        manager = make_manager(coord, metadata_workers=1)
        ctx = _Ctx("req-1")
        out = manager.prepare_store([key(1)], ctx)
        self.assertIsNotNone(out)
        manager.complete_store([key(1)], ctx, success=True)
        time.sleep(0.2)                 # let the worker record the False
        manager.on_schedule_end()       # first poll: one rejection absorbed
        self.assertFalse(manager.closed)
        # Drive resubmissions past the attempt bound: now it is evidence.
        # (Each iteration advances one attempt every other poll: the
        # resubmitted future completes asynchronously after dispatch.)
        for _ in range(2 * (manager._METADATA_MAX_ATTEMPTS + 3)):
            time.sleep(0.06)
            manager._poll_metadata()
            manager._dispatch_metadata()
        self.assertTrue(manager.closed)
        self.assertEqual(manager.degraded_reason, "coordinator_callback_rejected")

    def test_lost_renewal_retires_without_disabling_tier(self):
        coord = FakeCoordinator(deadline_mode="valid")
        manager = make_manager(coord, metadata_workers=1)
        ctx = _Ctx("req-1")
        self.assertIs(manager.lookup(key(1), ctx), True)
        coord.renew_result = False      # the coordinator now loses renewals
        # Force a renewal through the margin path, then let it fail.
        coord.deadlines[manager.loads[("req-1", key(1))].token] = \
            time.monotonic() + 1.0      # inside the renew margin
        manager._renew_requested.clear()
        manager.on_schedule_end()       # scans leases, dispatches the renew
        time.sleep(0.2)
        manager.on_schedule_end()       # polls the failed renew
        self.assertFalse(manager.closed)
        self.assertIn(("req-1", key(1)), manager._retiring_loads)


class TestMetadataPumpRunsWithoutScheduleEnd(unittest.TestCase):
    """Review §2.3, FIXED: metadata ACKs must process on a wall-clock bound,
    not only inside on_schedule_end. Pre-fix, an idle engine (no batch, hence
    no hook) never drained renewals/releases, so has_pending_work() stayed
    true forever and reset_cache() could never recover a poisoned engine —
    the lookup-gate deadlock one level down."""

    def test_idle_engine_drains_without_the_hook(self):
        coord = FakeCoordinator()
        manager = make_manager(coord, metadata_workers=1)
        ctx = _Ctx("req-1")
        self.assertIs(manager.lookup(key(1), ctx), True)
        manager.complete_load([key(1)], ctx)   # schedules the release ACK
        manager.on_request_finished(ctx)       # request ends; owner leaves the gate
        time.sleep(0.3)                         # worker finished long ago
        # The core-visible proof: pending work resolves without any hook call.
        self.assertFalse(manager.has_pending_work())
        self.assertNotIn(("req-1", key(1)), manager.loads)


class TestStoreBudgetCoupling(unittest.TestCase):
    """Review §2.4: the store admission budget is the lookup scan budget."""

    def test_store_keys_left_tracks_lookup_budget(self):
        coord = FakeCoordinator()
        manager = make_manager(coord, lookup_keys_per_step=8)
        manager.on_schedule_end()
        self.assertEqual(manager._store_keys_left, 8)
        manager._store_keys_left = 3
        manager.on_schedule_end()
        self.assertEqual(manager._store_keys_left, 8)


class TestSynchronousAdmissions(unittest.TestCase):
    """Review §3.1: reserve RPCs run on the calling (scheduler) thread even in
    async metadata mode — admissions are not handed to the executor."""

    def test_prepare_store_runs_reserves_on_calling_thread(self):
        coord = FakeCoordinator(reserve_load_result=None)
        manager = make_manager(coord, metadata_workers=2)
        ctx = _Ctx("req-1")
        out = manager.prepare_store([key(1)], ctx)
        self.assertIsNotNone(out)
        self.assertEqual(coord.call_threads, {threading.get_ident()})
        self.assertIn("reserve_store", coord.calls)


class TestEvictionClock(unittest.TestCase):
    """Review §2.2: reads must extend a committed object's eviction eligibility,
    and recovery must not make the whole restored cache instantly eligible."""

    def _store(self, clock):
        root = tempfile.mkdtemp(prefix="rpkv-test-")
        return DiskStore(root, small_limits(), clock=lambda: clock[0])

    def test_read_extends_expiry_after_fix(self):
        clock = [1000.0]
        store = self._store(clock)
        k = b"hot-key"
        commit_object(store, NS, k)
        clock[0] += 400.0
        lease = store.reserve_read(NS, k, "reader")
        self.assertTrue(store.read_into(lease, [bytearray(100)]))
        row = store.db.execute("SELECT expiry FROM objects WHERE ns=? AND key=?",
                               (NS, k)).fetchone()
        self.assertGreater(row[0], clock[0])
        store.close()

    def test_recovery_grants_fresh_eligibility_after_fix(self):
        clock = [1000.0]
        root = tempfile.mkdtemp(prefix="rpkv-test-")
        store = DiskStore(root, small_limits(), clock=lambda: clock[0])
        k = b"warm-key"
        commit_object(store, NS, k)
        store.close()
        clock[0] += 400.0
        store2 = DiskStore(root, small_limits(), clock=lambda: clock[0])
        row = store2.db.execute("SELECT expiry FROM objects WHERE ns=? AND key=?",
                                (NS, k)).fetchone()
        self.assertIsNotNone(row)
        self.assertGreater(row[0], clock[0])
        store2.close()


if __name__ == "__main__":
    unittest.main()
