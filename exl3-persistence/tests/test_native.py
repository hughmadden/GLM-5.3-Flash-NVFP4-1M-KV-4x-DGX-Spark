# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from types import SimpleNamespace as NS
import threading

import pytest

from recipe_persistence.geometry import Geometry
from recipe_persistence.handlers import DiskLoadStoreSpec, DiskTransferPump


class GPU:
    def __init__(self, block_ids, group_sizes, block_indices):
        self.block_ids, self.group_sizes, self.block_indices = block_ids, group_sizes, block_indices

    @staticmethod
    def medium():
        return "GPU"


class CPU:
    def __init__(self, block_ids):
        self.block_ids = block_ids


@dataclass
class Result:
    job_id: int
    success: bool
    transfer_size: int = 0
    transfer_time: float = 0


class Row:
    def __init__(self, size):
        self.data = bytearray(size)

    def numpy(self):
        return self.data


class Copy:
    def __init__(self, tensors, name, trace):
        self.dst_tensors = tensors
        self.src_tensors = [NS(shape=(10000, len(t[0].data))) for t in tensors]
        self.name, self.trace = name, trace
        self.active = []
        self.done = False
        self.fail = False
        self.throw = False
        self.closed = False

    def transfer_async(self, job_id, src_spec, dst_spec):
        self.trace.append((self.name, job_id, (src_spec, dst_spec)))
        self.active.append(job_id)
        if self.throw:
            raise OSError("native launch failed after starting")
        return True

    def get_finished(self):
        if not self.done:
            return []
        result = [Result(i, not self.fail) for i in self.active]
        self.active.clear()
        return result

    def wait(self, ids):
        self.done = True

    def shutdown(self):
        assert not self.active
        self.closed = True


class Store:
    def __init__(self):
        self.payload = {}
        self.calls = []
        self.gate = None
        self.started = threading.Event()
        self.fail = False

    def write(self, token, buffers):
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(3)
        self.calls.append(("write", token))
        if self.fail:
            raise OSError("fsync failed")
        self.payload[token] = b"".join(buffers)
        return True

    def read_into(self, token, buffers):
        self.calls.append(("read", token))
        if self.fail or token not in self.payload:
            return False
        payload = self.payload[token]
        offset = 0
        for buffer in buffers:
            buffer[:] = payload[offset:offset + len(buffer)]
            offset += len(buffer)
        return offset == len(payload)


def transfer(pump, job_id, spec):
    """Submit through the direction-explicit API the connector worker uses.

    Upstream f237e16b41 (#45053) split transfer_async into submit_store and
    submit_load. OffloadingConnectorWorker picks the leg from the metadata
    entry's src_spec/dst_spec; this helper does the same from the old
    (src, dst) tuple so the existing cases keep exercising both legs.
    """
    src, dst = spec
    if isinstance(src, GPU):
        return pump.submit_store(job_id, src, dst)
    return pump.submit_load(job_id, src, dst)


def worker_fake(store_handler, load_handler):
    """Stand-in for CPUOffloadingWorker: two private direction handlers and a
    shutdown that drains both, as of upstream f237e16b41 (#45053)."""
    native = NS(_store_handler=store_handler, _load_handler=load_handler)
    def shutdown():
        native._store_handler.shutdown()
        native._load_handler.shutdown()
    native.shutdown = shutdown
    return native


def key(number=0, group=0):
    return bytes([number + 1]) + group.to_bytes(4, "big")


def disk(keys, prefix="t"):
    return DiskLoadStoreSpec(tuple(keys), tuple((prefix + str(i),) for i in range(len(keys))),
                             "namespace", "request")


def pump(rows=1, cap=8):
    geometry = Geometry((16, 32), (((0, 7), (1, 21)), ((0, 4),)))
    tensors = [[Row(16) for _ in range(rows)], [Row(32) for _ in range(rows)]]
    trace = []
    native = worker_fake(Copy(tensors, "store", trace), Copy(tensors, "load", trace))
    store = Store()
    drains = []
    def drain():
        drains.append(True)
        for handler in (native._store_handler, native._load_handler):
            handler.active.clear()
    p = DiskTransferPump(geometry=geometry, cpu_gpu=native, store=store, rank=0,
                         rows=rows, max_pending_keys=cap, gpu_spec_cls=GPU,
                         cpu_spec_cls=CPU, result_cls=Result, drain_cuda=drain)
    return p, native, store, trace, drains


def test_store_row_held_until_cuda_and_fsync():
    p, native, store, trace, _ = pump()
    store.gate = threading.Event()
    try:
        assert transfer(p, 1, (GPU([0], [1, 0], [0, 0]), disk([key()])))
        assert p.get_finished() == []
        assert not p.free and store.calls == []
        native._store_handler.done = True
        assert p.get_finished() == []
        assert store.started.wait(1)
        assert not p.free
        store.gate.set()
        p.wait({1})
        result = p.get_finished()
        assert len(result) == 1 and result[0].success
        assert len(store.payload["t0"]) == 28
        assert len(p.free) == 1
    finally:
        store.gate.set()
        p.shutdown()


def test_failed_read_does_not_copy_or_zero_fill():
    p, native, store, trace, _ = pump()
    try:
        p.cpu_tensors[0][0].data[:] = b"x" * 16
        assert transfer(p, 1, (disk([key()]), GPU([5], [1, 0], [0, 0])))
        p.wait({1})
        result = p.get_finished()
        assert not result[0].success
        assert not trace
        assert p.cpu_tensors[0][0].data == b"x" * 16
    finally:
        p.shutdown()


def test_fair_one_row_waves_and_group_specs():
    p, native, store, trace, _ = pump(rows=2)
    try:
        assert transfer(p, 1, (GPU([0, 1, 2], [2, 1], [0, 0]),
                                    disk([key(0), key(1), key(2, 1)], "a")))
        assert transfer(p, 2, (GPU([3], [1, 0], [0, 0]), disk([key(3)], "b")))
        p.get_finished()
        assert len(trace) == 2  # not three waves from the first job
        assert [entry[2][0].block_ids for entry in trace] == [[0], [3]]
        p.wait({1, 2})
        assert all(r.success for r in p.get_finished())
        assert trace[-1][2][0].group_sizes == [0, 1]
        assert len(store.payload["a2"]) == 4
    finally:
        p.shutdown()


def test_queue_bound_and_malformed_specs():
    p, _, _, _, _ = pump(cap=1)
    try:
        assert not transfer(p, 1, (GPU([1, 2], [2, 0], [0, 0]), disk([key(), key(1)])))
        assert not transfer(p, 1, (GPU([1], [0, 1], [0, 0]), disk([key()])))
        assert transfer(p, 1, (GPU([1], [1, 0], [0, 0]), disk([key()])))
        assert not transfer(p, 2, (GPU([2], [1, 0], [0, 0]), disk([key(1)])))
        p.cancel({1})
        assert not p.get_finished()[0].success
        assert p.pending_keys == 0 and len(p.free) == 1
    finally:
        p.shutdown()


@pytest.mark.parametrize("failure", ["fsync", "native", "launch"])
def test_failures_produce_results(failure):
    p, native, store, _, drains = pump()
    try:
        store.fail = failure == "fsync"
        native._store_handler.fail = failure == "native"
        native._store_handler.throw = failure == "launch"
        transfer(p, 1, (GPU([1], [1, 0], [0, 0]), disk([key()])))
        p.wait({1})
        assert not p.get_finished()[0].success
        assert len(p.free) == 1
        assert bool(drains) == (failure == "launch")
    finally:
        p.shutdown()


def test_cancel_drains_active_cuda_and_shutdown_is_idempotent():
    p, native, _, _, _ = pump()
    transfer(p, 1, (GPU([1, 2], [2, 0], [0, 0]), disk([key(), key(1)])))
    p.get_finished()
    assert native._store_handler.active
    p.cancel({1})
    assert not native._store_handler.active
    assert not p.get_finished()[0].success
    p.shutdown()
    p.shutdown()
    assert native._store_handler.closed and native._load_handler.closed


def manager(capacity=8, sizes=((28,), (4,)), lookup_keys_per_step=None, **metadata):
    from recipe_persistence.native import _Manager
    from recipe_persistence.coordinator import Ticket
    class Coordinator:
        def __init__(self):
            self.live = {}
            self.invalidated = []
            self.completed = []
            self.lookup_calls = []
            self.miss = False
        def _reserve(self, namespace, key, owner):
            token = str(len(self.live)) + owner + key.hex()
            ticket = Ticket(token, namespace, key, (token,), sizes[int.from_bytes(key[-4:], "big")])
            self.live[token] = ticket
            return ticket
        def reserve_load(self, namespace, key, owner):
            self.lookup_calls.append((owner, key))
            return None if self.miss else self._reserve(namespace, key, owner)
        def reserve_store(self, namespace, key, owner, size_by_rank):
            return self._reserve(namespace, key, owner)
        def release(self, ticket):
            self.live.pop(ticket.token, None)
        def complete_store(self, ticket, success):
            self.completed.append(success)
            self.release(ticket)
        def invalidate(self, namespace, key):
            self.invalidated.append((namespace, key))
        def renew(self, ticket):
            return ticket.token in self.live
        def lease_deadline(self, ticket):
            import time
            return time.monotonic() + 60 if ticket.token in self.live else None
    coordinator = Coordinator()
    return _Manager(coordinator, "ns", sizes, capacity, DiskLoadStoreSpec, NS, NS,
                    lookup_keys_per_step=lookup_keys_per_step, **metadata), coordinator


def test_can_store_ignores_transient_pressure_and_step_deferral():
    m, _ = manager(capacity=1, lookup_keys_per_step=1)
    ctx = NS(req_id="a")
    try:
        assert m.can_store()
        assert m.lookup(key(), ctx) is True
        assert m.lookup(key(1), ctx) is False  # Capacity, not permanent disablement.
        assert m.can_store()
        assert m.prepare_store([key(2)], ctx) is None
        assert m.can_store()
        m.on_request_finished(ctx)
        m._store_keys_left = 0
        assert m.prepare_store([key(2)], ctx) is None
        assert m.can_store()
    finally:
        m.shutdown()
    assert not m.can_store()


def test_can_store_rejects_permanent_degradation():
    m, _ = manager()
    try:
        m._degrade("coordinator_callback_failed")
        assert not m.can_store()
        m.on_schedule_end()
        assert not m.can_store()
    finally:
        m.shutdown()


def metadata_steps_until(m, predicate, timeout=3):
    import time
    from concurrent.futures import wait, FIRST_COMPLETED
    deadline = time.monotonic() + timeout
    steps = 0
    while not predicate():
        assert time.monotonic() < deadline, "metadata work did not converge"
        m.on_schedule_end()
        steps += 1
        if m._metadata_futures:
            wait(tuple(m._metadata_futures), timeout=0.001, return_when=FIRST_COMPLETED)
    return steps


@pytest.mark.parametrize("completion", ["load", "cancel", "failure"])
def test_4219_async_cleanup_yields_heartbeats_and_retains_credits_until_ack(completion):
    import time
    from concurrent.futures import wait, FIRST_COMPLETED
    m, c = manager(capacity=5000, sizes=((2_250_000,),), metadata_workers=2,
                   metadata_max_submitted=8, metadata_shutdown_timeout=2)
    ctx, hot = NS(req_id="full-prefix"), NS(req_id="active-decode")
    keys = [i.to_bytes(8, "big") + bytes(4) for i in range(4219)]
    hot_key = b"hot" + bytes(4)
    assert all(m.lookup(k, ctx) for k in keys)
    assert m.lookup(hot_key, hot)
    m.prepare_load([hot_key], hot)
    if completion != "cancel":
        m.prepare_load(keys, ctx)
    gate, entered = threading.Event(), threading.Event()
    original = c.release
    main_thread = threading.get_ident()
    active, maximum = [0], [0]
    lock = threading.Lock()
    def release(ticket):
        assert threading.get_ident() != main_thread
        with lock:
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
        entered.set()
        try:
            assert gate.wait(5)
            return original(ticket)
        finally:
            with lock:
                active[0] -= 1
    c.release = release
    try:
        started = time.monotonic()
        if completion != "cancel":
            m.complete_load(keys, ctx)
            if completion == "failure":
                m.on_load_failure(keys, ctx)
        else:
            m.on_request_finished(ctx)
        assert time.monotonic() - started < 0.3
        assert entered.wait(1)
        assert len(m.loads) == 4220 and len(c.live) == 4220
        assert sum(t.sizes[0] for t in m.loads.values()) > 9_000_000_000
        heartbeats = []
        for step in range(5):
            m.on_schedule_end()
            assert m.lookup(hot_key, hot) is True
            heartbeats.append(step)
            assert len(m._metadata_futures) <= 8 and len(m.loads) == 4220
        assert heartbeats == list(range(5)) and m.has_pending_work()
        gate.set()
        deadline = time.monotonic() + 5
        while m._retiring_loads:
            assert time.monotonic() < deadline
            m.on_schedule_end()
            assert m.lookup(hot_key, hot) is True
            heartbeats.append(len(heartbeats))
            assert len(m._metadata_futures) <= 8
            wait(tuple(m._metadata_futures), timeout=0.001, return_when=FIRST_COMPLETED)
        assert maximum[0] <= 2 and len(m.loads) == 1
        m.complete_load([hot_key], hot)
        m._metadata_drain(2)
        assert not c.live and not m._metadata_pending()
    finally:
        gate.set()
        if (hot.req_id, hot_key) in m.prepared:
            m.complete_load([hot_key], hot)
        m.shutdown()


def test_background_ack_does_not_mutate_manager_maps_before_main_thread_poll():
    from concurrent.futures import wait
    m, c = manager(metadata_workers=1, metadata_max_submitted=1)
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    m.complete_load([key()], ctx)
    wait(tuple(m._metadata_futures), timeout=1)
    assert not c.live  # Worker really acknowledged release.
    assert len(m.loads) == 1 and m._retiring_loads  # Main ownership not reconciled yet.
    m.on_schedule_end()
    assert not m.loads and not m._retiring_loads
    m.shutdown()


def test_failed_release_keeps_ticket_capacity_and_shutdown_retry_ownership():
    from recipe_persistence.handlers import _CloseOnce
    m, c = manager(capacity=1, metadata_workers=1, metadata_max_submitted=1,
                   metadata_shutdown_timeout=0.03)
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    m.prepare_load([key()], ctx)
    original = c.release
    c.release = lambda ticket: False
    closed = []
    m._close_provider = _CloseOnce(lambda: closed.append(True))
    m.complete_load([key()], ctx)
    metadata_steps_until(m, lambda: m.closed)
    assert len(m.loads) == len(c.live) == 1 and m._retiring_loads
    assert m.lookup(key(), ctx) is False
    with pytest.raises(RuntimeError, match="metadata cleanup has not drained"):
        m.shutdown()
    assert not closed and not m._shutdown_complete and len(m.loads) == 1
    c.release = original
    m._metadata_drain(1)
    m.shutdown()
    assert closed == [True] and not m.loads


def test_shutdown_proves_worker_exit_within_deadline_before_provider_close():
    from recipe_persistence.handlers import _CloseOnce
    m, _ = manager(metadata_workers=1, metadata_max_submitted=1, metadata_shutdown_timeout=0.02)
    closed = []
    m._close_provider = _CloseOnce(lambda: closed.append(True))
    class UnprovedThread:
        alive = True
        def join(self, timeout):
            assert 0 <= timeout <= 0.02
        def is_alive(self):
            return self.alive
    thread = UnprovedThread()
    m._metadata_executor._threads.add(thread)
    with pytest.raises(RuntimeError, match="metadata workers have not drained"):
        m.shutdown()
    assert not closed and not m._shutdown_complete
    thread.alive = False
    m.shutdown()
    assert closed == [True] and m._shutdown_complete


def test_invalidation_veto_covers_other_owners_and_never_releases_active_reader():
    m, c = manager(capacity=8, metadata_workers=2, metadata_max_submitted=2)
    a, b, active = [NS(req_id=name) for name in ("failed", "unprepared", "active")]
    for ctx in (a, b, active):
        assert m.lookup(key(), ctx)
    m.prepare_load([key()], a)
    m.prepare_load([key()], active)
    active_token = m.loads[(active.req_id, key())].token
    release_gate, invalidation_gate = threading.Event(), threading.Event()
    released = []
    original = c.release
    def release(ticket):
        assert release_gate.wait(3)
        released.append(ticket.token)
        return original(ticket)
    def invalidate(*args):
        assert invalidation_gate.wait(3)
        return True
    c.release, c.invalidate = release, invalidate
    try:
        m.complete_load([key()], a)
        m.on_load_failure([key()], a)
        assert all(m.lookup(key(), ctx) is False for ctx in (a, b, active, NS(req_id="new")))
        invalidation_gate.set()
        metadata_steps_until(m, lambda: m._invalidations.get(key()) is True)
        assert active_token not in released
        assert m.lookup(key(), active) is False  # ACK alone never removes local veto.
        release_gate.set()
        metadata_steps_until(m, lambda: len(m.loads) == 1)
        assert active_token not in released and (active.req_id, key()) in m.prepared
        m.complete_load([key()], active)
        m._metadata_drain(2)
        assert not m.loads and m.lookup(key(), NS(req_id="new")) is False
        m.reset_cache()
        m.on_schedule_end()
        assert not m._invalidations
    finally:
        release_gate.set()
        invalidation_gate.set()
        if (active.req_id, key()) in m.prepared:
            m.complete_load([key()], active)
        m.shutdown()


def test_expired_4219_key_scan_has_only_bounded_candidate_passes_and_submissions():
    import time
    m, c = manager(capacity=5000, metadata_workers=2, metadata_max_submitted=8)
    ctx = NS(req_id="expired-prefix")
    keys = [i.to_bytes(8, "big") + bytes(4) for i in range(4219)]
    assert all(m.lookup(k, ctx) for k in keys)
    class CountedOwnership(dict):
        passes = 0
        def items(self):
            self.passes += 1
            return super().items()
    m.loads = CountedOwnership(m.loads)
    c.lease_deadline = lambda ticket: None
    gate = threading.Event()
    def renew(ticket):
        assert gate.wait(5)
        return True
    c.renew = renew
    try:
        started = time.monotonic()
        assert all(m.lookup(k, ctx) is None for k in keys)
        assert time.monotonic() - started < 0.3
        assert len(m._metadata_futures) == 8 and m.loads.passes <= 8
        previous = m.loads.passes
        assert all(m.lookup(k, ctx) is None for k in keys)
        assert m.loads.passes == previous and len(m._metadata_futures) == 8
        assert len(m._renew_requested) == 4219
    finally:
        gate.set()
        m.on_request_finished(ctx)
        m.shutdown()


def test_metadata_numeric_proofs_and_timeouts_reject_boolean_values():
    with pytest.raises(ValueError, match="shutdown timeout"):
        manager(metadata_workers=1, metadata_max_submitted=1, metadata_shutdown_timeout=True)
    m, c = manager(metadata_workers=1, metadata_max_submitted=1)
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    ticket = m.loads[(ctx.req_id, key())]
    c.lease_deadline = lambda ticket: True
    assert not m._lease_valid(ticket)
    m.shutdown()


def test_async_renewal_uses_finite_non_io_proof_and_defers_until_ack():
    import time
    m, c = manager(metadata_workers=1, metadata_max_submitted=1)
    ctx = NS(req_id="reader")
    assert m.lookup(key(), ctx)
    proof = [None]
    c.lease_deadline = lambda ticket: proof[0]
    gate, entered = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    def renew(ticket):
        assert threading.get_ident() != main_thread
        entered.set()
        assert gate.wait(2)
        proof[0] = time.monotonic() + 60
        return True
    c.renew = renew
    try:
        assert m.lookup(key(), ctx) is None
        assert entered.wait(1)
        m.on_schedule_end()
        assert m.lookup(key(), ctx) is None and len(m._metadata_futures) == 1
        assert len(c.lookup_calls) == 1
        gate.set()
        metadata_steps_until(m, lambda: not m._renew_requested)
        assert m.lookup(key(), ctx) is True
        proof[0] = time.monotonic() - 1
        assert m.lookup(key(), ctx) is None  # Never use a previous True forever.
    finally:
        gate.set()
        m.on_request_finished(ctx)
        m.shutdown()


def test_submission_overflow_fails_closed_without_dropping_owned_tickets():
    m, c = manager(metadata_workers=1, metadata_max_submitted=1)
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    executor = m._metadata_executor
    class Full:
        def submit(self, *args, **kwargs):
            raise RuntimeError("queue full")
    m._metadata_executor = Full()
    m.complete_load([key()], ctx)
    assert m.closed and len(m.loads) == len(c.live) == 1
    assert m._retiring_loads and not m._metadata_futures
    m._metadata_executor = executor
    m.shutdown()
    assert not m.loads and not c.live


def test_quarantine_overflow_reclaims_only_acknowledged_slots_after_global_disable():
    m, c = manager(capacity=2, metadata_workers=1, metadata_max_submitted=1)
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    # Inject a saturated historical veto map to exercise defensive overflow.
    m._invalidations = {key(10): True, key(11): True}
    m.on_load_failure([key()], ctx)
    assert m.closed and key() in m._invalidations and len(m._invalidations) == 2
    assert m.loads and m.lookup(key(10), NS(req_id="another")) is False
    m.shutdown()
    assert not c.live and m.closed


def test_unacknowledged_quarantine_overflow_is_explicit_namespace_fail_stop():
    m, c = manager(capacity=2, metadata_workers=1, metadata_max_submitted=1)
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    m._invalidations = {key(10): False, key(11): False}
    try:
        with pytest.raises(RuntimeError, match="namespace quarantine overflow"):
            m.on_load_failure([key()], ctx)
        assert m.closed and m._namespace_quarantined and len(m.loads) == len(c.live) == 1
        assert len(m._invalidations) == 2 and m.has_pending_work()
        with pytest.raises(RuntimeError, match="external recovery"):
            m.reset_cache()
        with pytest.raises(RuntimeError, match="external recovery"):
            m.shutdown()
        assert not m._shutdown_complete
    finally:
        # No external resources in this injected fake-provider fail-stop case;
        # retire the unused test executor without claiming provider cleanup ACK.
        m._metadata_executor.shutdown(wait=True)


def test_cleanup_type_rotation_prevents_store_starvation_behind_long_load_queue():
    m, c = manager(capacity=16, metadata_workers=1, metadata_max_submitted=1)
    reader, writer = NS(req_id="reader"), NS(req_id="writer")
    keys = [key(i) for i in range(8)]
    assert all(m.lookup(k, reader) for k in keys)
    c.miss = True
    store_key = key(99)
    m.prepare_store([store_key], writer)
    order = []
    release, complete = c.release, c.complete_store
    def record_release(ticket):
        order.append("release")
        return release(ticket)
    def record_complete(ticket, success):
        order.append("store")
        return complete(ticket, success)
    c.release, c.complete_store = record_release, record_complete
    m.complete_load(keys, reader)
    m.complete_store([store_key], writer)
    m._metadata_drain(2)
    assert order.index("store") < 4
    m.shutdown()


def test_complete_store_is_background_and_unacknowledged_store_stays_owned():
    m, c = manager(metadata_workers=1, metadata_max_submitted=1)
    ctx = NS(req_id="writer")
    c.miss = True
    out = m.prepare_store([key()], ctx)
    gate = threading.Event()
    original = c.complete_store
    main_thread = threading.get_ident()
    def complete(ticket, success):
        assert threading.get_ident() != main_thread
        assert gate.wait(2)
        return original(ticket, success)
    c.complete_store = complete
    try:
        m.complete_store(out.keys_to_store, ctx, success=False)
        assert len(m.stores) == len(c.live) == 1 and m.has_pending_work()
        m.on_request_finished(ctx)  # GPU drain happened, but metadata still owns it.
        assert len(m.stores) == 1
        gate.set()
        m._metadata_drain(2)
        assert not m.stores and c.completed == [False]
    finally:
        gate.set()
        m.shutdown()


def test_store_attempt_budget_is_separate_and_never_skips_unattempted_keys():
    m, c = manager(lookup_keys_per_step=1)
    ctx = NS(req_id="writer")
    c.miss = True
    out = m.prepare_store([key(), key(1)], ctx)
    assert out.keys_to_store == [key()] and out.skipped_keys == []
    assert m.prepare_store([key(1)], ctx) is None
    assert m._lookup_keys_left == 1
    m.complete_store([key()], ctx)
    m.on_schedule_end()
    out = m.prepare_store([key(1)], ctx)
    assert out.keys_to_store == [key(1)] and out.skipped_keys == []
    m.complete_store([key(1)], ctx)
    m.shutdown()


def full_prefix_scan(m, keys, context):
    # Pinned _maximal_prefix_lookup and SWA lookup continue scanning after None;
    # they defer the aggregate prefix, rather than treating deferral as a miss.
    results = [m.lookup(k, context) for k in keys]
    assert all(result is not False for result in results)
    return None if any(result is None for result in results) else True


def test_4219_key_lookup_yields_each_step_and_other_request_keeps_scheduling():
    m, c = manager(capacity=5000, sizes=((2_250_000,),), lookup_keys_per_step=8)
    cold, hot = NS(req_id="cold-prefix"), NS(req_id="active-decode")
    hot_key = b"hot" + bytes(4)
    assert m.lookup(hot_key, hot) is True
    m.prepare_load([hot_key], hot)
    m.on_schedule_end()
    keys = [i.to_bytes(8, "big") + bytes(4) for i in range(4219)]
    heartbeats = []
    for step in range(600):
        previous = len(c.lookup_calls)
        outcome = full_prefix_scan(m, keys, cold)
        assert len(c.lookup_calls) - previous <= 8
        # The simulated scheduler regains control after each complete scan and
        # still schedules the other request's decode heartbeat on EVERY step.
        assert m.lookup(hot_key, hot) is True
        assert len(c.lookup_calls) - previous <= 8
        heartbeats.append(step)
        if outcome is True:
            break
        assert m.has_pending_work()
        m.on_schedule_end()
    else:
        pytest.fail("full prefix did not finish within bounded owner turns")
    assert len(heartbeats) == 528
    assert len(m.loads) == 4220 and len(c.lookup_calls) == 4220
    assert sum(t.sizes[0] for t in m.loads.values()) > 9_000_000_000
    metadata = m.prepare_load(keys, cold)
    assert len(metadata.keys) == 4219 and not m._lookup_owners
    m.complete_load(keys, cold)
    m.complete_load([hot_key], hot)
    m.on_request_finished(cold)
    m.on_request_finished(hot)
    assert not m.has_pending_work() and not c.live
    m.shutdown()


def test_two_cold_owners_get_round_robin_turns_despite_fixed_scan_order():
    m, c = manager(capacity=64, lookup_keys_per_step=8)
    contexts = [NS(req_id="first"), NS(req_id="second")]
    keys = [key(i) for i in range(17)]
    done = set()
    turns = []
    for _ in range(6):
        previous = len(c.lookup_calls)
        for ctx in contexts:  # Adversarially, the first owner is always scanned first.
            if ctx.req_id not in done and full_prefix_scan(m, keys, ctx) is True:
                m.prepare_load(keys, ctx)
                m.complete_load(keys, ctx)
                done.add(ctx.req_id)
        fresh = c.lookup_calls[previous:]
        assert len(fresh) <= 8
        assert len({owner for owner, _ in fresh}) <= 1
        turns.append(fresh[0][0])
        m.on_schedule_end()
    assert turns == ["first", "second"] * 3
    assert done == {"first", "second"} and not m.has_pending_work()
    for ctx in contexts:
        m.on_request_finished(ctx)
    m.shutdown()


@pytest.mark.parametrize("reverse", [False, True])
def test_cached_prefix_hits_do_not_consume_current_owners_budget(reverse):
    m, c = manager(capacity=20, lookup_keys_per_step=2)
    first, second = NS(req_id="first"), NS(req_id="second")
    keys = [key(i) for i in range(3)]
    if reverse:
        keys.reverse()  # SWA-style backward scan uses the same deferral contract.
    assert full_prefix_scan(m, keys, first) is None
    assert full_prefix_scan(m, keys, second) is None
    m.on_schedule_end()
    assert m._lookup_turn_owner == "second" and m._lookup_keys_left == 2
    assert full_prefix_scan(m, keys, first) is None
    assert m._lookup_keys_left == 2 and len(c.lookup_calls) == 2
    assert full_prefix_scan(m, keys, second) is None
    assert m._lookup_keys_left == 0 and len(c.lookup_calls) == 4
    m.on_request_finished(first)
    m.on_request_finished(second)
    assert not c.live and not m.has_pending_work()


def test_deferred_lookup_keeps_idle_engine_stepping_and_cancel_removes_state():
    m, c = manager(capacity=2, lookup_keys_per_step=1)
    first, second, third = [NS(req_id=name) for name in ("first", "second", "third")]
    assert m.lookup(key(), first) is True
    assert m.lookup(key(1), first) is None
    assert not m.prepared and not m.stores and m.has_pending_work()
    assert m.lookup(key(1), second) is None
    assert len(m._lookup_owners) == 2
    assert m.lookup(key(2), third) is False  # No unbounded untracked-owner queue.
    assert len(m._lookup_owners) == m.capacity
    # The real cancellation lifecycle callback, not an invented cancel API.
    m.on_request_finished(first)
    m.on_request_finished(second)
    assert not c.live and not m._lookup_owners and not m.has_pending_work()
    m.on_schedule_end()
    assert not m.has_pending_work()
    m.shutdown()


def test_miss_removes_turn_but_preserves_confirmed_prefix_for_prepare_load():
    m, c = manager(lookup_keys_per_step=1)
    ctx = NS(req_id="partial-prefix")
    assert m.lookup(key(), ctx) is True
    assert m.lookup(key(1), ctx) is None
    m.on_schedule_end()
    c.miss = True
    assert m.lookup(key(1), ctx) is False
    assert not m._lookup_owners and len(c.live) == 1
    m.prepare_load([key()], ctx)
    m.complete_load([key()], ctx)
    m.on_request_finished(ctx)
    assert not m.has_pending_work()


@pytest.mark.parametrize("action", ["reset", "shutdown", "degrade"])
def test_lookup_gate_cleanup_never_spins_a_disabled_or_dead_owner(action):
    m, c = manager(lookup_keys_per_step=1)
    ctx = NS(req_id="cancelled")
    assert m.lookup(key(), ctx) is True
    assert m.lookup(key(1), ctx) is None
    if action == "reset":
        m.reset_cache()
    elif action == "shutdown":
        m.shutdown()
    else:
        m._degrade("lease_renewal_lost")
        m.on_request_finished(ctx)
    assert not m._lookup_owners and not m.has_pending_work() and not c.live


def test_lookup_reserves_before_true_and_releases_unused_request_promises():
    m, c = manager(capacity=2)
    ctx = NS(req_id="r")
    assert m.lookup(key(), ctx) is True
    assert len(c.live) == 1
    assert m.lookup(key(), ctx) is True and len(c.live) == 1
    assert m.lookup(key(1), ctx) is True
    assert m.lookup(key(2), ctx) is False
    spec = m.prepare_load([key()], ctx)
    assert spec.tokens[0][0] in c.live
    with pytest.raises(RuntimeError):
        m.reset_cache()
    with pytest.raises(RuntimeError):
        m.on_request_finished(ctx)
    m.complete_load([key()], ctx)
    m.on_load_failure([key()], ctx)
    assert c.invalidated == [("ns", key())]
    m.on_request_finished(ctx)
    assert not c.live
    m.shutdown()


def test_manager_clean_miss_failed_store_and_owner_isolation():
    m, c = manager()
    a, b = NS(req_id="a"), NS(req_id="b")
    c.miss = True
    assert m.lookup(key(), a) is False
    c.miss = False
    assert m.lookup(key(), a) and m.lookup(key(), b)
    m.complete_load([key()], a)
    assert len(c.live) == 1
    c.miss = True
    out = m.prepare_store([key(2), key(3)], a)
    assert out.keys_to_store == [key(2), key(3)]
    assert out.evicted_keys == []
    m.complete_store(out.keys_to_store, a, success=False)
    assert c.completed == [False, False]
    m.on_request_finished(b)
    assert not c.live


def test_full_prefix_metadata_exceeds_4219_keys_and_nine_gigabytes():
    # Logical queue credit only; this test allocates no payload/staging bytes.
    m, c = manager(capacity=32768, sizes=((2_000_000,),))
    ctx = NS(req_id="full-prefix")
    keys = [i.to_bytes(8, "big") + bytes(4) for i in range(5000)]
    assert all(m.lookup(k, ctx) is True for k in keys)
    assert sum(t.sizes[0] for t in c.live.values()) == 10_000_000_000
    spec = m.prepare_load(keys, ctx)
    assert len(spec.tokens) == 5000
    m.complete_load(keys, ctx)
    assert not c.live


def test_many_keys_stream_through_two_physical_rows():
    p, native, store, trace, _ = pump(rows=2, cap=100)
    try:
        keys = [key(i) for i in range(40)]
        assert transfer(p, 1, (GPU(list(range(40)), [40, 0], [0, 0]), disk(keys)))
        p.get_finished()
        assert len(trace) == 1  # one active row per job, not full-prefix allocation
        assert len(p.free) == 1
        p.wait({1})
        assert p.get_finished()[0].success
        assert len(store.payload) == 40 and len(p.free) == 2
    finally:
        p.shutdown()


@pytest.fixture
def fake_vllm(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    cache_config = NS(prefix_caching_hash_algo="sha256")
    import sys
    import types
    import recipe_persistence.native as module
    from abc import ABC, abstractmethod
    names = ("NodeLocalDiskOffloadingSpec", "NodeLocalDiskManager", "NativeDiskLoadStoreSpec",
             "DiskOffloadingWorker")
    for name in names:
        module.__dict__.pop(name, None)
    # LoadStoreSpec became a plain class and lost medium() (c46ced1ee3, #46544).
    class LoadStoreSpec:
        pass
    class OffloadingSpec(ABC):
        def __init__(self, config):
            self.config = config
            self.extra_config = config.extra_config
            self.blocks_per_chunk = self.extra_config.get("factor",
                                                          config.cache.blocks_per_chunk)
            self.tokens_per_block = (16, 16)
            self.replicated_layout = config.replicated_layout
        @classmethod
        def build_metric_definitions(cls, extra_config):
            return {}
    # OffloadingHandler/get_handlers were replaced by OffloadingWorker/get_worker
    # with an explicit direction (f237e16b41, #45053).
    class OffloadingWorker(ABC):
        @abstractmethod
        def submit_store(self, job_id, src_spec, dst_spec): ...
        @abstractmethod
        def submit_load(self, job_id, src_spec, dst_spec): ...
        @abstractmethod
        def get_finished(self): ...
        @abstractmethod
        def wait(self, job_ids): ...
    # Disk-tier lookup observability: metric metadata from
    # vllm.v1.kv_offload.base and the flat stats container from
    # vllm...offloading.metrics, both consumed lazily by _load_native.
    class OffloadingCounterMetadata:
        def __init__(self, documentation, labelnames=()):
            self.documentation = documentation
            self.labelnames = labelnames
    class OffloadingConnectorStats:
        def __init__(self):
            self.data = {"types": {}, "data": {}}
        def increase_counter(self, name, value=1, labelvalues=()):
            self.data["types"].setdefault(name, "counter")
            values = self.data["data"].setdefault(name, {})
            values[labelvalues] = values.get(labelvalues, 0) + value
        def is_empty(self):
            return not self.data["data"]
    def install(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value
    install("vllm")
    # prefix_caching_hash_algo left OffloadingConfig (a9531edfa6, #48150); it is
    # now read from the ambient vLLM config.
    install("vllm.config",
            get_current_vllm_config=lambda: NS(cache_config=cache_config))
    install("vllm.v1")
    install("vllm.v1.kv_offload")
    install("vllm.v1.kv_offload.base", GPULoadStoreSpec=GPU, LoadStoreSpec=LoadStoreSpec,
            OffloadingSpec=OffloadingSpec, OffloadingManager=ABC,
            OffloadingWorker=OffloadingWorker, LookupResult=NS(MISS=0, HIT=1, HIT_PENDING=2, RETRY=3),
            ScheduleEndContext=NS, TransferResult=Result,
            PrepareStoreOutput=NS, RequestOffloadingContext=NS,
            OffloadingCounterMetadata=OffloadingCounterMetadata)
    install("vllm.distributed")
    install("vllm.distributed.kv_transfer")
    install("vllm.distributed.kv_transfer.kv_connector")
    install("vllm.distributed.kv_transfer.kv_connector.v1")
    install("vllm.distributed.kv_transfer.kv_connector.v1.offloading")
    install("vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics",
            OffloadingConnectorStats=OffloadingConnectorStats)
    install("vllm.v1.kv_offload.cpu")
    install("vllm.v1.kv_offload.cpu.common", CPULoadStoreSpec=CPU)
    allocations, factory_calls = [], []
    def cpu_gpu(kv_caches, blocks_per_chunk, num_cpu_chunks, mmap_region=None,
                canonical_layout=False):
        assert mmap_region is None and not canonical_layout
        allocations.append((blocks_per_chunk, num_cpu_chunks))
        tensors = [[Row(t.page_size_bytes) for _ in range(num_cpu_chunks)]
                   for t in kv_caches.tensors]
        return worker_fake(Copy(tensors, "store", []), Copy(tensors, "load", []))
    install("vllm.v1.kv_offload.cpu.gpu_worker", CPUOffloadingWorker=cpu_gpu)
    install("torch", cuda=NS(synchronize=lambda: None))
    _, coordinator = manager(sizes=((28, 28), (4, 4)))
    def factory(**kwargs):
        factory_calls.append(kwargs)
        return NS(coordinator=coordinator, store=Store(), size_by_group=((28, 28), (4, 4)),
                  max_pending_keys=100, layout_fingerprint="b" * 64)
    install("test_provider", create=factory)
    # OffloadingConfig-shaped input (a9531edfa6, #48150): groups/cache/parallel/
    # model/engine_id/extra_config replace the old (vllm_config, kv_cache_config).
    config = NS(extra_config=dict(
        disk_root="/configured/cache", trusted_single_tenant=True, tenant_namespace="tenant",
        cache_fingerprint="explicit-model-layout-fingerprint", coordinator_module_path="test_provider",
        coordinator_factory="create", staging_bytes=100, staging_rows=20, max_pending_keys=100),
        engine_id="engine", model=NS(name="model", dtype="float16"),
        cache=NS(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=NS(rank=1, world_size=2, data_parallel_size=1),
        replicated_layout=False, canonical_layout=False)
    caches = NS(tensors=[NS(page_size_bytes=16), NS(page_size_bytes=32)],
                group_data_refs=[[NS(tensor_idx=0, page_size_bytes=7), NS(tensor_idx=1, page_size_bytes=21)],
                                 [NS(tensor_idx=0, page_size_bytes=4)]])
    yield module, config, caches, allocations, factory_calls, OffloadingWorker, LoadStoreSpec
    for name in names:
        module.__dict__.pop(name, None)


def tracked_factory(spec, closed, mutate=None):
    original = spec.factory
    def factory(**kwargs):
        provider = original(**kwargs)
        provider.close = lambda: closed.append(kwargs["role"])
        if mutate is not None:
            mutate(provider)
        return provider
    spec.factory = factory


@pytest.mark.parametrize("layout", ["", "a" * 63, "A" * 64, "g" * 64, None])
def test_scheduler_requires_valid_layout_digest_and_cleans_provider(fake_vllm, layout):
    module, config, *_ = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    closed = []
    tracked_factory(spec, closed, lambda p: setattr(p, "layout_fingerprint", layout))
    with pytest.raises(ValueError, match="layout_fingerprint"):
        spec.get_manager()
    assert closed == ["scheduler"]


def test_worker_checks_supplied_layout_digest(fake_vllm):
    module, config, caches, allocations, *_ = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    closed = []
    tracked_factory(spec, closed, lambda p: setattr(p, "layout_fingerprint", "invalid"))
    with pytest.raises(ValueError, match="layout_fingerprint"):
        spec.get_worker(caches)
    assert closed == ["worker"] and allocations == []


def test_namespace_binds_ref_order_even_with_identical_byte_sizes(fake_vllm):
    import hashlib
    import json
    module, config, caches, *_ = fake_vllm
    first_geometry = Geometry.from_canonical(caches)
    caches.group_data_refs[0].reverse()
    reordered_geometry = Geometry.from_canonical(caches)
    assert first_geometry.group_bytes == reordered_geometry.group_bytes
    assert first_geometry.padded_pages == reordered_geometry.padded_pages
    assert first_geometry.group_refs != reordered_geometry.group_refs
    namespaces = []
    for geometry in (first_geometry, reordered_geometry):
        # The provider computes the census digest; native consumes its opaque,
        # validated value rather than guessing a worker layout from byte totals.
        census = [(geometry.padded_pages, geometry.group_refs)] * 2
        digest = hashlib.sha256(json.dumps(census, separators=(",", ":")).encode()).hexdigest()
        spec = module.NodeLocalDiskOffloadingSpec(config)
        tracked_factory(spec, [], lambda p: setattr(p, "layout_fingerprint", digest))
        manager = spec.get_manager()
        namespaces.append(manager.namespace)
        manager.shutdown()
    assert namespaces[0] != namespaces[1]


def test_provider_closes_once_and_reset_does_not_close(fake_vllm):
    module, config, caches, *_ = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    closed = []
    tracked_factory(spec, closed)
    m = spec.get_manager()
    m.reset_cache()
    assert closed == []
    worker = spec.get_worker(caches)
    m.shutdown()
    m.shutdown()
    assert closed == ["scheduler"]
    worker.shutdown()
    assert closed == ["scheduler", "worker"]


def test_manager_shutdown_keeps_provider_until_prepared_load_drains():
    from recipe_persistence.handlers import _CloseOnce
    m, _ = manager()
    closed = []
    m._close_provider = _CloseOnce(lambda: closed.append(True))
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    m.prepare_load([key()], ctx)
    with pytest.raises(RuntimeError, match="active"):
        m.shutdown()
    assert closed == []
    m.complete_load([key()], ctx)
    m.shutdown()
    m.shutdown()
    assert closed == [True]


def test_pump_unsafe_shutdown_does_not_close_provider():
    from recipe_persistence.handlers import _CloseOnce
    p, native, _, _, _ = pump()
    closed = []
    p._close_provider = _CloseOnce(lambda: closed.append(True))
    transfer(p, 1, (GPU([1], [1, 0], [0, 0]), disk([key()])))
    p.get_finished()
    original_wait = native._store_handler.wait
    def fail(*args):
        raise RuntimeError("unsafe drain")
    native._store_handler.wait = fail
    p.drain_cuda = fail
    with pytest.raises(RuntimeError, match="unsafe"):
        p.shutdown()
    assert closed == [] and p.jobs and not p.free
    native._store_handler.wait = original_wait
    p.shutdown()
    p.shutdown()
    assert closed == [True]
    assert not p.get_finished()[0].success


@pytest.mark.parametrize("failure", ["geometry", "queue", "missing_store", "missing_coordinator"])
def test_startup_validation_closes_provider(fake_vllm, failure):
    module, config, caches, allocations, *_ = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    closed = []
    def mutate(provider):
        if failure == "geometry":
            provider.size_by_group = ((1, 1), (4, 4))
        elif failure == "queue":
            provider.max_pending_keys = 99
        elif failure == "missing_store":
            provider.store = None
        else:
            provider.coordinator = None
    tracked_factory(spec, closed, mutate)
    with pytest.raises(ValueError):
        if failure == "missing_coordinator":
            spec.get_manager()
        else:
            spec.get_worker(caches)
    assert closed == ["scheduler" if failure == "missing_coordinator" else "worker"]
    assert not allocations


@pytest.mark.parametrize("failure", ["allocation", "pump_constructor"])
def test_worker_startup_failure_closes_native_and_provider(fake_vllm, monkeypatch, failure):
    import sys
    module, config, caches, *_ = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    closed, allocated = [], []
    tracked_factory(spec, closed)
    gpu_worker = sys.modules["vllm.v1.kv_offload.cpu.gpu_worker"]
    original = gpu_worker.CPUOffloadingWorker
    def allocate(*args, **kwargs):
        if failure == "allocation":
            raise RuntimeError("allocation failed")
        native = original(*args, **kwargs)
        allocated.append(native)
        return native
    monkeypatch.setattr(gpu_worker, "CPUOffloadingWorker", allocate)
    if failure == "pump_constructor":
        def broken(*args, **kwargs):
            raise RuntimeError("pump constructor failed")
        monkeypatch.setattr(DiskTransferPump, "__init__", broken)
    with pytest.raises(RuntimeError):
        spec.get_worker(caches)
    assert closed == ["worker"]
    for native in allocated:
        assert native._store_handler.closed and native._load_handler.closed


def test_unsafe_startup_cleanup_retains_provider_and_native(fake_vllm, monkeypatch):
    import sys
    module, config, caches, *_ = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    closed = []
    tracked_factory(spec, closed)
    gpu_worker = sys.modules["vllm.v1.kv_offload.cpu.gpu_worker"]
    original = gpu_worker.CPUOffloadingWorker
    def fail(*args, **kwargs):
        raise RuntimeError("unsafe startup drain")
    def allocate(*args, **kwargs):
        native = original(*args, **kwargs)
        native._store_handler.shutdown = fail
        return native
    monkeypatch.setattr(gpu_worker, "CPUOffloadingWorker", allocate)
    monkeypatch.setattr(DiskTransferPump, "__init__", fail)
    with pytest.raises(RuntimeError, match="unsafe"):
        spec.get_worker(caches)
    assert closed == [] and spec._startup_cleanup is not None
    with pytest.raises(RuntimeError, match="previous native startup"):
        spec.get_worker(caches)
    native, close = spec._startup_cleanup
    native._store_handler.shutdown = lambda: None
    spec._cleanup_startup(native, close)
    assert closed == ["worker"] and spec._startup_cleanup is None


def test_pump_provider_close_follows_disk_and_both_native_shutdowns():
    from recipe_persistence.handlers import _CloseOnce
    p, native, store, _, _ = pump()
    closed = []
    def close():
        assert not p.jobs
        assert native._store_handler.closed and native._load_handler.closed
        assert store.calls == [("write", "t0")]
        closed.append(True)
    p._close_provider = _CloseOnce(close)
    transfer(p, 1, (GPU([1], [1, 0], [0, 0]), disk([key()])))
    p.get_finished()
    native._store_handler.done = True
    p.get_finished()
    assert store.started.wait(1)
    p.shutdown()
    p.shutdown()
    assert closed == [True]


@pytest.mark.parametrize("failure", ["exception", "rejection"])
def test_manager_close_failure_is_nonsecret_and_retried_until_ack(caplog, failure):
    from recipe_persistence.handlers import _CloseOnce
    m, _ = manager()
    attempts = []
    def close():
        attempts.append(True)
        if len(attempts) == 1:
            if failure == "exception":
                raise OSError("PRIVATE-CLOSE-DETAIL")
            return False
        return True if failure == "exception" else None
    m._close_provider = _CloseOnce(close)
    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="has not drained") as error:
            m.shutdown()
        assert "PRIVATE-CLOSE-DETAIL" not in str(error.value)
        assert not m._shutdown_complete and m._close_provider.callback is not None
        m.shutdown()
        m.shutdown()
    assert attempts == [True, True] and m.degraded_reason == "provider_close_failed"
    assert m._shutdown_complete and m._close_provider.callback is None
    assert "PRIVATE-CLOSE-DETAIL" not in caplog.text
    assert sum(r.name == "recipe_persistence.native" for r in caplog.records) == 1


def test_pump_retries_provider_close_after_unproved_drain():
    from recipe_persistence.handlers import _CloseOnce
    p, native, _, _, _ = pump()
    attempts = []
    def close():
        attempts.append(True)
        return False if len(attempts) == 1 else None
    p._close_provider = _CloseOnce(close)
    with pytest.raises(RuntimeError, match="has not drained"):
        p.shutdown()
    assert not p._shutdown_complete and p._close_provider.callback is not None
    assert native._store_handler.closed and native._load_handler.closed
    p.shutdown()
    p.shutdown()
    assert attempts == [True, True] and p._shutdown_complete


def test_startup_provider_close_failure_retains_retry_callback(fake_vllm):
    module, config, *_ = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    attempts = []
    original = spec.factory
    def factory(**kwargs):
        provider = original(**kwargs)
        provider.max_pending_keys = 99
        def close():
            attempts.append(True)
            return False if len(attempts) == 1 else None
        provider.close = close
        return provider
    spec.factory = factory
    with pytest.raises(RuntimeError, match="startup provider cleanup has not drained"):
        spec.get_manager()
    assert spec._startup_cleanup is not None
    spec._cleanup_startup(*spec._startup_cleanup)
    assert attempts == [True, True] and spec._startup_cleanup is None


def test_manager_degradation_logs_static_reason_once(caplog):
    m, c = manager()
    secret = "PRIVATE-KEY-TOKEN-AND-PATH"
    def broken(*args):
        raise OSError(secret)
    c.invalidate = broken
    with caplog.at_level("WARNING"):
        m.on_load_failure([key()], NS(req_id=secret))
        m.on_load_failure([key(1)], NS(req_id=secret))
    assert m.degraded_reason == "coordinator_callback_failed"
    records = [r for r in caplog.records if r.name == "recipe_persistence.native"]
    assert len(records) == 1
    assert records[0].getMessage() == "Persistence disabled: coordinator_callback_failed"
    assert secret not in caplog.text


@pytest.mark.parametrize("method,reason", [
    ("renew", "lease_renewal_lost"), ("invalidate", "coordinator_callback_rejected")])
def test_manager_degradation_reason_for_explicit_rejection(method, reason):
    m, c = manager()
    setattr(c, method, lambda *args: False)
    assert m._notify(method, object()) is False
    assert m.closed and m.degraded_reason == reason


@pytest.mark.parametrize("field,message", [
    ("canonical_layout", "canonical host layout"),
    ("replicated_layout", "replicated host layout"),
])
def test_native_refuses_shared_host_layouts(fake_vllm, field, message):
    """Both shared-page layouts would silently drop this rank's bytes.

    canonical_layout (2c4d348848, #48408) elects one writer per canonical page;
    replicated_layout makes only rank 0 a store writer. Every rank owns its own
    node-local objects here, so both must be refused, not adapted.
    """
    module, config, *_ = fake_vllm
    setattr(config, field, True)
    with pytest.raises(ValueError, match=message):
        module.NodeLocalDiskOffloadingSpec(config)


def test_manager_adapts_tristate_lookup_and_step_hook_to_native_abc(fake_vllm, monkeypatch):
    """The two OffloadingManager signature changes are absorbed in the subclass.

    lookup returns LookupResult since bb61177e49 (#46363) and on_schedule_end
    takes a ScheduleEndContext since 0fc2512094 (#46450). _Manager keeps its
    vLLM-free tri-state and no-argument forms.
    """
    import sys
    from recipe_persistence.native import _Manager
    module, config, *_ = fake_vllm
    result = sys.modules["vllm.v1.kv_offload.base"].LookupResult
    m = module.NodeLocalDiskOffloadingSpec(config).get_manager()
    ctx = NS(req_id="owner")
    for value, expected in ((True, result.HIT), (None, result.RETRY), (False, result.MISS)):
        monkeypatch.setattr(_Manager, "lookup", lambda self, k, c, v=value: v)
        assert m.lookup(key(), ctx) == expected
    # RETRY, never HIT_PENDING: an unproven lease must not admit a prefix.
    monkeypatch.setattr(_Manager, "lookup", lambda self, k, c: None)
    assert m.lookup(key(), ctx) != result.HIT_PENDING
    seen = []
    monkeypatch.setattr(_Manager, "on_schedule_end", lambda self: seen.append(True))
    m.on_schedule_end(NS(new_req_ids=("r",), preempted_req_ids=()))
    m.on_schedule_end()
    assert seen == [True, True]
    m.shutdown()


def test_disk_tier_lookup_counters_emit_registered_tiering_metrics(fake_vllm):
    """A disk-served prefix is published under the upstream tiering names.

    Registration is what OffloadPromMetrics.observe() asserts against, so the
    spec class must declare both names with the tier label. get_stats() reports
    per-interval deltas only and stays None (no empty series) while idle.
    """
    from recipe_persistence.native import (
        DISK_TIER_LABEL, TIER_CHUNK_HITS_METRIC, TIER_CHUNK_QUERIES_METRIC,
    )
    module, config, *_ = fake_vllm
    definitions = module.NodeLocalDiskOffloadingSpec.build_metric_definitions(
        config.extra_config)
    assert set(definitions) == {TIER_CHUNK_QUERIES_METRIC, TIER_CHUNK_HITS_METRIC}
    assert all(d.labelnames == ("tier",) for d in definitions.values())

    _, coordinator = manager(sizes=((28,),))
    native = module.NodeLocalDiskManager(coordinator, "ns", ((28,),), 8,
                                         DiskLoadStoreSpec, NS, NS)
    assert native.get_stats() is None
    ctx = NS(req_id="owner")
    assert native.lookup(key(), ctx)  # HIT
    stats = native.get_stats()
    assert stats.data["data"][TIER_CHUNK_QUERIES_METRIC] == {(DISK_TIER_LABEL,): 1}
    assert stats.data["data"][TIER_CHUNK_HITS_METRIC] == {(DISK_TIER_LABEL,): 1}
    assert native.get_stats() is None  # deltas consumed, idle

    coordinator.miss = True
    assert not native.lookup(key(1), ctx)  # MISS
    stats = native.get_stats()
    assert stats.data["data"][TIER_CHUNK_QUERIES_METRIC] == {(DISK_TIER_LABEL,): 1}
    assert TIER_CHUNK_HITS_METRIC not in stats.data["data"]
    native.shutdown()


def test_native_external_factory_and_picklable_metadata(fake_vllm):
    import pickle
    module, config, caches, allocations, calls, worker_cls, metadata_cls = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    manager = spec.get_manager()
    # get_worker replaced the two-yield get_handlers route table with a single
    # bidirectional worker (f237e16b41, #45053); one shared bounded ring keeps
    # loads and stores competing fairly, as the route table did.
    worker = spec.get_worker(caches)
    assert spec.get_worker(caches) is worker
    assert isinstance(worker, worker_cls)
    assert issubclass(module.NativeDiskLoadStoreSpec, metadata_cls)
    assert allocations == [(1, 2)]  # 48 physical bytes/row, not unpadded 28+4
    assert [c["role"] for c in calls] == ["scheduler", "worker"]
    assert calls[1]["rank"] == 1 and calls[1]["geometry"].group_bytes == (28, 4)
    assert manager.capacity == 100 and manager.lookup_keys_per_step == 8
    assert manager._async_metadata and manager._metadata_limit == 8
    manager.shutdown()
    metadata = module.NativeDiskLoadStoreSpec((key(),), (("rank0", "rank1"),), "ns", "owner")
    assert pickle.loads(pickle.dumps(metadata)) == metadata
    worker.shutdown()


@pytest.mark.parametrize("field,value", [
    ("factor", 2), ("coordinator_module_path", ""), ("coordinator_factory", ""),
    ("cache_fingerprint", ""), ("tenant_namespace", ""), ("trusted_single_tenant", False),
    ("disk_root", "relative/path"), ("staging_bytes", 0),
    ("lookup_keys_per_step", 0), ("lookup_keys_per_step", -1),
    ("lookup_keys_per_step", True), ("lookup_keys_per_step", 1.5),
    ("lookup_keys_per_step", None),
    ("metadata_workers", 0), ("metadata_workers", 33),
    ("metadata_max_submitted", 1), ("metadata_max_submitted", 4097),
    ("metadata_shutdown_timeout", 0), ("metadata_shutdown_timeout", float("nan")),
])
def test_native_fail_closed_config(fake_vllm, field, value):
    module, config, *_ = fake_vllm
    config.extra_config[field] = value
    with pytest.raises(ValueError):
        module.NodeLocalDiskOffloadingSpec(config)


def test_native_async_mode_requires_non_io_lease_proof_accessor(fake_vllm):
    module, config, *_ = fake_vllm
    spec = module.NodeLocalDiskOffloadingSpec(config)
    closed = []
    tracked_factory(spec, closed, lambda p: setattr(p.coordinator, "lease_deadline", None))
    with pytest.raises(ValueError, match="lease_deadline"):
        spec.get_manager()
    assert closed == ["scheduler"]


def test_native_lookup_budget_can_be_explicitly_configured(fake_vllm):
    module, config, *_ = fake_vllm
    config.extra_config["lookup_keys_per_step"] = 3
    spec = module.NodeLocalDiskOffloadingSpec(config)
    manager = spec.get_manager()
    assert manager.lookup_keys_per_step == 3
    manager.shutdown()


def test_native_rejects_geometry_mismatch_before_allocating(fake_vllm):
    module, config, caches, allocations, *_ = fake_vllm
    caches.group_data_refs[0][0].page_size_bytes = 6
    spec = module.NodeLocalDiskOffloadingSpec(config)
    with pytest.raises(ValueError, match="canonical"):
        spec.get_worker(caches)
    assert not allocations


@pytest.mark.parametrize("seed", ["", "random", "-1", "4294967296", "1.5"])
def test_native_requires_fixed_startup_hash_seed(fake_vllm, monkeypatch, seed):
    module, config, *_ = fake_vllm
    monkeypatch.setenv("PYTHONHASHSEED", seed)
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        module.NodeLocalDiskOffloadingSpec(config)


def test_namespace_changes_with_hash_seed_and_rejects_other_hash(fake_vllm, monkeypatch):
    module, config, *_ = fake_vllm
    first = module.NodeLocalDiskOffloadingSpec(config).get_manager().namespace
    monkeypatch.setenv("PYTHONHASHSEED", "456")
    second = module.NodeLocalDiskOffloadingSpec(config).get_manager().namespace
    assert first != second
    monkeypatch.setenv("PYTHONHASHSEED", "0456")
    padded = module.NodeLocalDiskOffloadingSpec(config).get_manager().namespace
    assert padded != second  # ab666 hashes the raw string, not numeric seed
    import sys
    sys.modules["vllm.config"].get_current_vllm_config().cache_config.prefix_caching_hash_algo = "xxhash"
    with pytest.raises(ValueError, match="sha256"):
        module.NodeLocalDiskOffloadingSpec(config)


def test_confirmed_durable_skip_without_releasing_an_active_load():
    m, c = manager()
    c.renew = lambda ticket: ticket.token in c.live
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    m.prepare_load([key()], ctx)
    out = m.prepare_store([key(), key(1)], ctx)
    assert out.keys_to_store == [] and out.skipped_keys == [key(), key(1)]
    assert out.store_spec.keys == ()
    assert len(c.live) == 1
    m.complete_load([key()], ctx)
    assert not c.live


@pytest.mark.parametrize("method", ["release", "complete_store", "invalidate", "renew"])
def test_callback_transport_failure_disables_admission_without_raising(method):
    m, c = manager()
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    m.prepare_load([key()], ctx)
    c.miss = True
    out = m.prepare_store([key(1)], ctx)
    def broken(*args):
        raise OSError("coordinator unavailable")
    setattr(c, method, broken)
    if method == "release":
        m.complete_load([key()], ctx)
    elif method == "complete_store":
        m.complete_store(out.keys_to_store, ctx, success=False)
    elif method == "invalidate":
        m.complete_load([key()], ctx)
        m.on_load_failure([key()], ctx)
    else:
        m.on_schedule_end()
    assert m.closed
    assert m.lookup(key(2), ctx) is False
    # Transport failure never frees active prepared ownership early.
    if method in {"renew", "complete_store"}:
        assert (ctx.req_id, key()) in m.prepared


@pytest.mark.parametrize("method", ["release", "complete_store", "invalidate"])
def test_callback_explicit_rejection_disables_admission(method):
    m, c = manager()
    setattr(c, method, lambda *args: False)
    assert m._notify(method, object()) is False
    assert m.closed


def test_cached_lookup_expired_lease_is_not_readvertised():
    m, c = manager()
    ctx = NS(req_id="owner")
    assert m.lookup(key(), ctx)
    c.renew = lambda ticket: False
    assert m.lookup(key(), ctx) is False
    assert not m.loads and not c.live


def test_load_holds_row_until_gpu_completion():
    p, native, store, trace, _ = pump()
    try:
        store.payload["t0"] = b"a" * 28
        transfer(p, 1, (disk([key()]), GPU([9], [1, 0], [0, 0])))
        p.get_finished()
        job = p.jobs[1]
        job.future.result(timeout=1)
        assert p.get_finished() == []
        assert native._load_handler.active and not p.free
        assert trace[0][0] == "load"
        p.wait({1})
        assert p.get_finished()[0].success
        assert len(p.free) == 1
    finally:
        p.shutdown()


def test_unsafe_cuda_drain_failure_retains_row_and_job():
    p, native, _, _, _ = pump()
    native._store_handler.throw = True
    def failed_drain():
        raise RuntimeError("device cannot drain")
    p.drain_cuda = failed_drain
    transfer(p, 1, (GPU([1], [1, 0], [0, 0]), disk([key()])))
    with pytest.raises(RuntimeError, match="drain"):
        p.get_finished()
    assert not p.free and 1 in p.jobs and not p.finished
    native._store_handler.throw = False
    p.drain_cuda = lambda: None
    p.shutdown()
    assert not p.get_finished()[0].success


@pytest.mark.parametrize("metadata_workers", [0, 2])
def test_two_rank_diskstore_manager_and_handler_roundtrip(tmp_path, metadata_workers):
    from recipe_persistence.coordinator import LocalCoordinator
    from recipe_persistence.native import _Manager
    from recipe_persistence.storage import DiskStore, Limits
    limits = Limits(quota=1_000_000, high=900_000, low=800_000,
                    index_bytes=65536, max_objects=8, free_bytes=0, free_inodes=0)
    stores = [DiskStore(tmp_path / ("rank" + str(rank)), limits) for rank in range(2)]
    pumps = [pump()[0] for _ in range(2)]
    sizes = ((28, 28), (4, 4))
    coordinator = LocalCoordinator(stores, sizes, max_pending_keys=8)
    namespace = "a" * 64
    m = _Manager(coordinator, namespace, sizes, 8, DiskLoadStoreSpec, NS, NS,
                 metadata_workers=metadata_workers)
    ctx = NS(req_id="write")
    try:
        out = m.prepare_store([key()], ctx)
        assert out.keys_to_store == [key()] and not out.skipped_keys
        for rank, p in enumerate(pumps):
            p.rank, p.store = rank, stores[rank]
            p.cpu_tensors[0][0].data[:] = bytes([rank + 1]) * 16
            p.cpu_tensors[1][0].data[:] = bytes([rank + 11]) * 32
            assert transfer(p, 1, (GPU([4], [1, 0], [0, 0]), out.store_spec))
            p.wait({1})
            assert p.get_finished()[0].success
        m.complete_store([key()], ctx)
        assert all(s.exists(namespace, key()) for s in stores)
        # A fresh manager discovers disk through all-rank reservations, not a
        # scheduler-local residency map; only the refs' unpadded bytes restore.
        restored = _Manager(coordinator, namespace, sizes, 8, DiskLoadStoreSpec, NS, NS,
                            metadata_workers=metadata_workers)
        read_ctx = NS(req_id="read")
        assert restored.lookup(key(), read_ctx) is True
        metadata = restored.prepare_load([key()], read_ctx)
        for rank, p in enumerate(pumps):
            for tensor in p.cpu_tensors:
                tensor[0].data[:] = b"x" * len(tensor[0].data)
            assert transfer(p, 2, (metadata, GPU([8], [1, 0], [0, 0])))
            p.wait({2})
            assert p.get_finished()[0].success
            assert p.cpu_tensors[0][0].data == bytes([rank + 1]) * 7 + b"x" * 9
            assert p.cpu_tensors[1][0].data == bytes([rank + 11]) * 21 + b"x" * 11
        restored.complete_load([key()], read_ctx)
        restored.shutdown()
    finally:
        for p in pumps:
            p.shutdown()
        m.shutdown()
        for store in stores:
            store.close()


def test_byte_deficit_fairness_small_loads_cannot_starve_stores():
    p, _, store, trace, _ = pump(rows=1, cap=100)
    try:
        big = [key(i) for i in range(20)]
        small = [key(i, 1) for i in range(60)]
        store.payload.update({"s" + str(i): b"abcd" for i in range(60)})
        assert transfer(p, 1, (GPU(list(range(20)), [20, 0], [0, 0]), disk(big, "b")))
        assert transfer(p, 2, (disk(small, "s"), GPU(list(range(60)), [0, 60], [0, 0])))
        p.wait({1, 2})
        assert all(r.success for r in p.get_finished())
        run = 0
        for direction, _, _ in trace:
            run = run + 1 if direction == "load" else 0
            assert run * 4 <= p.quantum
        assert [direction for direction, _, _ in trace[:2]] == ["store", "load"]
        assert len(store.calls) == 80
    finally:
        p.shutdown()


def test_rejects_gpu_out_of_bounds_and_fractional_ids():
    p, _, _, trace, _ = pump()
    try:
        for block in (-1, 10000, 0.5):
            assert not transfer(p, 1, (GPU([block], [1, 0], [0, 0]), disk([key()])))
        assert not trace and len(p.free) == 1
    finally:
        p.shutdown()
