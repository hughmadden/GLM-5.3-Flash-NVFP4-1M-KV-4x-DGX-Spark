"""Adversarial concurrency tests for recipe_persistence.handlers / .native.

STDLIB ONLY. No pytest, no torch, no vLLM, no engine, no network. Every object the
pump or the manager needs is a duck-typed fake defined below; `vllm`/`torch` are
never imported (the "absent vLLM" refusal path is simulated with a meta_path
finder, see `block_vllm`).

Surfaces hammered:
  * handlers.DiskTransferPump  -- submit_store/submit_load, wait, cancel, shutdown,
    _row_done, get_finished, rows/max_pending_keys back-pressure, fault paths.
  * handlers._CloseOnce        -- concurrent/idempotent cleanup.
  * native._Manager            -- load/store accounting, capacity bounds, leases,
    lookups, metadata drain, quarantine.
  * native._prefix_hash_algorithm / _load_native / __getattr__ with vLLM blocked.

Tests assert the CORRECT behaviour, so a genuine defect shows up as a FAIL whose
message carries the observed-vs-expected evidence.
"""
from __future__ import annotations

import importlib.abc
import logging
import math
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import conc_stubs as S
from recipe_persistence.handlers import (
    DiskLoadStoreSpec,
    DiskTransferPump,
    _CloseOnce,
)

logging.getLogger("recipe_persistence").setLevel(logging.CRITICAL)

NAMESPACE = S.dummy_namespace("conc")
GROUP_SIZES = (16, 32, 64)          # 3 cache groups, one canonical tensor
ROW_BYTES = 256

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def gkey(group: int, tag: bytes) -> bytes:
    """A key whose trailing 4 bytes select `group` (see geometry.key_group)."""
    return b"k:" + tag + b"|" + group.to_bytes(4, "big")


class FakeGeometry:
    def __init__(self, group_sizes=GROUP_SIZES, row_bytes=ROW_BYTES):
        # group_refs is keyed by group; each group is a tuple of (tensor_idx, size)
        # references, exactly like geometry.Geometry.from_canonical builds it.
        self.group_refs = tuple(((0, size),) for size in group_sizes)
        self.group_bytes = tuple(group_sizes)
        self.row_bytes = row_bytes

    def buffers(self, cpu_tensors, row, group):
        return [memoryview(bytearray(size)) for _, size in self.group_refs[group]]


class FakeGpuSpec:
    def __init__(self, block_ids, group_sizes, block_indices):
        self.block_ids = tuple(block_ids)
        self.group_sizes = tuple(group_sizes)
        self.block_indices = tuple(block_indices)


class FakeCpuSpec:
    def __init__(self, rows):
        self.rows = tuple(rows)


class FakeResult:
    def __init__(self, job_id, success, transfer_size, transfer_time):
        self.job_id = job_id
        self.success = success
        self.transfer_size = transfer_size
        self.transfer_time = transfer_time


class _Shape:
    def __init__(self, n=1 << 20):
        self.shape = (n,)


class FakeDirectionalHandler:
    def __init__(self, native, direction):
        self._native = native
        self.direction = direction
        self.src_tensors = [_Shape()]
        self.dst_tensors = [_Shape()]
        self.waits = 0

    def transfer_async(self, native_id, src, dst):
        return self._native.on_transfer(self, native_id, src, dst)

    def get_finished(self):
        native = self._native
        if native.poll_error:
            raise RuntimeError("injected get_finished failure")
        out = list(native.finished[self.direction])
        native.finished[self.direction].clear()
        return out

    def wait(self, ids):
        self.waits += 1
        return self._native.wait(self, ids)

    def shutdown(self):
        self._native.shutdown_calls.append(self.direction)


class FakeNative:
    """Duck-typed CPUOffloadingWorker: _store_handler/_load_handler + lifecycle."""

    def __init__(self, mode="immediate", reject=False, raise_submit=False,
                 poll_error=False, result_success=True, on_complete=None):
        self.mode = mode                      # "immediate" | "deferred"
        self.reject = reject
        self.raise_submit = raise_submit
        self.poll_error = poll_error
        self.result_success = result_success
        self.on_complete = on_complete
        self._store_handler = FakeDirectionalHandler(self, "store")
        self._load_handler = FakeDirectionalHandler(self, "load")
        self.finished = {"store": [], "load": []}
        self.pending = {}                     # native_id -> handler
        self.submitted = []
        self.drains = 0
        self.shutdown_calls = []
        self.max_pending = 0

    def on_transfer(self, handler, native_id, src, dst):
        if self.raise_submit:
            raise RuntimeError("injected native submit failure")
        if self.reject:
            return False
        self.submitted.append(native_id)
        self.pending[native_id] = handler
        self.max_pending = max(self.max_pending, len(self.pending))
        if self.mode == "immediate":
            self.finish(native_id, self.result_success)
        return True

    def finish(self, native_id, success):
        handler = self.pending.pop(native_id, None)
        if handler is None:
            return False
        self.finished[handler.direction].append(FakeResult(native_id, bool(success), 0, 0.0))
        if self.on_complete is not None:
            self.on_complete(native_id)
        return True

    def finish_all(self, success=None):
        value = self.result_success if success is None else success
        for native_id in list(self.pending):
            self.finish(native_id, value)

    def wait(self, handler, ids):
        if self.mode == "deferred":
            for native_id in list(ids):
                if native_id in self.pending and self.pending[native_id] is handler:
                    self.finish(native_id, self.result_success)

    def drain(self):
        self.drains += 1

    def shutdown(self):
        self.shutdown_calls.append("native")


class FakeStore:
    """Instant node-local store; optional latency/fault injection, concurrency probe."""

    def __init__(self, fail_after=None, raise_after=None, delay=0.0):
        self.fail_after = fail_after
        self.raise_after = raise_after
        self.delay = delay
        self.writes = 0
        self.reads = 0
        self.max_concurrent = 0
        self._live = 0
        self._lock = threading.Lock()

    def _run(self):
        with self._lock:
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
        try:
            if self.delay:
                time.sleep(self.delay)
            return True
        finally:
            with self._lock:
                self._live -= 1

    def write(self, token, buffers):
        self.writes += 1
        if self.raise_after is not None and self.writes > self.raise_after:
            raise RuntimeError("injected disk failure")
        if self.fail_after is not None and self.writes > self.fail_after:
            return False
        return self._run()

    def read_into(self, token, buffers):
        self.reads += 1
        if self.raise_after is not None and self.reads > self.raise_after:
            raise RuntimeError("injected disk failure")
        if self.fail_after is not None and self.reads > self.fail_after:
            return False
        return self._run()


def make_pump(rows=4, cap=64, io_threads=1, native=None, store=None, geometry=None,
              close_provider=None, rank=0):
    native = native or FakeNative(mode="immediate")
    store = store or FakeStore()
    geometry = geometry or FakeGeometry()
    pump = DiskTransferPump(
        geometry=geometry, cpu_gpu=native, store=store, rank=rank, rows=rows,
        max_pending_keys=cap, gpu_spec_cls=FakeGpuSpec, cpu_spec_cls=FakeCpuSpec,
        result_cls=FakeResult, drain_cuda=native.drain, io_threads=io_threads,
        close_provider=close_provider)
    return pump, native, store, geometry


def specs(pump, groups, tag, ns=None, owner="r0", block0=0):
    """Build a (gpu_spec, disk_spec) pair whose group sequence is `groups` (sorted)."""
    groups = tuple(groups)
    keys = tuple(gkey(g, tag + b"-%d" % i) for i, g in enumerate(groups))
    tokens = tuple((f"tok-{tag.decode()}-{i}",) for i in range(len(groups)))
    disk = DiskLoadStoreSpec(keys, tokens, ns or NAMESPACE, owner)
    counts = [0] * len(pump.geometry.group_refs)
    for group in groups:
        counts[group] += 1
    gpu = FakeGpuSpec(list(range(block0, block0 + len(groups))), counts, [0] * len(counts))
    return gpu, disk


def submit(pump, job_id, direction="store", groups=(0,), ns=None, owner="r0", block0=0):
    gpu, disk = specs(pump, groups, b"j%d" % job_id, ns=ns, owner=owner, block0=block0)
    if direction == "store":
        return pump.submit_store(job_id, gpu, disk)
    return pump.submit_load(job_id, disk, gpu)


def collect(pump, expected, deadline=8.0):
    """Drain get_finished until `expected` ids are seen or the deadline passes."""
    seen = {}
    end = time.monotonic() + deadline
    while set(expected) - set(seen) and time.monotonic() < end:
        for result in pump.get_finished():
            seen[result.job_id] = result
        if set(expected) - set(seen):
            time.sleep(0.0005)
    return seen


def safe_shutdown(pump):
    try:
        pump.shutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# _CloseOnce
# ---------------------------------------------------------------------------


def test_close_once_concurrent_calls_fire_callback_once():
    calls = []
    close = _CloseOnce(lambda: calls.append(1))
    results, errors, alive = S.run_threads([close, close, close, close], timeout=10.0)
    assert not any(errors), f"concurrent close raised: {errors[0]!r}"
    assert not any(alive), "close_once hung under concurrent calls"
    assert calls == [1], f"cleanup callback fired {len(calls)} times, expected exactly once"


def test_close_once_non_callable_rejected():
    try:
        _CloseOnce(42)
    except ValueError:
        return
    raise AssertionError("_CloseOnce accepted a non-callable provider close")


def test_close_once_failure_is_retryable_and_not_acknowledged():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("provider busy")
        return True

    close = _CloseOnce(flaky)
    try:
        close()
    except RuntimeError as exc:
        assert "not drained" in str(exc), f"unexpected error: {exc}"
    else:
        raise AssertionError("failed cleanup was acknowledged")
    close()  # must retry, not raise "already closed"
    assert state["n"] == 2, f"retry did not re-invoke the callback (n={state['n']})"


# ---------------------------------------------------------------------------
# DiskTransferPump construction / validation
# ---------------------------------------------------------------------------


def test_pump_constructor_bounds():
    for kwargs in (dict(rows=0), dict(cap=1, rows=4), dict(io_threads=0), dict(rank=-1)):
        try:
            make_pump(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"constructor accepted {kwargs}")
    try:
        make_pump(close_provider=42)
    except ValueError:
        return
    raise AssertionError("constructor accepted a non-callable close_provider")


def test_pump_requires_native_staging_handlers():
    bad = SimpleNamespace()  # no _store_handler/_load_handler
    try:
        DiskTransferPump(geometry=FakeGeometry(), cpu_gpu=bad, store=FakeStore(), rank=0,
                         rows=2, max_pending_keys=4, gpu_spec_cls=FakeGpuSpec,
                         cpu_spec_cls=FakeCpuSpec, result_cls=FakeResult,
                         drain_cuda=lambda: None)
    except ValueError as exc:
        assert "staging API" in str(exc), f"unhelpful message: {exc}"
        return
    raise AssertionError("pump accepted a native worker without staging handlers")


def test_pump_rejects_malformed_specs():
    pump, _, _, _ = make_pump(rows=2, cap=8)
    try:
        good_gpu, good_disk = specs(pump, (0,), b"ok")
        cases = [
            ("wrong gpu class", SimpleNamespace(block_ids=(0,), group_sizes=(1,), block_indices=(0,)), good_disk),
            ("non-disk metadata", good_gpu, SimpleNamespace(keys=good_disk.keys, tokens=good_disk.tokens,
                                                            namespace=NAMESPACE, owner="r0")),
            ("empty keys", FakeGpuSpec((), (0,), (0,)), DiskLoadStoreSpec((), (), NAMESPACE, "r0")),
            ("token/rank mismatch", good_gpu, DiskLoadStoreSpec(good_disk.keys, ((),), NAMESPACE, "r0")),
            ("blank owner", good_gpu, DiskLoadStoreSpec(good_disk.keys, good_disk.tokens, NAMESPACE, "")),
            ("unknown group", good_gpu, DiskLoadStoreSpec(
                (gkey(9, b"bad"),), (("t",),), NAMESPACE, "r0")),
            ("negative block", FakeGpuSpec((-1,), (1,), (0,)), good_disk),
            ("block beyond tensor", FakeGpuSpec((1 << 20,), (1,), (0,)), good_disk),
        ]
        for label, gpu, disk in cases:
            assert pump.submit_store(1000 + len(pump.jobs), gpu, disk) is False, f"accepted {label}"
        assert pump.jobs == {}, "a rejected submit still created a job"
    finally:
        safe_shutdown(pump)


# ---------------------------------------------------------------------------
# DiskTransferPump happy paths
# ---------------------------------------------------------------------------


def test_pump_store_job_completes_and_reports_size():
    pump, native, store, _ = make_pump(rows=4, cap=16, native=FakeNative(mode="immediate"))
    try:
        assert submit(pump, 1, "store", groups=(0, 1, 2)) is True
        seen = collect(pump, [1])
        assert set(seen) == {1}, f"job never completed: {seen}"
        result = seen[1]
        assert result.success is True, f"store job reported failure: {result.success}"
        assert result.transfer_size == sum(GROUP_SIZES), f"size={result.transfer_size}"
        assert store.writes == 3, f"expected 3 disk writes, got {store.writes}"
        assert native.submitted, "no native copy was ever submitted"
        assert pump.free and None not in pump.free, f"row leak/poison: {list(pump.free)}"
        assert pump.pending_keys == 0, f"pending_keys={pump.pending_keys}"
    finally:
        safe_shutdown(pump)


def test_pump_load_job_uses_read_into():
    pump, _, store, _ = make_pump(rows=4, cap=16, native=FakeNative(mode="immediate"))
    try:
        assert submit(pump, 7, "load", groups=(0, 0)) is True
        seen = collect(pump, [7])
        assert set(seen) == {7}, "load job never completed"
        assert seen[7].success is True, "load reported failure"
        assert store.reads == 2, f"expected 2 read_into calls, got {store.reads}"
        assert store.writes == 0, "a load performed a disk write"
    finally:
        safe_shutdown(pump)


def test_pump_deferred_native_needs_poll_or_wait():
    pump, native, _, _ = make_pump(rows=2, cap=8, native=FakeNative(mode="deferred"))
    try:
        submit(pump, 3, "store")
        assert pump.get_finished() == [], "deferred job finished without a wait/poll driver"
        native.finish_all(True)
        seen = collect(pump, [3])
        assert set(seen) == {3}, "job did not complete after native completion"
    finally:
        safe_shutdown(pump)


def test_pump_stress_many_waves_exactly_one_outcome_each():
    total, rows = 300, 2
    pump, _, _, _ = make_pump(rows=rows, cap=4, native=FakeNative(mode="immediate"))
    try:
        done, nxt = {}, 0
        end = time.monotonic() + 20.0
        while len(done) < total and time.monotonic() < end:
            while nxt < total and submit(pump, nxt, "store"):
                nxt += 1
            for result in pump.get_finished():
                assert result.job_id not in done, f"duplicate outcome for job {result.job_id}"
                done[result.job_id] = result
        assert len(done) == total, f"only {len(done)}/{total} jobs completed (submitted {nxt})"
        assert all(r.success for r in done.values()), "a successful store was reported as failed"
        assert sorted(pump.free) == list(range(rows)), f"row leak: free={sorted(pump.free)}"
        assert pump.pending_keys == 0 and pump.finished == [], "pump did not drain clean"
    finally:
        safe_shutdown(pump)


def test_pump_rows_bound_never_exceeded_under_saturation():
    rows = 3
    total = 20
    pump, native, store, _ = make_pump(rows=rows, cap=4096, native=FakeNative(mode="deferred"))
    try:
        for job_id in range(total):
            assert submit(pump, job_id, "store") is True, f"submit {job_id} refused"
        done = {}
        end = time.monotonic() + 10.0
        while len(done) < total and time.monotonic() < end:
            native.finish_all(True)
            for result in pump.get_finished():
                done[result.job_id] = result
        assert len(done) == total, f"only {len(done)}/{total} finished under saturation"
        assert native.max_pending <= rows, (
            f"more native rows in flight ({native.max_pending}) than rows={rows}")
        assert store.max_concurrent <= rows, f"disk concurrency {store.max_concurrent} > rows={rows}"
        assert sorted(pump.free) == list(range(rows)), f"free={sorted(pump.free)}"
    finally:
        safe_shutdown(pump)


def test_pump_pending_keys_capacity_is_exact_without_polling():
    cap = 5
    pump, _, _, _ = make_pump(rows=2, cap=cap, native=FakeNative(mode="deferred"))
    try:
        admitted = [jid for jid in range(20) if submit(pump, jid, "store")]
        assert len(admitted) == cap, (
            f"admitted {len(admitted)} jobs for max_pending_keys={cap} with no polling")
        assert pump.pending_keys == cap, f"pending_keys={pump.pending_keys} != cap={cap}"
        assert submit(pump, 999) is False, "capacity overrun admitted after saturation"
    finally:
        safe_shutdown(pump)


def test_pump_backpressure_frees_capacity_after_collection():
    pump, native, _, _ = make_pump(rows=2, cap=3, native=FakeNative(mode="immediate"))
    try:
        assert submit(pump, 1) and submit(pump, 2) and submit(pump, 3)
        assert submit(pump, 4) is False, "capacity not enforced"
        collect(pump, [1, 2, 3])
        assert pump.pending_keys == 0 and pump.finished == []
        assert submit(pump, 5) is True, "capacity not released after get_finished"
        collect(pump, [5])
    finally:
        safe_shutdown(pump)


# ---------------------------------------------------------------------------
# wait / cancel / shutdown
# ---------------------------------------------------------------------------


def test_pump_wait_does_not_consume_results():
    pump, _, _, _ = make_pump(rows=3, cap=8, native=FakeNative(mode="immediate"))
    try:
        for jid in (10, 11, 12):
            submit(pump, jid)
        pump.wait({10, 11, 12})
        seen = pump.get_finished()
        assert {r.job_id for r in seen} == {10, 11, 12}, (
            f"wait() consumed/dropped results: {[r.job_id for r in seen]}")
    finally:
        safe_shutdown(pump)


def test_pump_wait_on_unknown_ids_returns_immediately():
    pump, _, _, _ = make_pump(rows=1, cap=4)
    try:
        t0 = time.monotonic()
        pump.wait({424242})
        assert time.monotonic() - t0 < 1.0, "wait() on unknown ids blocked"
    finally:
        safe_shutdown(pump)


def test_pump_concurrent_wait_and_cancel_race_is_safe():
    rounds = 20
    failures = []
    for attempt in range(rounds):
        pump, native, _, _ = make_pump(rows=4, cap=64, native=FakeNative(mode="deferred"))
        try:
            ids = list(range(16))
            for jid in ids:
                assert submit(pump, jid)
            # one thread cancels the first half while the driver waits for all of it
            _, errors, alive = S.run_threads(
                [lambda: pump.cancel(set(ids[:8])), lambda: pump.wait(set(ids))], timeout=8.0)
            if any(alive):
                failures.append((attempt, "HANG: a poll thread never returned"))
                break
            errs = [e for e in errors if e]
            if errs:
                failures.append((attempt, f"{type(errs[0]).__name__}: {errs[0]}"))
                break
        finally:
            safe_shutdown(pump)
    assert not failures, (
        f"concurrent wait/cancel on one pump is unsafe ({len(failures)}/{rounds} rounds failed "
        f"before the first error): {failures[:3]}; _poll mutates self.jobs/self.native_jobs "
        "while iterating them")


def test_pump_cancel_recycles_row_for_next_job():
    pump, native, _, _ = make_pump(rows=1, cap=8, native=FakeNative(mode="deferred"))
    try:
        assert submit(pump, 1) is True
        pump.get_finished()                    # assigns the single row, native pending
        assert pump.jobs[1].row == 0, "row was not assigned to the in-flight job"
        assert submit(pump, 2) is True         # admitted, must wait for the row
        assert pump.jobs[2].row is None, "second job got a row while the only row was busy"
        pump.cancel({1})
        assert 1 not in pump.jobs, "cancelled job was not finished"
        assert pump.free == deque([0]) or pump.jobs[2].row == 0, (
            f"row not returned/recycled after cancel: free={list(pump.free)}, "
            f"job2.row={pump.jobs.get(2) and pump.jobs[2].row}")
        seen = {}
        end = time.monotonic() + 5.0
        while 2 not in seen and time.monotonic() < end:
            native.finish_all(True)
            for result in pump.get_finished():
                seen[result.job_id] = result
        assert 2 in seen and seen[2].success is True, (
            f"job after cancel never ran: seen={sorted(seen)}")
    finally:
        safe_shutdown(pump)


def test_pump_cancel_unknown_id_is_noop():
    pump, _, _, _ = make_pump(rows=1, cap=4)
    try:
        pump.cancel({77})
        assert pump.jobs == {}
    finally:
        safe_shutdown(pump)


def test_pump_shutdown_drains_and_cancels_inflight():
    pump, native, _, _ = make_pump(rows=4, cap=32, native=FakeNative(mode="deferred"))
    for jid in range(10):
        assert submit(pump, jid)
    seen = {}
    try:
        pump.shutdown()
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"shutdown raised during a clean drain: {exc!r}")
    for result in pump.get_finished():
        seen[result.job_id] = result
    assert pump.closed is True and pump._shutdown_complete is True, "shutdown left the pump open"
    assert pump.jobs == {}, f"shutdown left {len(pump.jobs)} jobs in flight (leak)"
    assert pump.pending_keys == 0, f"pending_keys={pump.pending_keys} after shutdown"
    assert len(seen) == 10, f"shutdown lost outcomes: {len(seen)}/10"
    assert all(r.success is False for r in seen.values()), "shutdown-reported job was marked success"
    assert native.shutdown_calls == ["native"], f"native.shutdown calls={native.shutdown_calls}"
    assert submit(pump, 999) is False, "pump accepted work after shutdown"


def test_pump_shutdown_is_idempotent():
    pump, native, _, _ = make_pump(rows=2, cap=8, native=FakeNative(mode="deferred"))
    submit(pump, 1)
    safe_shutdown(pump)
    before = list(native.shutdown_calls)
    try:
        pump.shutdown()
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"second shutdown raised: {exc!r}")
    assert native.shutdown_calls == before, (
        f"second shutdown re-shut the native worker: {native.shutdown_calls}")


def test_pump_shutdown_retry_after_provider_close_failure_is_safe():
    attempts = []

    def close_provider():
        attempts.append(1)
        raise RuntimeError("provider refuses to close")

    pump, native, _, _ = make_pump(rows=2, cap=8, native=FakeNative(mode="deferred"),
                                   close_provider=close_provider)
    submit(pump, 1)
    for attempt in range(2):
        try:
            pump.shutdown()
        except Exception:
            pass
        else:
            raise AssertionError(f"shutdown #{attempt + 1} acknowledged a failed provider close")
    assert len(attempts) == 2, f"provider close should be retried after failure, got {len(attempts)}"
    assert native.shutdown_calls.count("native") == 1, (
        "REGRESSION: retried shutdown called native.shutdown() "
        f"{native.shutdown_calls.count('native')}x (must be idempotent, row/region double-free risk)")


def test_pump_reentrant_shutdown_from_close_provider_does_not_deadlock():
    holder = {}

    def close_provider():
        holder["pump"].shutdown()      # provider finalises the pump it is closing

    pump, native, _, _ = make_pump(rows=1, cap=4, native=FakeNative(mode="immediate"),
                                   close_provider=close_provider)
    holder["pump"] = pump
    error = {}

    def run():
        try:
            pump.shutdown()
        except BaseException as exc:  # noqa: BLE001
            error["exc"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(3.0)
    if thread.is_alive():
        raise AssertionError(
            "HANG: pump.shutdown() called from its own close_provider self-deadlocks on "
            "_CloseOnce's non-reentrant lock and never returns")


def test_pump_native_rejected_submit_fails_the_job_not_hangs():
    native = FakeNative(mode="deferred", reject=True)
    pump, _, _, _ = make_pump(rows=2, cap=8, native=native)
    try:
        submit(pump, 1, "store")
        seen = collect(pump, [1], deadline=3.0)
        assert set(seen) == {1}, "rejected native submit left the job hanging"
        assert seen[1].success is False, "rejected native submit reported success"
        assert native.drains > 0, "device drain not requested after a rejected native submit"
        assert sorted(pump.free) == [0, 1], f"row leaked: {sorted(pump.free)}"
    finally:
        safe_shutdown(pump)


def test_pump_native_submit_exception_fails_the_job():
    native = FakeNative(mode="deferred", raise_submit=True)
    pump, _, _, _ = make_pump(rows=2, cap=8, native=native)
    try:
        submit(pump, 1, "store")
        seen = collect(pump, [1], deadline=3.0)
        assert set(seen) == {1}, "native submit exception left the job hanging"
        assert seen[1].success is False, "native submit exception reported success"
    finally:
        safe_shutdown(pump)


def test_pump_native_poll_failure_failstops_its_jobs():
    native = FakeNative(mode="immediate", poll_error=True)
    pump, _, _, _ = make_pump(rows=2, cap=8, native=native)
    try:
        submit(pump, 1, "store")
        submit(pump, 2, "load")
        seen = collect(pump, [1, 2], deadline=3.0)
        assert set(seen) == {1, 2}, f"get_finished failure leaked jobs: {seen}"
        assert all(r.success is False for r in seen.values()), "failed drain still reported success"
        assert native.drains > 0, "fail-stop did not force a device drain"
        assert sorted(pump.free) == [0, 1], f"row leaked after fail-stop: {sorted(pump.free)}"
    finally:
        safe_shutdown(pump)


def test_pump_disk_write_failure_fails_store_job():
    pump, _, store, _ = make_pump(rows=2, cap=8, native=FakeNative(mode="immediate"),
                                  store=FakeStore(fail_after=1))
    try:
        submit(pump, 1, "store", groups=(0, 0))
        seen = collect(pump, [1])
        assert set(seen) == {1}, "mid-transfer disk failure hung the job"
        assert seen[1].success is False, "disk write failure reported success"
        assert store.writes == 2, f"expected the failure on the 2nd write, got {store.writes}"
        assert sorted(pump.free) == [0, 1], f"row leaked: {sorted(pump.free)}"
    finally:
        safe_shutdown(pump)


def test_pump_disk_exception_fails_store_job():
    pump, _, _, _ = make_pump(rows=2, cap=8, native=FakeNative(mode="immediate"),
                              store=FakeStore(raise_after=0))
    try:
        submit(pump, 1, "store")
        seen = collect(pump, [1])
        assert set(seen) == {1} and seen[1].success is False, (
            f"disk exception not converted into a failed result: {seen}")
    finally:
        safe_shutdown(pump)


def test_pump_row_done_twice_on_final_row_must_not_crash():
    pump, native, _, _ = make_pump(rows=2, cap=8, native=FakeNative(mode="deferred"))
    try:
        submit(pump, 1, "store", groups=(0,))
        pump.get_finished()
        job = pump.jobs[1]
        assert job.row is not None
        pump._row_done(job, True)
        try:
            pump._row_done(job, True)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"_row_done called twice for the same job raised {type(exc).__name__}: {exc} "
                "(no idempotency guard; also underflows pending_keys)")
        assert None not in pump.free, f"double _row_done poisoned the free list: {list(pump.free)}"
    finally:
        safe_shutdown(pump)


def test_pump_row_done_twice_on_nonfinal_row_must_not_poison_rows():
    pump, native, _, _ = make_pump(rows=2, cap=8, native=FakeNative(mode="deferred"))
    try:
        submit(pump, 1, "store", groups=(0, 1))
        pump.get_finished()
        job = pump.jobs[1]
        pump._row_done(job, True)                 # first row of two
        pump._row_done(job, True)                 # duplicate completion
        assert None not in pump.free, (
            f"double _row_done pushed None onto the free-row deque: {list(pump.free)}")
    finally:
        safe_shutdown(pump)


def test_pump_concurrent_submit_accounting_is_consistent():
    workers, per, cap = 8, 250, 400
    pump, native, _, _ = make_pump(rows=8, cap=cap, native=FakeNative(mode="deferred"))

    def work(worker, iteration):
        submit(pump, worker * per + iteration, "store")

    previous = sys.getswitchinterval()
    try:
        sys.setswitchinterval(1e-6)   # force bytecode-level interleaving
        result = S.hammer(work, workers=workers, iterations=per, timeout=20.0)
        assert not any(result["alive"]), "concurrent submitters hung"
        assert not any(result["errors"]), S.summarize(result, "concurrent submit")
        admitted = len(pump.jobs)
        expected_keys = sum(len(job.disk.keys) for job in pump.jobs.values())
        assert admitted <= cap and pump.pending_keys == expected_keys, (
            f"RACE under {workers} unsynchronized submitting threads: admitted={admitted} "
            f"(cap={cap}), pending_keys={pump.pending_keys} vs {expected_keys} keys actually "
            f"in flight -- check-then-act admission and non-atomic 'pending_keys += n'")
    finally:
        sys.setswitchinterval(previous)
        safe_shutdown(pump)


# ---------------------------------------------------------------------------
# throughput
# ---------------------------------------------------------------------------


def _measure(io_threads, total=400, rows=8, cap=128, delay=0.0003):
    pump, _, _, _ = make_pump(rows=rows, cap=cap, io_threads=io_threads,
                              native=FakeNative(mode="immediate"), store=FakeStore(delay=delay))
    done, nxt = 0, 0
    try:
        t0 = time.perf_counter()
        while done < total:
            pending = False
            while nxt < total and submit(pump, nxt, "store"):
                nxt += 1
                pending = True
            if pending:
                pump.wait(set(pump.jobs))
            got = pump.get_finished()
            done += len(got)
            if not got and not pending:
                raise AssertionError("throughput loop made no progress")
        elapsed = time.perf_counter() - t0
    finally:
        safe_shutdown(pump)
    assert done == total, f"throughput run lost jobs: {done}/{total}"
    return total / elapsed


def test_pump_throughput_vs_io_concurrency():
    rates = {}
    for io_threads in (1, 2, 4, 8):
        rates[io_threads] = _measure(io_threads)
    for io_threads, rate in rates.items():
        print(f"[perf] io_threads={io_threads} jobs/sec={rate:.0f}", flush=True)
    assert all(rate > 100 for rate in rates.values()), f"implausible throughput: {rates}"
    assert rates[8] >= rates[1], f"throughput fell with more io threads: {rates}"


def test_pump_parallel_independent_pumps():
    counts = {}
    for pumps in (1, 2, 4, 8):
        results = {}

        def run(slot, pumps=pumps):
            results[slot] = _measure(1, total=200, rows=4, cap=32)

        fns = [(lambda slot=slot: run(slot)) for slot in range(pumps)]
        t0 = time.perf_counter()
        _, errors, alive = S.run_threads(fns, timeout=30.0)
        elapsed = time.perf_counter() - t0
        assert not any(alive), f"{pumps} independent pumps hung"
        assert not any(errors), f"independent pump raised: {[e for e in errors if e][:1]}"
        counts[pumps] = sum(results.values())
        print(f"[perf] independent_pumps={pumps} aggregate_jobs/sec={counts[pumps]:.0f}", flush=True)
    assert counts[8] > counts[1], f"independent pumps did not scale: {counts}"


# ---------------------------------------------------------------------------
# _Manager fakes
# ---------------------------------------------------------------------------

SIZES = ((16, 32, 64),)  # one cache group, world size 3
MGR_NAMESPACE = S.dummy_namespace("conc-manager")


class FakeTicket:
    def __init__(self, namespace, key, token, sizes, leases):
        self.namespace = namespace
        self.key = key
        self.token = token
        self.sizes = tuple(sizes)
        self.leases = tuple(leases)


class FakeCoordinator:
    def __init__(self, sizes=SIZES, renew_result=True, can_store_value=True,
                 barrier=None, ack=True, deadline=3600.0, fail_reserve_load=False,
                 raise_on_reserve_load=False, always_load=True):
        self.sizes = sizes
        self.renew_result = renew_result
        self.can_store_value = can_store_value
        self.barrier = barrier
        self.ack = ack
        self.deadline = deadline
        self.fail_reserve_load = fail_reserve_load
        self.raise_on_reserve_load = raise_on_reserve_load
        # always_load=True: reserve_load always proves durable data (lookup HIT).
        # always_load=False: reserve_load is a clean miss unless the key is seeded
        # into `durable`, which is what prepare_store's skip path keys off.
        self.always_load = always_load
        self.durable = set()
        self.calls = {}
        self.tickets = []

    def _bump(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def _ticket(self, namespace, key, tag):
        token = f"tok-{tag}-{len(self.tickets)}"
        group = int.from_bytes(key[-4:], "big")
        leases = [f"lease-{token}-{i}" for i in range(len(self.sizes[group]))]
        ticket = FakeTicket(namespace, key, token, self.sizes[group], leases)
        self.tickets.append(ticket)
        return ticket

    def reserve_load(self, namespace, key, owner):
        self._bump("reserve_load")
        if self.barrier is not None:
            self.barrier.wait(timeout=10.0)
        if self.raise_on_reserve_load:
            raise RuntimeError("coordinator transport exploded")
        if self.fail_reserve_load:
            return None
        if not self.always_load and (namespace, key) not in self.durable:
            return None
        return self._ticket(namespace, key, "load")

    def reserve_store(self, namespace, key, owner, sizes):
        self._bump("reserve_store")
        if self.barrier is not None:
            self.barrier.wait(timeout=10.0)
        return self._ticket(namespace, key, "store")

    def renew(self, ticket):
        self._bump("renew")
        return self.renew_result

    def release(self, ticket):
        self._bump("release")
        return self.ack

    def complete_store(self, ticket, success=True):
        self._bump("complete_store")
        return self.ack

    def invalidate(self, namespace, key):
        self._bump("invalidate")
        return self.ack

    def can_store(self):
        self._bump("can_store")
        return self.can_store_value

    def lease_deadline(self, ticket):
        self._bump("lease_deadline")
        return time.monotonic() + self.deadline


class Ctx:
    def __init__(self, req_id="r0"):
        self.req_id = req_id


class FakeOutput:
    def __init__(self, keys_to_store, store_spec, evicted_keys, skipped_keys):
        self.keys_to_store = list(keys_to_store)
        self.store_spec = store_spec
        self.evicted_keys = list(evicted_keys)
        self.skipped_keys = list(skipped_keys)


class FakeDiskSpec:
    def __init__(self, *args):
        self.keys, self.tokens, self.namespace, self.owner = args


def make_manager(coordinator=None, capacity=8, **kwargs):
    from recipe_persistence.native import _Manager
    coordinator = coordinator or FakeCoordinator()
    manager = _Manager(coordinator, MGR_NAMESPACE, SIZES, capacity,
                       FakeDiskSpec, FakeOutput, Ctx, **kwargs)
    return manager, coordinator


def mkey(tag, group=0):
    return gkey(group, tag)


# ---------------------------------------------------------------------------
# _Manager validation
# ---------------------------------------------------------------------------


def test_manager_constructor_validation():
    from recipe_persistence.native import _Manager
    bad = [
        dict(capacity=0),
        dict(capacity=-3),
        dict(sizes=()),
        dict(lookup_keys_per_step=0),
        dict(lookup_keys_per_step=True),
        dict(lookup_keys_per_step="8"),
        dict(metadata_workers=33),
        dict(metadata_workers=True),
        dict(metadata_workers=1.5),
        dict(metadata_workers=2, metadata_max_submitted=1),
        dict(metadata_max_submitted=4097),
        dict(metadata_shutdown_timeout=math.inf),
        dict(metadata_shutdown_timeout=0),
        dict(metadata_shutdown_timeout=301),
    ]
    for override in bad:
        kwargs = dict(capacity=8)
        kwargs.update(override)
        capacity = kwargs.pop("capacity")
        sizes = kwargs.pop("sizes", SIZES)
        try:
            _Manager(FakeCoordinator(), MGR_NAMESPACE, sizes, capacity,
                     FakeDiskSpec, FakeOutput, Ctx, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"_Manager accepted invalid config: {override}")
    # background metadata without lease_deadline/renew must be refused
    bare = SimpleNamespace(can_store=lambda: True)
    try:
        _Manager(bare, MGR_NAMESPACE, SIZES, 8, FakeDiskSpec, FakeOutput, Ctx,
                 metadata_workers=1)
    except ValueError:
        return
    raise AssertionError("async metadata accepted a coordinator without lease_deadline/renew")


# ---------------------------------------------------------------------------
# _Manager accounting
# ---------------------------------------------------------------------------


def test_manager_lookup_reserves_once_and_is_idempotent():
    manager, coord = make_manager(capacity=4)
    key = mkey(b"a")
    ctx = Ctx("r0")
    assert manager.lookup(key, ctx) is True
    assert manager.lookup(key, ctx) is True, "repeat lookup lost the reservation"
    assert coord.calls.get("reserve_load") == 1, (
        f"repeat lookup reserved again: {coord.calls.get('reserve_load')}")
    assert len(manager.loads) == 1, f"loads={manager.loads}"
    manager.complete_load([key], ctx)
    assert manager.loads == {}, "complete_load did not release the reservation"
    assert coord.calls.get("release") == 1, "complete_load did not notify the coordinator"


def test_manager_capacity_bound_sequential():
    manager, coord = make_manager(capacity=3)
    ctx = Ctx("r0")
    keys = [mkey(b"a"), mkey(b"b"), mkey(b"c")]
    assert all(manager.lookup(k, ctx) is True for k in keys), "capacity=3 refused an admit"
    assert manager.lookup(mkey(b"d"), ctx) is False, "capacity=3 admitted a 4th key"
    assert coord.calls.get("reserve_load") == 3, f"reserve_load={coord.calls.get('reserve_load')}"
    manager.complete_load([keys[0]], ctx)
    assert manager.lookup(mkey(b"d"), ctx) is True, "capacity not released after complete_load"


def test_manager_prepare_load_requires_unique_reserved_hits():
    manager, _ = make_manager(capacity=4)
    ctx = Ctx("r0")
    key = mkey(b"a")
    try:
        manager.prepare_load([key], ctx)
    except ValueError:
        pass
    else:
        raise AssertionError("prepare_load accepted an unreserved key")
    manager.lookup(key, ctx)
    spec = manager.prepare_load([key], ctx)
    assert key in spec.keys, "prepare_load built a spec without the key"
    try:
        manager.prepare_load([key], ctx)
    except ValueError:
        return
    raise AssertionError("prepare_load accepted the same key twice")


def test_manager_prepare_store_skips_durable_and_dedupes():
    coord = FakeCoordinator(always_load=False)
    manager, coord = make_manager(coordinator=coord, capacity=8)
    ctx = Ctx("r0")
    durable, fresh = mkey(b"dur"), mkey(b"fresh")
    coord.durable.add((MGR_NAMESPACE, durable))
    manager.lookup(durable, ctx)
    out = manager.prepare_store([durable, fresh, fresh], ctx)
    assert out is not None, "prepare_store returned nothing for a fresh key"
    assert out.skipped_keys == [durable], f"skipped={out.skipped_keys}"
    assert out.keys_to_store == [fresh], f"selected={out.keys_to_store} (duplicate not deduped)"
    assert len(manager.stores) == 1, f"stores={manager.stores}"
    manager.complete_store([fresh], ctx)
    assert manager.stores == {}, "complete_store did not clear the store reservation"


def test_manager_prepare_store_respects_capacity():
    manager, _ = make_manager(coordinator=FakeCoordinator(always_load=False), capacity=1)
    ctx = Ctx("r0")
    first = manager.prepare_store([mkey(b"a")], ctx)
    assert first is not None and first.keys_to_store, "first store not admitted"
    second = manager.prepare_store([mkey(b"b")], ctx)
    assert second is None, "capacity=1 admitted a second store reservation"


def test_manager_prepare_store_ignores_invalidated_keys():
    manager, coord = make_manager(capacity=4, metadata_workers=1)
    ctx = Ctx("r0")
    key = mkey(b"bad")
    manager.lookup(key, ctx)
    manager.on_load_failure([key], ctx)
    try:
        out = manager.prepare_store([key], ctx)
        assert out is None or out.keys_to_store == [], (
            f"prepare_store re-persisted an invalidated key: {out and out.keys_to_store}")
    finally:
        manager._metadata_executor.shutdown(wait=False)


def test_manager_on_load_failure_invalidates_synchronously():
    manager, coord = make_manager(capacity=4)
    ctx = Ctx("r0")
    key = mkey(b"a")
    manager.lookup(key, ctx)
    manager.complete_load([key], ctx)
    manager.on_load_failure([key], ctx)
    assert coord.calls.get("invalidate") == 1, (
        f"sync on_load_failure did not invalidate: {coord.calls}")


def test_manager_quarantine_on_invalidation_overflow():
    manager, coord = make_manager(capacity=1, metadata_workers=1,
                                  metadata_shutdown_timeout=0.5)
    ctx = Ctx("r0")
    try:
        manager.on_load_failure([mkey(b"a")], ctx)
        try:
            manager.on_load_failure([mkey(b"b")], ctx)
        except RuntimeError as exc:
            assert "quarantine" in str(exc), f"unexpected error: {exc}"
            assert manager._namespace_quarantined is True, "quarantine flag not set"
            assert manager.degraded_reason is not None, "overflow did not degrade the manager"
            return
        raise AssertionError("invalidation overflow silently evicted a tracked key")
    finally:
        manager._metadata_executor.shutdown(wait=False)


def test_manager_on_request_finished_refuses_active_transfers():
    manager, _ = make_manager(capacity=4)
    ctx = Ctx("r0")
    key = mkey(b"a")
    manager.lookup(key, ctx)
    manager.prepare_load([key], ctx)
    try:
        manager.on_request_finished(ctx)
    except RuntimeError:
        return
    raise AssertionError("on_request_finished released a request with prepared loads")
    manager.complete_load([key], ctx)
    manager.on_request_finished(ctx)


def test_manager_shutdown_releases_leases_and_closes_once():
    closed = []
    manager, coord = make_manager(capacity=4, close_provider=lambda: closed.append(1))
    ctx = Ctx("r0")
    manager.lookup(mkey(b"a"), ctx)
    manager.shutdown()
    assert manager.loads == {}, "shutdown leaked load reservations"
    assert coord.calls.get("release") == 1, f"shutdown did not release the lease: {coord.calls}"
    assert closed == [1], f"provider close called {len(closed)} times"
    assert manager._shutdown_complete is True
    manager.shutdown()
    assert closed == [1], "second shutdown re-closed the provider"


def test_manager_shutdown_refuses_active_store_without_closing_provider():
    closed = []
    manager, _ = make_manager(coordinator=FakeCoordinator(always_load=False), capacity=4,
                              close_provider=lambda: closed.append(1))
    ctx = Ctx("r0")
    manager.prepare_store([mkey(b"a")], ctx)
    try:
        manager.shutdown()
    except RuntimeError:
        assert closed == [], "provider closed while a store reservation was still live"
        assert manager.closed is False, "manager marked closed after a refused shutdown"
        return
    raise AssertionError("shutdown proceeded with an active store reservation")


def test_manager_degrade_on_coordinator_exception():
    coord = FakeCoordinator(raise_on_reserve_load=True)
    manager, _ = make_manager(coordinator=coord, capacity=4)
    assert manager.lookup(mkey(b"a"), Ctx("r0")) is False, "lookup succeeded despite transport failure"
    assert manager.degraded_reason == "coordinator_callback_failed", (
        f"degraded_reason={manager.degraded_reason!r}")
    assert manager.closed is True, "transport failure did not disable the manager"


def test_manager_degrade_when_coordinator_reports_no_capability():
    manager, _ = make_manager(coordinator=FakeCoordinator(can_store_value=False), capacity=4)
    assert manager.can_store() is False
    assert manager.degraded_reason == "coordinator_callback_failed"
    assert manager.lookup(mkey(b"a"), Ctx("r0")) is False


def test_manager_lookup_step_budget_and_rotation():
    manager, _ = make_manager(capacity=8, lookup_keys_per_step=2)
    a, b = Ctx("a"), Ctx("b")
    assert manager.lookup(mkey(b"a1"), a) is True
    assert manager.lookup(mkey(b"a2"), a) is True
    assert manager.lookup(mkey(b"a3"), a) is None, "per-step key budget not enforced"
    assert manager.lookup(mkey(b"b1"), b) is None, "a non-turn owner consumed the step budget"
    manager.on_schedule_end()
    assert manager.lookup(mkey(b"b2"), b) is True, "turn did not rotate to the next owner"
    assert manager.lookup(mkey(b"b3"), b) is True
    assert manager.lookup(mkey(b"b4"), b) is None, "budget not enforced after rotation"


def test_manager_pending_work_and_passive_hooks():
    manager, _ = make_manager(capacity=4, lookup_keys_per_step=2)
    ctx = Ctx("r0")
    assert manager.has_pending_work() is False
    manager.lookup(mkey(b"a"), ctx)
    assert manager.has_pending_work() is True, "an admitted lookup is not reported as pending"
    assert manager.touch([mkey(b"a")], ctx) is None
    assert manager.take_events() == ()
    manager.complete_load([mkey(b"a")], ctx)
    manager._clear_lookup_gate()
    assert manager.has_pending_work() is False


# ---------------------------------------------------------------------------
# _Manager background metadata
# ---------------------------------------------------------------------------


def test_manager_async_metadata_shutdown_drains():
    closed = []
    manager, coord = make_manager(capacity=4, metadata_workers=2,
                                  metadata_shutdown_timeout=3.0,
                                  close_provider=lambda: closed.append(1))
    ctx = Ctx("r0")
    key = mkey(b"a")
    assert manager.lookup(key, ctx) is True
    manager.complete_load([key], ctx)
    manager.shutdown()
    assert coord.calls.get("release", 0) >= 1, f"async release never dispatched: {coord.calls}"
    assert manager.loads == {} and manager._retiring_loads == {}, "async leases leaked"
    assert closed == [1], "provider not closed after a clean metadata drain"
    assert manager._shutdown_complete is True


def test_manager_metadata_drain_times_out_loudly():
    closed = []
    manager, coord = make_manager(capacity=4, metadata_workers=1,
                                  metadata_shutdown_timeout=0.4,
                                  close_provider=lambda: closed.append(1),
                                  coordinator=FakeCoordinator(ack=False))
    ctx = Ctx("r0")
    key = mkey(b"a")
    manager.lookup(key, ctx)
    manager.complete_load([key], ctx)
    try:
        manager.shutdown()
    except RuntimeError as exc:
        assert "not drained" in str(exc), f"unexpected error: {exc}"
        assert closed == [], "provider closed even though metadata had not drained"
        return
    finally:
        manager._metadata_executor.shutdown(wait=False)
    raise AssertionError("shutdown acknowledged a non-draining metadata queue")


def test_manager_concurrent_lookup_exceeds_capacity_bound():
    workers, capacity = 8, 4
    barrier = threading.Barrier(workers)
    coord = FakeCoordinator(barrier=barrier)
    manager, _ = make_manager(coordinator=coord, capacity=capacity, lookup_keys_per_step=None)

    def one(worker):
        return manager.lookup(mkey(b"c%d" % worker), Ctx(f"owner{worker}"))

    results, errors, alive = S.run_threads([(lambda w=w: one(w)) for w in range(workers)], timeout=15.0)
    assert not any(alive), "concurrent lookups hung"
    assert not any(errors), f"concurrent lookups raised: {[e for e in errors if e][:1]}"
    live = len(manager.loads)
    assert live <= capacity, (
        f"CAPACITY BOUND VIOLATED under {workers} concurrent lookups: {live} live load "
        f"reservations for max_pending_keys={capacity} (lookup results={results}); "
        "_Manager has no lock around the check-then-reserve sequence")


def test_manager_concurrent_prepare_store_exceeds_capacity_bound():
    workers, capacity = 8, 4
    barrier = threading.Barrier(workers)
    coord = FakeCoordinator(barrier=barrier, always_load=False)
    manager, _ = make_manager(coordinator=coord, capacity=capacity)

    def one(worker):
        return manager.prepare_store([mkey(b"s%d" % worker)], Ctx(f"owner{worker}"))

    results, errors, alive = S.run_threads([(lambda w=w: one(w)) for w in range(workers)], timeout=15.0)
    assert not any(alive), "concurrent prepare_store hung"
    assert not any(errors), f"concurrent prepare_store raised: {[e for e in errors if e][:1]}"
    live = len(manager.stores)
    assert live <= capacity, (
        f"CAPACITY BOUND VIOLATED under {workers} concurrent prepare_store calls: {live} live "
        f"store reservations for max_pending_keys={capacity}")


# ---------------------------------------------------------------------------
# native failure paths with vLLM absent
# ---------------------------------------------------------------------------


class _BlockVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


class block_vllm:
    """Simulate a runtime with no vLLM installed (never import torch/vllm here)."""

    def __enter__(self):
        self.finder = _BlockVllm()
        self.saved = {name: sys.modules.pop(name) for name in list(sys.modules)
                      if name == "vllm" or name.startswith("vllm.")
                      or name == "torch" or name.startswith("torch.")}
        sys.meta_path.insert(0, self.finder)
        return self

    def __exit__(self, *exc):
        try:
            sys.meta_path.remove(self.finder)
        finally:
            sys.modules.update(self.saved)
        return False


def test_native_module_imports_without_vllm():
    import recipe_persistence.native as native
    assert sys.modules.get("vllm") is None, "importing native.py pulled in vLLM"
    assert sys.modules.get("torch") is None, "importing native.py pulled in torch"
    assert hasattr(native, "_Manager"), "metadata-only surface missing"


def test_native_unknown_attribute_raises_attribute_error():
    import recipe_persistence.native as native
    try:
        native.definitely_not_a_symbol
    except AttributeError:
        return
    raise AssertionError("unknown module attribute did not raise AttributeError")


def test_prefix_hash_algorithm_refuses_without_vllm():
    import recipe_persistence.native as native
    with block_vllm():
        try:
            algorithm = native._prefix_hash_algorithm()
        except ImportError as exc:
            assert "vllm" in str(exc), f"refusal does not name vllm: {exc!r}"
            print(f"[refusal] _prefix_hash_algorithm -> {type(exc).__name__}: {exc}", flush=True)
            return
        else:
            raise AssertionError(
                f"_prefix_hash_algorithm() succeeded with vLLM absent, returned {algorithm!r}")


def test_load_native_refuses_without_vllm():
    import recipe_persistence.native as native
    with block_vllm():
        try:
            native._load_native()
        except ImportError as exc:
            assert "vllm" in str(exc), f"refusal does not name vllm: {exc!r}"
            print(f"[refusal] _load_native -> {type(exc).__name__}: {exc}", flush=True)
            return
        else:
            raise AssertionError("_load_native() succeeded with vLLM absent")


def test_native_getattr_known_symbol_without_vllm_is_import_error():
    import recipe_persistence.native as native
    with block_vllm():
        try:
            native.NodeLocalDiskManager
        except AttributeError as exc:
            raise AssertionError(
                f"KNOWN symbol raised AttributeError instead of a refusal: {exc!r}")
        except ImportError as exc:
            assert "vllm" in str(exc), f"refusal does not name vllm: {exc!r}"
            return
        raise AssertionError("known native symbol was importable without vLLM")
