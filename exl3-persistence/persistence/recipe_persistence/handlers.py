# SPDX-License-Identifier: Apache-2.0
# Architecture derived from nodelocal_disk_spec.py (revision 6729433),
# Apache-2.0; native CPU/GPU transfer API copyright vLLM contributors.
"""Bounded rank-local disk pump. Importing metadata never imports CUDA/vLLM.

One key occupies one physical row. Each job has at most one active wave; the
ready queue uses byte-deficit round-robin across jobs and directions. Rows remain owned until
both durable disk I/O and native GPU operations have finished. Native calls run
on the calling worker thread, not the disk executor's threads.

Direction is explicit (submit_store / submit_load), matching the upstream
OffloadingWorker API since vLLM f237e16b41 (#45053). It is no longer inferred
from LoadStoreSpec.medium(), which upstream deleted in c46ced1ee3 (#46544).
"""
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from dataclasses import dataclass
import threading
import time

from .geometry import key_group

_LOG = logging.getLogger(__name__)

# One-shot path tracing for transfer submission. A load that is never submitted
# at all looks identical, from the engine's counters, to a load that was
# submitted and never finished; these notices separate the two. Static codes
# only -- no key, path, token or exception message is logged.
_NOTE_SEEN: set[str] = set()
_NOTE_COUNTS: dict[str, int] = {}


def _note(code: str) -> None:
    _NOTE_COUNTS[code] = _NOTE_COUNTS.get(code, 0) + 1
    if code not in _NOTE_SEEN:
        _NOTE_SEEN.add(code)
        _LOG.warning("Persistence trace: %s", code)


def note_counts() -> dict[str, int]:
    """Snapshot of submitted transfer kinds (diagnostics only)."""
    return dict(_NOTE_COUNTS)


class _CloseOnce:
    """One acknowledged cleanup; retain ownership/callback on unproved drain.

    Re-entrant safe. A provider whose cleanup drains the object that owns this
    guard (e.g. a close_provider that finalizes the pump, or a callback that calls
    itself) re-enters ``__call__`` from inside the callback. With a plain Lock that
    self-deadlocked: the outer call held the lock, the inner call blocked on it
    forever, and ``DiskTransferPump.shutdown()``/``_Manager.shutdown()`` never
    returned. The inner call now returns immediately; only the outermost call runs
    the callback, exactly once, and the callback is still retained for retry when
    the drain is not proved.
    """
    def __init__(self, callback=None):
        if callback is not None and not callable(callback):
            raise ValueError("provider close must be callable or None")
        self.callback = callback
        self._lock = threading.RLock()
        self._closing = False

    def __call__(self):
        with self._lock:
            if self.callback is None or self._closing:
                return
            self._closing = True
            try:
                try:
                    result = self.callback()
                except Exception:
                    raise RuntimeError("provider cleanup has not drained") from None
                if result is not None and result is not True:
                    raise RuntimeError("provider cleanup has not drained")
                self.callback = None
            finally:
                self._closing = False



_REJECT_SEEN: set[str] = set()
_REJECT_COUNTS: dict[str, int] = {}


def _reject(code: str) -> bool:
    """Record why a transfer submission was refused, and return False.

    ``_submit`` has several independent admission checks and used to return a
    bare ``False`` from each. A refused load then simply never completed, with
    nothing anywhere saying which check had refused it, which made a silent
    stall indistinguishable from a slow transfer. The codes are static and
    bounded, matching the package rule that no key, path, token or exception
    message ever reaches a log line.
    """
    _REJECT_COUNTS[code] = _REJECT_COUNTS.get(code, 0) + 1
    if code not in _REJECT_SEEN:
        _REJECT_SEEN.add(code)
        _LOG.warning("Transfer submission refused: %s", code)
    return False


def reject_counts() -> dict[str, int]:
    """Snapshot of refusal codes seen by this process (diagnostics only)."""
    return dict(_REJECT_COUNTS)


@dataclass(frozen=True)
class DiskLoadStoreSpec:
    keys: tuple[bytes, ...]
    tokens: tuple[tuple[str, ...], ...]  # key-major, then distributed world rank
    namespace: str
    owner: str

    # Retained as our own storage-medium marker for logging and namespacing.
    # Upstream's LoadStoreSpec.medium() abstract method was deleted; nothing
    # in the native API calls this any more.
    @staticmethod
    def medium():
        return "NODE_LOCAL_DISK"


@dataclass
class _Job:
    job_id: int
    disk: DiskLoadStoreSpec
    gpu: object
    store: bool
    groups: tuple[int, ...]
    index: int = 0
    row: int | None = None
    native_id: int | None = None
    future: object = None
    cancelled: bool = False
    success: bool = True
    size: int = 0
    started: float = 0.0
    credit: int = 0


class DiskTransferPump:
    def __init__(self, *, geometry, cpu_gpu, store, rank, rows, max_pending_keys,
                 gpu_spec_cls, cpu_spec_cls, result_cls, drain_cuda, io_threads=1,
                 close_provider=None):
        if rows < 1 or max_pending_keys < rows or io_threads < 1 or rank < 0:
            raise ValueError("invalid bounded transfer capacity or rank")
        self.geometry, self.native, self.store = geometry, cpu_gpu, store
        # Usually row_bytes dominates; repeated canonical refs can make an
        # unpadded key larger. Every key must fit one scheduling quantum.
        self.quantum = max(geometry.row_bytes, max(geometry.group_bytes))
        self.rank, self.max_pending_keys = rank, max_pending_keys
        self.gpu_cls, self.cpu_cls, self.result_cls = gpu_spec_cls, cpu_spec_cls, result_cls
        self.drain_cuda = drain_cuda
        # CPUOffloadingWorker composes one SingleDirectionOffloadingHandler per
        # direction and exposes them as private attributes since f237e16b41.
        # Bind them once here so the rest of the pump never reaches into
        # upstream privates, and fail closed with a legible message if a future
        # upstream rename removes them.
        try:
            self._store_native = cpu_gpu._store_handler
            self._load_native = cpu_gpu._load_handler
        except AttributeError:
            raise ValueError(
                "native CPU worker does not expose _store_handler/_load_handler; "
                "the pinned upstream staging API has changed") from None
        self.cpu_tensors = self._store_native.dst_tensors
        self.free = deque(range(rows))
        self.jobs = {}
        self.ready = deque()
        self.finished = []
        self.native_jobs = {}
        self.next_native_id = 0
        self.pending_keys = 0
        self.closed = False
        self._shutdown_complete = False
        self._native_shutdown_done = False
        self._close_provider = _CloseOnce(close_provider)
        self.executor = ThreadPoolExecutor(max_workers=min(rows, io_threads),
                                           thread_name_prefix="kv-disk")

    def submit_store(self, job_id, src_spec, dst_spec):
        """Async GPU -> node-local disk."""
        _note("submit_store")
        return self._submit(job_id, src_spec, dst_spec, True)

    def submit_load(self, job_id, src_spec, dst_spec):
        """Async node-local disk -> GPU."""
        _note("submit_load")
        return self._submit(job_id, src_spec, dst_spec, False)

    def _submit(self, job_id, src_spec, dst_spec, storing):
        if self.closed or job_id in self.jobs:
            return _reject("closed_or_duplicate")
        gpu, disk = (src_spec, dst_spec) if storing else (dst_spec, src_spec)
        try:
            if not isinstance(gpu, self.gpu_cls) or not isinstance(disk, DiskLoadStoreSpec):
                return _reject("spec_type")
            n = len(disk.keys)
            if n < 1 or n != len(gpu.block_ids) or n != len(disk.tokens):
                return _reject("arity_keys_vs_blocks")
            if self.pending_keys + len(self.finished) + n > self.max_pending_keys:
                return _reject("pending_capacity")
            groups = tuple(key_group(k, len(self.geometry.group_refs)) for k in disk.keys)
            expected = tuple(g for g, count in enumerate(gpu.group_sizes) for _ in range(count))
            if (groups != expected or len(gpu.group_sizes) != len(self.geometry.group_refs)
                    or len(gpu.block_indices) != len(gpu.group_sizes)):
                return _reject("group_shape")
            if (not isinstance(disk.namespace, str) or not disk.namespace
                    or not isinstance(disk.owner, str) or not disk.owner):
                return _reject("identity")
            if any(self.rank >= len(tokens) or not isinstance(tokens[self.rank], str)
                   or not tokens[self.rank] for tokens in disk.tokens):
                return _reject("rank_token")
            gpu_tensors = self._store_native.src_tensors
            for block, group in zip(gpu.block_ids, groups):
                if int(block) != block or int(block) < 0:
                    return _reject("block_value")
                if any(int(block) >= gpu_tensors[index].shape[0]
                       for index, _ in self.geometry.group_refs[group]):
                    return _reject("block_bounds")
        except (ValueError, TypeError, AttributeError, IndexError):
            return _reject("exception")
        job = _Job(job_id, disk, gpu, storing, groups, started=time.monotonic(), credit=self.quantum)
        self.jobs[job_id] = job
        self.pending_keys += n
        self.ready.append(job_id)
        # Submission only admits metadata. get_finished/wait drive native copies.
        return True

    def _handler(self, job):
        return self._store_native if job.store else self._load_native

    def _finish(self, job):
        self.pending_keys -= len(job.disk.keys)
        del self.jobs[job.job_id]
        # TransferResult moved to kv_offload/base.py and lost transfer_type.
        self.finished.append(self.result_cls(
            job_id=job.job_id, success=job.success and not job.cancelled,
            transfer_size=job.size, transfer_time=time.monotonic() - job.started))

    def _row_done(self, job, success):
        job.success &= bool(success)
        self.free.append(job.row)
        job.row = None
        job.future = None
        job.native_id = None
        if not job.success or job.cancelled:
            self._finish(job)
            return
        size = self.geometry.group_bytes[job.groups[job.index]]
        job.size += size
        job.credit -= size
        job.index += 1
        if job.index == len(job.disk.keys):
            self._finish(job)
        elif job.credit >= self.geometry.group_bytes[job.groups[job.index]]:
            self.ready.appendleft(job.job_id)
        else:
            job.credit += self.quantum
            self.ready.append(job.job_id)

    def _submit_native(self, job):
        group = job.groups[job.index]
        sizes = [0] * len(self.geometry.group_refs)
        sizes[group] = 1
        gpu = self.gpu_cls([int(job.gpu.block_ids[job.index])], sizes,
                           [0] * len(sizes))  # factor 1: offsets do not affect copies
        cpu = self.cpu_cls([job.row])
        native_id = self.next_native_id
        self.next_native_id += 1
        job.native_id = native_id
        self.native_jobs[native_id] = job
        try:
            src, dst = (gpu, cpu) if job.store else (cpu, gpu)
            accepted = self._handler(job).transfer_async(native_id, src, dst)
        except Exception:
            accepted = False
        if not accepted:
            # A native exception can occur after launching CUDA but before event
            # registration. A device drain is required even for rejected submits.
            self.drain_cuda()
            self.native_jobs.pop(native_id, None)
            self._row_done(job, False)

    def _submit_disk(self, job):
        try:
            buffers = self.geometry.buffers(self.cpu_tensors, job.row, job.groups[job.index])
            token = job.disk.tokens[job.index][self.rank]
            operation = self.store.write if job.store else self.store.read_into
            job.future = self.executor.submit(operation, token, buffers)
        except Exception:
            self._row_done(job, False)

    def _poll(self):
        for handler in (self._store_native, self._load_native):
            try:
                results = handler.get_finished()
            except Exception:
                # Do not recycle any row if this drain itself fails: fail-stop.
                self.drain_cuda()
                affected = [j for j in list(self.native_jobs.values())
                            if self._handler(j) is handler]
                for job in affected:
                    self.native_jobs.pop(job.native_id, None)
                    self._row_done(job, False)
                continue
            for result in results:
                job = self.native_jobs.pop(result.job_id, None)
                if job is None:
                    continue
                job.native_id = None
                if result.success and job.store and not job.cancelled:
                    self._submit_disk(job)
                else:
                    self._row_done(job, result.success)
        for job in list(self.jobs.values()):
            if job.future is not None and job.future.done():
                try:
                    success = bool(job.future.result())
                except Exception:
                    success = False
                job.future = None
                if success and not job.store and not job.cancelled:
                    self._submit_native(job)
                else:
                    self._row_done(job, success)
        while self.free and self.ready:
            job = self.jobs.get(self.ready.popleft())
            if job is None:
                continue
            if job.cancelled:
                self._finish(job)
                continue
            job.row = self.free.popleft()
            if job.store:
                self._submit_native(job)
            else:
                self._submit_disk(job)

    def get_finished(self):
        self._poll()
        result, self.finished = self.finished, []
        return result

    def wait(self, job_ids):
        # Results stay available to get_finished after this blocking drain.
        while job_ids.intersection(self.jobs):
            self._poll()
            # Snapshot: a concurrent cancel()/get_finished() mutates self.jobs, and
            # iterating the live view raised 'dictionary changed size during iteration'.
            futures = [j.future for j in list(self.jobs.values()) if j.future is not None]
            if futures:
                futures_wait(futures, return_when="FIRST_COMPLETED")
            for handler in (self._store_native, self._load_native):
                ids = {i for i, j in list(self.native_jobs.items()) if self._handler(j) is handler}
                if ids:
                    try:
                        handler.wait(ids)
                    except Exception:
                        self.drain_cuda()
                        for i in ids:
                            job = self.native_jobs.pop(i)
                            self._row_done(job, False)

    def cancel(self, job_ids):
        for job_id in job_ids:
            if job_id in self.jobs:
                self.jobs[job_id].cancelled = True
        self.wait(set(job_ids))

    def shutdown(self):
        if self._shutdown_complete:
            return
        self.closed = True
        self.cancel(set(self.jobs))
        self.executor.shutdown(wait=True)
        # A retry after a provider-close failure re-enters here; native.shutdown()
        # is not documented idempotent (row/region double-free risk), so run it once.
        if not self._native_shutdown_done:
            self.native.shutdown()
            self._native_shutdown_done = True
        self._close_provider()
        self._shutdown_complete = True
