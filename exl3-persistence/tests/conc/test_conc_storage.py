"""Adversarial concurrency tests for recipe_persistence.storage.DiskStore.

STDLIB ONLY. Dummy data only. No engine, no GPU, no network, no sleeps longer
than a few ms. Every test builds its own private store under /tmp.

Rule followed here: the package is NOT edited. Tests assert the *correct*
behaviour; a FAIL therefore is the evidence of a defect, and every assertion
message carries the observed values. Perf measurements are emitted on stderr as
`##PERF## ...` lines so the JSON stream on stdout stays machine-readable.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time

import conc_stubs as S
from recipe_persistence.storage import DiskStore

MIB = 1024 * 1024


def _perf(line):
    print("##PERF## " + line, file=sys.stderr, flush=True)


class FakeClock:
    """Driven clock; no real waiting anywhere in this module."""

    def __init__(self, t=1000.0):
        self.t = float(t)
        self._lk = threading.Lock()

    def __call__(self):
        with self._lk:
            return self.t

    def advance(self, delta):
        with self._lk:
            self.t += delta
            return self.t


def _store(name, clock=time.time, **over):
    return DiskStore(S.dummy_root(name), S.dummy_limits(**over), clock=clock)


def _commit(store, ns, key, blob, owner="owner"):
    tok = store.reserve_write(ns, key, len(blob), owner)
    assert tok is not None, f"reserve_write refused a fresh key {key!r}"
    assert store.write(tok, [blob]) is True, f"write failed for {key!r}"
    return tok


def _rows(store, sql, args=()):
    with store._lock:
        return store.db.execute(sql, args).fetchall()


def _count(store, where="1", args=()):
    return _rows(store, f"SELECT COUNT(*) FROM objects WHERE {where}", args)[0][0]


def _state(store, key):
    ns_rows = _rows(store, "SELECT state FROM objects WHERE key=?", (key,))
    return ns_rows[0][0] if ns_rows else None


def _obs(store):
    """Best-effort observation snapshot; never raises."""
    out = {}
    for name, fn in (
        ("failed", lambda: store.failed),
        ("closed", lambda: store.closed),
        ("usage", lambda: store.usage()),
        ("objects_total", lambda: _count(store)),
        ("objects_c", lambda: _count(store, "state='C'")),
        ("objects_w", lambda: _count(store, "state='W'")),
        ("objects_t", lambda: _count(store, "state='T'")),
        ("leases", lambda: _rows(store, "SELECT COUNT(*) FROM leases")[0][0]),
    ):
        try:
            out[name] = fn()
        except BaseException as exc:  # noqa: BLE001
            out[name] = f"<{type(exc).__name__}: {exc}>"
    return out


# --------------------------------------------------------------------------
# A. plain concurrency at the store interface
# --------------------------------------------------------------------------

def test_distinct_keys_no_loss_under_race():
    """8 threads x distinct keys: nothing lost, every payload round-trips."""
    store = _store("distinct", lease_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    workers, per = 8, 6
    tok_of = {}
    lk = threading.Lock()

    def work(w, i):
        key = f"w{w}-k{i}".encode()
        blob = S.dummy_blob(1024, seed=w * 100 + i)
        tok = store.reserve_write(ns, key, len(blob), f"owner{w}")
        assert tok is not None, f"reserve_write refused fresh key {key!r}"
        assert store.write(tok, [blob]) is True, f"write failed for {key!r}"
        with lk:
            tok_of[key] = blob

    r = S.hammer(work, workers=workers, iterations=per)
    assert not any(r["alive"]), "HANG: " + S.summarize(r, "distinct writes")
    assert not any(r["errors"]), S.summarize(r, "distinct writes")

    missing = [k for k in tok_of if not store.exists(ns, k)]
    assert not missing, f"{len(missing)} committed keys not visible: {missing[:3]}"

    bad = []
    for key, blob in tok_of.items():
        rt = store.reserve_read(ns, key, "reader")
        if rt is None:
            bad.append((key, "reserve_read returned None for a committed key"))
            continue
        buf = bytearray(len(blob))
        if store.read_into(rt, [memoryview(buf)]) is not True or bytes(buf) != blob:
            bad.append((key, "read_into mismatch"))
        store.release(rt)
    assert not bad, f"round-trip failures under concurrency: {bad[:3]}"
    store.close()


def test_same_key_reserve_race_has_exactly_one_winner():
    """One key, 8 simultaneous reservers: exactly one token, and one write winner."""
    store = _store("samekey", lease_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    n = 8
    toks = [None] * n

    def mk(i):
        def run():
            toks[i] = store.reserve_write(ns, b"hot", 4096, f"o{i}")
        return run

    _, errs, alive = S.run_threads([mk(i) for i in range(n)], timeout=25.0)
    assert not any(alive), "HANG: same-key reserve race"
    assert not any(errs), f"reserve race raised {[f'{type(e).__name__}: {e}' for e in errs if e]}"
    winners = [t for t in toks if t]
    assert len(winners) == 1, (
        f"expected exactly 1 reservation for one key, got {len(winners)}: {winners}")
    winner = winners[0]

    blob = S.dummy_blob(4096, 5)
    flags = [None] * n

    def mw(i):
        def run():
            flags[i] = store.write(winner, [blob])
        return run

    _, errs, alive = S.run_threads([mw(i) for i in range(n)], timeout=25.0)
    assert not any(alive), "HANG: same-token write race"
    assert not any(errs), f"write race raised {[f'{type(e).__name__}: {e}' for e in errs if e]}"
    assert flags.count(True) == 1, (
        f"a single reservation was committed {flags.count(True)} times: {flags}")
    assert store.exists(ns, b"hot") is True
    store.close()


def test_writer_vs_readers_never_yields_a_torn_read():
    """Readers racing an invalidate+rewrite must see exactly one of two payloads."""
    store = _store("torn", lease_seconds=30.0, grace_seconds=0.0)
    ns = S.dummy_namespace("rank-0")
    key = b"contended"
    a = S.dummy_blob(64 * 1024, seed=1)
    b = S.dummy_blob(64 * 1024, seed=2)
    _commit(store, ns, key, a)

    torn, served, refused = [], [], []
    lk = threading.Lock()

    def writer():
        for _ in range(6):
            store.invalidate(ns, key)
            tok = store.reserve_write(ns, key, len(b), "writer")
            if tok is not None:
                store.write(tok, [b])

    def reader(w):
        def run():
            for _ in range(12):
                rt = store.reserve_read(ns, key, f"r{w}")
                if rt is None:
                    with lk:
                        refused.append(1)
                    continue
                buf = bytearray(len(a))
                ok = store.read_into(rt, [memoryview(buf)])
                got = bytes(buf)
                store.release(rt)
                if ok is not True or got not in (a, b):
                    with lk:
                        torn.append((ok, got[:16]))
                else:
                    with lk:
                        served.append(1)
        return run

    fns = [writer] + [reader(w) for w in range(4)]
    _, errs, alive = S.run_threads(fns, timeout=25.0)
    assert not any(alive), "HANG: writer vs readers"
    assert not any(errs), f"writer-vs-readers raised {[f'{type(e).__name__}: {e}' for e in errs if e]}"
    assert not torn, (
        f"{len(torn)} torn/partial reads observed while a lease was held: {torn[:3]}")
    assert served, "no reader ever obtained a lease; the race was vacuous"
    store.close()


# --------------------------------------------------------------------------
# B. leases vs eviction / collection
# --------------------------------------------------------------------------

def test_read_lease_vetoes_eviction():
    """Eviction must not select an object that a live read lease holds."""
    store = _store("veto", index_bytes=1 * MIB, quota=8 * MIB, high=6 * MIB, low=4 * MIB,
                   max_object_bytes=2 * MIB, lease_seconds=30.0, grace_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(MIB, seed=7)
    keys = [f"o{i}".encode() for i in range(4)]
    toks = [_commit(store, ns, k, blob) for k in keys]
    for t in toks:
        store.release(t)  # release == make immediately evictable, still state 'C'
    assert store.usage() >= store.limits.high, (
        f"setup did not cross the high watermark: {_obs(store)}")

    rt = store.reserve_read(ns, keys[0], "reader")
    assert rt is not None, "reserve_read refused a committed object"
    marked = store.evict()
    assert marked > 0, f"evict marked nothing; setup invalid: {_obs(store)}"
    assert _state(store, keys[0]) == "C", (
        "evict tombstoned an object with a live read lease: "
        f"state={_state(store, keys[0])} obs={_obs(store)}")

    buf = bytearray(len(blob))
    assert store.read_into(rt, [memoryview(buf)]) is True, (
        "read_into failed for a leased object that eviction was supposed to protect")
    assert bytes(buf) == blob, "leased read returned wrong bytes"
    store.release(rt)
    store.close()


def test_collect_force_never_removes_a_leased_object():
    """collect(force=True) documents that it never bypasses leases."""
    store = _store("forcelease", lease_seconds=30.0, grace_seconds=0.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(4096, 11)
    _commit(store, ns, b"k", blob)
    rt = store.reserve_read(ns, b"k", "reader")
    assert rt is not None
    assert store.invalidate(ns, b"k") is True
    assert _state(store, b"k") == "T"
    store.collect(force=True)
    assert _rows(store, "SELECT id FROM objects WHERE key=?", (b"k",)), (
        "collect(force=True) unlinked an object while a read lease was live")
    buf = bytearray(len(blob))
    assert store.read_into(rt, [memoryview(buf)]) is True, (
        "leased reader lost its object to collect(force=True)")
    assert bytes(buf) == blob
    store.release(rt)
    store.close()


def test_evict_racing_readers_never_breaks_a_granted_lease():
    """Fuzz: any token handed out by reserve_read must stay fully readable."""
    store = _store("evictrace", index_bytes=1 * MIB, quota=16 * MIB, high=10 * MIB,
                   low=8 * MIB, max_object_bytes=2 * MIB, lease_seconds=30.0,
                   grace_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(2048, 13)
    keys = [f"e{i}".encode() for i in range(12)]
    for k in keys:
        t = _commit(store, ns, k, blob)
        store.release(t)  # immediately evictable, state stays 'C'

    bad, granted = [], [0]
    lk = threading.Lock()

    def reader(w):
        def run():
            for i in range(12):
                k = keys[(w * 5 + i) % len(keys)]
                rt = store.reserve_read(ns, k, f"r{w}")
                if rt is None:
                    continue
                buf = bytearray(len(blob))
                ok = store.read_into(rt, [memoryview(buf)])
                store.release(rt)
                with lk:
                    granted[0] += 1
                    if ok is not True or bytes(buf) != blob:
                        bad.append((k, ok))
        return run

    def evictor():
        for _ in range(8):
            store.evict()
            time.sleep(0.005)

    fns = [evictor] + [reader(w) for w in range(4)]
    _, errs, alive = S.run_threads(fns, timeout=25.0)
    assert not any(alive), "HANG: evict racing readers"
    assert not any(errs), f"evict race raised {[f'{type(e).__name__}: {e}' for e in errs if e]}"
    assert not bad, (
        f"{len(bad)} reads failed although reserve_read had granted a lease: {bad[:3]}")
    assert granted[0] > 0, "no reader ever got a lease; the race was vacuous"
    store.close()


# --------------------------------------------------------------------------
# C. invalidate / renew / lease expiry
# --------------------------------------------------------------------------

def test_invalidate_racing_reserve_read():
    """invalidate vs reserve_read: a granted lease must always be servable."""
    store = _store("invrace", lease_seconds=30.0, grace_seconds=0.5)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(4096, 17)
    keys = [f"i{i}".encode() for i in range(8)]
    for k in keys:
        _commit(store, ns, k, blob)

    bad, granted = [], [0]
    lk = threading.Lock()

    def reader(w):
        def run():
            for i in range(20):
                k = keys[(w * 3 + i) % len(keys)]
                rt = store.reserve_read(ns, k, f"r{w}")
                if rt is None:
                    continue
                buf = bytearray(len(blob))
                ok = store.read_into(rt, [memoryview(buf)])
                with lk:
                    granted[0] += 1
                    if ok is not True or bytes(buf) != blob:
                        bad.append((k, ok))
                store.release(rt)
        return run

    def invalidator(w):
        def run():
            time.sleep(0.003)  # let readers acquire leases first; this is a race, not a sweep
            for i in range(20):
                store.invalidate(ns, keys[(w * 3 + i) % len(keys)])
        return run

    fns = [reader(w) for w in range(3)] + [invalidator(w) for w in range(3)]
    _, errs, alive = S.run_threads(fns, timeout=25.0)
    assert not any(alive), "HANG: invalidate vs reserve_read"
    assert not any(errs), f"invalidate race raised {[f'{type(e).__name__}: {e}' for e in errs if e]}"
    assert not bad, (
        f"{len(bad)} reads failed although reserve_read had granted a lease: {bad[:3]}")
    assert granted[0] > 0, "no reader ever got a lease; the race was vacuous"
    left = [k for k in keys if store.exists(ns, k)]
    assert not left, f"keys still visible after invalidate: {left}"
    store.close()


def test_lease_valid_agrees_with_read_into_after_invalidate():
    """lease_valid() is the 'retry receipt' check; it must not contradict read_into()."""
    store = _store("leasevalid", lease_seconds=30.0, grace_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(4096, 19)
    _commit(store, ns, b"k", blob)
    rt = store.reserve_read(ns, b"k", "reader")
    assert rt is not None
    assert store.lease_valid(rt) is True
    assert store.invalidate(ns, b"k") is True  # state -> 'T'
    buf = bytearray(len(blob))
    served = store.read_into(rt, [memoryview(buf)])
    lv = store.lease_valid(rt)
    assert served is True, "read_into refused a live lease on a tombstoned object"
    assert lv is True, (
        "lease_valid()=False while read_into() served the same live lease (object state 'T'): "
        f"served={served} lease_valid={lv} obs={_obs(store)}")
    store.release(rt)
    store.close()


def test_renew_cannot_resurrect_an_expired_lease():
    """After the lease window closes, renew must be False and reads must fail."""
    clock = FakeClock(1000.0)
    store = _store("renew", clock=clock, lease_seconds=10.0, grace_seconds=1.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(4096, 23)
    _commit(store, ns, b"k", blob)

    rt = store.reserve_read(ns, b"k", "reader")
    assert rt is not None
    assert store.renew(rt) is True, "renew refused a lease inside its window"
    clock.advance(11.0)
    store.evict()  # runs _expire()
    assert store.renew(rt) is False, (
        "renew() returned True for an expired lease; it extended nothing durably")
    buf = bytearray(len(blob))
    assert store.read_into(rt, [memoryview(buf)]) is False, (
        "read_into served a lease that renew() reported as un-renewable")
    assert store.lease_valid(rt) is False

    # same, for a queued *write* reservation
    wt = store.reserve_write(ns, b"w", 4096, "writer")
    assert wt is not None
    clock.advance(11.0)
    store.evict()
    assert store.renew(wt) is False, "renew() resurrected an expired write reservation"
    assert store.write(wt, [blob]) is False, "write committed after its reservation expired"
    assert store.exists(ns, b"w") is False
    store.close()


def test_write_after_reservation_expiry_fails_cleanly():
    """An expired write reservation must not commit, and must not leave a payload."""
    clock = FakeClock(1000.0)
    store = _store("expwrite", clock=clock, lease_seconds=5.0, grace_seconds=1.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(8192, 29)
    tok = store.reserve_write(ns, b"k", len(blob), "writer")
    assert tok is not None
    clock.advance(6.0)
    assert store.write(tok, [blob]) is False, (
        "write() committed a payload whose reservation lease had already expired")
    assert store.exists(ns, b"k") is False
    leftover = [p.name for p in store.objects.iterdir()]
    assert not leftover, f"refused write left files behind: {leftover}"
    assert _obs(store)["objects_c"] == 0
    store.close()


# --------------------------------------------------------------------------
# D. close(), shutdown, and the public API contract
# --------------------------------------------------------------------------

def test_close_racing_inflight_ops_is_safe():
    """close() during in-flight work: no exception escapes, nothing hangs."""
    store = _store("closerace", lease_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    _commit(store, ns, b"seed", S.dummy_blob(1024, 1))
    rounds = [0] * 4

    def mk(w):
        def run():
            for i in range(60):
                rounds[w] += 1
                store.exists(ns, b"seed")
                store.reserve_read(ns, b"seed", f"o{w}")
                k = f"c{w}-{i}".encode()
                tok = store.reserve_write(ns, k, 1024, f"o{w}")
                if tok is not None:
                    store.write(tok, [S.dummy_blob(1024, i)])
                store.invalidate(ns, k)
                store.renew(b"0" * 32)
                store.lease_valid(b"0" * 32)
                store.evict()
                store.collect()
            return "done"
        return run

    def closer():
        time.sleep(0.02)
        store.close()
        return "closed"

    fns = [mk(w) for w in range(4)] + [closer]
    res, errs, alive = S.run_threads(fns, timeout=25.0)
    assert not any(alive), "HANG: close() racing in-flight operations blocked forever"
    raised = [f"{type(e).__name__}: {e}" for e in errs if e]
    assert not raised, (
        "an operation raised instead of returning a benign falsy value after close(): "
        f"{raised[:3]} obs={_obs(store)}")
    assert store.closed is True
    # post-close contract: everything is falsy, nothing raises
    post = []
    for name, call in (
        ("reserve_write", lambda: store.reserve_write(ns, b"post", 1024, "o")),
        ("write", lambda: store.write(b"a" * 32, [b"x"])),
        ("exists", lambda: store.exists(ns, b"seed")),
        ("reserve_read", lambda: store.reserve_read(ns, b"seed", "o")),
        ("read_into", lambda: store.read_into(b"a" * 32, [bytearray(4)])),
        ("release", lambda: store.release(b"a" * 32)),
        ("renew", lambda: store.renew(b"a" * 32)),
        ("lease_valid", lambda: store.lease_valid(b"a" * 32)),
        ("invalidate", lambda: store.invalidate(ns, b"seed")),
        ("evict", lambda: store.evict()),
        ("collect", lambda: store.collect()),
    ):
        try:
            post.append((name, call()))
        except BaseException as exc:  # noqa: BLE001
            post.append((name, f"<RAISED {type(exc).__name__}: {exc}>"))
    bad = [p for p in post if isinstance(p[1], str) and p[1].startswith("<RAISED")]
    assert not bad, f"post-close calls raised: {bad}"
    store.close()  # idempotent


def test_usage_after_close_matches_the_rest_of_the_api():
    """Every post-close call returns a benign falsy value; usage() must not raise."""
    store = _store("usageclose")
    store.close()
    raised = None
    try:
        got = store.usage()
    except BaseException as exc:  # noqa: BLE001
        got, raised = None, f"{type(exc).__name__}: {exc}"
    assert raised is None, (
        "usage() raises after close() while every other public method returns a benign "
        f"falsy value -- a caller logging usage during shutdown crashes: {raised}")
    store.close()


def test_post_close_maintenance_calls_do_not_mark_the_store_failed():
    """A programming error must not be recorded as a durable storage fault."""
    store = _store("closeflag")
    store.close()
    store.evict()
    store.collect()
    assert store.failed is False, (
        "evict()/collect() on a closed store set the failed flag (sqlite3.ProgrammingError "
        "'Cannot operate on a closed database' is caught by the same _ERRORS handler used "
        "for real I/O faults). Harmless here because the store is closed, but it shows any "
        "caught exception -- not just I/O faults -- is escalated to failed=True")
    store.close()


def test_lock_queue_wait_does_not_silently_expire_a_write_reservation():
    """A granted write reservation must not be invalidated by pure lock queueing.

    The store serializes every call -- including the whole fsync'd payload write --
    behind one global RLock. A caller that reserves, then calls write(), can spend
    seconds waiting for that lock; the reservation's lease_seconds is wall-clock,
    and the expiry check happens *after* the lock is acquired. The payload is then
    dropped with a bare `False` that is indistinguishable from an I/O fault.
    """
    store = _store("slowwrite", lease_seconds=0.05, grace_seconds=0.05,
                   io_chunk_bytes=4096, index_bytes=4 * MIB, quota=64 * MIB,
                   high=48 * MIB, low=32 * MIB, max_object_bytes=16 * MIB)
    ns = S.dummy_namespace("rank-0")
    chunk = S.dummy_blob(64 * 1024, seed=53)
    pieces = [chunk] * 128          # 8 MiB payload, cheap to build
    payload = chunk * 128

    # control: same payload, no lock contention
    tok = store.reserve_write(ns, b"solo", len(payload), "solo")
    assert tok is not None
    t0 = time.perf_counter()
    alone_ok = store.write(tok, pieces)
    alone = time.perf_counter() - t0
    assert alone_ok is True, f"uncontended 8 MiB write refused in {alone:.3f}s"

    # contended: 3 threads saturate the store's single lock
    tok = store.reserve_write(ns, b"big", len(payload), "big")
    assert tok is not None, "reserve_write refused while idle"
    stop = threading.Event()

    def churn(w):
        def run():
            for i in range(40):
                if stop.is_set():
                    return
                t = store.reserve_write(ns, f"churn{w}-{i}".encode(), 1024, f"o{w}")
                if t is not None:
                    store.release(t)
        return run

    threads = [threading.Thread(target=churn(w), daemon=True) for w in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.002)
    t0 = time.perf_counter()
    ok = store.write(tok, pieces)
    contended = time.perf_counter() - t0
    stop.set()
    for t in threads:
        t.join(20.0)
    assert not any(t.is_alive() for t in threads), "HANG: churn threads never finished"

    assert ok is True, (
        f"contended write returned False after {contended:.3f}s (uncontended control: "
        f"{alone:.3f}s) -- the reservation ({store.limits.lease_seconds}s lease) expired "
        "while the call was queued behind the store's own global lock, and the entire "
        "8 MiB payload was silently discarded; the caller cannot distinguish this from an "
        f"I/O error. obs={_obs(store)}")
    assert store.exists(ns, b"big") is True
    store.close()


# --------------------------------------------------------------------------
# E. bounds under concurrency
# --------------------------------------------------------------------------

def test_max_leases_is_enforced_under_race():
    store = _store("leases", max_leases=4, lease_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    _commit(store, ns, b"k", S.dummy_blob(1024, 3))
    n = 8
    toks = [None] * n

    def mk(i):
        def run():
            toks[i] = store.reserve_read(ns, b"k", f"o{i}")
        return run

    _, errs, alive = S.run_threads([mk(i) for i in range(n)], timeout=25.0)
    assert not any(alive), "HANG: max_leases race"
    assert not any(errs), f"max_leases race raised {[f'{type(e).__name__}: {e}' for e in errs if e]}"
    granted = [t for t in toks if t]
    assert len(granted) == 4, f"max_leases=4 but {len(granted)} leases were granted"
    assert _obs(store)["leases"] <= 4, f"lease rows exceeded max_leases: {_obs(store)}"
    store.close()


def test_max_objects_is_enforced_under_race():
    store = _store("objects", max_objects=8, lease_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    n = 16
    toks = [None] * n

    def mk(i):
        def run():
            blob = S.dummy_blob(1024, i)
            toks[i] = store.reserve_write(ns, f"k{i}".encode(), len(blob), f"o{i}")
        return run

    _, errs, alive = S.run_threads([mk(i) for i in range(n)], timeout=25.0)
    assert not any(alive), "HANG: max_objects race"
    assert not any(errs), f"max_objects race raised {[f'{type(e).__name__}: {e}' for e in errs if e]}"
    granted = [t for t in toks if t]
    assert len(granted) == 8, f"max_objects=8 but {len(granted)} reservations were granted"
    assert _count(store) <= 8, f"object rows exceeded max_objects: {_obs(store)}"
    store.close()


def test_quota_never_exceeded_under_race():
    """Concurrent reservers must never overshoot the quota, and a granted write must commit."""
    store = _store("quota", index_bytes=1 * MIB, quota=8 * MIB, high=6 * MIB, low=4 * MIB,
                   max_object_bytes=1 * MIB, lease_seconds=30.0, grace_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    granted, refused = [0], [0]
    lk = threading.Lock()

    def work(w, i):
        key = f"q{w}-{i}".encode()
        blob = S.dummy_blob(256 * 1024, seed=w * 10 + i)
        tok = store.reserve_write(ns, key, len(blob), f"o{w}")
        if tok is None:
            with lk:
                refused[0] += 1
            return
        with lk:
            granted[0] += 1
        assert store.write(tok, [blob]) is True, (
            f"write() failed for a reservation that reserve_write had already granted "
            f"(key={key!r}) -- reservation granted but data never committed")
        rt = store.reserve_read(ns, key, "r")
        assert rt is not None, f"granted+written key {key!r} is not readable"
        buf = bytearray(len(blob))
        assert store.read_into(rt, [memoryview(buf)]) is True
        assert bytes(buf) == blob, f"payload mismatch for {key!r}"
        store.release(rt)

    r = S.hammer(work, workers=16, iterations=4)
    assert not any(r["alive"]), "HANG: quota race"
    assert not any(r["errors"]), S.summarize(r, "quota race")
    assert granted[0] > 0 and refused[0] > 0, (
        f"test did not exercise both branches: granted={granted[0]} refused={refused[0]}")
    usage = store.usage()
    assert usage <= store.limits.quota, (
        f"usage {usage} exceeded quota {store.limits.quota} under concurrency: {_obs(store)}")
    store.close()


def test_store_never_reports_failure_under_a_clean_mix():
    """No fault is injected, so failed must stay False and nothing may deadlock."""
    store = _store("mixed", lease_seconds=30.0, index_bytes=4 * MIB, quota=64 * MIB,
                   high=48 * MIB, low=32 * MIB, max_object_bytes=2 * MIB,
                   grace_seconds=0.5)
    ns = S.dummy_namespace("rank-0")

    def work(w, i):
        key = f"m{w}-{i % 5}".encode()
        blob = S.dummy_blob(2048, seed=w * 10 + i)
        tok = store.reserve_write(ns, key, len(blob), f"o{w}")
        if tok is not None:
            assert store.write(tok, [blob]) is True
            rt = store.reserve_read(ns, key, f"o{w}")
            if rt is not None:
                buf = bytearray(len(blob))
                assert store.read_into(rt, [memoryview(buf)]) is True
                store.release(rt)
        store.exists(ns, key)
        store.invalidate(ns, b"never-seen")
        store.usage()
        store.renew(b"0" * 32)
        store.lease_valid(b"0" * 32)
        if i % 3 == 0:
            store.evict()
        if i % 4 == 0:
            store.collect()
        if i % 2 == 0:
            store.invalidate(ns, key)

    r = S.hammer(work, workers=8, iterations=10)
    assert not any(r["alive"]), "HANG: mixed operations never finished"
    assert not any(r["errors"]), S.summarize(r, "mixed ops")
    assert store.failed is False, (
        "store.failed became True under a pure concurrency mix with no injected fault; "
        f"the whole store is now permanently disabled: {_obs(store)}")
    store.close()


# --------------------------------------------------------------------------
# F. suspected availability / error-handling defects (these are expected to FAIL)
# --------------------------------------------------------------------------

def test_invalidate_then_rewrite_is_not_refused_for_the_grace_window():
    """invalidate() then an immediate rewrite of the SAME key must be possible."""
    clock = FakeClock(1000.0)
    store = _store("regrace", clock=clock, lease_seconds=30.0, grace_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(4096, 31)
    _commit(store, ns, b"k", blob)
    assert store.invalidate(ns, b"k") is True
    assert store.exists(ns, b"k") is False
    tok = store.reserve_write(ns, b"k", len(blob), "writer")
    assert tok is not None, (
        "reserve_write refused an immediate rewrite of a key that invalidate() had just "
        "removed: the tombstone row still matches the state-agnostic "
        "'SELECT 1 FROM objects WHERE ns=? AND key=?' guard, and collect() cannot unlink it "
        "until grace_seconds (30.0 here; 60.0 by default) has elapsed. obs=" + str(_obs(store)))
    # control: after the grace window the same call succeeds
    clock.advance(31.0)
    store.collect()
    assert store.reserve_write(ns, b"k", len(blob), "writer") is not None, (
        "key still unwritable after the grace window (control failed)")
    store.close()


def test_invalidated_objects_do_not_block_max_objects():
    """Tombstoned, unreachable objects must not count against max_objects."""
    store = _store("tombmax", max_objects=4, lease_seconds=30.0, grace_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(4096, 37)
    for i in range(4):
        _commit(store, ns, f"k{i}".encode(), blob)
    for i in range(4):
        assert store.invalidate(ns, f"k{i}".encode()) is True
    assert all(not store.exists(ns, f"k{i}".encode()) for i in range(4))
    tok = store.reserve_write(ns, b"fresh", len(blob), "writer")
    assert tok is not None, (
        "4 invalidated objects (all state='T', exists()==False, unreachable) still count "
        "toward max_objects=4, so the store refuses every new write for the whole "
        f"grace window. obs={_obs(store)}")
    store.close()


def test_index_exhaustion_is_distinguishable_and_not_permanent():
    """A full SQLite index must not look like 'key already exists' or brick the store.

    quota is set to 1 GiB so it can never be the binding constraint: the only
    thing that can refuse a write here is the 16-page max_page_count implied by
    index_bytes=65536.
    """
    store = _store("indexfull", index_bytes=65536, quota=1 * 1024 * MIB, high=900 * MIB,
                   low=200000, max_objects=100000, max_object_bytes=1 * MIB,
                   lease_seconds=30.0, grace_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    ok = 0
    refused_key = None
    for i in range(600):
        key = f"k{i}".encode()
        blob = S.dummy_blob(1024, seed=i)
        tok = store.reserve_write(ns, key, len(blob), "writer")
        if tok is None:
            refused_key = key
            break
        if store.write(tok, [blob]) is True:
            ok += 1
        else:
            refused_key = key  # write() itself hit the failure (db full at commit)
            break
    assert refused_key is not None, f"never reached a refusal in 600 objects: {_obs(store)}"

    # Page accounting + an independent proof of the underlying error on the same
    # connection: run exactly the INSERT that reserve_write would have run.
    pages = None
    direct = None
    try:
        with store._lock:
            pages = {
                "page_count": store.db.execute("PRAGMA page_count").fetchone()[0],
                "max_page_count": store.db.execute("PRAGMA max_page_count").fetchone()[0],
                "freelist": store.db.execute("PRAGMA freelist_count").fetchone()[0],
            }
            store.db.execute("BEGIN IMMEDIATE")
            store.db.execute(
                "INSERT INTO objects VALUES(?,?,?,?,?,'W',?,?,?)",
                ("f" * 32, ns, b"direct-probe", 1024, 12288, 0.0, 0.0, "probe"))
    except BaseException as exc:  # noqa: BLE001
        direct = f"{type(exc).__name__}: {exc}"
        with store._lock:
            if store.db.in_transaction:
                store.db.execute("ROLLBACK")
    else:
        with store._lock:
            store.db.execute("ROLLBACK")

    probe_key = b"never-reserved-at-all"
    probe = store.reserve_write(ns, probe_key, 1024, "writer")
    probe_exists = store.exists(ns, probe_key)

    # Does the store ever accept writes again once free capacity is returned?
    recovered = None
    try:
        for i in range(3):
            store.invalidate(ns, f"k{i}".encode())
        store.collect(force=True)
        r2 = store.reserve_write(ns, b"after-invalidate-and-collect", 1024, "writer")
        recovered = repr(r2)
        if r2 is not None:
            store.release(r2)
    except BaseException as exc:  # noqa: BLE001
        recovered = f"<RAISED {type(exc).__name__}: {exc}>"

    obs = _obs(store)
    info = (f"ok={ok} refused_key={refused_key!r} pages={pages} "
            f"direct_objects_insert={direct!r} fresh_key_reserve={probe!r} "
            f"fresh_key_exists={probe_exists!r} after_invalidate_collect={recovered} obs={obs}")

    assert obs["usage"] < store.limits.quota, (
        f"quota, not the index, was the binding constraint: {info}")
    assert direct and "full" in direct.lower(), (
        f"expected the SQLite index to be full at {ok} objects, but replaying "
        f"reserve_write's own INSERT did not report it: {info}")

    # Is the refusal distinguishable from a benign 'key already exists'?
    assert probe is not None or store.failed, (
        "reserve_write returned the same plain None for a hard 'database or disk is full' "
        "error as for a benign 'key already exists' refusal: the caller cannot tell a "
        f"capacity failure from a cache hit, and nothing is logged. {info}")

    assert obs["failed"] is False, (
        "index exhaustion permanently poisoned the store: collect()/evict() caught "
        "sqlite3.OperationalError('database or disk is full') and set failed=True, after "
        f"which every operation is refused forever with no recovery path. {info}")
    store.close()


# --------------------------------------------------------------------------
# G. performance / contention
# --------------------------------------------------------------------------

def test_perf_write_read_throughput_by_concurrency():
    """Round-trip ops/sec at concurrency 1/2/4/8; asserts no catastrophic collapse."""
    ns = S.dummy_namespace("rank-0")
    per = 5
    out = {}
    for workers in (1, 2, 4, 8):
        store = _store(f"perf{workers}", lease_seconds=30.0)

        def work(w, i):
            key = f"p{w}-{i}".encode()
            blob = S.dummy_blob(1024, seed=w * 7 + i)
            tok = store.reserve_write(ns, key, len(blob), f"o{w}")
            assert tok is not None, f"reserve refused fresh key {key!r}"
            assert store.write(tok, [blob]) is True
            rt = store.reserve_read(ns, key, f"o{w}")
            assert rt is not None
            buf = bytearray(len(blob))
            assert store.read_into(rt, [memoryview(buf)]) is True
            assert bytes(buf) == blob
            store.release(rt)

        with S.Stopwatch() as sw:
            r = S.hammer(work, workers=workers, iterations=per)
        assert not any(r["alive"]), f"HANG at concurrency {workers}"
        assert not any(r["errors"]), S.summarize(r, f"perf c={workers}")
        ops = workers * per
        out[workers] = ops / sw.seconds
        _perf(f"c={workers:2d} round-trips={ops:3d} seconds={sw.seconds:7.3f} "
              f"round-trips/sec={out[workers]:8.1f}")
        store.close()
    ratio = out[8] / out[1]
    _perf(f"scaling: c=8/c=1 throughput ratio = {ratio:.2f}x (1.0 == fully serialized)")
    assert ratio > 0.35, (
        f"throughput collapsed under concurrency: {out} (c=8/c=1={ratio:.2f}x)")


def test_perf_concurrent_reads_of_direct_keys_do_not_parallelize():
    """Measure the global-lock cost: N distinct-key reads in parallel vs sequential."""
    store = _store("readpar", index_bytes=4 * MIB, quota=512 * MIB, high=400 * MIB,
                   low=300 * MIB, max_object_bytes=64 * MIB, io_chunk_bytes=1 * MIB,
                   lease_seconds=30.0, grace_seconds=30.0)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(8 * MIB, seed=41)
    keys = [f"r{i}".encode() for i in range(4)]
    for k in keys:
        _commit(store, ns, k, blob)

    def one(k):
        rt = store.reserve_read(ns, k, "r")
        assert rt is not None
        buf = bytearray(len(blob))
        assert store.read_into(rt, [memoryview(buf)]) is True
        assert bytes(buf) == blob
        store.release(rt)

    with S.Stopwatch() as sw:
        for k in keys:
            one(k)
    seq = sw.seconds

    fns = [(lambda k=k: one(k)) for k in keys]
    t0 = time.perf_counter()
    _, errs, alive = S.run_threads(fns, timeout=25.0)
    par = time.perf_counter() - t0
    assert not any(alive), "HANG: parallel distinct-key reads"
    assert not any(errs), f"parallel reads raised {[f'{type(e).__name__}: {e}' for e in errs if e]}"
    _perf(f"4x8MiB distinct-key reads: sequential={seq:.3f}s parallel(4 threads)={par:.3f}s "
          f"speedup={seq / par:.2f}x (4.0 == fully parallel)")

    # single 1 KiB write cost, to show the fixed fsync/serialization floor
    small = S.dummy_blob(1024, 43)
    with S.Stopwatch() as sw2:
        for i in range(10):
            k = f"s{i}".encode()
            _commit(store, ns, k, small)
    _perf(f"single-threaded 1KiB write+commit: {sw2.seconds / 10 * 1000:.1f} ms/op "
          f"({10 / sw2.seconds:.1f} ops/sec)")
    store.close()


def test_perf_collect_cost_per_tombstone_under_the_global_lock():
    """collect() unlinks + fsyncs + commits one transaction per tombstone, all
    while holding the single global lock, so its latency is a direct stall on
    every other thread's reserve/read."""
    store = _store("collectcost", lease_seconds=30.0, grace_seconds=30.0,
                   index_bytes=4 * MIB, quota=256 * MIB, high=200 * MIB, low=32 * MIB,
                   max_object_bytes=1 * MIB)
    ns = S.dummy_namespace("rank-0")
    blob = S.dummy_blob(4096, seed=61)
    n = 80
    with S.Stopwatch() as sw_w:
        for i in range(n):
            k = f"c{i}".encode()
            _commit(store, ns, k, blob)
            assert store.invalidate(ns, k) is True
    assert _count(store, "state='T'") == n
    with S.Stopwatch() as sw_c:
        store.collect(force=True)
    assert _count(store) == 0, f"collect left tombstones behind: {_obs(store)}"
    _perf(f"{n} tombstones: write+invalidate={sw_w.seconds:.3f}s "
          f"({sw_w.seconds / n * 1000:.1f} ms/object); collect(force=True)="
          f"{sw_c.seconds:.3f}s ({sw_c.seconds / n * 1000:.1f} ms/tombstone) "
          f"-- the whole collect holds the single global lock")
    store.close()
