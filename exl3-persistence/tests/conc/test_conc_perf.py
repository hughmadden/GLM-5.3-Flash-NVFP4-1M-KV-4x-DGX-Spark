"""Performance and hang hypotheses for the disk store under concurrency.

These are measurements with generous assertions: they exist to surface a
CONTENTION CLIFF or a DEADLOCK, not to police microbenchmarks. The numbers are
printed so a run leaves evidence behind.
"""
import threading
import time
import conc_stubs as S


def _throughput(workers, ops_per_worker, blob=4096, ns=None, read_ratio=0.0):
    store = S.dummy_store(f"perf{workers}")
    namespace = ns or S.dummy_namespace(f"perf{workers}")
    # preload so reads have something to find
    for i in range(workers * ops_per_worker):
        t = store.reserve_write(namespace, f"p{i}".encode(), blob, "seed")
        if t is None:
            break
        store.write(t, [S.dummy_blob(blob, i)])
        store.release(t)

    def work(w, i):
        key = f"p{(w * ops_per_worker + i) % (workers * ops_per_worker)}".encode()
        if read_ratio and (i % 100) < read_ratio * 100:
            t = store.reserve_read(namespace, key, "r")
            if t is not None:
                buf = bytearray(blob)
                store.read_into(t, [memoryview(buf)])
                store.release(t)
            return
        t = store.reserve_write(namespace, f"w{w}-{i}".encode(), blob, "o")
        if t is not None:
            store.write(t, [S.dummy_blob(blob, w * 1000 + i)])
            store.release(t)

    r = S.hammer(work, workers=workers, iterations=ops_per_worker, timeout=120)
    total_ops = workers * ops_per_worker
    store.close()
    return total_ops / r["seconds"], r


def test_perf_write_throughput_vs_concurrency():
    """Measure write ops/s at C=1,2,4,8. A store whose rate COLLAPSES with threads
    is serialising on a global lock plus fsync; that is a real scalability finding."""
    curve = {}
    for c in (1, 2, 4, 8):
        rate, r = _throughput(c, 20 if c > 1 else 40)
        curve[c] = round(rate, 1)
        assert not any(r["alive"]), f"HANG at concurrency {c}"
        assert not any(r["errors"]), S.summarize(r, f"C={c}")
    print(f"  write ops/s by concurrency: {curve}")
    assert curve[8] >= curve[1] * 0.5, (
        f"write throughput collapsed under concurrency: {curve} "
        f"(C=8 is {curve[8] / curve[1]:.2f}x C=1)"
    )


def test_perf_read_throughput_vs_concurrency():
    curve = {}
    for c in (1, 2, 4, 8):
        rate, r = _throughput(c, 20 if c > 1 else 40, read_ratio=1.0)
        curve[c] = round(rate, 1)
        assert not any(r["alive"]), f"HANG at concurrency {c}"
    print(f"  read ops/s by concurrency: {curve}")
    assert curve[8] >= curve[1] * 0.5, f"read throughput collapsed: {curve}"


def test_no_hang_when_close_races_inflight_writes():
    """close() while writers are mid-flight must never deadlock or hang."""
    store = S.dummy_store("closerace")
    ns = S.dummy_namespace("closerace")
    stop = threading.Event()

    def writer(w, i):
        t = store.reserve_write(ns, f"c{w}-{i}".encode(), 2048, f"o{w}")
        if t is not None:
            store.write(t, [S.dummy_blob(2048, w * 100 + i)])
            store.release(t)

    def closer():
        for _ in range(50):
            if stop.is_set():
                return
            time.sleep(0.01)
        store.close()

    threads = [threading.Thread(target=writer, args=(w, i), daemon=True)
               for w in range(4) for i in range(10)]
    threads.append(threading.Thread(target=closer, daemon=True))
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    alive = [t for t in threads if t.is_alive()]
    assert not alive, f"HANG: {len(alive)} thread(s) still alive after close() race"
    assert time.perf_counter() - t0 < 20, "close() race took too long"
    try:
        store.close()
    except Exception:
        pass


def test_no_hang_when_collect_races_live_readers():
    """collect(force=True) is documented as never bypassing leases. Racing it against
    live readers must not deadlock and must not unlink a leased object."""
    store = S.dummy_store("collectrace")
    ns = S.dummy_namespace("collectrace")
    keys = []
    for i in range(32):
        k = f"k{i}".encode()
        t = store.reserve_write(ns, k, 2048, "o")
        if t is not None:
            store.write(t, [S.dummy_blob(2048, i)])
            store.release(t)
            keys.append(k)

    torn = []

    def reader(w, i):
        k = keys[(w * 8 + i) % len(keys)]
        t = store.reserve_read(ns, k, f"r{w}")
        if t is None:
            return
        buf = bytearray(2048)
        try:
            store.read_into(t, [memoryview(buf)])
        except Exception:
            torn.append(k)
            raise
        finally:
            store.release(t)

    def collector(w, i):
        store.collect(force=True)

    fns = []
    for w in range(4):
        for i in range(8):
            fns.append((lambda w=w, i=i: reader(w, i)))
    fns.append(lambda: collector(0, 0))
    results, errors, alive = S.run_threads(fns, timeout=30)
    assert not any(alive), "HANG: collect(force=True) vs live readers deadlocked"
    assert not torn, f"torn read while collect(force=True) raced a leased reader: {torn[:3]}"
    store.close()
