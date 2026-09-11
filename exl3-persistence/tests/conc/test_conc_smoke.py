"""Harness smoke test: proves the runner + stubs work and the store survives a race."""
import conc_stubs as S


def test_stubs_are_instant():
    with S.Stopwatch() as sw:
        store = S.dummy_store("smoke")
        store.close()
    assert sw.seconds < 5.0, f"dummy_store took {sw.seconds}s; stubs must be instant"


def test_store_concurrent_writes_no_loss():
    store = S.dummy_store("race")
    workers, per = 8, 25
    ns = S.dummy_namespace("rank-0")

    def work(w, i):
        key = f"w{w}-k{i}".encode()
        blob = S.dummy_blob(1024, seed=w * 100 + i)
        token = store.reserve_write(ns, key, len(blob), f"owner{w}")
        store.write(token, [blob])
        store.release(token)

    r = S.hammer(work, workers=workers, iterations=per)
    assert not any(r["errors"]), S.summarize(r, "concurrent write")
    assert not any(r["alive"]), "worker thread did not finish (deadlock?)"
    missing = [f"w{w}-k{i}".encode() for w in range(workers) for i in range(per)
               if not store.exists(ns, f"w{w}-k{i}".encode())]
    assert not missing, f"{len(missing)} keys lost under concurrency, e.g. {missing[:3]}"
    store.close()


def test_store_concurrent_read_roundtrip():
    store = S.dummy_store("roundtrip")
    ns = S.dummy_namespace("rank-0")
    blobs = {}
    for w in range(4):
        for i in range(10):
            key = f"r{w}-{i}".encode()
            blob = S.dummy_blob(2048, seed=w * 10 + i)
            blobs[key] = blob
            tok = store.reserve_write(ns, key, len(blob), f"o{w}")
            store.write(tok, [blob])
            store.release(tok)

    mismatches = []
    lock = __import__("threading").Lock()

    def work(w, i):
        key = f"r{w}-{i}".encode()
        tok = store.reserve_read(ns, key, f"o{w}")
        buf = bytearray(len(blobs[key]))
        store.read_into(tok, [memoryview(buf)])
        store.release(tok)
        if bytes(buf) != blobs[key]:
            with lock:
                mismatches.append(key)

    r = S.hammer(work, workers=4, iterations=10)
    assert not any(r["errors"]), S.summarize(r, "concurrent read")
    assert not mismatches, f"read/write mismatch under concurrency: {mismatches[:3]}"
    store.close()
