"""Instant dummy data for recipe_persistence concurrency tests.

STDLIB ONLY, and deliberately cheap: nothing here starts an engine, touches a GPU,
binds a real rank's address, or sleeps for a real duration. Everything a test needs
is synthesised in-process so a whole module loads and runs inside the 30 s harness
budget with zero engine start time and zero inference.

Real waits are the enemy of a fast loop, so `dummy_limits` shrinks every bound to
something a test can exhaust on purpose in milliseconds.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time

MIB = 1024 * 1024


def dummy_limits(**over):
    """A small, self-consistent Limits: big enough to hold test data, small enough
    to hit quota/lease/object bounds deliberately inside one test.

    index_bytes is NOT shrunk to the validator minimum: the store turns it into
    `PRAGMA max_page_count = index_bytes // page_size`, so a 64 KiB index caps the
    whole database at 16 pages (~104 objects) and every later reserve_write fails
    with sqlite3.OperationalError("database or disk is full") -- which reserve_write
    swallows into a None return, indistinguishable from "key already exists".
    Keep it realistically large; override it deliberately to test index exhaustion.
    """
    base = dict(
        quota=32 * MIB,
        high=24 * MIB,
        low=16 * MIB,
        index_bytes=4 * MIB,
        max_objects=4096,
        max_leases=256,
        max_object_bytes=1 * MIB,
        free_bytes=0,
        free_inodes=0,
        lease_seconds=5.0,
        grace_seconds=0.5,
        io_chunk_bytes=4096,
    )
    base.update(over)
    from recipe_persistence.storage import Limits
    return Limits(**base)


def dummy_root(name="store"):
    """A fresh 0700 private root; DiskStore refuses anything else."""
    root = tempfile.mkdtemp(prefix=f"conc-{name}-")
    os.chmod(root, 0o700)
    return root


def dummy_store(name="store", **limit_over):
    from recipe_persistence.storage import DiskStore
    return DiskStore(dummy_root(name), dummy_limits(**limit_over))


def dummy_namespace(seed="rank-0"):
    """DiskStore._identity requires a 64-hex SHA-256 namespace; a bare label raises."""
    import hashlib
    return hashlib.sha256(str(seed).encode()).hexdigest()


def dummy_blob(size=4096, seed=0):
    """Deterministic bytes, no RNG cost at scale."""
    return bytes(((i * 31 + seed) & 0xFF) for i in range(size))


def dummy_infos(ranks=4, pages=(2304, 4, 2304, 2304, 2304, 2304, 64),
                groups=(((0, 0), (0, 1)), ((2, 2),), ((6, 64), (6, 65)))):
    """Rank-ordered census rows for layout_fingerprint.

    `group_refs` is a list of GROUPS, each group a list of [tensor_idx, page_bytes]
    refs -- that is what geometry_identity/layout_fingerprint unpack. An earlier
    version emitted a flat list of refs, which raised
    `TypeError: cannot unpack non-iterable int object` in both consumers.
    """
    refs = [[[int(i), int(s)] for i, s in group] for group in groups]
    return [{"padded_pages": [int(p) for p in pages],
             "group_refs": [[list(ref) for ref in group] for group in refs]}
            for _ in range(ranks)]


def dummy_geometry(pages=(2304, 4, 2304, 2304, 2304, 2304, 64),
                   groups=(((0, 0), (0, 1)), ((2, 2),), ((6, 64), (6, 65)))):
    """Duck-typed geometry for geometry_identity().

    group_refs is a list of groups (see dummy_infos); group_bytes is the per-group
    byte total, so len(group_bytes) == len(group_refs) as the validator requires.
    """
    from types import SimpleNamespace
    refs = [[[int(i), int(s)] for i, s in group] for group in groups]
    return SimpleNamespace(
        padded_pages=[int(p) for p in pages],
        group_refs=refs,
        group_bytes=[sum(s for _, s in group) for group in refs],
    )


def dummy_token(n=48):
    """Printable-ASCII credential of a legal length (32..256)."""
    import secrets
    import string
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def dummy_token_file(path=None, n=48):
    fh = open(path or os.path.join(tempfile.mkdtemp(), "token"), "w")
    tok = dummy_token(n)
    fh.write(tok + "\n")
    fh.close()
    os.chmod(fh.name, 0o600)
    return fh.name, tok


def run_threads(fns, timeout=30.0):
    """Start every callable at the same instant behind a Barrier so races are real.

    Returns (results, errors, still_alive). A live thread after `timeout` is a
    deadlock/hang signal the caller is expected to assert on.
    """
    n = len(fns)
    barrier = threading.Barrier(n)
    results = [None] * n
    errors = [None] * n

    def wrap(i, fn):
        try:
            barrier.wait(timeout=timeout)
        except threading.BrokenBarrierError as exc:
            errors[i] = exc
            return
        try:
            results[i] = fn()
        except BaseException as exc:  # noqa: BLE001
            errors[i] = exc

    threads = [threading.Thread(target=wrap, args=(i, f), daemon=True) for i, f in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    return results, errors, [t.is_alive() for t in threads]


def hammer(fn, workers=8, iterations=25, timeout=60.0):
    """`workers` threads each calling fn(worker, i) `iterations` times, started together."""
    def make(w):
        def run():
            for i in range(iterations):
                fn(w, i)
        return run
    fns = [make(w) for w in range(workers)]
    started = time.perf_counter()
    results, errors, alive = run_threads(fns, timeout=timeout)
    return {"seconds": round(time.perf_counter() - started, 3), "errors": errors, "alive": alive}


class Stopwatch:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.perf_counter() - self.t0, 4)
        return False


def summarize(result, label=""):
    """Compact evidence string for a failure message."""
    errs = [f"{type(e).__name__}: {e}" for e in (result.get("errors") or []) if e]
    return (f"{label} seconds={result.get('seconds')} alive={sum(1 for a in result.get('alive', []) if a)} "
            f"errors={len(errs)}" + (f" first={errs[0]}" if errs else ""))
