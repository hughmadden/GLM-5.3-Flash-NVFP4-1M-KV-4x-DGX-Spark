"""Reduced, confirmed defects — central coordination file for the fix/re-run cycle.

One test per defect, each a minimal instant reproduction with dummy data. A FAILING
test here is the defect's proof until the central fix lands; then it is the
regression guard.
"""
import sqlite3
import conc_stubs as S


def test_defect1_hard_storage_failure_is_indistinguishable_from_benign_refusal():
    """DEFECT-1 (storage): reserve_write swallows hard failures into None.

    `index_bytes` becomes `PRAGMA max_page_count = index_bytes // page_size`. When
    the index fills, the INSERT raises sqlite3.OperationalError("database or disk is
    full"); reserve_write's blanket `except _ERRORS: return None` converts that into
    the SAME None it returns for the benign "key already exists" case, and — unlike
    write/exists/release — it does NOT set `self.failed`.

    Consequence in production: when the disk (or the index budget) fills, persistence
    silently stops storing while the store keeps reporting itself healthy. The caller
    cannot distinguish "nothing to do" from "I have lost the ability to persist".

    Expected: a hard storage error marks the store failed. Observed: it does not.
    """
    store = S.dummy_store("defect1", index_bytes=128 * 1024)
    ns = S.dummy_namespace("defect1")

    # 1. benign refusal: the same key twice
    tok = store.reserve_write(ns, b"dup", 1024, "o")
    assert tok is not None, "first reserve_write must succeed"
    store.write(tok, [S.dummy_blob(1024, 0)])
    store.release(tok)
    dup = store.reserve_write(ns, b"dup", 1024, "o")
    assert dup is None, "duplicate key must be refused with None"

    # 2. hard failure: exhaust the SQLite page budget
    first_bad = None
    for i in range(2000):
        t = store.reserve_write(ns, f"k{i}".encode(), 1024, "o")
        if t is None:
            first_bad = i
            break
        store.write(t, [S.dummy_blob(1024, i)])
        store.release(t)
    assert first_bad is not None, "expected the index budget to be exhausted"

    # 3. the defect: distinguishable AND recoverable
    assert store.failure_count > 0 and store.last_failure, (
        "DEFECT-1: reserve_write refused a fresh key after a hard storage failure "
        "but recorded nothing, so the caller cannot tell a capacity failure from a "
        "benign 'key already exists' refusal"
    )
    assert "full" in (store.last_failure or "").lower() or "disk" in (store.last_failure or "").lower(), (
        f"recorded failure does not name the condition: {store.last_failure!r}"
    )
    assert store.failed is False, (
        "DEFECT-1b: a recoverable capacity failure latched self.failed, which "
        "refuses every operation forever with no recovery path"
    )

    # 4. recovery: reclaim, then explicitly clear the fail-closed latch.
    #    invalidate() latches self.failed on the same capacity error (deliberate
    #    fail-closed for a durability fault); there was no way back at all.
    for i in range(first_bad):
        store.invalidate(ns, f"k{i}".encode())
    store.collect(force=True)
    assert store.failed is True, (
        "expected invalidate()/collect() to latch failed on the capacity error"
    )
    assert store.clear_failure() is True, "clear_failure() must reset a latched store"
    assert store.failed is False
    assert store.reserve_write(ns, b"after-recovery", 1024, "o") is not None, (
        "the store did not recover after collect(force=True)+clear_failure(); a "
        "recoverable capacity condition must not require reopening the store"
    )
    store.close()


def test_defect1b_bad_caller_input_must_not_poison_the_store():
    """Companion guard: a CALLER error (bad namespace) must not mark the store failed.

    The fix for DEFECT-1 must separate caller errors (ValueError, e.g. a namespace
    that is not a 64-hex SHA-256) from storage errors (OSError/sqlite3.Error). A bad
    key from one caller must not take the whole rank's store out of service.
    """
    store = S.dummy_store("defect1b")
    assert store.reserve_write("not-a-fingerprint", b"k", 1024, "o") is None
    assert store.failed is False, "a caller error must not fail the store"
    tok = store.reserve_write(S.dummy_namespace("ok"), b"k", 1024, "o")
    assert tok is not None, "store must still work after a caller error"
    store.close()
