# FIX: the disk-tier lookup gate could deadlock (2026-09-11 AEST)

A code fix, verified by a test that fails without it and passes with it, and
observed live as a hard stall.

---

## 1. What was wrong

`_Manager._admit_lookup` admits one request's worth of **new** disk reservations
per turn and answers `RETRY` (`None`) to every other request. The turn was
advanced in exactly one place: `_Manager.on_schedule_end`.

That is a circular dependency:

```
    request's lookups all answered RETRY
        -> the request produces no batch
            -> the scheduler has nothing to run
                -> on_schedule_end is not called
                    -> the turn never advances
                        -> every later lookup is RETRY again
```

Nothing in that loop can break it, and the request never completes. On the engine
it presents as:

```
Engine 000: ... Running: 0 reqs, Waiting: 1 reqs, Deferred: 1 reqs,
             GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

with the process healthy, `/health` returning 200, no exception anywhere, and the
client hanging until it times out. It was reproducible on a **cold** prompt with
nothing on the disk tier at all, which is what pointed at the gate rather than at
the store: a cold lookups should simply miss and terminate the scan.

## 2. The fix

The gate now advances on wall-clock as well as from the scheduler hook, through a
single `_rotate_lookup_turn()` helper used by both paths. One owner still gets its
whole per-step key budget within a single scan, so the throttle is unchanged; what
changes is that the gate can no longer be stranded by a scheduler that has nothing
to do.

```
_LOOKUP_TURN_SECONDS = 0.05   # wall-clock lifetime of one owner's turn
```

`_clear_lookup_gate` (the degraded/closed path) also resets the new timestamp, so
a disabled manager does not retain turn state.

## 3. Verification

`tests/test_lookup_gate_liveness.py` drives the real manager with a fake
in-process coordinator and, critically, **never calls `on_schedule_end`** —
reproducing a scheduler that has no batch to run. No engine, no torch, no socket,
no filesystem; the file runs in 0.12 s.

| Test | Without the fix | With the fix |
|---|---|---|
| `test_second_owner_is_admitted_without_the_scheduler_hook` | **FAIL** — `owner B was never admitted without on_schedule_end` | pass |
| `test_repeated_retry_always_resolves` | **FAIL** | pass |
| `test_gate_clears_on_degrade` | **FAIL** | pass |
| `test_budget_still_bounds_one_turn` | pass | pass |

Three of four fail on the reverted tree and all four pass on the fixed tree. The
fourth is retained deliberately: it asserts that liveness was not bought by
deleting the throttle, since a fix that removes the budget would pass the other
three and be wrong.

Full package suite after the change: **254 passed** (250 before, plus these four).

## 4. Relationship to the other two load-path defects

Three distinct things were wrong, and they had to be separated because each one
masks the others:

| # | Defect | Kind | Status |
|---|---|---|---|
| 1 | `PERSIST_LOOKUP_KEYS_PER_STEP=8` throttled almost every real prompt into RETRY | configuration | **fixed** (`2048`), see `FIX-LOOKUP-BUDGET-20260911.md` |
| 2 | The lookup gate deadlocked because its only rotation hook does not run for a request it deferred | code, liveness | **fixed and verified here** |
| 3 | `prepare_load` is never reached, so a resolved disk hit is never turned into a transfer | upstream interaction | **open**, O1/O3 |

With 1 and 2 fixed, the engine no longer hangs and sessions complete, which is what
makes the session matrix in the accompanying report measurable at all. Defect 3 is
unchanged: `kv_offload_load_bytes_total` is still 0, so the tier still does not
serve, and the report says so plainly rather than presenting the fixed hangs as a
working tier.

The general lesson, which cost several engine restarts to learn: **a throttling
signal must not be able to depend for its own release on a subsystem that the
throttle itself stalls.** `RETRY` was the right word for "later", but the only
thing that could make it later was the work it had just blocked.
