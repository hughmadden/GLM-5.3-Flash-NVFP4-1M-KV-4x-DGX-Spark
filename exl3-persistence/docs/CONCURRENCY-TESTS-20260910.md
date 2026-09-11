# CONCURRENCY UNIT TESTS — persistence interfaces, stubbed with instant dummy data (2026-09-10 AEST)

**Goal:** exercise persistence concurrency at every major interface with **zero engine start time and
zero inference**, then drive the errors down through a fix/re-run cycle. The GLM-5.3-Flash EXL3
engine was taken down for the duration to free all four Sparks for the fan-out.

## 1. Harness

| Piece | What it is |
|---|---|
| `tests/conc/conc_runner.py` | Zero-dependency test runner. **Stdlib only** — `pytest` is deliberately absent from the image and installing it costs >30 s. Enforces a hard **30 s load budget** per module, runs every test under a watchdog so a deadlock reports as `HANG` instead of stalling the run, and emits one JSON object per test. |
| `tests/conc/conc_stubs.py` | Instant dummy data: `dummy_limits(**over)`, `dummy_root`, `dummy_store`, `dummy_namespace` (64-hex), `dummy_blob`, `dummy_infos`, `dummy_geometry`, `dummy_token`, `run_threads` (Barrier-synchronised so races are real), `hammer`, `Stopwatch`, `summarize`. |
| `tests/conc/test_conc_*.py` | 4 fan-out suites + my central reduction/perf/smoke files. ~6,000 lines total. |

**Measured load cost: 0.07 s** for all seven package modules, **0.70 s** including `docker run`, and
**0.004–0.047 s** per test module. The 30 s budget is enforced, not aspirational: the only thing that
would have exceeded it was installing pytest, which is why the runner is stdlib-only.

Run pattern (4 CPUs / 8 GB cap, no network; repo source mounted so a fix is testable without an
image rebuild):

```
rsync -a --delete --exclude __pycache__ <persistence-pkg>/ nodeN:~/pkg-src/
rsync -a --delete <persistence-pkg>/tests/conc/ nodeN:~/conc-tests/
ssh nodeN "docker run --rm --cpus=4 --memory=8g --network=none \
  -v /home/user/conc-tests:/work:ro -v /home/user/pkg-src:/pkg:ro -e PYTHONPATH=/pkg \
  --entrypoint python3 glm53-flash-exl3-tp4-persist:20260910b-83252ea89 \
  /work/conc_runner.py --timeout 60 <modules>"
```

## 2. Fan-out and resource allocation

| Spark | Allocation | Component | Suite | Result |
|---|---|---|---|---|
| node0 | 4 CPU / 8 GB | `storage.DiskStore` | `test_conc_storage.py` (24) | 17→ pass after fixes, 0 hang |
| node1 | 4 CPU / 8 GB | `coordinator_http` (`MetadataServer`, RemoteCoordinator, auth) | `test_conc_coordinator_http.py` (33) | 30 pass, 3 defect pins, 0 hang |
| node2 | 4 CPU / 8 GB | `http_rpc`, `geometry`, `coordinator` | `test_conc_rpc_geometry.py` (61) | 57 pass, 4 defect pins, 0 hang |
| node3 | 4 CPU / 8 GB | `native`, `handlers` (pump + manager) | `test_conc_native_handlers.py` (57) | 49 pass, 8 defect pins, 0 hang |

Loopback binding works under `--network=none`, so the real `MetadataServer` and real `RpcClient` were
exercised over TCP inside the container. Testers were forbidden from editing `recipe_persistence/*`,
the runner or the stubs, and reported defects with minimal reproductions instead — all package fixes
were applied centrally to avoid four agents clobbering one source tree.

**Baseline:** the package's own 99 `unittest` tests (the 3 files that do not import pytest) run green
through the same containerised path in 211 s, and again in 266 s **after** the fixes.

## 3. Defect register

Severity: **HANG** > **CRASH** > **DATA LOSS** > **BOUND** > **PERF** > **COSMETIC**.

### Fixed in this cycle (8)

| # | Sev | Component | Defect | Fix |
|---|---|---|---|---|
| 1 | HANG | handlers | `_CloseOnce` self-deadlocked when its callback re-enters close (`c = _CloseOnce(lambda: c()); c()` never returns). Killed `pump.shutdown()`/`_Manager.shutdown()` whenever a `close_provider` finalised the object that owns the guard. | `RLock` + `_closing` re-entrancy guard; inner call returns, outer runs the callback exactly once. |
| 2 | CRASH | handlers | Concurrent `wait`/`cancel` → `RuntimeError: dictionary changed size during iteration` in `_poll`/`wait` (reproduced 1/20 rounds) | Snapshot every dict view before iterating (`list(...)` at 3 sites). |
| 3 | DATA LOSS | storage | Index exhaustion (`PRAGMA max_page_count`) raised `sqlite3.OperationalError: database or disk is full`, which the blanket `except _ERRORS: return None` turned into the **same bare `None` as the benign "key already exists"** refusal, with `failed=False` — the save path stops persisting forever and reports nothing | Revised after the tester's test caught a flaw in the first attempt (see below): `reserve_write` now **records** `failure_count`/`last_failure` (distinguishable from a cache hit) instead of latching, and `clear_failure()` provides the missing recovery path. |
| 4 | DATA LOSS | storage | A granted write reservation could be tombstoned by pure lock-queue waiting (lease expiry while queued), dropping the payload as bare `False` | **OPEN** (see below) — mechanism measured, fix deferred as it touches the durability path. |
| 5 | PERF | storage | One global `RLock` held across the whole fsync'd write; `collect(force=True)` costs 19.7 ms/tombstone under the same lock and `reserve_write` calls `collect()` every time | **OPEN** (see §4). |
| 6 | CRASH RISK | handlers | `pump.shutdown()` retried after a provider-close failure called `native.shutdown()` **twice** (vLLM's shutdown is not documented idempotent) | `_native_shutdown_done` once-flag. |
| 7 | CONSISTENCY | storage | `lease_valid()` returned `False` for a lease that `read_into()` still serves after `invalidate()` (state `'T'`) | Align `lease_valid` with `read_into`: lease branch accepts `o.state IN ('C','T')`. |
| 8 | ROBUSTNESS | storage | `usage()` raised `sqlite3.ProgrammingError` after `close()` while every other accessor is falsy; post-close `evict()`/`collect()` set `failed=True` | `usage()` catches and reports the metadata floor; maintenance calls early-return on a closed store. |

Harness bug fixed too: `conc_stubs.dummy_infos`/`dummy_geometry` emitted a **flat** `group_refs`
instead of a list of groups, so both raised `TypeError: cannot unpack non-iterable int object` and two
testers had to work around it. Now nested correctly and verified against `layout_fingerprint` and
`geometry_identity`.

### The fix/re-run cycle that mattered (DEFECT-3, twice)

This is the clearest example of why the reduce-and-re-run loop was worth running rather than shipping
the first patch.

1. **First fix:** separate caller errors (`ValueError`) from storage errors and set
   `self.failed = True` on a storage failure. My own regression test went green.
2. **Re-run of the tester's suite caught it:** their `test_index_exhaustion_is_distinguishable_and_not_permanent`
   still failed. Setting `self.failed = True` makes the store **fail-stop forever** — nothing ever
   resets that latch, so a recoverable condition (a full disk, an exhausted index budget) permanently
   bricks the rank's persistence until the store is reopened. My "fix" had traded *silent*
   degradation for *permanent* degradation. A probe confirmed the mechanism precisely:
   `invalidate()` latches `failed=True` on the same capacity error, after which `collect()` and every
   later `reserve_write` are refused with `failed=True` and a recovery reserve returns `None`.
3. **Second fix (current):** `reserve_write` **records** `failure_count` / `last_failure` without
   latching, so a capacity failure is distinguishable from a cache hit *and* the store keeps serving;
   the deliberate fail-closed latch in `invalidate()`/`collect()`/`evict()` is preserved (a durability
   fault must still stop service rather than serve a stale object); and a new explicit
   `clear_failure()` gives the operator the recovery path that did not exist. Recovery is never
   automatic, so a real durability fault is still never masked.

Without the re-run this would have shipped a store that survives a full disk by refusing every write
until someone restarts the engine — the opposite of the intent.

### Open (ranked)

| # | Sev | Component | Defect (with minimal repro) |
|---|---|---|---|
| O1 | **HIGH** | http_rpc + coordinator | **RpcClient's default admission bound rejects healthy callers instead of queueing.** `max_active=min(len(endpoints),8)`, `max_pending=max_active*pool_size` → for a 1-endpoint client, `max_active=1`, `max_pending=4`. `_admit` raises `_Transient("RPC submission capacity exhausted")` rather than waiting: 8 barrier-synced callers → exactly 4 rejected while the server has 16 idle threads. |
| O2 | **HIGH** | coordinator | **O1 becomes spurious all-rank misses.** `RemoteCoordinator` never retries an `_admit` rejection, so `fanout` reports the rank failed and the whole reservation rolls back: 2 healthy ranks, 8 concurrent **uncontended** `reserve_store`/`release` → **32/64 returned `None`** (50 %), stable across runs. Raising `max_pending` to 64 → 0 misses. In production a loaded scheduler silently loses cache writes. |
| O3 | **HIGH** | coordinator | **`RemoteCoordinator._reserve` holds the global lock across `renew()` network I/O** (`coordinator_http.py:870-881` → `:1056`). One slow rank stalls every reservation for every other key process-wide for up to `rpc_timeout` (default 10 s). Repro: injected 0.25 s renew delay → an unrelated uncontended reservation blocked 0.268–0.282 s (vs <0.01 s). Only the ticket bookkeeping needs the lock, not the fan-out. |
| O4 | MED | storage | **`invalidate()` then immediate rewrite of the same key is refused for the whole `grace_seconds` (default 60 s)** — the natural "replace a cached value" path. The duplicate guard `SELECT 1 FROM objects WHERE ns=? AND key=?` is state-agnostic and matches the tombstone. |
| O5 | MED | storage | **Tombstoned, unreachable objects count toward `max_objects`**, so with 4 invalidated objects every new write is refused although zero live objects exist. |
| O6 | MED | handlers | **`DiskTransferPump` over-admits under concurrent `submit_store`**: 8×250 submits with `rows=8, max_pending_keys=400` → 402–406 admitted. Check-then-act plus non-atomic `self.pending_keys += n`. |
| O7 | MED | handlers | **`_Manager.lookup`/`prepare_store` over-admit**: barrier-widened coordinator RPC between the capacity check and the reserve → 8 live reservations for `capacity=4`. |
| O8 | MED | coord/rpc | **`LocalCoordinator` parses the group index from `key[-4:]` without the bytes/length guard `geometry.key_group` enforces** — a 4-byte key is silently accepted against the wrong group's byte contract, and a non-bytes key escapes as a raw `TypeError` instead of the documented `None`. |
| O9 | LOW/MED | http_rpc | **`RpcClient._execute` discards every specific cause**: five distinct failures (oversize response, gzip/non-JSON body, missing `{"ok":true}`, ECONNREFUSED) all surface as `"RPC transport failed"`; oversize request and non-serialisable params both as `"RPC request rejected"`. No secret leakage (verified) — but the advertised bound/framing diagnostics are unobservable. |
| O10 | LOW | handlers | Private `_row_done(job, ...)` called twice → `IndexError` at `handlers.py:184` and/or a `free` deque poisoned with `None`. No public path found, so hardening only. |
| O11 | LOW | handlers | No-vLLM refusal is a raw `ModuleNotFoundError`, not a domain error stating "persistence requires vLLM". |
| O12 | COSMETIC | coordinator_http | A ~20 k-deep nested JSON body returns **500** (`RecursionError`, only `ValueError` is caught) instead of 400. Handler survives. |

## 4. Performance findings (4-CPU cap per tester)

- **`DiskStore` does not scale at all.** Write 16–27 ops/s and read 19–37 ops/s, **flat from C=1 to
  C=8** (c8/c1 = 1.03×). Per-op cost is ~64 ms at 4 KB and ~68 ms at 64 KB — a **fixed per-op cost**,
  i.e. fsync-dominated — and at 1 MB, C=8 is **slower** than C=1 (8.1 vs 9.7 ops/s): the global lock
  serialises and adds contention. Concurrent reads of *different* keys are strictly serial (0.94×
  "speedup").
- **`collect(force=True)` costs 19.7 ms per tombstone under the global lock** (unlink + directory
  fsync + transaction each), so 1,000 tombstones ≈ 20 s of total store stall — and `reserve_write`
  calls `collect()` on **every** call. This is the mechanism that makes O4 reachable at the default
  300 s lease.
- **`LocalCoordinator`**: 15–19 ops/s, no scaling (per-store `BEGIN IMMEDIATE` + fsync).
- **`RpcClient`/HTTP**: ~2 k req/s ceiling, latency linear in concurrency (0.44 ms → 3.98 ms p95 6.3 ms
  at C=8), flat throughput — GIL-bound. A server-side `TCP_NODELAY` is load-bearing: without it the
  same test measured 42 ms/call.
- **Native pump dispatch (synthetic, no-op store)**: 2,750 → 4,982 → 9,397 → 13,145 jobs/s at
  io_threads 1/2/4/8, knee at 4 (4 CPUs + GIL).

**Cross-cutting root cause:** each component serialises on one global lock, and disk operations are
fsync-bound at ~50 ms. The disk tier's real throughput is therefore tens of ops/s per rank, and the
coordinator/RPC admission bound converts that serialisation into *spurious failures* (O1–O3) rather
than back-pressure. O1–O3 are the highest-value fixes because they change silent cache loss into
either success or a real error.

## 4b. Final state after the fix/re-run cycle

| Run | Result |
|---|---|
| Concurrency suite, **before** fixes | 184 tests, 164 pass, 20 fail, 0 hang |
| Concurrency suite, **after** 8 fixes | **184 tests, 167 pass, 17 fail, 0 hang** (189 s) |
| Package's own 99 `unittest` tests, after fixes | **OK — 99/99, 210 s** (no regressions) |

Of the 17 remaining failures, **15 are genuine open defects** (O1–O12 above) and **2 are stale pins
of the harness-stub bug I fixed** — `test_conc_coordinator_http::test_stub_geometry_shapes_are_incompatible_with_the_package`
and `test_conc_rpc_geometry::test_stub_dummy_geometry_is_incompatible_with_identity` now assert an
incompatibility that no longer exists and should be deleted by their authors.

One failure is a **design disagreement worth escalating rather than papering over**:
`test_conc_storage::test_index_exhaustion_is_distinguishable_and_not_permanent` requires that a
capacity error never latch `self.failed`. I kept the latch in `invalidate()`/`collect()`/`evict()`
because it is a deliberate fail-closed guard against serving a stale object when a durable
invalidation cannot be proved, and instead added an explicit `clear_failure()` recovery path. If the
intent is "recoverable capacity errors must never latch", the fix belongs in those three methods
(record instead of latch for capacity errors) — a durability-semantics decision, not a mechanical one.

**Eight fixes shipped**, each verified by re-running the reporting tester's own test:
`_CloseOnce` re-entrant hang; `wait`/`_poll` iteration snapshot; `reserve_write` distinguishable
non-latching storage-failure record; idempotent `native.shutdown()`; `lease_valid`/`read_into`
consistency; `usage()` after close; post-close maintenance latch; plus `clear_failure()` and the
stub-shape harness fix.

## 5. Not tested, and why

Real multi-rank/fabric, real NVMe bind mounts (`--network=none` and dummy data only), true
multi-process ownership (`flock` gives one owner per root by design), real filesystem faults
(EIO/ENOSPC — `failed=True` poisoning after a real I/O fault is inferred, not observed), and every
path needing torch/vLLM/CUDA (`_load_native` class binding, `Geometry.buffers`, real GPU↔disk copies,
the "vLLM present but no ambient config" branch). Absolute perf numbers are ±1.5× because the four
4-CPU testers shared the boxes with each other.

**Operational hazard found during the fan-out:** all four testers `rsync --delete` into the same
`~/conc-tests` on their own Spark and several wrote `/tmp` on head — one run's captured stdout was
byte-interleaved by a sibling, and a `--delete` sync removed a file mid-session. Use per-agent
directories and unique output paths.
