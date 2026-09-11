# FIXES — request cap + persistence decode penalty (2026-09-11 AEST)

Both defects identified, fixed, and validated on the live engine. Reproduction for each is below.
Companion documents: `FIX-PLAN-20260911.md` (the wider defect register) and
`TEST-ITERATION-AND-PROFILING.md` (fast iteration + instrumentation).

---

## FIX 1 — the request cap: UMA over-commitment (VALIDATED)

### Root cause

Not a prompt-size bug and not a persistence bug. The deployment over-committed GB10 **unified**
memory, leaving essentially no host headroom, so any sizable transient allocation failed inside the
NVIDIA driver.

Measured budget at `GPU_MEM_UTIL=0.85` (121.69 GiB UMA):

| component | GiB |
|---|---|
| weights + non-torch (`Actual usage … consumed memory`) | 45.66 |
| KV cache pool (`Available KV cache memory`) | 55.77 |
| engine host-side footprint (container cgroup) | ~10.4 |
| **host MemAvailable** | **0.86** |

The engine's own planner believed it had 18.26 GiB spare (it sizes the pool from *total* device
memory, not from what is actually free), but `free -m` showed `available 884 MB` with swap untouched.

### Evidence — the memory timeline captured at the death

A 1 Hz sampler on all four ranks walked straight into the failure:

```
t+ 0s  MemAvailable=2484 MB      t+41s  MemAvailable=1631 MB
t+11s  MemAvailable=2434 MB      t+51s  MemAvailable= 859 MB
t+21s  MemAvailable=2425 MB      t+62s  MemAvailable= 743 MB
t+31s  MemAvailable=2086 MB      t+72s  MemAvailable= 568 MB   <- NVIDIA driver fails here
                                 t+82s  MemAvailable=6089 MB   <- engine releases
                                 t+92s  MemAvailable=115900 MB
```

Kernel log at the same instant: `NVRM: … Out of memory [NV_ERR_NO_MEMORY] … _memdescAllocInternal`.
The engine then shuts down through vLLM's SIGTERM handler (`launcher.py:126-136`), exits **0** with
`OOMKilled=false`, and the API server **hangs draining the client socket** — needing a manual
`down` + `up` (~7 min). Never trust exit 0 here; the real signals are the NVRM lines and `/health`.

### Fix

`GPU_MEM_UTIL=0.85` → **`0.75`** in `env.tp4.fleet`. That is it — a configuration value, no code
change.

### Validation (engine, measured)

| | before (0.85) | after (0.75) |
|---|---|---|
| KV pool | 55.77 GiB | **43.2–43.9 GiB** |
| KV tokens | 4,333,333 | 3,398,717–3,441,025 (4.0× at 850k) |
| host available | 884 MB | **16,400–18,200 MB** |
| largest surviving prompt | **~83k tokens** (died at 90k) | **255,393 tokens survived** |

Ladder with `PERSISTENCE=off` at 0.75: 101,324 / 153,527 / **255,393** tokens all SURVIVED (52 s,
80 s, 68 s). Then re-validated with the full stack (`PERSISTENCE=on` + WAL) — see the final ladder
receipt.

**Trade-off, stated plainly:** the GPU KV pool is ~12.5 GiB smaller, so fewer tokens stay resident in
GPU. That is a good trade here precisely because there is a disk tier to absorb the overflow — the
cap fix makes the persistence tier *earn its keep* rather than be a liability.

**Still to decide (not done):** `MAX_MODEL_LEN` is still **850,000**. The safe prefill bound is now
at least 255k but has not been bisected above that. Until it is, a prompt between the measured safe
bound and 850k can still kill the engine. The right end state is `MAX_MODEL_LEN` set to the measured
safe prefill bound so an over-large request is **refused with a clean 400** instead of dying.

### Reproduce

```bash
# 1 Hz UMA sampler on each rank
ssh nodeN 'setsid nohup /tmp/uma_sampler.sh nodeN 1800 >/dev/null 2>&1 < /dev/null &'
# ascend until it dies; the failing size is the cap
ssh node0 'docker run --rm --cpus=2 --memory=4g --network=host -v /tmp/ladder.py:/ladder.py:ro \
  --entrypoint python3 <image> -u /ladder.py 100000,150000,250000,400000'
# confirm the CAUSE, not just the symptom
ssh node0 'sudo dmesg -T | grep -c NV_ERR_NO_MEMORY; free -m'
ssh node0 'docker inspect glm53-exl3-tp4-head --format "{{.State.ExitCode}} {{.State.OOMKilled}}"'
```

---

## FIX 2 — the persistence decode penalty: a durable fsync per lease grant (VALIDATED)

### Root cause

`_Manager.on_schedule_end()` runs **every scheduler step**, and the cache-lookup path grants a lease
through a **synchronous coordinator RPC**, which lands on `DiskStore.reserve_read` — where the lease
`INSERT` **commits with `PRAGMA synchronous=EXTRA`**. Every lease grant therefore paid a full fsync.

Measured on the unit harness (no engine, dummy data):

| operation | latency |
|---|---|
| lookup **miss** | 0.022 ms |
| lookup **hit** (lease granted) | **8.697 ms median / 11.39 ms p95** |

The engine metric told the same story: `vllm:kv_offload_lookup_sync_delay_seconds` ≈ **9.3 s of every
10 s window** across ~800 lookups — the engine was blocked on persistence lookups ~93 % of the time.

The store holds a **reconstructible cache**, not a journal (`kv_load_failure_policy=recompute`), so
`synchronous=EXTRA` bought no real guarantee while costing an fsync per commit.

### Fix — durability becomes an explicit policy

1. **`storage.py`**: `DiskStore(..., journal_mode=..., synchronous=...)`, validated against
   allowlists. **Defaults are unchanged (`delete` + `extra`)**, so the library contract and every
   existing test still hold.
2. **`coordinator_http.py`**: the engine's store reads `sqlite_journal_mode` /
   `sqlite_synchronous` from the spec's `extra_config`.
3. **`start-tp4.sh` + `env.tp4.fleet`**: the launcher renders those two keys from
   `PERSIST_SQLITE_JOURNAL_MODE` / `PERSIST_SQLITE_SYNCHRONOUS`; the fleet sets **`wal` + `normal`**.
4. **Directory audit**: the store's root allowlist now permits `index.sqlite-wal` / `-shm`, which WAL
   creates (the audit previously rejected them — this is what the two `cache root contains
   unaccounted files` test errors were catching, and it is now handled properly rather than by
   weakening the mode).

### Measured effect

| configuration | lease-grant latency |
|---|---|
| DELETE + EXTRA (original) | 8.697 ms |
| DELETE + NORMAL | 4.498 ms |
| DELETE + OFF | 0.054 ms |
| **WAL + NORMAL (chosen)** | **0.024 ms** |

**~360× faster**, and WAL additionally removes reader/writer serialisation (the concurrency suite
went from 50 s to 12 s wall).

### Validation (engine, measured)

Decode, warmed, median of 5, measured from `usage.completion_tokens` (never chunk counts):

| arm | median decode |
|---|---|
| `PERSISTENCE=on`, before the fix | **~12 tok/s** |
| `PERSISTENCE=off` (upper bound) | 77.2 tok/s |
| **`PERSISTENCE=on`, after the fix** | **58.1 tok/s** (range 37–80.6) |

A **4.8× recovery**, now above Mia's own published prose figure (27.1 tok/s) and into her
"structured" band at the top end. Median acceptance length 0.57; median prefill 1,375 tok/s.

> **CORRECTION (2026-09-11, see `CORRECTION-LOCKFIX-AND-DECODE-20260911.md`).** This table was
> **confounded** and must not be quoted as a 4.8× end-to-end win. Throughput is
> `(acceptance × 7 + 1) / ms/step`, and the arms above differed in acceptance, in whether the
> page-cache flusher was still running, and in leftover benchmark containers competing for the
> engine. A controlled re-run with the flusher stopped, the engine idle and 8 reps each gives:
> **lock-fix reverted 42.1 tok/s at ~83 ms/step; lock-fix applied 41.0 tok/s at ~83 ms/step** —
> i.e. the engine's *per-step speed did not change at all*, and **there is no decode bug**: our
> ~83 ms/step is faster than the ~118–125 ms/step implied by Mia's own published figures. The
> isolated durability measurement below (8.697 ms → 0.024 ms per lease grant) remains valid and is
> the reason to keep that fix.

### Reproduce

```bash
# unit level — seconds, no engine (the fast path used to develop this)
ssh node0 "docker run --rm --cpus=4 --memory=8g --network=none \
  -v /home/user/conc-tests:/work:ro -v /home/user/pkg-src:/pkg:ro -e PYTHONPATH=/pkg \
  --entrypoint python3 <image> /work/probe_miss.py"     # prints MISS / HIT medians
# engine level — confirm the live store actually switched
ssh node0 'sudo python3 -c "import sqlite3;c=sqlite3.connect(\"/home/user/kv-persist/rank0/rank-0/index.sqlite\");print(c.execute(\"PRAGMA journal_mode\").fetchone())"'
ssh node0 'docker run --rm --network=host -v /tmp/decode_bench.py:/p.py:ro --entrypoint python3 <image> -u /p.py'
```

Note `PRAGMA synchronous` is **per connection** — a host-side probe reports its own default, so use
`journal_mode` (persistent) to prove the engine applied the policy.

---

## Fast-iteration hook added for this work

`PERSIST_SRC_OVERLAY=<path to recipe_persistence>` makes the launcher bind-mount the repo package over
the installed one:

```
-v '<dir>:/usr/local/lib/python3.12/dist-packages/recipe_persistence:ro'
```

Python-side fixes are then validated on the real engine **with no image rebuild** — this is how both
fixes above were proven. It must be pushed to **all four ranks** (each container mounts its own
node's path); a rank left stale produced `NameError: name 'journal_mode' is not defined` on the
workers during this session. Deploying for real still requires an image rebuild + re-stage.

## Lesson worth keeping

Every engine death in this campaign exited **0** with `OOMKilled=false`. Only `dmesg`'s NVRM lines and
`/health` distinguish "died" from "fine". And the two most valuable measurements of the whole effort
came from instruments that cost nothing to run: a **1 Hz memory sampler** (found the cap) and a
**warmed, usage-based decode bench** (found the 5–6× chunk-counting error and the lease-fsync cost).
