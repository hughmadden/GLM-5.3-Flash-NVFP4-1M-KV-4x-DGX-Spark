# FIX ROUND 19 — the flood verdict: write-behind absorbs the burst, restores 2.7× faster than eager (2026-09-12 ~16:30 AEST)

Goal-round 19.

## 1. The allocation-stall fix (live-learned)

The first evict-only flood crashed in `get_new_blocks → popleft_n:
AssertionError`: the scheduler's accounting assumes every refcount-zero
block is in the free queue, and the sink's held blocks are invisible to
it — a flood drained the queue to the assertion. **Demand-aware holding**
(0009, `1e77fdd4`): the sink declines new holds under a free-queue
watermark, and the drain proactively returns the newest holds when
pressure appears. Same-thread as allocation, so the check is race-free.
The design doc's "never a stall" claim now holds by construction, not
hope.

## 2. The flood-scale A/B (eviction probe, 400 × 9k = 3.6M tokens > 3.4M pool)

| metric | eager (round 11) | **write-behind evict-only** |
|---|---:|---:|
| Flood throughput | 2,241 tok/s | 2,522 tok/s (+12%) |
| Engine through flood | healthy | **healthy** (no stall) |
| Written during flood | 73.8 GB | **2.07 GB** (only genuine evictions) |
| Allocation declines | 228 | **11** |
| Cross-session loads during flood | 7.17 GB | 4.48 GB |
| **Restore after GPU eviction** | **4.15 s, 1.03 GB** | **1.51 s, 341 MB — 2.7× faster, 3× fewer bytes** |

`SERVE PATH AFTER EVICTION FLOOD: disk tier`, at 1.51s. Evict-only stores
exactly what the GPU evicted (not whole-prompt copies), so the restore
transfers less and finishes sooner. The design's value case #1
(heavy-concurrency overflow absorbed, restores better than cold) is
measured in both modes — and the write-behind wins.

One anomaly recorded honestly: phase-1 (settled warm repeat) missed both
tiers (4.15s recompute, no hits) while phase-3 (post-flood) hit disk
cleanly. Suspected: phase-0's blocks were themselves sink-held at
phase-1 time (not yet re-queued, hence no GPU hit) and the disk probe
raced the in-flight copies. Not a correctness issue (pure may-miss);
flagged for the tuning round.

## 3. State

Fleet: r21 image (`f0b52729`), evict-only, fingerprint pin retained,
health 200. Commits through `1e77fdd4`. Suite/chain green.

## 4. Round 20

The PO'd A/B matrix: prefill/decode × concurrency (1/2/4/8) × sizes
(2k→256k+, including 100k-600k-class agentic contexts per Hugh), OFF vs
eager vs evict-only, with the skip counters surfaced; then the full
benchmarked report PO.
