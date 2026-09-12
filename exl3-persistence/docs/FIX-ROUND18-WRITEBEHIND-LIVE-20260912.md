# FIX ROUND 18 — the write-behind tier is LIVE (evict-only, R1 achieved) (2026-09-12 ~15:30 AEST)

Goal-round 18. Patch 0009 baked (`ffb9f100`, loaded `43d54a41`), fleet
running `store_mode=evict_only`, `PERSIST_MIN_DISK_LOOKUP_TOKENS=4096`,
fingerprint pin retained. Engine healthy through the full ladder.

## 1. The bring-up bugs (three live bug classes, all now pinned engine-free)

1. Drain comprehension crash (`TypeError: cannot unpack`) — a mangled
   port line; fixed and pinned by an AST drain test.
2. `TransferJob.__init__() got an unexpected keyword argument` — the real
   `TransferJob` is {req_id, src_spec, dst_spec} only; the status lives in
   `TransferJobStatus` registered in `self._jobs` (the completion loop
   keys on it). Fixed and pinned by the drain shape test
   (`test_drain_eviction_queue_registers_job_and_releases_declined`).
3. Worktree contamination during patch generation (context-mangled
   first 0009) — caught by the AST suite's method bindings; the final
   0009 is regenerated from verified-pristine trees by a one-pass
   idempotent script with per-anchor diagnostics.

Plus two ops findings: Romeo's qemu binfmt needed reinstall; the fleet
launcher had DIVERGED from the repo (sudo-token preflight + quoting) —
restored from backup and patched in place; the repo launcher needs the
sudo-token fix back-ported (tidy-up item).

## 2. The evict-only ladder (probe_prefill_tax, step-driver, settle 30)

| size | cold wall (tok/s) | cold store | eager cold store | OFF cold wall |
|---:|---:|---:|---:|---:|
| 2k | 1.73s (1,111) | **0 MB** | 1.3 MB | 1.66s |
| 8k | 7.19s (1,081) | **0 MB** | 253 MB | 5.04s |
| 16k | 8.11s (1,970) | **0 MB** | 462 MB | 8.04s |
| 32k | 15.77s (1,992) | **0 MB** | 951 MB | 15.51s |
| 64k | 32.30s (2,003) | **0 MB** | 1,469 MB | 31.46s |

Warm rows: GPU-served at parity (0.73–1.82s; cached up to 64,512/65,536)
with **zero warm re-stores** (eager paid ~159 MB per warm recompute).
Background eviction copies drain visibly (159 MB per warm-probe window —
the prior sessions' evicted blocks, held and copied at background
priority), engine healthy throughout. A 32k warm-probe also served
16,128 external hits from disk (cross-session restore works).

## 3. Findings for the design

- **R1 achieved**: zero store bytes on any request path, every size.
- **Residual tax**: prompts ≥ the 4096-token gate that MISS everywhere
  pay the lookup scan (~1–2s at 8k: 7.19s vs 5.04s OFF; invisible at
  16k+ where it amortizes). This is the design's accepted may-miss cost;
  the dial is the threshold and a scan bound (queued for tuning).
- 2k rows: gate skips the disk entirely (R3) — parity with OFF.

## 4. Queue for round 19

1. Burst/restore under evict-only (eviction probe at flood scale: the
   sink must keep up; skip rate is the health metric).
2. The 8k-class lookup-miss tax: threshold tuning and/or a bounded scan.
3. Then the PO'd A/B matrix (concurrency × sizes incl. 100k+) and the
   full report.
