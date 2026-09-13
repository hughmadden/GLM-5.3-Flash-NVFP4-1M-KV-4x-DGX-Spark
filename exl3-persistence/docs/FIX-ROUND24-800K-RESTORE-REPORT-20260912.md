# FIX ROUND 24 — 800k restore measured honestly; report + public fork updated (2026-09-12 ~23:30 AEST)

## 1. The 800k benchmark, three attempts, three lessons

1. **Seed-then-restart failed honestly**: eager seed of 819,696 tokens
   landed in 500.6s, but the finish-drain (~1 key/step at idle) never
   completed before the mode-switch restart; recovery tombstoned the
   un-drained tail (`W→T`). The restore then hit exactly 183 durable
   chunks (~421k tokens) and missed at the first gap (trace reason
   `reserve_none`). LESSON: eager's tail-land latency at idle is
   hours-class; restarting mid-drain destroys the tail.
2. **Flood-method (the round-19 protocol, scaled)**: seeded 64k + 800k,
   flooded 300×9k in 1,509s (300/300 OK), re-requested: both restored
   as FULL RECOMPUTES (64k 33.9s, 800k 542.7s, cached=0). Trace census:
   `reserve_none=298,346` — the sessions' keys were never captured.
3. **The capture window is the measured boundary**: write-behind holds
   evicted blocks in a 4096-block registry while copies drain at
   8 keys/step; the flood evicted far more than the window holds, so
   holds past the cap were declined and freed unstored. Not a defect —
   the design's dial (`eviction_hold_cap`, `eviction_store_per_step`).
   Measured partial credit: the 800k seed pass restored 423,936 tokens
   (184 chunks) from disk mid-request in 299.7s total.

## 2. What the report now says (committed `89670ee1`)

§4.2 carries all four rows (9k in-window 1.51s; 64k/800k not-captured
with the census; the 424k partial restore) plus the capture-window
explanation and the dial. The 990k answer is measurements-bracketed
extrapolation, labeled as such.

## 3. Public fork updated and pushed (`21819d7`)

`exl3-persistence/` gains: patches 0005–0009 + manifest + AST suite
(45 tests), the full package incl. 0010, sanitized probes
(ab_matrix/probe_prefill_tax/probe_session_restore; internal
identities env-parameterized), configs, the report, and docs
(FIX-ROUND15–23, implementation docs). `verify_public_sanitization.sh`:
zero findings in our lane (remaining hits are pre-existing tracked
history). Share-page publish NOT done — awaits Hugh's explicit word.

## 4. Fleet state

Evict-only, gate 32768, package 0010+0010b live, trace flag off in
env (effective next restart). Engine healthy.
