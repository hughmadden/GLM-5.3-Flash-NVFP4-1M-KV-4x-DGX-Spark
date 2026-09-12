# FIX ROUND 21 — the matrix complete, both POs sent (2026-09-12 ~22:15 AEST)

Goal-round 21. The full A/B matrix finished cleanly after the driver
fixes (health-wait on the idempotent mode path, fire-and-forget up
issuance). All 18 cells (3 modes × 3 sizes × 2 concurrencies, 2 reps)
recorded in `WRITEBEHIND-BENCHMARK-REPORT-20260912.md`.

Headline: evict-only dominates eager on every measured axis — cheaper
prefill at every cell (64k conc1: −12% vs OFF, eager −25%), decode at
OFF parity (eager −37% at 256k conc1), flood absorption with 36× less
write amplification, restores 2.7× faster with 3× fewer bytes. Spec
acceptance flat across modes. Evict-only is the recommended default;
eager remains as the dial's other endpoint.

Both Pushover alerts sent at priority −1 (A/B speeds; full benchmarked
report). Fleet left healthy in evict-only mode. Repo launcher
sudo-token preflight back-ported. Public republish awaits Hugh's
explicit ask.
