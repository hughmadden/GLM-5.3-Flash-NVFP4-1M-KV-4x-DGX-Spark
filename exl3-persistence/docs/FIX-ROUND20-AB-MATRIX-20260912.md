# FIX ROUND 20 — the A/B matrix, in flight (2026-09-12 ~18:00 AEST)

Goal-round 20. The full matrix driver (`probes/ab_matrix.py`) is running:
modes off/eager/evict-only × sizes 8k/64k/256k × concurrency 1/4,
2 reps, streaming TTFT (real prefill tok/s) + 200-token decode +
acceptance. First real cells: OFF 8k conc1 prefill 1,868 tok/s,
decode 56.5 tok/s; conc4 per-request prefill 348 (4-way split).

Driver bring-up lessons (three): the launcher `up` ssh hangs unless the
background child is fully detached (`</dev/null`); `pkill -f` self-kills
when the pattern matches the invoking shell (use `pkill -f "x.p[y]"` or
`kill $(pgrep ...)`); GLM streams reasoning deltas in the `reasoning`
field, so TTFT must key on content OR reasoning_content OR reasoning.

The matrix runs ~2-2.5h across the mode restarts; results land in
`/tmp/ab_matrix_v4.out` and will be compiled into the report doc with
the PO in the next round.
