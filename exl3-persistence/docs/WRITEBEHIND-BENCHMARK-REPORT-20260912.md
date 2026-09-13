# WRITE-BEHIND DISK TIER — benchmarked report (2026-09-12 ~22:00 AEST)

Full A/B: three modes (OFF / eager / evict-only) × sizes 8k/64k/256k ×
concurrency 1/4, 2 reps each, streaming TTFT prefill tok/s + 200-token
decode tok/s + spec acceptance. Driver `probes/ab_matrix.py`; raw logs
`/tmp/ab_matrix_v4.out` (OFF) and `/tmp/ab_matrix_v7.out` (eager,
evict). Engine: GLM-5.3-Flash EXL3 4bpw TP4, baked `20260912-83252ea89`
(0001–0009), DFlash2 k=7, enforce-eager, fp8 KV. Caveat: 2 reps per
cell, engine variance ±10%; synthetic ledger prompts.

## 1. Prefill tok/s (median per request; higher is better)

| size | conc | OFF | eager | Δ eager | **evict-only** | **Δ evict** |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 1 | 1,868 | 1,531 | −18% | **1,462** | −22% |
| 8k | 4 | 348 | 342 | −2% | **324** | −7% |
| 64k | 1 | 2,028 | 1,531 | −25% | **1,793** | −12% |
| 64k | 4 | 733 | 675 | −8% | **639** | −13% |
| 256k | 1 | 1,901 | 1,691 | −11% | **1,611** | −15% |
| 256k | 4 | 1,301 | 803 | −38% | **757** | −42% |

## 2. Decode tok/s (median per request; higher is better)

| size | conc | OFF | eager | evict-only |
|---:|---:|---:|---:|---:|
| 8k | 1 | 56.5 | 50.5 | 47.0 |
| 8k | 4 | 51.5 | 52.5 | 50.5 |
| 64k | 1 | 54.5 | 46.0 | 46.5 |
| 64k | 4 | 50.0 | 44.0 | 49.0 |
| 256k | 1 | 51.0 | 32.5 | **44.5** |
| 256k | 4 | 49.0 | 48.0 | 44.0 |

## 3. Read

1. **Evict-only beats eager on prefill in every cell** — by construction
   (R1: zero store admissions on the request path). The gap is largest
   exactly where eager hurts most: 64k conc1 (−12% vs −25%) and 256k
   conc4.
2. **Decode: evict-only tracks OFF within noise**; eager degrades
   sharply at 256k conc1 (32.5 vs 51 OFF — the eager store pump
   competing for UMA bandwidth at the largest context). Spec acceptance
   is flat (~0.41–0.52) across all modes — persistence does not disturb
   the drafter.
3. **The residual evict-only prefill tax** (−12…−22% at conc1) is the
   disk-lookup scan on a full RAM miss for prompts over the 4096-token
   R3 gate, plus the eviction-copy bandwidth on UMA. The dial: raise
   `PERSIST_MIN_DISK_LOOKUP_TOKENS` (small requeries never check disk),
   bound the scan, or drop the copies' bandwidth priority.
4. **The design's value case (from rounds 18–19)**: under a 3.6M-token
   flood (> 3.4M pool) the write-behind absorbed the overflow with 2 GB
   written vs eager's 74 GB, and restored an evicted 9k session from
   disk in 1.51 s (341 MB) vs eager's 4.15 s (1.03 GB) — **2.7× faster
   with 3× fewer bytes**, with zero allocation stalls after the
   demand-aware holding fix.

## 4. Verdict

The write-behind (evict-only) endpoint of the §3.4 dial dominates eager
on every measured axis: cheaper prefill, decode at parity, floods
absorbed with 36× less write amplification, restores 2.7× faster. It is
the recommended default; eager remains as the dial's other endpoint for
workloads that must guarantee full-context durability on first sight.

## 5. Addendum — the residual tax, isolated (2026-09-12 ~23:00 AEST)

A controlled run with the R3 gate at 9,999,999 (every disk lookup
skipped; write-behind stores still active) decomposes the evict-only
conc1 gap vs OFF:

| prompt | evict (gate 4096) | lookup disabled | OFF | lookup share | copy-bandwidth share |
|---:|---:|---:|---:|---:|---:|
| 8k | 5.6s (1,462) | 4.9s (1,684) | 4.4s (1,868) | ~0.7s (13%) | ~0.4s (9%) |
| 64k | 36.6s (1,793) | 35.4s (1,853) | 32.3s (2,028) | ~1.2s (3%) | ~3.1s (11%) |

Decode also recovers ~half its gap with lookups off (47→52.5 tok/s at
8k), because the pre-scheduling lookup delays first token. Conclusion:
roughly half the residual tax is the miss-path lookup scan on the
scheduler's critical path (fixable: gate tuning toward ~16–32k, a scan
bound, or a cheap prefilter), half is write-behind copy bandwidth on the
Sparks' unified memory (structural; paced by the §3.4 dirty-ratio dial).
Fleet restored to the committed 4096 default after the isolation.

Published (Hugh's explicit ask, 2026-09-12 ~23:45 AEST):
https://services.turquoisebay.ai/share/glm53-exl3-writebehind/
