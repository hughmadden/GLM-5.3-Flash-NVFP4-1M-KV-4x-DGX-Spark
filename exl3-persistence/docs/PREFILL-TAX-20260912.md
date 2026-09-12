# PREFILL TAX — persistence ON vs OFF across context sizes, cache-hit vs cold (2026-09-12 ~13:30 AEST)

Measured on the baked `20260912-83252ea89` image (all eight engine patches
+ the full package), flusher stopped, step-driver active (a tiny request
every ~3s so the finish-drain advances on an otherwise-idle engine —
identical load in both arms). Unique seeded ledger prompts per size
(guaranteed cold), max_tokens=8 to isolate prefill. Single sample per
cell; the engine's prefill variance is ±20–30%, so read the trend, not
the last digit. Raw logs: `/tmp/tax_on5.out` (ON), `/tmp/tax_off.out`
(OFF).

## Cold (first sight of a prompt — the tax)

| Context | OFF wall (tok/s) | ON wall (tok/s) | Tax | Stored |
|---:|---:|---:|---:|---:|
| 2,048 | 1.66s (1,153) | 1.99s (964) | **+20%** | 1.3 MB |
| 8,192 | 5.04s (1,542) | 7.84s (990) | **+55%** | 253 MB |
| 16,384 | 8.04s (1,990) | 8.82s (1,815) | **+10%** | 462 MB |
| 32,768 | 15.51s (2,025) | 22.73s (1,382) | **+47%** | 951 MB |
| 65,536 | 31.46s (2,057) | 38.30s (1,690) | **+22%** | 1,469 MB |

The tax scales with admitted store bytes: synchronous all-rank admission
RPCs plus pump writes sit on the prefill path (review §3.1 — async
admissions is the known fix). At 2k the tax is barely visible; at 8–32k
it is 2.8–7.2s on a 5–22s prefill. **Amortized view: the tax is paid
once per unique context; a single repeat erases it (below).**

## Warm (immediate-context repeat — cache hit)

| Context | OFF warm | ON warm | Served by |
|---:|---:|---:|---|
| 8,192 | 0.98s | 1.18s | GPU prefix cache, 6,912/8,192 tokens (84%) |
| 16,384 | 1.61s | 1.64s | GPU prefix cache, 13,824/16,384 (84%) |
| 32,768 | 1.26s | 1.52s | GPU prefix cache, 29,952/32,768 (91%) |
| 65,536 | 0.72s | 1.02s | GPU prefix cache, 64,512/65,536 (98%) |

Warm latency is statistically identical across arms (ON is +0.2–0.3s,
within noise — the disk-tier lookup scan). Both arms serve a warm 64k
context in ~1s versus 31–38s cold: the cache (GPU while resident) is
worth 30–60×.

## Disk-tier serve (GPU cache could NOT hold the context)

Separate measurement (round 11, eviction probe): after a 3.6M-token flood
forced the GPU pool to evict a 9k session, the re-request resumed from
the **disk tier in 4.15s (1.03GB transferred) versus 14.8s full
recompute** — a 3.6× win exactly in the "free cache RAM cannot fit the
context" regime. That is the persistence tier's reason to exist; the
cold tax above is its price.

## Measurement findings worth keeping

1. **The finish-drain creeps ~1 key per scheduler step** (visible in the
   store-batch traces), so a fresh prompt's disk snapshot completes only
   after many post-response steps. An idle engine advances the drain only
   when a later request drives steps — a lone cold→warm pair can never
   see its own disk snapshot, and even the GPU prefix cache won't serve
   the immediate repeat until the hold releases. The probe therefore
   uses a step driver plus a two-phase warm (the first repeat drives the
   drain, the second is the measurement).
2. With the drain landed, the GPU prefix cache serves warm repeats at
   up to 98% of the prompt (64k), and the disk snapshot stands behind it
   for eviction (verified round 11).
3. ON-arm warm at 32k cold also recorded 16,128 external hits — partial
   cross-session prefix overlap on the ledger template; the seeded
   bodies are unique but early token runs can coincide.

## State after measurement

Engine restored: persistence ON, `VLLM_LOGGING_LEVEL=INFO`,
`PERSIST_DEBUG_TRACE=0`, flusher running, health 200, baked 20260912
image. Probe: `probes/probe_prefill_tax.py` (step driver + two-phase
warm + `--settle`).
