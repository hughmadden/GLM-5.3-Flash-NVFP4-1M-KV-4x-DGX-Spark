# GLM-5.3-Flash EXL3 4bpw at TP4 with a no-tax write-behind disk KV tier

Four DGX Spark (GB10) nodes serving **GLM-5.3-Flash EXL3 TR3 4bpw** (Mia-AiLab build) at
tensor parallelism 4 with DFlash2 speculative decoding — plus a disk KV persistence tier
that adds ~3.5M token-equivalents of session context with **zero tax on unpressurized
traffic**: an idle engine writes nothing, decode under a full pool costs ~5%, and a cached
200k session retrieves in **3.0 s** versus 112 s cold.

Full measured report: https://services.turquoisebay.ai/share/glm53-exl3-writebehind/
(source: `report/index.html` in this directory).

## What this directory contains

| Path | What it is |
|---|---|
| `patches/persistence-flash/` | The engine patch series `0001-0012` (hash-pinned `manifest.json`, `apply.py check/apply/verify`, AST suite `test_native.py`) applied to vLLM `83252ea899` at image build |
| `persistence/recipe_persistence/` | The out-of-tree spec + store + coordinator package (bind-mounts over dist-packages; ships by rsync + process restart) |
| `configs/` | The launch environment and fleet launcher (`PERSIST_*` knobs) |
| `benchmark/` | Every probe used in the report (`bench_c1c6.py` the Local Inference Labs standard, `ab_matrix.py`, `probe_prefill_tax.py`, `probe_evict_refill.py`, `probe_session_restore.py`, `probe_pressure_tax.py`, `probe_200k_restore.py`) |
| `tests/` | Engine-free unit suites (package pytest/unittest + concurrency harness) |
| `report/` `docs/` | The public report and the per-round fix/implementation records |

## Reproduce

1. **Image**: build from the repo root (`docker/`; build host needs qemu/binfmt for the
   arm64 stages). The series applies at image build via
   `patches/persistence-flash/apply.py`; `manifest.json` pins every patch and every
   touched file's before/after hash.
2. **Configure** (see `configs/`): `PERSISTENCE=on`,
   `PERSIST_STORE_MODE=evict_only` (the supported mode; eager was retired after the A/B
   showed 11-38% prefill and 37% decode costs for no restore advantage),
   `PERSIST_MIN_DISK_LOOKUP_TOKENS=32768`,
   `PERSIST_EVICTION_STORE_HIGH_WATERMARK=<~20% of pool blocks>` (pressure gate; 0=off),
   `PERSIST_LOOKUP_KEYS_PER_STEP=2048` (anything smaller hangs restores).
   Idle-flusher knobs: `PERSIST_IDLE_FLUSH_STALE_SECONDS=3600`,
   `PERSIST_IDLE_FLUSH_PER_SCAN=32`, `PERSIST_IDLE_FLUSH_SCAN_SECONDS=5`.
   Pin `CACHE_FINGERPRINT` across image changes or the persisted tier re-namespaces.
3. **Launch**: one container per rank via the fleet launcher; the package overlays from
   `persistence/recipe_persistence/`.
4. **Measure**:
   ```
   python3 benchmark/bench_c1c6.py --url http://<head>:8888 --model <served-name> --rounds 3
   python3 benchmark/ab_matrix.py --modes off,evict --sizes 8192,65536,262144 --conc 1,4
   python3 benchmark/probe_evict_refill.py --n 400 --tokens 9000 --settle 45   # flood + restore
   ```
   (`ab_matrix.py` mode switches need `PERSIST_FLEET_HOST` and `PERSIST_BENCH_URL` set.)

## What the patches do (one line each)

- **0001-0004** load-path contracts: drained-outcome API, failure frontiers, grouped
  recovery ordering, finished-store frontier.
- **0005-0008** made the tier actually serve: pending-key frontier walk, finish-drain
  liveness bound, finished partial-chunk store, non-cacheable-group exclusion (kpool).
- **0009** the write-behind design: evict-only stores via a deferred-free sink fed by
  GPU prefix-cache evictions, demand-aware under allocation pressure; R3 small-prefill
  read gate.
- **0010/0010b** lookup cost: in-RAM durable-key index (misses cost a set membership
  test, not per-key sqlite reads on four ranks) + in-process dispatch for the rank-local
  coordinator leg. Took the residual prefill tax from -12% to ~0%.
- **0011** the pressure gate: an idle pool writes nothing.
- **0012** the idle-time staleness flusher: stale APC content lands on disk during
  decode-free steps, in-place, fenced - durability without memory pressure and zero
  decode cost by construction.

## Current measured state (2026-09-13)

Cold prefill at OFF parity (64k: 2,026 vs 2,028 tok/s OFF); decode within noise except
deep-context under pressure; 400x9k flood absorbed with 2 GB written (eager wrote 74 GB)
and an evicted 9k session restored from disk in 1.42 s; production ladder 32k-400k clean,
850k refused at the configured max. Practical single-request ceiling ~405k tokens (UMA;
see report §4.5 for the baseline-comparison question). Known open item: large-session
cold disk restore beyond the draft window (report §4.2).

## Credits

Tony (d2)Wild (TP4 playbook and baseline), Mia AI Lab (EXL3 build and engine lineage),
Local Inference Labs (Spark builds, LMCache design lineage, the standard benchmark),
the vLLM project and community (engine + carried fixes). Full attribution in the repo's
`CREDITS.md`. The persistence tier is Turquoise Bay AI's additive work.
