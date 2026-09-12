# FIX ROUND 15 — CUTOVER COMPLETE: the fleet runs the baked 0001–0008 image (2026-09-12 ~12:15 AEST)

Goal-round 15. The recipe's full fix stack is now BAKED into the fleet
image — no container injection — with the persisted tier preserved.

---

## 1. The cutover

- Image `glm53-flash-exl3-tp4-persist:20260912-83252ea89` (all eight
  engine patches + full package incl. B5/B6) verified in-image, staged to
  all four Sparks, and deployed via the standard ritual.
- **Fingerprint preservation (delta recorded in the fleet env and here)**:
  `CACHE_FINGERPRINT=2ec52df57e01bf0cfed88c584ac100cb` is pinned so the
  new image does not re-namespace the ~74GB tier. Basis: the 0001–0008
  patches and package fixes change store/load bookkeeping, NOT KV bytes;
  the census-enforced layout_fingerprint is unchanged. (Also recorded:
  the content-addressed export round-trips to a different Id on the ranks
  than Romeo's in-docker copy for identical bytes — IMAGE_ID is pinned to
  what the ranks present, `a9c0f657…`.)

## 2. The boot regression and its root cause (efficiency note: engine
down for ~90 min across the cutover attempts; sole user, no external
traffic impact)

Two failed boots (both images) with
`ValueError: Unable to configure handler/formatter 'vllm'` — a CIRCULAR
IMPORT in vLLM's logging dictConfig. Root cause was NOT the image:
**round 10's `VLLM_LOGGING_LEVEL` removal left the launcher passing
`-e VLLM_LOGGING_LEVEL=` (set-EMPTY)**, which vLLM's dictConfig rejects
at import time. Rounds 10–12 survived only because those restarts were
`docker restart` (Env baked at the earlier create, still DEBUG); the
cutover's down/up recreated the containers with the empty value.
Reproduced in a throwaway container (`-e VLLM_LOGGING_LEVEL=` → crash;
unset or INFO → clean), fixed at the launcher
(`${VLLM_LOGGING_LEVEL:-INFO}`, with a comment explaining why empty is
fatal), fleet env set to INFO.

## 3. Verification (all on the baked image, no injection)

- In-container: 0008 guard present (5 `prefix_cacheable` refs), class
  name intact, B5/B6 present (`state-aware`).
- `health=200`; head container from the new image.
- Acceptance probe: pass 1 **40/40 tier hits + 6,912 ext_hits**; pass 2
  **1.6s** (GPU APC + disk combined serve). The fingerprint pin worked —
  the tier was visible from the first probe.

## 4. State and queue

Fleet: baked 20260912 image + overlay (manager instrumentation, B1–B3,
pump retry live — all also in the image's package). Commits through this
record. Remaining: B9/B10, the write race, §3.1 async admissions,
report/fork updates (republish awaits Hugh's ask), and the in-flight
ops-hygiene item of re-syncing the repo `env.tp4.fleet`-equivalent
documentation (the fleet env is authoritative on the launcher host).
