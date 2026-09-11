# REQUEST CAP — LOCATION IDENTIFIED (2026-09-11 AEST)

The cap is not a vague "UMA over-commitment". It is **one named allocation in the sparse-indexer
profiling branch**, and the lever is a documented environment variable. Everything below is derived
from the code plus measurement; the confirmation experiment is at the end.

---

## 1. The location

**`vllm/model_executor/layers/sparse_attn_indexer_kpool.py:327`**, inside
`sparse_attn_indexer_kpool()` (line 260), in its **memory-profiling branch**:

```python
# Reserve profiler-visible memory for the worst-case decode logits,
# whose shape is [B * next_n, max_model_len]. This profiling branch
# returns before invoking the logits kernel itself.
worst_decode_tokens = min(sched.max_num_seqs * (num_spec + 1),
                          sched.max_num_batched_tokens)          # :319
decode_logits_elems = worst_decode_tokens * max_model_len * 4    # :324  float32, uint8 sentinel
prefill_cap_elems   = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024   # :325
max_logits_elems    = max(decode_logits_elems, prefill_cap_elems)            # :326
_ = torch.empty(max_logits_elems, dtype=torch.uint8, device=...)             # :327
```

For our configuration (`MAX_MODEL_LEN=850000`, `MAX_NUM_SEQS=4`, `num_spec=7`,
`MAX_NUM_BATCHED_TOKENS=7168`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` unset):

| term | value |
|---|---|
| `worst_decode_tokens = min(4 × (7+1), 7168)` | 32 |
| `decode_logits_elems = 32 × 850000 × 4` | 103.8 MiB |
| `prefill_cap_elems = 512 × 1 MiB` (default is **512**) | **512 MiB** |
| `max_logits_elems = max(103.8, 512)` | **512 MiB** |

Two properties make this the prime suspect:

1. **The 512 MiB env cap binds, not `max_model_len`.** At util 0.85 the failure margin was ~884 MB,
   so this single reservation is ~58 % of all the headroom the engine had. Corollary: **lowering
   `MAX_MODEL_LEN` would NOT have shrunk this buffer** — it only becomes the binding term above
   `max_model_len ≈ 4.19 M`. The lever is `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`.
2. **It is a reservation, not an allocation.** The result is assigned to `_`, so the reference is
   dropped immediately — but PyTorch's caching allocator keeps the block *reserved*, so the driver
   still holds it. It is taken on **first indexer use**, i.e. the first prefill — which is exactly
   when every death occurred.

## 2. Also ruled out, by arithmetic (no engine needed)

The **sparse-indexer workspace** is *not* the culprit:

```
entries = min(max_num_seqs, MNBT) × cdiv(max_model_len + num_spec, compress_ratio)
        = min(4, 7168) × cdiv(850007, 4) = 4 × 212502 = 850,008 entries
bytes   = 850,008 × 132 B (128 fp8 + 4 B scale) + 1 MiB radix = 112.2 MiB
```

and it is **profile-time locked and subtracted from the KV pool** — so it never competes with prefill
transients. Note the `GLM53_INDEXER_WORKSPACE=rightsize` patch returns its 4,173 MiB saving **to the
KV pool**, which is preallocated: the patch is good for the pool but does nothing for headroom.

## 3. Confirmation experiment (run, measured)

Back to the **failing** configuration (`GPU_MEM_UTIL=0.85`), changing only
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512 → 96`:

| | util 0.85, cap 512 (did fail) | util 0.85, cap 96 |
|---|---|---|
| host MemAvailable | **884 MB** | **2,866 MB** |
| KV pool | 55.77 GiB | 56.74 GiB |
| a ~150k-token prompt | **killed the engine** (died at ~83–90k) | **engine survived**; request sat `waiting_by_reason="deferred"` |

The post-fix state is diagnostic in itself: `num_requests_waiting_by_reason{reason="deferred"} = 1`
with `running = 0` — the request is now held by the **KV-transfer** constraint, not by memory. The
UMA death is gone; the next limit in line is the persistence connector's staging/admission, which is
non-fatal.

So: the 512 MiB reservation is a **binding contributor** to the request cap, and cutting it recovers
~2 GB of headroom and the margin needed to survive large prefills.

## 4. How the location was found — stub harnesses vs instrumentation

**Stubbed paths first (instant, no engine).** Both quantities above are *closed-form functions of the
config*, so I evaluated them in one line of Python. That is a stub harness for the engine's memory
planner: it is exact for profile-time reservations, costs nothing, and — the important part — it
decided *which lever matters* (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, not `MAX_MODEL_LEN`) before any
launch. The same discipline produced the persistence fixes: `conc_runner.py` + `conc_stubs.py` run
the real store with dummy data in 0.07 s, which is how the lease-fsync cost (8.697 ms → 0.024 ms) was
found and fixed with no engine at all. The repo also already carries a no-GPU proof harness for the
indexer chunking (`tests/test_indexer_workspace.py`, exhaustion over randomized legal batches).

**What cannot be stubbed.** A driver-level `NV_ERR_NO_MEMORY` is not fakeable: it depends on a 128 GiB
UMA carrying a ~101 GiB carve-out. So the *existence and size* of the cap had to be measured on the
engine. The stubs narrowed the search space; the engine only confirmed.

**Instrumentation used, in order of value:**

| instrument | what it gave |
|---|---|
| **1 Hz UMA sampler** (`MemAvailable`/`Cached`/cgroup) on all four ranks | the smoking-gun timeline: 2484 → 2086 → 1631 → 859 → 743 → **568 MB**, then release to 115.9 GB |
| `sudo dmesg -T \| grep NVRM` | names the failing driver call (`_memdescAllocInternal`) and its exact timestamps |
| `docker inspect --format '{{.State.ExitCode}} {{.State.OOMKilled}}'` | proved it was **not** a container OOM: exit 0, OOMKilled=false |
| `vllm:num_requests_waiting_by_reason` | revealed the post-fix state is a KV-transfer deferral, not memory |
| `VLLM_DEBUG_WORKSPACE=1` (upstream switch, already in the codebase) | logs every workspace resize with its call site — the fastest way to see profile-time reservations |
| `vllm:spec_decode_*` counters | exposed that the "decode slowdown" was acceptance-length variance, not engine speed |

**Next instrument if this needs to go further:** `torch.cuda.memory._record_memory_history()` +
`_dump_snapshot()` around one failing request gives every allocation with its Python stack — the
definitive attribution if another multi-hundred-MB reservation is hiding. `nsys profile
--cuda-memory-usage=true` is the driver-level fallback.

## 5. State left behind

`GPU_MEM_UTIL=0.75` (validated: 404,749-token prompt survives) **plus**
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=96`. `MAX_MODEL_LEN` is still 850,000 — now that the biggest
context-independent reservation is bounded, the safe prefill bound should be re-bisected and
`MAX_MODEL_LEN` set to it so an over-large request is refused with a clean 400 instead of deferring
indefinitely.
