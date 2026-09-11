# FINDINGS — first real boot of the EXL3 TP4 + persistence lane (2026-09-10 AEST)

**Author lane:** `glm-5.3-flash-exl3-tp4-persistence/`. **Written on head, 10 September 2026 AEST**
(all timestamps Sydney). Everything below is a measured result or a sourced statement; inferences
are labelled. The lane reached the private deployment runbook §3H end state — GLM-5.3-Flash EXL3 TR3-4bpw at TP4 on
node0-3 with node-local disk KV persistence live, serving through head LiteLLM and head dsh — after
**five printed-recipe defects were fixed and one image rebuild**.

Read this before executing the private deployment runbook. That runbook was written from build-host and CPU-side evidence
and had never been run on a GPU; these are the places where it could not execute as printed.

---

## 1. Defects found and fixed

### 1.1 The image Id does not survive `docker save` → `docker load` (the private deployment runbook §1.2 / §3.2 / §3.3)

`IMAGE_ID=sha256:6543e3ad71712b13963580e4076f8d391a428e07992d96a6839220c9dd140a85` is the Id on
**build-host**, whose docker 29.7.2 uses the `io.containerd.snapshotter.v1` store. Loading the saved
tarball on the Sparks (docker 29.2.1, classic store) produced a **different** image Id:

```
build-host   glm53-flash-exl3-tp4-persist:20260910-83252ea89  sha256:6543e3ad…  34.4 GB (docker images)
node0  glm53-flash-exl3-tp4-persist:20260910-83252ea89  sha256:9d29fa37…  23.2 GB
```

The content is identical — both images have **the same 55 `RootFS.Layers` digests**
(`docker image inspect --format '{{json .RootFS.Layers}}' | md5sum` → `e1e4de0d239f62368f5c6db82b3bbbb1`
on both) — only the recomputed config digest differs, which is exactly what the OCI→docker-archive
conversion does. build-host's `.Size` is also compressed accounting (10,815,588,098 B) versus the Sparks'
uncompressed layer total (23,243,385,483 B).

**Consequence:** `preflight` compares every rank's live `docker image inspect --format '{{.Id}}'`
against the pinned `IMAGE_ID` and would refuse the fleet on all four ranks.

**Resolution:** after `docker load`, pin the **loaded** Id (the private deployment runbook §3.2's permitted post-rebuild
`IMAGE`/`IMAGE_ID` substitution, applied in the working copy on node0 only, with the new sha256
re-recorded). Two builds were pinned this way: `9d29fa37…` for `:20260910-83252ea89` and
`113024ec…` for the fixed `:20260910b-83252ea89`. All four ranks agreed bit-for-bit each time.

### 1.2 Launcher: `$HOME` mounts were single-quoted into a remote shell (FIX-9)

`launch_rank()` built the `docker run` command as a string handed to `on_rank`, which executes it via
`bash -lc` (rank 0) or `ssh <host>` (ranks 1-3). Inside that string the three cache mounts were
written `-v '\$HOME/.cache/vllm:/root/.cache/vllm'`, so the **remote shell saw single quotes and did
not expand `$HOME`**; docker then received a literal `$HOME` and refused:

```
docker: Error response from daemon: create $HOME/.triton/cache: "$HOME/.triton/cache" includes
invalid characters for a local volume name … If you intended to pass a host directory, use absolute path
```

The same bug in the preceding `mkdir -p '\$HOME/.cache/vllm' …` line created a junk tree
`/home/user/$HOME/.cache/vllm` on the first node launched (node3).

**Resolution (working copy on node0):** six single-quote pairs around `\$HOME…` removed so the
remote shell expands the path; junk tree deleted. `bash -n` clean.
`start-tp4.sh` sha256 `5d8ba267…` (as printed) → `5dbf9cbb…` (FIX-9) → `98df6847…` (with 1.4 below).

### 1.3 `--limit-mm-per-prompt` must be a JSON dict on this base

`env.tp4.fleet` carries Mia's `LIMIT_MM=100`, and the inner script emits
`--limit-mm-per-prompt "${LIMIT_MM}"`. On vLLM `83252ea89` the field is a `dict[str,int]`, so the
engine died before loading a single weight:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
limit_mm_per_prompt
  Input should be a valid dictionary [type=dict_type, input_value=100, input_type=int]
```

Tested inside the image (`FlexibleArgumentParser` + `EngineArgs.add_cli_args`, with a GPU visible
because arg parsing needs device inference):

```
FORM '100'         -> 100                (int → pydantic rejects)
FORM 'image=100'   -> REJECTED exit 2    (key=value form not supported for this field)
FORM '{"image":100}' -> {'image': 100}   (accepted)
```

**Resolution:** working-copy env line became `LIMIT_MM='{"image":100}'` (the quotes are required so
the sourcing shell keeps the braces and inner double quotes). Nothing else about Mia's intent
changed.

### 1.4 Preflight could not stat the persistence token as `user`

`provision-persistence-token.py` creates `/run/glm53` mode `0700 root:root` with the token
`0600 root` inside — deliberately. `preflight` ran `test -f` / `stat` **as user**, which cannot
traverse a `0700` root directory, and failed closed:

```
token file /run/glm53/persistence-token absent or a symlink -- see provision-persistence-token.py
preflight failed -- nothing was started
```

**Resolution (working copy on node0):** the four metadata commands
(`test -f`, `test ! -L`, `stat -c %a|%U|%s`) now run under `sudo` — passwordless sudo is already a
prerequisite of this recipe (the memory ritual and flusher both hard-exit without it). **The token
value is still never read, echoed or logged.** Directory mode and file mode are unchanged.

Note: the private deployment runbook's own §2.7 verification command (`ssh $n 'stat -c … /run/glm53/persistence-token'`)
has the same defect and only works with `sudo`.

Token size is **49 B**, not the 65 B the private deployment runbook predicts. The same value is installed on all four
nodes (verified by sha256 of the file, not by reading it).

### 1.5 Persistence spec construction had no ambient vLLM config — image rebuild required

The boot reached the weight load and then died 6 minutes in:

```
File ".../recipe_persistence/native.py", line 706, in __init__
File ".../recipe_persistence/native.py", line 635, in _prefix_hash_algorithm
ValueError: persistence requires an active vLLM cache configuration
```

Call path (all file:line read from the image):

```
EngineCoreProc.__init__            vllm/v1/engine/core.py:1085
  Scheduler.__init__               vllm/v1/core/sched/scheduler.py:339
    KVConnectorFactory.create_connector
      OffloadingConnector.__init__ offloading_connector.py:66
        OffloadingSpecFactory.create_spec → NodeLocalDiskOffloadingSpec.__init__
```

`recipe_persistence` derives the prefix-cache hash algorithm from the **ambient** vLLM config,
because `OffloadingConfig` carries no `cache_config` since vLLM `a9531edfa6` (#48150) — its
docstring says so explicitly and refuses rather than assuming `sha256`. That is correct on the
worker path (vLLM wraps workers in `set_current_vllm_config`), but **the scheduler path constructs
the spec inside `EngineCoreProc.__init__`, and `vllm/v1/engine/core.py` never calls
`set_current_vllm_config` anywhere**. `OffloadingConnectorScheduler.__init__` also reads the same
value via `spec.get_manager()`, so deferring resolution inside `native.py` would not have helped:
both reads happen in the same construction.

**Resolution:** a new anchored, hash-pinned build patch —
`patches/vllm/patch_offloading_ambient_config.py` — wraps spec construction (and the
scheduler/worker-side object built from it) in `with set_current_vllm_config(vllm_config)`. It is
fail-closed: the target file must hash to BEFORE `6fb06a0c…`, both anchors must appear exactly once,
the result must hash to AFTER `c7a8611e…`, and a re-run is a no-op. It accepts both the source-root
and bare-`vllm`-package layouts, like `persistence-flash/apply.py`. Wired into `configs/build/Dockerfile`
immediately after the `persistence-pkg` install layer. **Rebuilt as
`glm53-flash-exl3-tp4-persist:20260910b-83252ea89`** (fleet Id `sha256:113024ec…`, tarball
10,628,123,697 B sha256 `198836ef…`, recipe stamp `d42c9da7be253046`).

After the fix: `/health=200` in 390 s, `coordinator rank 0..3 listening on 198.51.100.{1,2,3,4}:8976`,
`all four coordinator listeners up`.

---

## 2. Other deltas worth knowing

- **Failed boots leave the workers resident.** When rank 0 dies, `up` errors out and the rank 1-3
  worker containers keep running with the model in memory (observed: 44 minutes, ~113 GB of
  anonymous memory per node; a subsequent memory ritual then had almost nothing to reclaim). Correct
  retry order is **`down` → memory ritual → flusher → `up`**, not ritual-then-`up`.
- **`pkill -f '…flusher-unconditional.sh'` over ssh self-matches the remote shell's own command
  line** when the same invocation also contains the flusher path, killing the session before the new
  flusher starts. Use a small script file, or the `[f]lusher` bracket trick, and verify with
  `ps -eo args | grep -c '[f]lusher-unconditional.sh'`.
- **The `.tar.zst` sidecar path is build-host-absolute** (`$STAGE/images/…`), so
  `sha256sum -c` from `~/flash-exl3-staging/images/` cannot work on a Spark. Verify the staged bytes
  against the pinned digest directly instead.
- **Do not sample the tarball size until the build log prints `Build complete.`** The file appears at
  its final path while `docker save | zstd` is still writing; a staged copy taken then is short and
  silently wrong (observed 6.2 GB of a 10.6 GB artifact). Verify size **and** sha256 on the node.
- **`du -Lsb` is the wrong instrument for the §2.2 byte checks.** It double-counts HF `blobs/` (once
  through `snapshots/` symlinks, once directly). The private deployment runbook's numbers are *resolved snapshot tree*
  sums: EXL3 `175,715,854,754` B (safetensors 175,642,157,752 B, 120 shards) and DFlash2
  `2,342,175,855` B, both reproduced byte-exact on all four nodes with
  `find -L <snapshots>/<rev> -type f -printf '%s\n' | awk '{s+=$1} END{print s}'`.
- **The gateway's live `upstreams` is a dict**, not a list (canon and deployed agree) — the paused
  entries restore straight back into it.
- **`DSH_*` names are reserved.** `the agent CLI env file` setting `DSH_LITELLM_API_KEY` makes **every** `dsh`
  invocation throw `sets "…", which only the launching environment may set`. Use a
  non-reserved name (`GLM53_LITELLM_API_KEY`).
- **dsh's default `thinkingFormat: deepseek` sends a `thinking` field** that LiteLLM rejects
  (`UnsupportedParamsError: openai does not support parameters: ['thinking']`). For a
  gateway-fronted GLM lane use `thinkingFormat: chat-template`, whose kwargs travel as
  `chat_template_kwargs` — already in the LiteLLM block's `allowed_openai_params` and accepted by
  vLLM.
- **A 20-token `max_tokens` returns `content: null`** on this model: it is a reasoning model and
  spends the whole budget on `reasoning_content` (`reasoning_tokens: 20`). Coherence checks need
  headroom.

---

## 3. MISS — S2-1 observer geometry receipt does not pass

The private deployment runbook's S2-1 requires (1) one receipt per rank, (2) an **complete** `engine_projection`
record, (3) exactly the three manifest-declared stages incomplete, (4) nothing else incomplete.

Observed on the successful persistence boot:

```
[STARTUP_OBSERVER_INCOMPLETE] stage=engine_projection error=NotImplementedError
[STARTUP_OBSERVER_INCOMPLETE] stage=worker_preallocation error=NotImplementedError
[STARTUP_OBSERVER_INCOMPLETE] stage=worker_postbind error=NotImplementedError
```

and only **rank 0** writes a receipt — node1/3/4's `<RECEIPTS>/observer/` is empty although
the bind mount is present on every rank
(`/home/user/persistence-receipts/observer -> /observer-receipts`).

So S2-1 fails on criteria (1) and (2). This is a diagnostic-subsystem gap, not a serving or
persistence fault: the engine, the offloading connector, the coordinator census and disk stores all
work (see §4). Recorded as a MISS, not backfilled. This is the private deployment runbook UNVERIFIED item #3 becoming a
finding.

---

## 4. What passed, with the measured numbers

- Weights byte-exact on all four nodes (resolved-tree and safetensors sums, §2.2).
- §3.4 in-image overlay runtime check: exit 0, `glm53 overlay runtime (import) verify OK`.
- §3.5 preflight: `preflight OK` on all four ranks (image Id, GIDs 5/5/3/3, hubs, drafter,
  persistence roots on ext4 with 2.3-3.3 TB free, token metadata, observer manifest match).
- §3.8 / S2-2 gate suite **PASS with PERSISTENCE=on**, fresh seeds: A8-1 `health=200`; A8-2 deep
  ×3 @30k (`GATE=PASS`, prompt_tokens 30,370/30,392/30,429, 160 completion tokens, coherent);
  A8-3 concurrent ×3 @20k (`GATE=PASS`); A8-4 repeat ×3 @30k (`GATE=PASS`); S2-3 round-trip @8k
  ×2 coherent with the disk store growing.
- **GPU KV cache 4,388,461 tokens** (56.18 GiB available) with persistence on, versus **4,406,410**
  with it off — persistence costs ≈0.4% of the pool.
- Hybrid APC groups: `MLAAttentionSpec[0]`, `MambaSpec[2,3,4,5]`, `SlidingWindowSpec[6]`,
  `eagle_group_ids=[6]` — the DFlash2 drafter-group annotation is live, closing kill-gate
  constraint 3.
- **KV on disk: 2,250,340,608 B (2.10 GiB) on every rank**, symmetric; `kv_offload_store_bytes_total`
  = 9,029,116,928 B over 392 store operations, `kv_offload_lookup_sync_delay_seconds_count`
  incrementing, no errors.
- **S2-3 is PARTIAL, and S2-4 was not run.** The storage direction is proven. The *load-back*
  direction is **not yet demonstrated**: this build exposes no disk-tier load/hit counter
  (`kv_offload_total_bytes_total` for `CPU_to_GPU`/`GPU_to_CPU` is the CPU-pinned tier and stays
  0.0), and the head log's `Prefix cache hit rate` read `0.0%` across the round-trip. The second
  identical 8k prompt returned coherent output, but that is equally consistent with the prefix still
  being resident in the GPU pool, so the private deployment runbook's S2-3 criterion ("second run shows a remote-KV hit
  in the head log") is **not met as written**. The test that would settle it is **S2-4 (restart
  retention)**: `down`, `up`, repeat seed 111, and look for the post-restart request hitting stored
  KV. That requires taking the handed-over service down, so it was **not** run under the §3H.6 STOP
  rule — it needs the operator's go. Note also the engine's own
  `Disabling fine-grained prefix-cache hits because these KV cache managers require block-aligned
  lookups: SlidingWindowManager` line, which makes exact-prefix hit accounting coarse on this model.
- **Not run, by design:** S2-5 (oversubscription across the measured KV boundary) and S2-7 (the A/B
  matrix) are hours-class and require the operator's explicit ask. S2-6 fault paths were not exercised.
- §3H handover: gateway `configuration valid models=1 upstreams=1`; LiteLLM advertises exactly
  `['glm53-flash-exl3']`; loopback and LAN client-key completions both return `READY`; and
  `dsh --profile headless "Reply with the single word READY."` returns `READY`, exit 0.
  Receipt: `<RECEIPTS>/the handover receipt` on head.
