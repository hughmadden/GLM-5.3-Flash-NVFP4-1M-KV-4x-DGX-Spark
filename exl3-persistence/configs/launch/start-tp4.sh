#!/usr/bin/env bash
# start-tp4.sh -- launch GLM-5.3-Flash EXL3 TP4 across our four DGX Sparks.
#
# Ported from Mia's start-tp4.sh (1,773 lines). This is NOT a line-for-line
# clone: her script also owns image build/pull/ship, HF weight download and
# rank-to-rank weight sync, none of which apply to us (our image is built on
# the build host and distributed by distribute.sh; our weights are already staged, see
# reports/spark-staging.md). What is ported faithfully is everything that is
# load-bearing at launch:
#
#   * worker-first rank ordering 3 -> 2 -> 1 -> 0 (a fresh rank that
#     rendezvouses with a dying one hangs; tear all four down before relaunching
#     any -- tonyd2wild hard-won rule #2);
#   * per-rank NCCL/GLOO pins, docker run flags and cache mounts;
#   * the head/worker inner `vllm serve` argument construction;
#   * /health-only liveness (never /v1/models -- it returns 200 with a dead
#     engine).
#
# Two gaps in her TP4 path are FIXED here (reports/mia-repo.md §6):
#   1. EXL3_FAT_GROUPED / GLM53_ADAPTIVE_K* / GLM53_DENSE_FP8 are forwarded to
#      every rank, so env.tp4.fleet can actually override them.
#   2. NIC / HCA / GID are genuinely per-rank, not one shared default.
#
# NO floor guards, no memory kill machinery, no watchdog. Recipe-native
# protections only. The page-cache flusher and the memory ritual are separate
# HOST scripts (../host/) run before this script, per tonyd2wild.
#
# Run on rank 0 (node0).
#
# Usage:
#   ./start-tp4.sh preflight     # checks; creates only the persistence root
#                                #   and the observer receipt dir, starts nothing
#   ./start-tp4.sh coordinator-check  # prove the four in-engine KV coordinator
#                                #   listeners are up (after `up`)
#   ./start-tp4.sh up            # stop all, then launch 3,2,1,0 and wait
#   ./start-tp4.sh down          # stop and remove all four containers
#   ./start-tp4.sh logs          # follow the head container
#   ./start-tp4.sh args          # print the resolved serve args, launch nothing
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/env.tp4.fleet}"

log()  { printf '\033[1;36m[glm53-tp4]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-tp4]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-tp4]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$ENV_FILE" ] || die "env file not found: $ENV_FILE"
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

# ---------------------------------------------------------------------------
# per-rank accessors
# ---------------------------------------------------------------------------
# NOTE (fixed 2026-09-10): the words of a `local` command are expanded BEFORE
# the builtin assigns any of them, so the original one-liner
#   local r="$1" name="$2" v="RANK${r}_${name}"
# expanded ${r} against the (unset) GLOBAL r and died under `set -u` with
# "line 52: r: unbound variable" on EVERY subcommand -- preflight, args and up
# alike. Split into separate assignments.
rank_var() {
    local r="$1" name="$2"
    local v="RANK${r}_${name}"
    printf '%s' "${!v:-}"
}
rank_lan()     { rank_var "$1" LAN; }
rank_fabric()  { rank_var "$1" FABRIC; }
rank_host()    { rank_var "$1" HOST; }
rank_ib()      { rank_var "$1" IB; }
rank_if()      { rank_var "$1" IF; }
rank_gid()     { rank_var "$1" GID; }
rank_hub()     { rank_var "$1" HUB; }
rank_persist() { rank_var "$1" PERSIST_ROOT; }
# FIX-3 (node3 DFlash2): a node's second hub directory, mounted read-only at
# $HUB2_CONTAINER. Empty means "this node has only one hub". node3 (rank 3)
# keeps its verified EXL3 weights under hub-local-staging but its DFlash2
# drafter under ~/.cache/huggingface/hub, so one mount cannot see both.
rank_hub2()    { rank_var "$1" HUB2; }
rank_ssh()     { rank_var "$1" SSH; }
rank_container() {              # same expansion-order fix as rank_var
    local r="$1"
    local v="CONTAINER${r}"
    printf '%s' "${!v}"
}

# rank 0 runs locally; 1..3 over ssh.
on_rank() {
    local r="$1"; shift
    if [ "$r" = "0" ]; then bash -lc "$*";
    else ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$(rank_ssh "$r")" "$*"; fi
}

HEAD_FABRIC="$(rank_fabric 0)"

# ---------------------------------------------------------------------------
# FIX-3: hub directories.
#
# Container-side search path, in priority order. HUB2_CONTAINER is only mounted
# on nodes that declare RANK<N>_HUB2. The inner script walks HUB_DIRS for both
# the model and the drafter, so a node whose weights and drafter live in two
# different hub trees resolves both.
# ---------------------------------------------------------------------------
HUB_CONTAINER=/root/.cache/huggingface/hub
HUB2_CONTAINER=/root/.cache/huggingface/hub-secondary

# Host-side equivalent, used by preflight only.
rank_hub_list() {
    local r="$1" h2; h2="$(rank_hub2 "$r")"
    printf '%s' "$(rank_hub "$r")${h2:+:$h2}"
}

# Resolve this rank's drafter snapshot ON THE NODE, the same way the inner
# script will. Prints the directory, or nothing if it cannot be resolved.
rank_draft_dir() {
    local r="$1" repo_dir
    repo_dir="models--${DFLASH_REPO//\//--}"
    on_rank "$r" "for h in \$(printf '%s' '$(rank_hub_list "$r")' | tr ':' ' '); do \
        d=\$(ls -d \"\$h/${repo_dir}/snapshots\"/* 2>/dev/null | sort | tail -1); \
        if [ -n \"\$d\" ] && [ -d \"\$d\" ]; then printf '%s' \"\$d\"; break; fi; \
    done" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# kv-transfer-config (PERSISTENCE=on)
#
# Every key below is rendered EXPLICITLY -- nothing is left to a library
# default -- and every key is one the pinned package actually reads. Sources:
#
#   persistence/recipe_persistence/native.py:110-153   (extra_config contract)
#   persistence/recipe_persistence/coordinator_http.py:1258-1300 (factory)
#   persistence/recipe_persistence/storage.py:45-79    (Limits fields)
#   persistence/README.md "Native configuration contract"
#   persistence/README-coordinator-http.md "Configuration and trust boundary"
#
# FIXED 2026-09-10 (LAUNCH-FIXES-20260910.md):
#   * coordinator_url REMOVED. No file in recipe_persistence reads that key.
#     factory() requires coordinator_endpoints: a list of exactly world_size
#     "host:port" strings in RANK ORDER, else ValueError
#     ("coordinator_endpoints must list one endpoint per rank",
#     coordinator_http.py:1274-1277). Rendered from RANK<N>_FABRIC.
#   * disk_quota_bytes REMOVED. The string "disk_quota_bytes" does not appear
#     anywhere in the package; the real per-rank quota is
#     coordinator_disk_limits.quota (storage.py:46), passed to Limits(**...)
#     at coordinator_http.py:1297-1301. The old key was silently ignored.
#   * coordinator_disk_limits rendered with all twelve Limits fields explicit.
#   * The bounded coordinator_* transport knobs are rendered explicitly.
#
# There is NO separate coordinator daemon. Each rank's WORKER-role factory call
# constructs and starts its own MetadataServer, bound to endpoints[rank], inside
# the blocking initialize_from_config (coordinator_http.py:1302-1310;
# README-coordinator-http.md "Startup and identity"). The scheduler role only
# builds an RpcClient. Do not start anything by hand. `coordinator-check`
# verifies all four listeners after readiness and fails closed if one is absent.
#
# The bearer token is passed as a FILE PATH only. Never a value, never argv.
# load_auth_token() (coordinator_http.py:93-124) opens it O_NOFOLLOW and rejects
# a symlink, a non-regular file, a foreign owner, any group/other bit, >4096
# bytes, non-ASCII, and a low-entropy secret.
# ---------------------------------------------------------------------------
coordinator_endpoints_csv() {
    local r out=""
    for r in 0 1 2 3; do
        out="${out}${out:+,}$(rank_fabric "$r"):${PERSIST_COORDINATOR_PORT}"
    done
    printf '%s' "$out"
}

kv_transfer_config() {
    local r="$1"
    PY_RANK="$r" \
    PY_ROOT="$(rank_persist "$r")" \
    PY_ENDPOINTS="$(coordinator_endpoints_csv)" \
    PY_TENANT="$PERSIST_TENANT" \
    PY_TOKEN_FILE="$PERSIST_TOKEN_FILE" \
    PY_STAGING_BYTES="$PERSIST_STAGING_BYTES" \
    PY_STAGING_ROWS="$PERSIST_STAGING_ROWS" \
    PY_MAX_PENDING="$PERSIST_MAX_PENDING_KEYS" \
    PY_MAX_PENDING_BYTES="$PERSIST_MAX_PENDING_BYTES" \
    PY_LOOKUP_KEYS="$PERSIST_LOOKUP_KEYS_PER_STEP" \
    PY_SQLITE_JOURNAL="$PERSIST_SQLITE_JOURNAL_MODE" \
    PY_SQLITE_SYNC="$PERSIST_SQLITE_SYNCHRONOUS" \
    PY_META_WORKERS="$PERSIST_METADATA_WORKERS" \
    PY_META_SUBMITTED="$PERSIST_METADATA_MAX_SUBMITTED" \
    PY_META_SHUTDOWN="$PERSIST_METADATA_SHUTDOWN_TIMEOUT" \
    PY_IO_THREADS="$PERSIST_DISK_IO_THREADS" \
    PY_FINGERPRINT="$CACHE_FINGERPRINT" \
    PY_STARTUP_TIMEOUT="$PERSIST_COORDINATOR_STARTUP_TIMEOUT" \
    PY_RPC_TIMEOUT="$PERSIST_COORDINATOR_RPC_TIMEOUT" \
    PY_SERVER_THREADS="$PERSIST_COORDINATOR_SERVER_THREADS" \
    PY_MAX_BODY="$PERSIST_COORDINATOR_MAX_BODY_BYTES" \
    PY_CLIENT_POOL="$PERSIST_COORDINATOR_CLIENT_POOL" \
    PY_IDEM_ENTRIES="$PERSIST_COORDINATOR_IDEMPOTENCY_ENTRIES" \
    PY_RENEW_MARGIN="$PERSIST_COORDINATOR_RENEW_MARGIN" \
    PY_LIMIT_QUOTA="$PERSIST_LIMIT_QUOTA" \
    PY_LIMIT_HIGH="$PERSIST_LIMIT_HIGH" \
    PY_LIMIT_LOW="$PERSIST_LIMIT_LOW" \
    PY_LIMIT_INDEX_BYTES="$PERSIST_LIMIT_INDEX_BYTES" \
    PY_LIMIT_MAX_OBJECTS="$PERSIST_LIMIT_MAX_OBJECTS" \
    PY_LIMIT_MAX_LEASES="$PERSIST_LIMIT_MAX_LEASES" \
    PY_LIMIT_MAX_OBJECT_BYTES="$PERSIST_LIMIT_MAX_OBJECT_BYTES" \
    PY_LIMIT_FREE_BYTES="$PERSIST_LIMIT_FREE_BYTES" \
    PY_LIMIT_FREE_INODES="$PERSIST_LIMIT_FREE_INODES" \
    PY_LIMIT_LEASE_SECONDS="$PERSIST_LIMIT_LEASE_SECONDS" \
    PY_LIMIT_GRACE_SECONDS="$PERSIST_LIMIT_GRACE_SECONDS" \
    PY_LIMIT_IO_CHUNK_BYTES="$PERSIST_LIMIT_IO_CHUNK_BYTES" \
    python3 -S -c '
import json, os
env = os.environ
extra = {
    # --- vLLM OffloadingSpecFactory (kv_offload/factory.py:30-49) -----------
    "spec_name": "NodeLocalDiskOffloadingSpec",
    "spec_module_path": "recipe_persistence.native",
    # base.py:602 defaults this to True; we offload decode KV too.
    "offload_prompt_only": False,
    # --- native.py:120-137 required identity --------------------------------
    "disk_root": env["PY_ROOT"],
    "trusted_single_tenant": True,
    "tenant_namespace": env["PY_TENANT"],
    "cache_fingerprint": env["PY_FINGERPRINT"],
    "coordinator_module_path": "recipe_persistence.coordinator_http",
    "coordinator_factory": "factory",
    # --- coordinator_http.factory required (coordinator_http.py:1273-1278) --
    "coordinator_endpoints": env["PY_ENDPOINTS"].split(","),
    "coordinator_auth_token_file": env["PY_TOKEN_FILE"],
    # --- physical staging ring (native.py:134-137, 242-250) -----------------
    "staging_bytes": int(env["PY_STAGING_BYTES"]),
    "staging_rows": int(env["PY_STAGING_ROWS"]),
    "disk_io_threads": int(env["PY_IO_THREADS"]),
    # --- logical metadata credits; manager/worker/provider must agree -------
    # (native.py:172-173 raises if extra != provider.max_pending_keys)
    "max_pending_keys": int(env["PY_MAX_PENDING"]),
    "max_pending_bytes": int(env["PY_MAX_PENDING_BYTES"]),
    # --- native step/callback bounds (native.py:130-143) --------------------
    "lookup_keys_per_step": int(env["PY_LOOKUP_KEYS"]),
    # --- SQLite durability policy (storage.py DiskStore) --------------------
    # The store is a reconstructible cache (kv_load_failure_policy=recompute), so
    # the historical DELETE+EXTRA fsync-per-commit bought no real guarantee while
    # costing 8.7 ms on every lease grant (the decode penalty). WAL+NORMAL keeps
    # crash consistency and removes reader/writer serialisation.
    "sqlite_journal_mode": env["PY_SQLITE_JOURNAL"],
    "sqlite_synchronous": env["PY_SQLITE_SYNC"],
    "metadata_workers": int(env["PY_META_WORKERS"]),
    "metadata_max_submitted": int(env["PY_META_SUBMITTED"]),
    "metadata_shutdown_timeout": float(env["PY_META_SHUTDOWN"]),
    # --- bounded coordinator transport (coordinator_http.py:1281-1289) ------
    "coordinator_startup_timeout": float(env["PY_STARTUP_TIMEOUT"]),
    "coordinator_rpc_timeout": float(env["PY_RPC_TIMEOUT"]),
    "coordinator_server_threads": int(env["PY_SERVER_THREADS"]),
    "coordinator_max_body_bytes": int(env["PY_MAX_BODY"]),
    "coordinator_client_pool": int(env["PY_CLIENT_POOL"]),
    "coordinator_idempotency_entries": int(env["PY_IDEM_ENTRIES"]),
    "coordinator_renew_margin": float(env["PY_RENEW_MARGIN"]),
    "coordinator_close_store_on_shutdown": True,
    # --- rank-local DiskStore limits: every storage.py:45-58 field -----------
    "coordinator_disk_limits": {
        "quota": int(env["PY_LIMIT_QUOTA"]),
        "high": int(env["PY_LIMIT_HIGH"]),
        "low": int(env["PY_LIMIT_LOW"]),
        "index_bytes": int(env["PY_LIMIT_INDEX_BYTES"]),
        "max_objects": int(env["PY_LIMIT_MAX_OBJECTS"]),
        "max_leases": int(env["PY_LIMIT_MAX_LEASES"]),
        "max_object_bytes": int(env["PY_LIMIT_MAX_OBJECT_BYTES"]),
        "free_bytes": int(env["PY_LIMIT_FREE_BYTES"]),
        "free_inodes": int(env["PY_LIMIT_FREE_INODES"]),
        "lease_seconds": float(env["PY_LIMIT_LEASE_SECONDS"]),
        "grace_seconds": float(env["PY_LIMIT_GRACE_SECONDS"]),
        "io_chunk_bytes": int(env["PY_LIMIT_IO_CHUNK_BYTES"]),
    },
    # _profile_from_config (coordinator_http.py:1128-1130) refuses anything
    # else; the engine default at this pin is also sha256 (config/cache.py:141)
    # and native.py:117-119 re-checks the LIVE cache_config value.
    "prefix_caching_hash_algo": "sha256",
}
print(json.dumps({
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": extra,
}, separators=(",", ":")))'
}

# The fingerprint must change whenever anything that changes KV bytes changes:
# image identity, quantization, KV dtype, context, kernel knobs. A stale
# fingerprint would let a new engine read another build's cached KV.
compute_fingerprint() {
    # FIX-4: the identity term is the PINNED IMAGE_ID from the env file, not a
    # live `docker image inspect`. The old form silently produced
    # id="unknown" -- and therefore a different fingerprint -- on any host
    # without that image loaded (e.g. rendering `args` on the control host for review), so
    # the reviewed JSON was not the JSON the fleet would run. preflight still
    # proves every rank actually presents this Id.
    local id
    id="$(image_id_expected)"
    printf '%s' "$(printf '%s|%s|%s|%s|%s|%s|%s' \
        "$id" "$MODEL_REPO" "$QUANTIZATION" "$KV_CACHE_DTYPE" \
        "$MAX_MODEL_LEN" "$PREFIX_MATCH_UNIT" "$EXL3_FAT_GROUPED" \
        | sha256sum | cut -c1-32)"
}

# ---------------------------------------------------------------------------
# FIX-2: startup-observer wiring.
#
# startup_observer.emit() (port/observer/startup_observer.py:379-394) is a
# no-op unless VLLM_STARTUP_OBSERVER=1, and then REQUIRES all three of
# VLLM_STARTUP_OBSERVER_RUN_ID, VLLM_STARTUP_OBSERVER_MANIFEST_SHA256 (64
# lowercase hex, checked at :387) and VLLM_STARTUP_OBSERVER_DIR. write_record()
# (:352-357) additionally requires the run id to match
# [A-Za-z0-9][A-Za-z0-9._-]{0,95} and the directory to be precreated, absolute
# and not a symlink; receipts are written 0600 as <run_id>.<host>.<pid>.jsonl.
#
# OBSERVER=on|off (default on). With OBSERVER=on a missing/!=64-hex manifest
# fingerprint is a hard stop here, so the receipt never silently goes missing.
# ---------------------------------------------------------------------------
observer_enabled() { [ "${OBSERVER:-on}" = "on" ]; }

observer_check_config() {
    observer_enabled || { log "observer OFF (OBSERVER=off)"; return 0; }
    case "${OBSERVER_MANIFEST_SHA256:-}" in
        [0-9a-f]*) : ;;
        *) die "OBSERVER=on but OBSERVER_MANIFEST_SHA256 is unset/not lowercase hex" ;;
    esac
    [ "${#OBSERVER_MANIFEST_SHA256}" = "64" ] \
        || die "OBSERVER_MANIFEST_SHA256 must be 64 lowercase hex chars (got ${#OBSERVER_MANIFEST_SHA256})"
    [ -n "${OBSERVER_DIR_HOST:-}" ] && [ -n "${OBSERVER_DIR:-}" ] \
        || die "OBSERVER=on requires OBSERVER_DIR_HOST and OBSERVER_DIR"
    case "$OBSERVER_DIR" in /*) : ;; *) die "OBSERVER_DIR must be absolute" ;; esac
}

# Confirm the fingerprint we forward is the manifest the image actually shipped.
# One read-only `sha256sum` container on rank 0; no GPU, no engine.
observer_verify_manifest_in_image() {
    observer_enabled || return 0
    [ "${OBSERVER_VERIFY_IN_IMAGE:-1}" = "1" ] || { warn "observer manifest in-image check SKIPPED"; return 0; }
    local got
    got="$(on_rank 0 "docker run --rm --entrypoint sha256sum '$IMAGE' '$OBSERVER_MANIFEST_IN_IMAGE' 2>/dev/null | awk '{print \$1}'" || true)"
    [ -n "$got" ] || die "cannot read $OBSERVER_MANIFEST_IN_IMAGE from $IMAGE"
    [ "$got" = "$OBSERVER_MANIFEST_SHA256" ] \
        || die "observer manifest in image is $got, env says $OBSERVER_MANIFEST_SHA256"
    log "observer manifest ${got} (matches image)"
}

# ---------------------------------------------------------------------------
# FIX-4: image identity pin.
#
# The tag is not the image. IMAGE_ID in the env file is the exact
# `docker image inspect --format '{{.Id}}'` of the build that this recipe was
# written against; every rank must present that Id or nothing starts.
# ---------------------------------------------------------------------------
image_id_expected() {
    case "${IMAGE_ID:-}" in
        sha256:[0-9a-f]*) [ "${#IMAGE_ID}" = "71" ] || die "IMAGE_ID is not a sha256:<64hex> digest" ;;
        *) die "IMAGE_ID must be pinned to sha256:<64hex> (see env file; update after each rebuild)" ;;
    esac
    printf '%s' "$IMAGE_ID"
}

# ---------------------------------------------------------------------------
# inner container script (identical text on every rank; NODE_RANK selects the
# head/worker behaviour, exactly as Mia's two inner scripts differ only by
# --headless and --node-rank)
# ---------------------------------------------------------------------------
INNER=/tmp/glm53-tp4-inner.sh
write_inner() {
    cat > "$INNER" <<'INNER_EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-tp4 rank ${NODE_RANK}] $*"; }

# Runtime-env-dependent overlay steps. Everything else is baked into the image;
# these two are re-run per boot because they read env at container start, which
# is exactly how Mia invokes them (Dockerfile --preflight, start.sh writes).
python3 /opt/glm53/overlay/patch_spinwait.py
python3 /opt/glm53/overlay/patch_glm_video_placeholders.py

# FIX-3: resolve the snapshot dir across EVERY hub this node mounts, in
# HUB_DIRS priority order, not one hardcoded path. On node2/node3 the weights
# and the DFlash2 drafter live under two different hub directories.
resolve_snapshot() {
    REPO="$1" python3 -S -c '
import os, sys, glob
repo = os.environ["REPO"].replace("/", "--")
roots = [p for p in os.environ["HUB_DIRS"].split(":") if p]
for hub in roots:
    hits = sorted(h for h in glob.glob(os.path.join(hub, "models--" + repo, "snapshots", "*"))
                  if os.path.isdir(h))
    if hits:
        print(hits[-1]); break
else:
    sys.exit("no snapshot for " + repo + " under " + ":".join(roots))'
}
if [ -z "${MODEL_DIR:-}" ]; then
    MODEL_DIR="$(resolve_snapshot "${MODEL_REPO}")"
fi
[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
say "model dir ${MODEL_DIR}"

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${HEAD_FABRIC}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-enable-flashinfer-autotune
    --prefix-match-unit "${PREFIX_MATCH_UNIT}"
)
[ "${NODE_RANK}" != "0" ] && ARGS+=(--headless)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ] && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

if [ "${SPEC_METHOD:-none}" = "dflash" ]; then
    # FIX-3: same multi-hub search. DRAFT_REPO_DIR (env) overrides outright.
    DFLASH_MODEL_DIR="${DFLASH_MODEL_DIR:-${DRAFT_REPO_DIR:-$(resolve_snapshot "${DFLASH_REPO}")}}"
    export DFLASH_MODEL_DIR
    ARGS+=(--speculative-config "$(python3 -S -c '
import json, os
spec = {
    "method": "dflash",
    "model": os.environ["DFLASH_MODEL_DIR"],
    "num_speculative_tokens": int(os.environ.get("DFLASH_TOKENS", "7")),
    "kv_cache_dtype": "auto",
    "draft_sample_method": "probabilistic",
    "rejection_sample_method": "standard",
}
tp = os.environ.get("DFLASH_DRAFT_TP", "").strip()
if tp:
    spec["draft_tensor_parallel_size"] = int(tp)
print(json.dumps(spec, separators=(",", ":")))')")
fi

if [ -n "${KV_TRANSFER_CONFIG:-}" ]; then
    ARGS+=(--kv-transfer-config "${KV_TRANSFER_CONFIG}")
    say "persistence ON -> recipe_persistence.native:NodeLocalDiskOffloadingSpec"
else
    say "persistence OFF -> in-GPU KV only"
fi

if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS}); ARGS+=("${EXTRA[@]}")
fi

if [ "${PRINT_ARGS_ONLY:-0}" = "1" ]; then
    printf 'vllm serve %s' "${MODEL_DIR}"; printf ' %q' "${ARGS[@]}"; printf '\n'; exit 0
fi

say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
INNER_EOF
    chmod +x "$INNER"
}

# ---------------------------------------------------------------------------
# preflight
#
# Read-only, with ONE deliberate exception documented in
# LAUNCH-FIXES-20260910.md: with PERSISTENCE=on it creates a missing per-rank
# persistence root (mkdir 0700) and the observer receipt directory, and says so
# loudly. Everything else only observes. Nothing is started.
# ---------------------------------------------------------------------------
preflight() {
    local r ok=1 id0="" id want_id=""
    want_id="$(image_id_expected)"
    observer_check_config
    for r in 0 1 2 3; do
        log "rank ${r} ($(rank_host "$r") $(rank_lan "$r")):"
        # FIX-4: image identity, not tag. Four nodes that each built the tag
        # locally get four different images (tonyd2wild hard-won rule #3), and
        # a `:latest` tag can move under us -- so every rank must present the
        # exact pinned IMAGE_ID, and they must all agree with each other.
        id="$(on_rank "$r" "docker image inspect '$IMAGE' --format '{{.Id}}' 2>/dev/null" || true)"
        if [ -z "$id" ]; then warn "  image $IMAGE ABSENT"; ok=0
        else
            [ -z "$id0" ] && id0="$id"
            if [ "$id" != "$want_id" ]; then
                warn "  image ${id} != pinned IMAGE_ID ${want_id}"; ok=0
            elif [ "$id" = "$id0" ]; then log "  image ${id}"
            else warn "  image ${id} != rank0 ${id0}  -- DIVERGENT"; ok=0; fi
        fi
        # GID: an empty gid at the pinned index is the silent-hang failure.
        local gid
        gid="$(on_rank "$r" "cat /sys/class/infiniband/$(rank_ib "$r")/ports/1/gids/$(rank_gid "$r") 2>/dev/null" || true)"
        case "$gid" in
            ""|0000:0000:0000:0000:0000:0000:0000:0000)
                warn "  GID index $(rank_gid "$r") on $(rank_ib "$r") is EMPTY -- NCCL will hang silently"; ok=0 ;;
            *) log "  GID[$(rank_gid "$r")] ${gid}" ;;
        esac
        # weights
        if on_rank "$r" "test -d '$(rank_hub "$r")/models--${MODEL_REPO//\//--}'"; then
            log "  weights $(rank_hub "$r")"
        else warn "  weights MISSING under $(rank_hub "$r")"; ok=0; fi
        # FIX-3: resolve the DRAFTER snapshot on this node across BOTH hub
        # directories, exactly the way the inner script will. node3 stages
        # DFlash2 under ~/.cache/huggingface/hub while its EXL3 weights are in
        # hub-local-staging, and the old single mount could not see both --
        # the engine died with `no snapshot for incoai--GLM-5.3-Flash-DFlash2`
        # after a full weight load. Nothing here is silently substituted: an
        # unresolvable drafter aborts.
        if [ "${SPEC_METHOD:-none}" = "dflash" ]; then
            local draft_dir
            draft_dir="$(rank_draft_dir "$r")"
            if [ -n "$draft_dir" ]; then log "  drafter ${draft_dir}"
            else warn "  drafter ${DFLASH_REPO} NOT FOUND under $(rank_hub_list "$r" | tr ':' ' ')"; ok=0; fi
        fi
        # FIX-1: persistence root, backing filesystem, free space, token file
        # and coordinator port. DiskStore takes an EXCLUSIVE lock on
        # disk_root/rank-<N> (storage.py:82-105) and the Limits accounting
        # assumes a private local filesystem, so an NFS/overlay/tmpfs backing
        # or a shared root is a hard stop, not a warning.
        if [ "${PERSISTENCE}" = "on" ]; then
            local proot fstype source free_b
            proot="$(rank_persist "$r")"
            if ! on_rank "$r" "test -d '$proot'"; then
                warn "  persist root $proot MISSING -- creating it (mkdir -m 0700)"
                on_rank "$r" "mkdir -p -m 0700 '$proot'" || { warn "  cannot create $proot"; ok=0; }
            fi
            if on_rank "$r" "test -d '$proot' && test -w '$proot'"; then
                on_rank "$r" "chmod 0700 '$proot'" >/dev/null 2>&1 || true
                fstype="$(on_rank "$r" "findmnt -T '$proot' -no FSTYPE" 2>/dev/null || true)"
                source="$(on_rank "$r" "findmnt -T '$proot' -no SOURCE" 2>/dev/null || true)"
                free_b="$(on_rank "$r" "df -PB1 '$proot' | awk 'NR==2{print \$4}'" 2>/dev/null || echo 0)"
                case "$fstype" in
                    ext4|xfs|btrfs|f2fs|ext3) log "  persist root $proot on $fstype ($source), free ${free_b} B" ;;
                    "") warn "  persist root $proot: findmnt returned nothing"; ok=0 ;;
                    *)  warn "  persist root $proot is on '$fstype' ($source) -- NOT a local block filesystem"; ok=0 ;;
                esac
                if [ "${free_b:-0}" -lt "$PERSIST_MIN_FREE_BYTES" ] 2>/dev/null; then
                    warn "  persist root $proot has ${free_b} B free, need >= ${PERSIST_MIN_FREE_BYTES} B"; ok=0
                fi
            else warn "  persist root $proot missing or not writable"; ok=0; fi
            # Token: metadata only. The value is never read, echoed or logged.
            if on_rank "$r" "test -f '$PERSIST_TOKEN_FILE_HOST' && test ! -L '$PERSIST_TOKEN_FILE_HOST'"; then
                local mode owner size
                mode="$(on_rank "$r" "stat -c %a '$PERSIST_TOKEN_FILE_HOST'")"
                owner="$(on_rank "$r" "stat -c %U '$PERSIST_TOKEN_FILE_HOST'")"
                size="$(on_rank "$r" "stat -c %s '$PERSIST_TOKEN_FILE_HOST'")"
                # load_auth_token (coordinator_http.py:104-124): regular file,
                # owned by the container euid (root), no group/other bits,
                # <=4096 B, and the secret itself 32..256 printable chars.
                [ "$mode" = "600" ] || [ "$mode" = "400" ] || { warn "  token file mode $mode, want 600 or 400"; ok=0; }
                [ "$owner" = "root" ] || { warn "  token file owner $owner, want root (the container euid)"; ok=0; }
                { [ "${size:-0}" -ge 32 ] && [ "${size:-0}" -le 4096 ]; } || { warn "  token file size ${size} B outside 32..4096"; ok=0; }
                [ "$ok" = "1" ] && log "  token file $PERSIST_TOKEN_FILE_HOST ($mode $owner ${size}B)"
            else warn "  token file $PERSIST_TOKEN_FILE_HOST absent or a symlink -- see the token provisioning utility"; ok=0; fi
            # The coordinator listener binds this rank's FABRIC address. The
            # address must exist on the node and the port must be free: the
            # MetadataServer is started inside initialize_from_config and a
            # bind failure there is a mid-load engine death.
            if ! on_rank "$r" "ip -o addr show | grep -qw '$(rank_fabric "$r")'"; then
                warn "  fabric address $(rank_fabric "$r") is not configured on this node"; ok=0
            fi
            if on_rank "$r" "ss -Hltn 'sport = :${PERSIST_COORDINATOR_PORT}' | grep -q ."; then
                warn "  coordinator port ${PERSIST_COORDINATOR_PORT} is ALREADY BOUND on this node"; ok=0
            fi
        fi
        # FIX-2: observer receipt directory. write_record refuses a directory
        # that is not precreated, absolute and non-symlink
        # (startup_observer.py:355-357).
        if observer_enabled; then
            if on_rank "$r" "test -L '$OBSERVER_DIR_HOST'"; then
                warn "  observer dir $OBSERVER_DIR_HOST is a SYMLINK -- receipts would be refused"; ok=0
            else
                on_rank "$r" "mkdir -p -m 0755 '$OBSERVER_DIR_HOST'" || { warn "  cannot create $OBSERVER_DIR_HOST"; ok=0; }
                on_rank "$r" "test -d '$OBSERVER_DIR_HOST'" || { warn "  observer dir $OBSERVER_DIR_HOST missing"; ok=0; }
                log "  observer dir $OBSERVER_DIR_HOST -> $OBSERVER_DIR"
            fi
        fi
        # container must not already be running
        if on_rank "$r" "docker ps --filter name=^/$(rank_container "$r")$ --format '{{.Names}}' | grep -q ."; then
            warn "  container $(rank_container "$r") is ALREADY RUNNING"; ok=0
        fi
    done
    observer_verify_manifest_in_image
    [ "$ok" = "1" ] || die "preflight failed -- nothing was started"
    log "preflight OK"
}

# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------
stop_all() {
    local r
    # Tear every rank down before relaunching any.
    for r in 0 1 2 3; do
        on_rank "$r" "docker rm -f '$(rank_container "$r")' >/dev/null 2>&1 || true"
    done
    log "all four ranks stopped"
}

launch_rank() {
    local r="$1" cname kvcfg="" ssh_t
    cname="$(rank_container "$r")"
    [ "${PERSISTENCE}" = "on" ] && kvcfg="$(kv_transfer_config "$r")"

    if [ "$r" != "0" ]; then
        ssh_t="$(rank_ssh "$r")"
        scp -q -o BatchMode=yes "$INNER" "${ssh_t}:${INNER}"
    fi

    # FIX-3 / FIX-2: optional second hub mount and the observer receipt dir.
    local hub2 hub2_opt="" hub_dirs="$HUB_CONTAINER" obs_opt="" obs_run=""
    hub2="$(rank_hub2 "$r")"
    if [ -n "$hub2" ] && on_rank "$r" "test -d '$hub2'"; then
        hub2_opt="-v '${hub2}:${HUB2_CONTAINER}:ro'"
        hub_dirs="${HUB_CONTAINER}:${HUB2_CONTAINER}"
    elif [ -n "$hub2" ]; then
        die "rank ${r}: RANK${r}_HUB2=${hub2} does not exist -- refusing to let docker create it"
    fi
    if observer_enabled; then
        obs_run="${OBSERVER_RUN_STAMP}-rank${r}"
        obs_opt="-v '${OBSERVER_DIR_HOST}:${OBSERVER_DIR}'"
    fi

    on_rank "$r" "mkdir -p '\$HOME/.cache/vllm' '\$HOME/.triton/cache' '\$HOME/.tilelang/cache'"

    # NOTE the docker run flag set: --network host + --ipc host + --shm-size 32g
    # + --device /dev/infiniband + --cap-add IPC_LOCK + memlock unlimited, all
    # from tonyd2wild's launcher (§7) and Mia's TP4 launcher, which agree.
    # NOT set: --privileged (neither uses it), --restart (v1 has no engine
    # restart path; a policy cannot recover a multi-node mp deployment), and no
    # --memory/--memory-swap cap -- tony's 112g pair is a host-RAM guard, and we
    # are told to add no floor guards or kill machinery.
    on_rank "$r" "docker run -d --name '$cname' \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v '$(rank_hub "$r"):${HUB_CONTAINER}:ro' \
        ${hub2_opt} \
        ${obs_opt} \
        -v '\$HOME/.cache/vllm:/root/.cache/vllm' \
        -v '\$HOME/.triton/cache:/root/.triton/cache' \
        -v '\$HOME/.tilelang/cache:/root/.tilelang/cache' \
        $([ -n "${PERSIST_SRC_OVERLAY:-}" ] && printf -- "-v '%s:/usr/local/lib/python3.12/dist-packages/recipe_persistence:ro' " "$PERSIST_SRC_OVERLAY") \
        $([ "${PERSISTENCE}" = "on" ] && printf -- "-v '%s:%s' " "$(rank_persist "$r")" "$(rank_persist "$r")") \
        $([ "${PERSISTENCE}" = "on" ] && printf -- "-v '%s:%s:ro' " "$PERSIST_TOKEN_FILE_HOST" "$PERSIST_TOKEN_FILE") \
        -v '${INNER}:/start.sh:ro' \
        -e NODE_RANK='$r' \
        -e VLLM_HOST_IP='$(rank_fabric "$r")' \
        -e HEAD_FABRIC='$HEAD_FABRIC' \
        -e NCCL_SOCKET_IFNAME='$(rank_if "$r")' \
        -e GLOO_SOCKET_IFNAME='$(rank_if "$r")' \
        -e TP_SOCKET_IFNAME='$(rank_if "$r")' \
        -e NCCL_IB_HCA='$(rank_ib "$r")' \
        -e NCCL_IB_GID_INDEX='$(rank_gid "$r")' \
        -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
        -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
        -e NCCL_IB_ADDR_RANGE='$NCCL_IB_ADDR_RANGE' \
        -e NCCL_CROSS_NIC='$NCCL_CROSS_NIC' \
        -e NCCL_NVLS_ENABLE=0 -e NCCL_IB_MERGE_NICS=0 \
        -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
        -e NCCL_DEBUG='$NCCL_DEBUG' -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
        -e SERVED_MODEL_NAME='$SERVED_MODEL_NAME' \
        -e PORT='$PORT' -e TP='$TP' -e NNODES='$NNODES' -e MASTER_PORT='$MASTER_PORT' \
        -e MODEL_REPO='$MODEL_REPO' -e DFLASH_REPO='$DFLASH_REPO' \
        -e MODEL_DIR='${MODEL_DIR:-}' \
        -e QUANTIZATION='$QUANTIZATION' -e KV_CACHE_DTYPE='$KV_CACHE_DTYPE' \
        -e MAX_MODEL_LEN='$MAX_MODEL_LEN' -e GPU_MEM_UTIL='$GPU_MEM_UTIL' \
        -e MAX_NUM_SEQS='$MAX_NUM_SEQS' -e MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' \
        -e ENFORCE_EAGER='$ENFORCE_EAGER' -e PREFIX_MATCH_UNIT='$PREFIX_MATCH_UNIT' \
        -e SPEC_METHOD='$SPEC_METHOD' -e DFLASH_TOKENS='$DFLASH_TOKENS' \
        -e DFLASH_DRAFT_TP='$DFLASH_DRAFT_TP' \
        -e LANGUAGE_MODEL_ONLY='$LANGUAGE_MODEL_ONLY' -e LIMIT_MM='$LIMIT_MM' \
        -e SKIP_MM_PROFILING='$SKIP_MM_PROFILING' -e CHAT_TEMPLATE='$CHAT_TEMPLATE' \
        -e EXL3_FAT_GROUPED='$EXL3_FAT_GROUPED' \
        -e EXL3_TEMP_ROWS_FUSED='$EXL3_TEMP_ROWS_FUSED' \
        -e GLM53_INDEXER_WORKSPACE='$GLM53_INDEXER_WORKSPACE' \
        -e GLM53_SPINWAIT_MS='$GLM53_SPINWAIT_MS' \
        -e GLM53_ADAPTIVE_K='$GLM53_ADAPTIVE_K' \
        -e GLM53_ADAPTIVE_K_SET='$GLM53_ADAPTIVE_K_SET' \
        -e GLM53_DENSE_FP8='$GLM53_DENSE_FP8' \
        -e HUB_DIRS='$hub_dirs' \
        -e DRAFT_REPO_DIR='$(rank_var "$r" DRAFT_REPO_DIR)' \
        -e VLLM_STARTUP_OBSERVER='$(observer_enabled && echo 1 || echo 0)' \
        -e VLLM_STARTUP_OBSERVER_DIR='${OBSERVER_DIR:-}' \
        -e VLLM_STARTUP_OBSERVER_RUN_ID='$obs_run' \
        -e VLLM_STARTUP_OBSERVER_MANIFEST_SHA256='${OBSERVER_MANIFEST_SHA256:-}' \
        -e PYTHONHASHSEED='${PYTHONHASHSEED:-0}' \
        -e EXTRA_ARGS='${EXTRA_ARGS:-}' \
        -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB='${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-}' \
        -e KV_TRANSFER_CONFIG='$kvcfg' \
        -e PRINT_ARGS_ONLY='${PRINT_ARGS_ONLY:-0}' \
        --entrypoint bash '$IMAGE' /start.sh" >/dev/null
    log "rank ${r} started on $(rank_host "$r") (if=$(rank_if "$r") hca=$(rank_ib "$r") gid=$(rank_gid "$r"))"
}

# ---------------------------------------------------------------------------
# FIX-1: prove the four in-engine coordinator listeners are actually up.
#
# There is no daemon to start: each rank's worker-role factory binds its own
# MetadataServer inside initialize_from_config. By the time /health returns 200
# the all-rank census must already have succeeded, so a missing listener here
# means something is wrong that the engine did not surface. Fail closed.
# TCP reachability only -- no request is made, no token is used.
# ---------------------------------------------------------------------------
coordinator_check() {
    [ "${PERSISTENCE}" = "on" ] || { log "PERSISTENCE=off -- no coordinator listeners expected"; return 0; }
    local r ok=1 host
    for r in 0 1 2 3; do
        host="$(rank_fabric "$r")"
        if on_rank 0 "timeout 5 bash -c '</dev/tcp/${host}/${PERSIST_COORDINATOR_PORT}' 2>/dev/null"; then
            log "coordinator rank ${r} listening on ${host}:${PERSIST_COORDINATOR_PORT}"
        else
            warn "coordinator rank ${r} NOT listening on ${host}:${PERSIST_COORDINATOR_PORT}"; ok=0
        fi
    done
    [ "$ok" = "1" ] || die "coordinator listeners incomplete -- persistence is not proven up"
    log "all four coordinator listeners up"
}

wait_for_health() {
    # /health only. /v1/models returns 200 with a dead engine; /health returns
    # 503 on EngineDeadError.
    local url="http://127.0.0.1:${PORT}/health" elapsed=0
    log "waiting for ${url} (timeout ${READY_TIMEOUT}s)"
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then log "healthy after ${elapsed}s"; return 0; fi
        if ! docker inspect -f '{{.State.Running}}' "$(rank_container 0)" 2>/dev/null | grep -q true; then
            die "head container exited during startup -- docker logs $(rank_container 0)"
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done
    die "not healthy within ${READY_TIMEOUT}s"
}

up() {
    [ "${PERSISTENCE}" = "on" ] || [ "${PERSISTENCE}" = "off" ] || die "PERSISTENCE must be on|off"
    CACHE_FINGERPRINT="${CACHE_FINGERPRINT:-$(compute_fingerprint)}"
    export CACHE_FINGERPRINT
    # FIX-2: one stamp for the whole launch; each rank appends -rank<N>.
    # write_record's run-id regex is [A-Za-z0-9][A-Za-z0-9._-]{0,95}.
    OBSERVER_RUN_STAMP="${OBSERVER_RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
    export OBSERVER_RUN_STAMP
    observer_enabled && log "observer run stamp ${OBSERVER_RUN_STAMP} -> ${OBSERVER_DIR_HOST}"
    log "cache_fingerprint=${CACHE_FINGERPRINT} persistence=${PERSISTENCE}"
    preflight
    write_inner
    stop_all
    local r
    for r in 3 2 1 0; do launch_rank "$r"; done
    wait_for_health
    coordinator_check
}

case "${1:-up}" in
    preflight) preflight ;;
    coordinator-check) coordinator_check ;;
    up)        up ;;
    down)      stop_all ;;
    logs)      docker logs -f --tail 200 "$(rank_container 0)" ;;
    args)      CACHE_FINGERPRINT="${CACHE_FINGERPRINT:-$(compute_fingerprint)}"
               echo "cache_fingerprint=${CACHE_FINGERPRINT}"
               echo "image=${IMAGE} image_id=${IMAGE_ID:-UNPINNED}"
               echo "coordinator_endpoints=$(coordinator_endpoints_csv)"
               for r in 0 1 2 3; do
                   echo "--- rank ${r} kv-transfer-config ---"
                   if [ "${PERSISTENCE}" = "on" ]; then kv_transfer_config "$r"; else echo "(PERSISTENCE=off)"; fi
               done ;;
    *)         die "usage: $0 {preflight|coordinator-check|up|down|logs|args}" ;;
esac
