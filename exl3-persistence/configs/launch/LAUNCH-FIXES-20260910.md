# Launch fixes, 2026-09-10 AEST — GLM-5.3-Flash TP4 + disk-KV persistence

These are the launch sources the image bakes into `/opt/glm53/launch`:

| File | What this document changes |
|---|---|
| `configs/launch/start-tp4.sh` | FIX-0 … FIX-4 |
| `configs/launch/env.tp4.fleet` | FIX-1 … FIX-4 |

The image must be **rebuilt** to bake these files. The original per-file sha256
values described the private deployment files; the published copies differ by the
sanitization substitutions recorded in `SANITIZATION-AUDIT.md`, so those digests
are not reproduced here.

Package paths below are relative to
`$WORK/fork-worktree/persistence/`.
The pinned vLLM tree is `$WORK/pinned-vllm`
(`83252ea899c6538eaa0c1fb31f28a92c661bbffc`).

Nothing here was run against the fleet. No ssh to any Spark, no build, no
commit. Everything below was validated on head only.

---

## FIX-0 (found while fixing the rest) — the launcher could not run at all

`rank_var()` and `rank_container()` were one-line `local` statements that
referenced a variable being assigned in the *same* statement:

```bash
rank_var() { local r="$1" name="$2" v="RANK${r}_${name}"; printf '%s' "${!v:-}"; }
```

Bash expands every word of a command **before** the builtin executes, so
`${r}` was expanded against the (unset) *global* `r`, and under the script's own
`set -euo pipefail` this aborted at the very first call —
`HEAD_FABRIC="$(rank_fabric 0)"`:

```
./start-tp4.sh: line 52: r: unbound variable
```

Reproduced on the pristine backup of both launchers (bash 5.2.21). This affected
**every** subcommand — `preflight`, `args`, `up`, `down`, `logs` — so no rendered
config in the private deployment runbook can have come from actually running these scripts. Fixed
by splitting the assignments. `mandate:` the scripts' own `set -u`.

---

## FIX-1 — persistence configuration contract

### 1a. `coordinator_url` → `coordinator_endpoints` (the blocking patch)

* **Mandated by** `recipe_persistence/coordinator_http.py:1273-1277`:
  ```python
  raw_endpoints = cfg.get("coordinator_endpoints")
  if (not isinstance(raw_endpoints, list) or len(raw_endpoints) != world_size
          or not all(isinstance(e, str) for e in raw_endpoints)):
      raise ValueError("coordinator_endpoints must list one endpoint per rank")
  ```
  and `README-coordinator-http.md` "Configuration and trust boundary" ("Supply
  exactly one endpoint per TP rank in rank order").
* `coordinator_url` is read by **nothing** in the package (`grep -rn coordinator_url
  recipe_persistence/` → no hits). It was silently ignored and the launch would
  have died in `factory()`.
* Rendered now, in rank order, from `RANK<N>_FABRIC` + `PERSIST_COORDINATOR_PORT`:
  `["198.51.100.1:8976","198.51.100.2:8976","198.51.100.3:8976","198.51.100.4:8976"]`.
  Each string is additionally accepted by the real `parse_endpoint`
  (`coordinator_http.py:127-167`: numeric IP, explicit port, no userinfo/path/
  query/fragment, unicast).

### 1b. `disk_quota_bytes` was a dead key

* `grep -rn disk_quota_bytes recipe_persistence/` → **no hits.** The per-rank
  quota is `Limits.quota` (`storage.py:46`), reached only through
  `coordinator_disk_limits`: `limits = Limits(**limits_kwargs)`
  (`coordinator_http.py:1297-1301`).
* The old value (1 TiB = 1,099,511,627,776) also would **not** have been a
  drop-in: `Limits.__post_init__` requires `0 < low < high <= quota`
  (`storage.py:71-72`), so raising quota without moving the watermarks is a
  separate, deliberate change. We render the package's frozen portable defaults
  instead: `quota=1_000_000_000_000` (1 TB decimal), `high=900e9`, `low=800e9`.

### 1c. Everything explicit

All twelve `Limits` fields (`storage.py:45-58`), the logical credits
(`max_pending_keys=32768`, `max_pending_bytes=64_000_000_000` —
`coordinator_http.py:55-56`), the native step/callback bounds
(`lookup_keys_per_step=8`, `metadata_workers=2`, `metadata_max_submitted=8`,
`metadata_shutdown_timeout=10` — `native.py:130-133`), and the bounded transport
knobs (`coordinator_http.py:1281-1289`) are now rendered rather than defaulted.
`native.py:172-173` raises unless the config's `max_pending_keys` equals
`provider.max_pending_keys`, so those two are driven from one env var.

`coordinator_renew_margin=30.0` is the *derived* default made explicit:
`max(1.0, min(30.0, 0.1*ttl))` with `ttl=lease_seconds=300`
(`coordinator_http.py:784-786`, bound `0 < margin < ttl`).

`prefix_caching_hash_algo: "sha256"` — `_profile_from_config`
(`coordinator_http.py:1128-1130`) rejects anything else, and `native.py:117-119`
re-checks the **live** engine value. The pinned engine default is also `sha256`
(`pinned-vllm/vllm/config/cache.py:141`), so no serve flag is needed.

### 1d. `PYTHONHASHSEED`

`native.py:111-116` refuses to start unless `PYTHONHASHSEED` is a fixed decimal
`0..4294967295` at process start, and keeps the **raw string** (leading zeroes
included) as part of the persistence namespace identity. It is now an explicit
`PYTHONHASHSEED=0` in the env file as well as a `-e` on every rank.

### 1e. Who serves each coordinator endpoint — no daemon, no sidecar

Each rank's **worker-role** factory call constructs and starts its own
`MetadataServer` bound to `endpoints[rank]`, inside the blocking
`initialize_from_config`:

```python
server = MetadataServer(store=store, identity=identity, rank=rank,
                        world_size=world_size, host=endpoints[rank][0],
                        port=endpoints[rank][1], token=token, ...)
server.start()
```
(`coordinator_http.py:1302-1310`; `README-coordinator-http.md` "Startup and
identity": *"Workers synchronously bind/start their local listeners before
querying peer geometry"*.) The **scheduler** role builds only an `RpcClient`
(`:1354-1380`). So there is nothing to launch by hand — the launcher starts no
extra process. Instead it **fails closed after the fact**: the new
`coordinator-check` subcommand (also run automatically at the end of `up`)
proves all four `RANK<N>_FABRIC:8976` listeners accept TCP, and `die`s if any
does not. TCP reachability only; no request is issued and the token is not used.

`coordinator_bind` and `coordinator_server_timeout` appear in
`README-coordinator-http.md` but are read **nowhere** in the code — they are not
rendered.

### 1f. Disk roots

`/srv/kv-persist/rank<N>` → **`/home/user/kv-persist/rank<N>`** (the Sparks'
NVMe-backed home). Bind-mounted into the container at the identical path, which
is what `disk_root` names. `DiskStore` creates `disk_root/rank-<N>` beneath it
with mode 0700 and takes an **exclusive** root lock (`storage.py:82-105`,
`coordinator_http.py:1303`), so two processes must never share a root.

`preflight` now, per rank:
1. creates the root if missing (`mkdir -p -m 0700`) and re-`chmod 0700`s it —
   *the one deliberate write in an otherwise read-only preflight*;
2. requires it writable;
3. `findmnt -T <root> -no FSTYPE,SOURCE` and accepts only
   `ext4|xfs|btrfs|f2fs|ext3` — NFS, overlay, tmpfs and an empty result all
   abort (the Limits accounting assumes a private local filesystem);
4. requires `df -PB1` free ≥ `PERSIST_MIN_FREE_BYTES` = **1,288,490,188,800 B
   (1,200 GiB)** — the 1 TB quota plus the 1 GiB `free_bytes` reserve plus index
   headroom;
5. requires this rank's `RANK<N>_FABRIC` address to exist on the node
   (`ip -o addr show`) and port `8760` to be **free** (`ss -Hltn`), because the
   listener binds mid-`initialize_from_config` and a bind failure there is a
   mid-load engine death.

### 1g. Token — path only, never a value

Contract: `load_auth_token` (`coordinator_http.py:93-124`) opens the path
`O_NOFOLLOW` and rejects a symlink, a non-regular file, a foreign owner
(`st_uid != geteuid()` — the container is root, so the **host** file must be
root-owned), any group/other bit, >4096 bytes, non-ASCII, and a low-entropy
secret (`_validate_secret`, `:83-92`: 32..256 printable non-space chars,
≥128 bits estimated).

Provisioning is operator-side and deliberately **not shipped** with this source
release: the private utility is wired to an internal secret store. Reproduce the
same contract yourself — a dry-run-first installer, payload on stdin only,
published by exclusive hard link, mode 0600, value never echoed and captured
output never surfaced:

```bash
# 1. generate the secret (never to a terminal or a file you keep):
python3 -c 'import secrets;print(secrets.token_hex(32))'

# 2. publish it to each node as a root-owned regular file, then re-verify:
for n in node0 node1 node2 node3; do   ssh $n 'stat -c "%n %a %U %s" /run/glm53/persistence-token'; done
# PASS: /run/glm53/persistence-token 600 root 65
```

Host `/run/glm53/persistence-token` → container
`/run/secrets/persistence-token` (`PERSIST_TOKEN_FILE_HOST` /
`PERSIST_TOKEN_FILE`, mounted `:ro`). `/run` is tmpfs — **the token is lost on
reboot and must be re-published.** `preflight` now checks presence, non-symlink,
mode `600`/`400`, owner `root` and size 32..4096 — metadata only; the value is
never read, echoed or logged, here or anywhere else.

### 1h. Census deadline — the one deliberate deviation, needs the operator

`coordinator_startup_timeout` defaults to **60 s** in code
(`coordinator_http.py:1282`; the README's "120 s" is wrong) and bounds the whole
all-rank census, which runs inside `initialize_from_config` on every rank —
i.e. *after* a ~175 GiB weight load whose duration differs per node. A 60 s
all-rank deadline would fail closed on a healthy fleet. Rendered as
**1800 s**, inside the package's own `top=3600.0`.
**This is the only value in the rendered JSON that is not the package default.
Get the operator's sign-off before the first `PERSISTENCE=on` launch.**

---

## FIX-2 — startup observer (the private deployment runbook S2-1)

`startup_observer.emit()` (`observer/startup_observer.py:379-394`) needs
`VLLM_STARTUP_OBSERVER=1` **plus** `VLLM_STARTUP_OBSERVER_RUN_ID`,
`VLLM_STARTUP_OBSERVER_MANIFEST_SHA256` (`re.fullmatch(r'[0-9a-f]{64}')`, `:387`)
and `VLLM_STARTUP_OBSERVER_DIR`. `write_record` (`:352-357`) requires the run id
to match `[A-Za-z0-9][A-Za-z0-9._-]{0,95}` and the directory to be
**precreated, absolute and not a symlink**. The launcher previously forwarded
only the first and mounted nothing.

Now:

* `OBSERVER=on|off`, default **on**, in the env file. With `on`, a missing or
  non-64-lowercase-hex `OBSERVER_MANIFEST_SHA256`, or a missing
  `OBSERVER_DIR_HOST`/`OBSERVER_DIR`, aborts the launch (`observer_check_config`).
  It never degrades silently to "no receipts".
* `VLLM_STARTUP_OBSERVER_MANIFEST_SHA256=5de9cf56268e81cc11e43a5e24b0480717cb34acf2dbce2222927f98f72dcd24`
  — verified on head as the sha256 of
  `observer/observer-callsites-83252ea89.manifest.json`, byte-identical to
  `patches/observer/observer-callsites-83252ea89.manifest.json`, which the
  Dockerfile copies to `/opt/glm53/observer/` (Dockerfile:685).
  `preflight` re-verifies it **against the image** with one read-only
  `docker run --rm --entrypoint sha256sum` on rank 0 (no GPU, no engine);
  `OBSERVER_VERIFY_IN_IMAGE=0` skips that container.
* `VLLM_STARTUP_OBSERVER_RUN_ID=<stamp>-rank<N>` where `<stamp>` is one
  `date +%Y%m%d-%H%M%S` taken once per `up`, so all four ranks share it.
  Matches the run-id regex.
* `VLLM_STARTUP_OBSERVER_DIR=/observer-receipts`, backed by
  `-v /home/user/persistence-receipts/observer:/observer-receipts` (rw) on every rank;
  `preflight` creates the host dir and refuses a symlink.
* Receipts land as `<run_id>.<host>.<pid>.jsonl`, **root:root 0600**
  (`write_record` refuses to write a file it does not own, `:366-368`). Read
  with sudo, or `sudo chown -R user:user <RECEIPTS>/observer` after the run.

**Known conflict with the private deployment runbook's S2-1 PASS criterion — not fixable here.**
`observer-callsites-83252ea89.manifest.json` itself declares
`known_incomplete_stages` for `engine_final`, `worker_preallocation` and
`worker_postbind` (fields dropped from the new call-site signatures at this
pin), each explicitly annotated `-> STARTUP_OBSERVER_INCOMPLETE`. So on this
manifest three of the four stages are *expected* to emit
`[STARTUP_OBSERVER_INCOMPLETE]`, while the private deployment runbook §3.10 S2-1 demands four complete
stages and no such line. Wiring the env correctly does not resolve that; the
receipt criterion needs restating or the call-site port extending.

---

## FIX-3 — node3 DFlash2 path

`notes/staging-audit.md` (2026-09-10 read-only audit, rows for node3):
EXL3 weights verified under `~/.cache/huggingface/hub-local-staging`, but
`~/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2` is
"**Complete, local. Goal already met.**" — i.e. the drafter is in the *other*
hub tree. The launcher mounted exactly one host hub at
`/root/.cache/huggingface/hub` and the inner script hardcoded that path for both
lookups, so rank 3 would have died with
`no snapshot for incoai--GLM-5.3-Flash-DFlash2` *after* a full weight load.

Fix — per-node second hub, generic rather than a node3 special case:

* `RANK<N>_HUB2` (env). Mounted read-only at
  `/root/.cache/huggingface/hub-secondary` **only when the host directory
  exists**; a declared-but-absent `HUB2` is a hard `die` (docker would otherwise
  create a root-owned empty directory and the drafter would vanish again).
* `-e HUB_DIRS=<primary>[:<secondary>]` is forwarded, and the inner script's new
  `resolve_snapshot()` walks it in priority order for **both** `MODEL_REPO` and
  `DFLASH_REPO` (TrellisMX: for the revision-pinned carrier and, under
  `SPECULATOR=dflash2`, the drafter).
* `RANK<N>_DRAFT_REPO_DIR` (container path) overrides the search outright.
* Per the audit only rank 3 needs it:
  `RANK3_HUB2=/home/user/.cache/huggingface/hub`; ranks 0–2 are empty (node0 has
  both models under `/srv/node0-share/huggingface/hub`, node1 both under
  `~/.cache/huggingface/hub`, node2 both under `hub-local-staging` — its
  DFlash2 rsync completed and verified).
* `preflight` now **resolves the drafter snapshot on each node** across the same
  hub list and aborts if it is absent (the old preflight did not check the
  drafter at all). TrellisMX's carrier check was widened the same way.

This makes the private deployment runbook's §2.2 "required fix on node3" (an rsync into
`hub-local-staging`) optional rather than mandatory — either path now works, and
preflight proves it either way.

---

## FIX-4 — image pin

* `IMAGE=…:latest` → the dated tags:
  * EXL3 `glm53-flash-exl3-tp4-persist:20260910-83252ea89`
  * TrellisMX `glm53-flash-trellismx-tp4-persist:20260910-83252ea89`
* New `IMAGE_ID=` in each env file, carrying a loud
  `>>> AFTER EVERY REBUILD: update IMAGE and IMAGE_ID together` banner, because
  **these images will be rebuilt to bake these launch files** and the Ids below
  are the pre-rebuild ones:
  * EXL3 `sha256:d1169125cc81bc9c279c8328e8739a4aef02ab31e14bab001da27839bfa43059`
  * TrellisMX `sha256:6812a405cfb86590a9c670f9b749de0610b4bd5bcaa75a45023727adc4c034c8`
* `preflight` requires **every** rank's `docker image inspect --format '{{.Id}}'`
  to equal `IMAGE_ID` (and, as before, to agree with rank 0 — tonyd2wild
  hard-won rule #3). `image_id_expected()` also rejects a malformed pin.
* `compute_fingerprint()` now takes its image term from `IMAGE_ID` rather than a
  live `docker image inspect`. The old form silently yielded `id="unknown"` on
  any host without the image loaded (e.g. rendering `args` on head for review), so
  the reviewed `cache_fingerprint` was **not** the one the fleet would use.
  Rendering is now host-independent and reproducible.

---

## Validation performed (all on head, 2026-09-10)

| Check | Result |
|---|---|
| `bash -n` the launcher | PASS |
| `bash -n` the extracted inner script | PASS |
| `python3 -m py_compile tests/validate_kv_transfer_config.py` | PASS |
| `./start-tp4.sh args`, ranks 0-3 | renders; per-rank `disk_root`, shared endpoint list |
| all 4 rendered JSONs → `tests/validate_kv_transfer_config.py` | **4/4 ACCEPTED** |
| pre-fix control (`coordinator_url` + `disk_quota_bytes`) → same validator | **REJECTED**: `coordinator_endpoints must list one endpoint per rank` |
| multi-hub `resolve_snapshot()` against a synthetic HF tree | model from hub 1, drafter from hub 2; single-hub case exits `no snapshot for incoai--GLM-5.3-Flash-DFlash2` |
| host-side `rank_draft_dir()` against the same tree | resolves via the second hub |

Receipts (diffs, rendered JSON, validator output) are under
`<RECEIPTS>/launch-fixes/`:
`start-tp4.sh.diff`, `env.tp4.fleet.diff`, `start-tp4-trellismx.sh.diff`,
`env.tp4.trellismx.diff`, `rendered-args-exl3.txt`,
`rendered-args-trellismx.txt`, `validator-acceptance.txt`.
The validator itself is `$WORK/tests/validate_kv_transfer_config.py`.

`tests/validate_kv_transfer_config.py` runs the package's *own* code
torch-free: the real `NodeLocalDiskOffloadingSpec.__init__` via
`native._load_native()` (with the pinned `vllm.v1.kv_offload.base` names stubbed —
its `OffloadingSpec.__init__` is copied verbatim from
`pinned-vllm/vllm/v1/kv_offload/base.py:586-607`), plus the real
`parse_endpoint`, `_positive`, `_profile_from_config` and `Limits`.

### Not validatable on head — carried forward as MISS

* Anything needing CUDA, a real engine, real KV geometry, the census handshake
  or an actual store: `create_worker`, `Geometry.from_canonical`, the
  all-rank registration, `layout_fingerprint`, lease/credit behaviour.
* `load_auth_token` against the real file (root-owned, on each Spark) — the
  provisioning script has not been executed; ssh to the Sparks was out of scope.
* The persistence roots' real backing filesystem and free space on node0-3
  (`findmnt`/`df` run only at preflight, on the nodes).
* The `IMAGE_ID` values: neither image exists on head. They are transcribed from
  the build record and will change at the next rebuild.
* `native._prefix_hash_algorithm()` reads the live vLLM cache config; the
  validator asserts the pinned `sha256` default instead.
