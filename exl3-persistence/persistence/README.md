# Rank-local disk KV persistence candidate

CPU-tested portable storage and native-worker implementation, **not hardware
qualification**. Supports the vLLM external offloading ABI at
`83252ea899c6538eaa0c1fb31f28a92c661bbffc` (ported from
`ab666069935c1f23e8ef56038b4659ac9e8f19f8`; see `../docs/BASE-DECISION.md`).
Requires the companion native failure/cursor/HMA patches from
`patches/persistence-flash/`: stock native workers assert on unsuccessful
transfers. Installing this package alone does not make that runtime fail-soft.

## Scope

Persist complete eligible prefix blocks, preserving every registered target,
indexer and draft group. This is **not** a snapshot of in-flight application,
stream, sampling/RNG or speculative state. No group is synthesized, silently
skipped or zero-filled. DFlash volatile tails and final partial blocks remain
subject to the native scheduler's eligibility rules. There is no full-session
host payload allocation, host tuning, inference launch or replacement gateway.

Initial support is an explicitly configured **trusted single-tenant canary**.
`tenant_namespace` is operator-controlled, not a client-supplied authorization
claim. Unrelated tenants require separate namespaces and a separately verified
trusted identity path. Request transfer parameters never select a tenant here.

## Components and ownership

- `recipe_persistence/storage.py`: CPU-only `DiskStore`, immutable per-key files,
  bounded SQLite index, quota and lease-safe tombstone collection.
- `recipe_persistence/coordinator.py`: exact `CoordinatorProtocol`, `Ticket`, and
  provider factory contract. `LocalCoordinator` is an in-process conformance
  fixture, **not a remote multi-node provider**.
- `recipe_persistence/coordinator_http.py`: separately configured authenticated
  HTTP metadata provider and real-loopback CPU transport fixtures. See
  `README-coordinator-http.md` for its configuration and qualification boundary.
- `recipe_persistence/geometry.py`: actual canonical group references and their
  unpadded bytes; physical ring sizing counts the unique padded tensors once.
- `recipe_persistence/handlers.py`: shared bounded load/write wave pump, native
  copy events, durable media completion, cancellation and ownership fences.
- `recipe_persistence/native.py`: lazy external `NodeLocalDiskOffloadingSpec` and
  metadata manager. A provider is mandatory; there is no scheduler-local index
  fallback that pretends to know another rank's disk.

The provider factory is called with
`factory(config=extra_config, role='scheduler'|'worker', rank=None|int,
world_size=int, geometry=None|Geometry)` and returns the attributes documented
in `coordinator.Provider`. Workers must register their **actual canonical
geometry** before the scheduler admits persistence. A scheduler's representative
uniform cache spec cannot supply the true mixed-layer byte census. Provider
transport, authentication, deadlines and lifecycle must be supplied explicitly.
The metadata endpoint and worker must share one `DiskStore` instance; separate
processes cannot open the same rank root concurrently. The optional provider
`close` callback is owned by native lifecycle code: a successful acknowledgement
is consumed once on startup cleanup or after shutdown drain, never during reset.
A failed/negative acknowledgement raises a static unproved-drain error and keeps
cleanup ownership for retry. Close the metadata listener and its active requests
before closing the shared store. Unsafe GPU drain retains ownership rather than
tearing down live resources.

## Storage contract

Each private rank root contains a fixed-budget SQLite database and immutable
objects. Namespace hashes bind the operator's model/draft/template/backend/layout
fingerprint, trusted tenant, actual group byte geometry, the provider's mandatory
all-rank `layout_fingerprint`, SHA256 hash algorithm and raw fixed launcher
`PYTHONHASHSEED`. The layout digest covers ordered canonical padded pages and
per-group references on every rank: a same-size reference reorder must miss.
Changing any such input intentionally misses.
The seed must be supplied **before Python starts**, not assigned inside the plugin.
A fingerprint must describe the pinned Python/hash serialization policy as well
as model revisions, tokenizer/template, RoPE, precision, TP/group ordering and
packed-cache format. Same byte sizes alone do not prove a compatible layout.

An object uses a fixed 4096-byte header: version magic, exact length, namespace,
key digest and payload SHA256. Header padding is validated and no unbounded JSON
header is parsed. Payload chunks are written/read through bounded memoryviews.
Publication order is file fsync -> atomic rename -> object-directory fsync ->
SQLite EXTRA-synchronous commit (including DELETE-journal directory sync). A write is never reported successful before all
four stages. Checksums are verified in staging **before** a GPU load is launched.
A lookup is a reserved, provisional metadata hit, not a checksum-valid restored
cache. A corrupt object discovered after lookup produces a failed result and
requires the companion all-rank invalidation/recompute path.

One process holds an exclusive rank-root lock. A restart discards old leases,
reconciles reservations/partials/orphans with the index using streaming scans,
rechecks committed file lengths, and rebuilds byte accounting. No filename list
or terabyte-scale residency dictionary is loaded into RAM. Truncated files and
uncommitted renamed files become misses. Checksums are checked on restore, not
by an unbounded startup payload scan. Unexpected root contents are rejected,
not silently ignored for quota purposes. Files are 0600; owned directories 0700.
This is access control, **not encryption at rest**; use an externally managed
protected/encrypted filesystem if required.

### Quota is not RAM

`Limits` defaults are decimal and **per rank**:

| Limit | Bytes |
|---|---:|
| Hard quota | 1,000,000,000,000 |
| Begin eviction | 900,000,000,000 |
| Target after eviction | 800,000,000,000 |
| SQLite database cap | 67,108,864 |
| SQLite cache | 2,097,152 |
| Maximum single-object payload | 67,108,864 |

The quota reserves two database budgets (database plus rollback journal), root
metadata, rounded object allocation including its header, and conservative
per-object directory overhead. Pending writes are charged for their full reserved
size **before** any partial is created. Partials and tombstones stay charged
until unlink and directory fsync complete. SQLite has a page cap, file-backed
temporaries, mmap disabled, and explicit object/lease count limits. Separate
filesystem-free byte/inode reserves reject admission before the filesystem fills.
The index/object-count caps may reject small-object workloads before the byte
quota; 1 TB is a hard ceiling, not guaranteed usable payload capacity. Increase
the bounded on-disk index budget explicitly if the measured object census needs
it; that does not increase the fixed SQLite memory cache.
The root is exclusively owned: external writers modifying its contents violate
the accounting contract. Filesystem compression/reflinks and thin-pool capacity
are not independently qualified by the logical reservation accounting.

Eviction persists tombstones before unlink, excludes active reader and writer
leases, and waits for configured grace. Existing leased readers may complete;
new lookups cannot acquire tombstoned objects. Expired **queued** leases can fail
cleanly. An active syscall holds the store lock, so time expiry never permits a
janitor to unlink a file still used by that operation. A slow syscall cannot be
forcibly cancelled safely: cancellation waits for drain, and unsafe CUDA drain
failure is fail-stop rather than recycled memory or false success.

There is one physical ring for reads and writes, no independent 4 GiB/1 GiB pools.
Its payload allocation is exactly
`rows * sum(canonical_tensor.page_size_bytes)`, including padding even when disk
objects omit it. Byte-deficit round robin shares a bounded quantum across jobs
and directions; every job holds at most one active row, and small group objects
cannot monopolize an unbounded burst or starve writes. Native pointer descriptors/events, Python metadata, SQLite and
OS/driver bookkeeping are **additional** memory. Buffered I/O page cache is not a
hard userspace allocation budget; advisory cache dropping does not guarantee an
OS memory ceiling. The previously measured small direct-I/O buffer is not a
qualified runtime ring budget. No staging default is advertised as fitting the
selected context limit; explicitly configure bytes/rows after geometry and
headroom qualification.

Metadata admission is distinct from ring occupancy. A 270K prefix can require
thousands of leased descriptors and several GB of **logical queued bytes** while
streaming through a few physical rows. The reference coordinator defaults to
32,768 pending descriptors and 64,000,000,000 logical bytes; this allocates neither
64 GB nor a whole session payload. The remote provider must use matching bounded
full-prefix descriptor capacity, not cap the whole prefix to ring bytes. GPU
admission remains vLLM-owned. Large-prefix fixture tests demonstrate this
separation, not a measured model layout or 270K inference acceptance.

## Native configuration contract

External hook:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "spec_name": "NodeLocalDiskOffloadingSpec",
    "spec_module_path": "recipe_persistence.native",
    "offload_prompt_only": false
  }
}
```

This snippet is intentionally **incomplete, not a launch command**. Supply:

- Explicit absolute `disk_root`, or rank-local `RECIPE_PERSISTENCE_ROOT`.
- `trusted_single_tenant: true`, `tenant_namespace`, `cache_fingerprint`.
- `coordinator_module_path` and `coordinator_factory`, plus provider configuration.
- Positive `staging_bytes`, `staging_rows`, `max_pending_keys`; manager, worker and
  provider queue bounds must agree. Optional `disk_io_threads` is additionally
  capped by physical rows.
- Fixed `PYTHONHASHSEED` at process launch and the existing `sha256` prefix hash.
- The coordinator's `lease_deadline(ticket)` must be a non-I/O monotonic validity
  snapshot. Its `can_store()` is a non-I/O permanent-capability snapshot, not an
  instantaneous free-capacity check. Native defaults are `lookup_keys_per_step=8`, `metadata_workers=2`,
  `metadata_max_submitted=8`, `metadata_shutdown_timeout=10` seconds. Fresh store
  attempts use a separate step quantum; callbacks/renewals run in bounded
  background work. These are conservative initial limits, not performance results.

Only native `blocks_per_chunk=1` (formerly `block_size_factor`), data parallel size
one, and the exact registered rank/group census are accepted. Omit the optional
offloaded chunk-size override.
No old divisor patch, hard-coded layer count, group-ID shortcut, allocator policy
or baseline context reduction is needed by this module.

All-rank lookup must reserve every rank or roll back every partial reservation.
Local durable shards alone are insufficient for a hit. One failed store vetoes
its shards; one failed load globally invalidates its affected keys only after
native issued copies drain. Unknown/failed peers must never be treated as an
acknowledgement. Store omissions caused by pressure remain retryable; only
confirmed durable existing keys are returned as `skipped_keys`. The companion core
must retain finished-request store frontiers and GPU ownership across partial or
zero admission and keep scheduling their work. `can_store()` is False only for
permanent native disablement/shutdown, never temporary pressure or an exhausted
step quantum; abandonment is not a successful/skipped store.

Normal metadata callback failures close further admission and retain bounded
retry ownership without synchronously blocking scheduler callbacks. The manager
exposes a static `degraded_reason` and logs the first failure once without exception
text, keys, paths or tokens. Accepted tickets retain accounting until actual
cleanup ACK; failed keys are vetoed immediately. Unsafe drain or an untrackable
quarantine overflow is explicit fail-stop/external recovery, not fabricated
cleanup. Reset/shutdown cannot silently discard active ownership. See
[the HTTP provider contract](README-coordinator-http.md) for the separate transport
and callback bounds.

## CPU tests and remaining qualification

From the checkout root:

```sh
PYTHONPATH=persistence python3 -m pytest tests -q
```

Storage and coordinator tests also run with standard-library unittest; native
fixtures use pytest and fake canonical/native objects, not CUDA. Importing
`recipe_persistence`, its storage or geometry modules does not import torch or
vLLM. Accessing the optional native class imports the native ABI; allocating its
handlers is the CUDA-dependent path. Tests use small temporary fixtures, injected
faults and metadata-only large-prefix counts, never a real 1 TB allocation.

Covered: checksums/short/missing files, write/fsync/rename errors, process crash
between rename and index commit, exact durability ordering, quota including
reservations/partials, restart/reduced-quota recovery, bounded SQLite/index,
leases/expiry/tombstones, all-rank failure rollback, mixed unpadding/physical
geometry, bounded wave ownership, cancellation/drain and lazy native imports.

Still required before operational claims: full pinned runtime/companion-patch
integration; actual target/indexer/draft/rank tensor census; byte-exact CUDA
roundtrips; unchanged maximum-context admission and measured UMA/driver/page-cache
peaks; real multi-node timeout/failure/restart tests; full-prefix idle drain;
performance/fairness under oversubscribed traffic; filesystem/power-loss durability
qualification. One object and fsync per logical block favors simple reviewable
correctness, **not a demonstrated throughput optimum**. No pack compaction or
arbitrary in-flight checkpoint mechanism is included.

See `NOTICE` for code provenance and licensing. No pushes, deployments or service
changes are side effects of these tests.
