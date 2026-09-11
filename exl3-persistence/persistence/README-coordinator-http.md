# Authenticated rank-local metadata coordinator

This is the concrete CPU-tested provider for `recipe_persistence.native`, not a
live TP4/CUDA qualification. It uses Linux/POSIX local filesystems and Python's
standard library. **KV payload never travels over HTTP.** Each worker owns one
`DiskStore`, shared by its metadata listener and native transfer handler.

## Configuration and trust boundary

Select these provider fields inside the native connector's extra configuration:

```json
{
  "coordinator_module_path": "recipe_persistence.coordinator_http",
  "coordinator_factory": "factory",
  "coordinator_endpoints": [
    "192.0.2.10:9300", "192.0.2.11:9300",
    "192.0.2.12:9300", "192.0.2.13:9300"
  ],
  "coordinator_auth_token_file": "/run/secrets/persistence-token",
  "disk_root": "/var/lib/recipe-persistence",
  "max_pending_keys": 32768,
  "max_pending_bytes": 64000000000,
  "lookup_keys_per_step": 8
}
```

These are documentation-only addresses and example paths, not a launch-ready
configuration. Replace them externally. The other native requirements in
[README.md](README.md), including actual-model/cache fingerprint, trusted tenant,
fixed raw hash seed and explicit physical staging limits, remain mandatory.
Never copy a benchmark's host-memory allocation as the staging budget.

* Supply exactly one endpoint per TP rank in rank order. Endpoints must use
  numeric IPv4 or bracketed IPv6 and an explicit port. DNS names, scoped IPv6,
  credentials in URLs, nonempty paths, query strings and fragments are rejected.
  Only `http` is supported; unspecified/multicast destination IPs are rejected.
* `coordinator_bind` may separately select the local bind address. Otherwise each
  worker binds its rank's advertised address. A wildcard bind is allowed only as
  a bind address, not an advertised destination. The package changes no firewall,
  route, host policy or gateway configuration.
* The bearer token comes only from a local regular file, not a literal config
  field, environment token value or argv. The file must belong to the process's
  effective UID and have no group/other permissions; use mode `0600` (or read-only
  `0400`). Symlinks and nonregular files are rejected. Root containers therefore
  require a root-owned token file, separately from any user-readable backend env
  file. Provision a CSPRNG-generated secret with at least 128 bits of entropy;
  syntactic weak-token checks cannot prove entropy.
* Bearer authentication is **not encryption**. Use a restricted private control
  network and an independently reviewed network policy. Do not expose these
  listeners through an inference gateway or to untrusted tenants. TLS and token
  rotation are not implemented. Keys and geometry metadata are sensitive even
  though payload bytes never cross this transport.
* Each rank uses `disk_root/rank-N`; its one `DiskStore` holds the exclusive root
  lock. Do not run a second daemon/index over the same root.

## Startup and identity

Workers synchronously bind/start their local listeners **before** querying peer
geometry. They never wait for a scheduler listener. Every rank publishes its
actual canonical padded pages, ordered unpadded group references, exact group
bytes, lease TTL and a hashed runtime profile. Receivers recompute geometry bytes
and fingerprints rather than trusting inconsistent claims.

At pinned vLLM `83252ea899c6538eaa0c1fb31f28a92c661bbffc`, worker `get_worker`
(formerly `get_handlers`) runs inside blocking `initialize_from_config`;
scheduler `get_manager` is constructed afterwards. Native handshake metadata
arrives too late to discover these endpoints. See the pinned
[engine initialization](https://github.com/vllm-project/vllm/blob/83252ea899c6538eaa0c1fb31f28a92c661bbffc/vllm/v1/engine/core.py)
(`EngineCore._initialize_kv_caches` calls `initialize_from_config` at `:354`).
This is a source-order contract, not a distributed startup observation.

The census requires the full rank set and matching ordered group layout and
runtime profile. The profile binds the full bounded caller cache fingerprint,
trusted tenant, raw `PYTHONHASHSEED` and SHA256 hash policy; the scheduler compares
workers against **its own** expected profile. No truncation is used. The returned
`layout_fingerprint` hashes the ordered all-rank canonical census and enters the
native disk namespace. Same-sized layout/reference reordering is not equivalent.

## Admission, failure and resource bounds

A load hit exists only after every rank grants a read lease. A store ticket exists
only after every rank reserves its exact local bytes. Logical descriptor/key/byte
credits are reserved **before** RPC submission, including concurrent pending
operations. These credits are not payload allocations. Rank failure or denial
vetoes the operation; later successful ranks are also rolled back. Request IDs
are atomically bound to the complete reservation method/parameters. Unexpired
retry receipts are never evicted to make room: receipt exhaustion returns a
bounded saturation failure. Receipt retries check the original lease without
extending its TTL. This prevents duplicate leases within the retained retry window
and prevents one request from borrowing another request's result.

Renewal success is conservatively anchored at the **start** of the all-rank
operation, not its slowest response. Cached renewal never deliberately extends
that window. An expired/unproved renewal is not a hit. Store completion requires
all-rank durability checks; partial completion invalidates all ranks. Unknown
cleanup is never acknowledgement. A bounded cleanup replay queue retains failed
operations; if it overflows, a permanent bounded uncertainty flag closes admission
rather than forgetting failures and later claiming success. Remote lease expiry
is a recovery backstop, not a fabricated acknowledgement.

The RPC client uses numeric `socket.connect`, owned sockets and one absolute
admission-based deadline covering queue wait, retry, status, headers and body.
`shutdown(SHUT_RDWR)` interrupts socket readers, including slow header/body drips.
Submitted operations and active calls have separate hard limits; cancelled queued
work is physically removed. Active credits remain held until the real callable
and its watchdog finish. Watchdogs are joined before their credits are released.
Only fully validated, completely consumed responses can return to the bounded
idle connection pool. TCP_NODELAY is retained. Request/response headers are capped
at 16 KiB; JSON bodies are separately bounded. Errors/logs do not echo credentials
or request data.

Provider shutdown stops admission, aborts/cancels owned transport work and proves
handler drain before closing the shared store. Unproved drain raises a static
error, keeps ownership and permits retry; it is not reported as a successful
close. Startup failures use the same rule. Native GPU/media drain must precede
the provider close callback; a successful callback is consumed once, while a
failed callback remains retryable. A permanently stuck non-socket operation still
requires fail-stop/operator recovery, not unsafe resource recycling.

### Scheduler fairness is separate from transfer fairness

The native adapter defaults to **8 new load reservation attempts per scheduler
step**. Cached hits do not consume that budget. A bounded deferred-owner queue
rotates owners between steps, preserving acquired prefix leases; deferred lookup
returns `None`, not a false terminal miss. Cancellation and preparation remove
queued owner state. The pinned scheduler keeps polling through
`has_pending_work` and calls `on_schedule_end` to reset the quantum.

The adapter also gives store admission a separate per-step attempt budget, using
the same configured quantum; an unattempted/denied store is never a durable skip.
The companion core must retain finished-request frontiers and GPU references
across partial/zero admission, continuing idle scheduling until issued jobs drain
and the frontier completes or is honestly abandoned. Native `can_store()` returns
False only for permanent disablement/shutdown, never temporary pressure. The
coordinator exposes the same non-I/O capability snapshot, including known client
shutdown or lost-cleanup state. Explicit terminal rank errors disable admission;
a sticky failed/closed rank store returns static HTTP 410 rather than disguising
that condition as an ordinary null lease. Valid misses and bounded quota/key/RPC
pressure remain retryable, not successful stores.
Completion, cancellation, store completion and renewals use a small background
metadata executor. Existing ownership maps hold bounded cleanup intents; at most
`metadata_max_submitted` futures are submitted, with `metadata_workers` callbacks
running. The scheduler only reconciles completed acknowledgements. Retiring
identities cannot be reused and retain ticket/byte/key credits until actual ACK.
Failed-key invalidation immediately vetoes local lookup; the native adapter keeps
that conservative per-key quarantine until a safe reset. Prepared GPU/media leases
are never released before their own physical drain. If quarantine capacity is
exhausted, admission closes before acknowledged veto slots can be reclaimed. If
all slots are still unacknowledged, a bounded namespace fail-stop refuses reset
and shutdown until external recovery; it never claims an unrecorded key was
retained or cleaned.

`CoordinatorProtocol.lease_deadline(ticket)` supplies a non-I/O, monotonic validity
snapshot. Cached lookup uses that finite proof, not an arbitrary-duration cached
True; expired/unknown proof defers lookup and queues renewal. This method must not
wait on a lock shared with disk or network operations. Unknown/negative callback
results retain bounded cleanup ownership and close new admission; scheduling
polls may retry cleanup, but enqueueing is never acknowledgement. Shutdown waits
only through its explicit metadata deadline, then raises and retains ownership if
drain remains unproved.

This bounds fresh RPC **count**, not wall-clock latency: eight slow calls can still
be slow, and one store attempt may require both a durable-read probe and a write
reservation. Cached prefix rescans and bounded ownership-map sweeps remain CPU
costs. Physical transfer uses a separate shared staging ring and byte-deficit
fairness. None of these mechanisms establishes actual decode latency or 270K
runtime headroom.

## Optional limits

| Extra-config key | Default | Meaning |
|---|---:|---|
| `coordinator_startup_timeout` | 120 s | Whole census deadline |
| `coordinator_rpc_timeout` | 10 s | Absolute per-operation RPC deadline, including retry |
| `coordinator_server_timeout` | 5 s | Server connection I/O/idle timeout |
| `coordinator_max_body_bytes` | 65536 | Request/response JSON body limit; maximum 1 MiB |
| `coordinator_server_threads` | 16 | Hard concurrent accepted-handler bound |
| `coordinator_client_pool` | 4 | Idle connections per endpoint and default submission sizing |
| `coordinator_idempotency_entries` | 32768 | Expire-only reservation retry receipts; saturation does not evict live IDs |
| `coordinator_renew_margin` | derived | Positive safety margin below the minimum rank lease TTL |
| `coordinator_close_store_on_shutdown` | true | Factory owns/closes the worker store by default |
| `coordinator_disk_limits` | `Limits` defaults | Explicit rank-local storage limits; see main README |
| `metadata_workers` | 2 | Native background cleanup/renewal callbacks |
| `metadata_max_submitted` | 8 | Native submitted callback futures, including queued/running work |
| `metadata_shutdown_timeout` | 10 s | Native metadata drain deadline; timeout retains ownership |

`RpcClient` also exposes explicit `max_active` and `max_pending` constructor
bounds for embedders. Default active calls are at most eight and at most the rank
count; default submitted calls are active-count times pool size. The native
`lookup_keys_per_step` default is an initial conservative choice, not a measured
optimal setting. Physical memory additionally includes bounded Python metadata,
SQLite cache, sockets, driver/native descriptors and buffered filesystem pages.

## CPU evidence and remaining gates

```sh
PYTHONPATH=persistence python3 -m pytest persistence/tests -q
PYTHONPATH=persistence python3 -m pytest -q -s \
  persistence/tests/test_coordinator_http.py::LongPrefixMetadataTests
```

Tests cover real loopback HTTP, SQLite stores, authentication/framing failures,
partial-rank rollback, concurrent admission/idempotency, slow peers, cancellation,
shutdown proof, profile drift, restart visibility and bounded large metadata
queues. The 4219-key fixture reserves over 9 GB **logically without allocating the
payload**, using the production 300-second lease TTL. It prints reservation and
release times separately; neither is disk restore throughput. A native CPU step
driver completes 4219 lookups across 528 steps with another request's heartbeat on
every step. That is scheduling-pattern evidence, not a GPU inference run.

Remaining gates include actual TP4 startup, canonical target/indexer/draft geometry,
CUDA/media drain under faults, complete payload roundtrip after restart, scheduler
latency under competing requests, host-memory/SSD limits and the unchanged 270000
context baseline. No fleet qualification, public exposure or performance claim is
implied by CPU test success.
