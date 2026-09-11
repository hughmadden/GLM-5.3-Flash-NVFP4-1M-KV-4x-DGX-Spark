#!/usr/bin/env python3
"""Validate a rendered --kv-transfer-config against the PINNED persistence package.

Torch-free, runs on head. It exercises the package's OWN validation code:

  * recipe_persistence.native.NodeLocalDiskOffloadingSpec.__init__
    (native.py:92-155) -- via the real _load_native(), with the pinned
    vllm.v1.kv_offload.base names stubbed. The stub's OffloadingSpec.__init__
    is copied verbatim from pinned-vllm vllm/v1/kv_offload/base.py:586-607.
  * recipe_persistence.coordinator_http.parse_endpoint / _positive /
    _profile_from_config (the real functions).
  * recipe_persistence.storage.Limits (the real dataclass + __post_init__).

What it CANNOT prove on head: anything needing CUDA, an engine, real geometry,
the census handshake, or the token file (root-owned, on the Spark).

Usage:
  PYTHONHASHSEED=0 python3 validate_kv_transfer_config.py \
      --package <path to persistence/> --rank 0 --world-size 4 [json file or -]
"""
import argparse, json, os, sys, types
from dataclasses import dataclass, field
from typing import Any


def install_vllm_stub():
    """Minimal stand-ins for the pinned vLLM names native.py imports."""
    for name in ("vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.base",
                 "vllm.config", "vllm.v1.kv_offload.config"):
        sys.modules.setdefault(name, types.ModuleType(name))
    base = sys.modules["vllm.v1.kv_offload.base"]

    class _Stub:  # placeholder ABCs; native subclasses them but we never call them
        pass

    @dataclass
    class OffloadingKVEventsConfig:
        enable_kv_cache_events: bool = False
        self_describing_kv_events: bool = False

    class OffloadingSpec:
        # verbatim from vllm/v1/kv_offload/base.py:586-607 @ 83252ea89
        def __init__(self, config):
            self.config = config
            self.extra_config = config.extra_config
            self.replicated_layout = False
            self.kv_events_config = OffloadingKVEventsConfig(
                enable_kv_cache_events=config.enable_kv_cache_events,
                self_describing_kv_events=bool(
                    self.extra_config.get("self_describing_kv_events", False)),
            )
            self.offload_prompt_only = bool(self.extra_config.get("offload_prompt_only", True))
            self.tokens_per_block = tuple(g.tokens_per_block for g in config.groups)
            self.tokens_per_hash = config.cache.tokens_per_hash
            self.blocks_per_chunk = config.cache.blocks_per_chunk

    for n in ("GPULoadStoreSpec", "LoadStoreSpec", "LookupResult", "OffloadingManager",
              "OffloadingWorker", "PrepareStoreOutput", "RequestOffloadingContext",
              "TransferResult"):
        setattr(base, n, type(n, (_Stub,), {}))
    base.OffloadingSpec = OffloadingSpec
    base.OffloadingKVEventsConfig = OffloadingKVEventsConfig


@dataclass
class _Group:
    tokens_per_block: int


@dataclass
class _Cache:
    tokens_per_hash: int = 4
    blocks_per_chunk: int = 1


@dataclass
class _Parallel:
    rank: int
    world_size: int
    data_parallel_size: int = 1


@dataclass
class _Config:
    extra_config: dict
    parallel: _Parallel
    groups: tuple = (_Group(4), _Group(128))
    cache: _Cache = field(default_factory=_Cache)
    enable_cache_events: bool = False
    enable_kv_cache_events: bool = False
    canonical_layout: bool = False


def check_factory_config(cfg, world_size):
    """Re-run coordinator_http.factory's config-only checks (no sockets)."""
    from recipe_persistence import coordinator_http as ch
    from recipe_persistence.storage import Limits
    out = {}
    raw = cfg.get("coordinator_endpoints")
    if (not isinstance(raw, list) or len(raw) != world_size
            or not all(isinstance(e, str) for e in raw)):
        raise ValueError("coordinator_endpoints must list one endpoint per rank")
    out["endpoints"] = [ch.parse_endpoint(e) for e in raw]
    out["max_pending_keys"] = ch._positive(cfg, "max_pending_keys", ch.DEFAULT_MAX_PENDING_KEYS, integer=True)
    out["max_pending_bytes"] = ch._positive(cfg, "max_pending_bytes", ch.DEFAULT_MAX_PENDING_BYTES, integer=True)
    out["startup_timeout"] = ch._positive(cfg, "coordinator_startup_timeout", 60.0, top=3600.0)
    out["rpc_timeout"] = ch._positive(cfg, "coordinator_rpc_timeout", 10.0, top=300.0)
    out["server_threads"] = ch._positive(cfg, "coordinator_server_threads", 16, top=128, integer=True)
    out["max_body"] = ch._positive(cfg, "coordinator_max_body_bytes", 65536, top=ch._MAX_JSON_SIZE, integer=True)
    out["pool_size"] = ch._positive(cfg, "coordinator_client_pool", 4, top=16, integer=True)
    out["idem_entries"] = ch._positive(cfg, "coordinator_idempotency_entries", 32768, top=65536, integer=True)
    root = cfg.get("disk_root")
    if not isinstance(root, str) or not root or not os.path.isabs(root):
        raise ValueError("worker role requires an absolute disk_root")
    limits_kwargs = cfg.get("coordinator_disk_limits") or {}
    if not isinstance(limits_kwargs, dict):
        raise ValueError("coordinator_disk_limits must be a Limits field mapping")
    out["limits"] = Limits(**limits_kwargs)
    out["profile"] = ch._profile_from_config(cfg)
    # RemoteCoordinator's renew-margin bound (coordinator_http.py:784-786).
    ttl = float(out["limits"].lease_seconds)
    margin = cfg.get("coordinator_renew_margin")
    margin = margin if margin is not None else max(1.0, min(30.0, 0.1 * ttl))
    if not 0 < margin < ttl:
        raise ValueError("renew margin must be inside the lease ttl")
    out["renew_margin"] = margin
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", nargs="?", default="-")
    ap.add_argument("--package", required=True, help="path to the persistence/ checkout root")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, args.package)
    install_vllm_stub()
    import recipe_persistence.native as native
    # native._prefix_hash_algorithm() reads the LIVE vLLM cache config; on head
    # there is none, so we assert the pinned value the engine will supply
    # (vllm/config/cache.py:141 default at 83252ea89 is "sha256").
    native._prefix_hash_algorithm = lambda: "sha256"

    blob = sys.stdin.read() if args.json_file == "-" else open(args.json_file).read()
    doc = json.loads(blob)
    for key, want in (("kv_connector", "OffloadingConnector"),
                      ("kv_role", "kv_both"),
                      ("kv_load_failure_policy", "recompute")):
        if doc.get(key) != want:
            raise SystemExit(f"FAIL: {key}={doc.get(key)!r}, expected {want!r}")
    extra = doc["kv_connector_extra_config"]

    if os.environ.get("PYTHONHASHSEED", "") == "":
        raise SystemExit("FAIL: run with PYTHONHASHSEED set (native.py:111-113)")

    spec_cls = getattr(native, extra["spec_name"])
    cfg = _Config(extra_config=extra, parallel=_Parallel(rank=args.rank, world_size=args.world_size))
    spec = spec_cls(cfg)                      # runs native.py:92-155 for real
    fac = check_factory_config(spec.recipe_config, args.world_size)

    print(json.dumps({
        "rank": args.rank,
        "ACCEPTED_BY": ["recipe_persistence.native.NodeLocalDiskOffloadingSpec.__init__",
                        "coordinator_http.parse_endpoint/_positive/_profile_from_config",
                        "storage.Limits"],
        "disk_root": spec.recipe_config["disk_root"],
        "world_size": spec.world_size,
        "offload_prompt_only": spec.offload_prompt_only,
        "hash_seed": spec.hash_seed,
        "hash_algorithm": spec.hash_algorithm,
        "coordinator_factory": f'{spec.factory.__module__}.{spec.factory.__name__}',
        "endpoints": [f"{h}:{p}" for h, p in fac["endpoints"]],
        "bound_endpoint_for_this_rank": "%s:%d" % fac["endpoints"][args.rank],
        "limits": fac["limits"].__dict__,
        "renew_margin": fac["renew_margin"],
        "startup_timeout": fac["startup_timeout"],
        "census_profile_sha256": fac["profile"],
        "max_pending_keys": fac["max_pending_keys"],
        "max_pending_bytes": fac["max_pending_bytes"],
    }, indent=1))


if __name__ == "__main__":
    main()
