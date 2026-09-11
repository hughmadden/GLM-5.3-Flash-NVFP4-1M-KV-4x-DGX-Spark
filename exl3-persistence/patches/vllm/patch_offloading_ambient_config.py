#!/usr/bin/env python3
"""Enter the ambient vLLM config around offloading-spec construction.

WHY (first real engine boot, 2026-09-10 AEST, node0-3):

    EngineCoreProc.__init__
      -> Scheduler.__init__                (vllm/v1/core/sched/scheduler.py:339)
      -> KVConnectorFactory.create_connector
      -> OffloadingConnector.__init__      (offloading_connector.py:66)
      -> OffloadingSpecFactory.create_spec
      -> recipe_persistence.native.NodeLocalDiskOffloadingSpec.__init__
      -> ValueError: persistence requires an active vLLM cache configuration

``recipe_persistence`` derives the prefix-cache hash algorithm from the ambient
vLLM config, because ``OffloadingConfig`` (vLLM a9531edfa6, #48150) no longer
carries ``cache_config``.  ``OffloadingConnectorScheduler`` -- and therefore
``spec.get_manager()``, the other reader of that algorithm -- is built in the
same constructor.  The scheduler path runs inside ``EngineCoreProc.__init__``,
which has no ``set_current_vllm_config`` context (``vllm/v1/engine/core.py``
never calls it; only the worker and model-runner paths do), so the ambient
config is legitimately absent and the spec fails closed.

The fix is to establish the ambient config for exactly the scope that needs it:
spec construction plus the scheduler/worker-side object built from that spec.
This is the contract the out-of-tree spec API already assumes -- the worker
path has it -- and it restores it for the scheduler path.

Fail-closed and idempotent: both anchors must appear exactly once, the file must
hash to BEFORE, and the result must hash to AFTER.  Anything else refuses.

Usage: patch_offloading_ambient_config.py <vllm-source-root>
       (the directory that contains ``vllm/``, i.e. the image's dist-packages)
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

REL = "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py"

BEFORE = "6fb06a0cf74c01ec2446c8d0bbd5630d3aec25416aceed5adba0261940b41b25"
AFTER = "c7a8611e7e9198a6a280df60bce152162e42f80f17d0293a80388fbd4798957c"

IMPORT_OLD = "from vllm.config import VllmConfig\n"
IMPORT_NEW = "from vllm.config import VllmConfig, set_current_vllm_config\n"

BLOCK_OLD = """        offloading_config = build_offloading_config(vllm_config, kv_cache_config)
        self._canonical_layout = offloading_config.canonical_layout
        spec = OffloadingSpecFactory.create_spec(offloading_config)

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(
                spec, vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(
                spec, vllm_config, kv_cache_config
            )
"""

BLOCK_NEW = """        offloading_config = build_offloading_config(vllm_config, kv_cache_config)
        self._canonical_layout = offloading_config.canonical_layout
        # An out-of-tree offloading spec may read the ambient vLLM config both in
        # its constructor and in get_manager(): recipe_persistence.native reaches
        # the prefix-cache hash algorithm that way, because OffloadingConfig no
        # longer carries cache_config. The scheduler path builds this connector
        # from Scheduler.__init__ inside EngineCoreProc.__init__, which runs
        # before any set_current_vllm_config context exists, so establish it
        # here for the spec and the scheduler-side manager built from it.
        with set_current_vllm_config(vllm_config):
            spec = OffloadingSpecFactory.create_spec(offloading_config)

            self.connector_scheduler: OffloadingConnectorScheduler | None = None
            self.connector_worker: OffloadingConnectorWorker | None = None
            if role == KVConnectorRole.SCHEDULER:
                self.connector_scheduler = OffloadingConnectorScheduler(
                    spec, vllm_config, kv_cache_config
                )
            elif role == KVConnectorRole.WORKER:
                self.connector_worker = OffloadingConnectorWorker(
                    spec, vllm_config, kv_cache_config
                )
"""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm-source-root>", file=sys.stderr)
        return 2
    root = pathlib.Path(sys.argv[1])
    # Accept both layouts, exactly like persistence-flash/apply.py: a source root
    # that contains ``vllm/``, or the ``vllm`` package directory itself (the bare
    # overlay layout the runtime image ships).
    candidates = (root / REL, root / REL.split("/", 1)[1])
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        print(
            f"REFUSED: neither {candidates[0]} nor {candidates[1]} is a file",
            file=sys.stderr,
        )
        return 3
    rel = str(path.relative_to(root))

    text = path.read_text()
    digest = sha256(text)

    if digest == AFTER:
        print(f"already applied: {rel} == AFTER")
        return 0
    if digest != BEFORE:
        print(
            f"REFUSED: {rel} sha256 {digest} is neither BEFORE {BEFORE} "
            f"nor AFTER {AFTER}",
            file=sys.stderr,
        )
        return 4

    for name, anchor in (("import", IMPORT_OLD), ("body", BLOCK_OLD)):
        count = text.count(anchor)
        if count != 1:
            print(
                f"REFUSED: {name} anchor occurs {count} times in {rel}, want 1",
                file=sys.stderr,
            )
            return 5

    patched = text.replace(IMPORT_OLD, IMPORT_NEW, 1).replace(BLOCK_OLD, BLOCK_NEW, 1)

    result = sha256(patched)
    if result != AFTER:
        print(
            f"REFUSED: patched {rel} sha256 {result} != expected AFTER {AFTER}",
            file=sys.stderr,
        )
        return 6

    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {rel}: {BEFORE} -> {AFTER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
