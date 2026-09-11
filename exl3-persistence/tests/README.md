# tests/ — engine-free test harnesses

Everything here runs **without an engine, without a GPU and without network
access to the deployment**. `recipe_persistence` imports neither torch nor vLLM
at module load; the CUDA-dependent path is only reached by explicitly allocating
native handlers.

## Package unit suite

The six `test_*.py` files at this level are the package's own suite. Three of them
(`test_coordinator.py`, `test_geometry.py`, `test_storage.py`) run under the
standard library's `unittest`; `test_native.py`, `test_coordinator_http.py` and
`test_http_rpc.py` use pytest fixtures. The package's native fixtures use fake
canonical/native objects, not CUDA.

```sh
# package at ../persistence, so put it on the path:
PYTHONPATH=persistence python3 -m pytest tests -q
```

Measured on the campaign hardware: the pytest-free subset (99 tests) runs green in
~210 s inside the containerised path.

## Concurrency stub harness — `conc/`

A deliberately dependency-free runner: **`pytest` is not used**, because in the
target image it is absent and installing it costs >30 s. The runner enforces a
hard **30 s load budget** per module and runs every test under a watchdog, so a
deadlock reports as `HANG` instead of stalling the run. It emits one JSON object
per test.

| Piece | What it is |
|---|---|
| `conc/conc_runner.py` | The runner. Stdlib only. |
| `conc/conc_stubs.py` | Instant dummy data: `dummy_limits`, `dummy_root`, `dummy_store`, `dummy_namespace` (64-hex), `dummy_blob`, `dummy_infos`, `dummy_geometry`, `dummy_token`, `run_threads` (barrier-synchronised, so races are real), `hammer`, `Stopwatch`, `summarize`. |
| `conc/test_conc_*.py` | The suites: storage, coordinator HTTP, RPC + geometry, native handlers, plus the central reduction (`test_conc_reduced`), perf and smoke files. |

```sh
cd tests/conc
PYTHONPATH=../../persistence python3 conc_runner.py --timeout 60 \
  test_conc_reduced test_conc_storage test_conc_native_handlers \
  test_conc_coordinator_http test_conc_rpc_geometry test_conc_perf test_conc_smoke
```

Add `--json` for machine-readable output, `--filter <substr>` to run one test, and
`--load-report` to re-assert the 30 s load budget. Load cost measured: **0.07 s**
for all seven package modules, 0.70 s including `docker run`.

Recorded result on the campaign hardware: **184 tests, 167 pass / 17 fail / 0
hang** after the eight fixes (up from 164/20). Of the 17, 15 are genuine open
defects and 2 are stale pins of a harness-stub bug that was fixed — see
[`../docs/CONCURRENCY-TESTS-20260910.md`](../docs/CONCURRENCY-TESTS-20260910.md).

## `validate_kv_transfer_config.py`

A torch-free validator that exercises the package's **own** code: the real
`NodeLocalDiskOffloadingSpec.__init__` via `native._load_native()` (with the
pinned `vllm.v1.kv_offload.base` names stubbed), plus the real `parse_endpoint`,
`_positive`, `_profile_from_config` and `Limits`. Feed it the rendered
`--kv-transfer-config` JSON from the launcher:

```sh
python3 tests/validate_kv_transfer_config.py <rendered-args.json>
```

A pre-fix control that used the dead keys `coordinator_url` and `disk_quota_bytes`
is **REJECTED** (`coordinator_endpoints must list one endpoint per rank`) — that
rejection is what proves the validator is load-bearing.

## What this harness cannot prove

No CUDA, no real engine, no real KV geometry, no census handshake, no real
filesystem faults (EIO/ENOSPC are inferred, not observed), no true multi-process
ownership (`flock` gives one owner per root by design), and no real multi-node
fabric (`--network=none` and dummy data only). Absolute performance numbers from
the concurrency suites are ±1.5× because the four testers shared the nodes.
