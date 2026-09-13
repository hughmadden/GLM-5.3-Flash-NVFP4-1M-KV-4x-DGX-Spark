# configs/build/ — the end-state ARM64 image recipe

The image that carries everything in this release: upstream vLLM at the pin, the
persistence connector series (from its fork branch, by pinned commit), the `recipe_persistence` package, the EXL3
overlay and kernels, the startup observer, the launch files and the chat
template, built as one ARM64 artifact for the four GB10 nodes.

**Source build required; not one-click replication.** Read
[What is NOT verified](#what-is-not-verified) before treating any of this as a
working build.

## Files

| File | What it is |
|---|---|
| `Dockerfile` | The as-built multi-stage build. **Published for review** — see the header note inside it. |
| `build.sh` | Host driver: sanity checks (buildx/binfmt/wheels), stages wheels and the vendor `.so` into the context, computes a content stamp, builds and `--load`s, writes an image receipt and a `tar.zst` + sha256 sidecar. |
| `distribute.sh` | Fleet distribution (`docker save \| zstd \| ssh "docker load"`) with per-node image-**Id** verification. Fleet-gated: it starts nothing, but it does move a 10.6 GB artifact. |
| `constraints.txt` | The pip constraints used by the dependency stage. |
| `README.md` | This file. |

### The published Dockerfile does not build as-copied

`Dockerfile` is reproduced from the private build tree, where the build context
is that tree's `image/` directory: `build.sh` sets `CONTEXT="$HERE/context"` and
every `COPY` is `context/<subtree>/...`. This release publishes the *contents* of
that context under a top-level engineering layout (`configs/`, `patches/`,
`persistence/`, `tests/`). Reconstruct the context as follows, then run
`build.sh` from the directory holding both it and `context/`:

| Context path (what the Dockerfile copies) | Reconstruct from |
|---|---|
| `context/constraints.txt` | `configs/build/constraints.txt` |
| `context/files/` | `configs/files/` |
| `context/launch/` | `configs/launch/` |
| `context/observer/` | `patches/observer/` |
| `context/overlay/` | `patches/overlay/` |
| `context/patches/` | `patches/vllm/` |
| `context/persistence-flash/` | `patches/persistence-flash/` |
| `context/fork-src/` | **Staged by `build.sh`**, not distributed: the eight series files fetched from the vLLM fork at the commit pinned in `patches/persistence-flash/manifest.json`, hash-verified at staging (`apply.py fetch`). |
| `context/persistence-pkg/` | `persistence/`, with `tests/` placed inside it as `tests/` |
| `context/tests/` | `patches/tests/` |
| `context/verify/` | `patches/verify/` |
| `context/ablit/` | `patches/ablit/` — **regenerate the `.pt` blobs first** (not distributed) |
| `context/host/` | `configs/host/` (present in the context but **not consumed by the Dockerfile**) |
| `context/ext/` | **Not distributed.** A prebuilt vendor `exllamav3_ext.cpython-312-aarch64-linux-gnu.so` (~101 MB) plus its provenance note. Provide your own build of the extension, or switch to `EXT_MODE=inimage`. |
| `context/wheels/` | **Not distributed.** Five pinned wheels staged on the build host by `build.sh`. |

Two build inputs therefore have no public substitute in this tree: the prebuilt
vendor extension and the pinned wheel set. Both are named exactly below so you
can resolve them at the pins rather than at a moving head.

## Pins

| Component | Pin |
|---|---|
| vLLM | commit `83252ea899c6538eaa0c1fb31f28a92c661bbffc`; wheel `vllm-0.28.1rc1.dev617+g83252ea89-cp38-abi3-manylinux_2_28_aarch64.whl` |
| torch | `2.13.0+cu130`, cp312, aarch64 |
| FlashInfer | `0.6.18.post1` (python / cubin / jit-cache, cu130) — the nightly metadata pins `post1`, not bare `0.6.18` |
| InstantTensor | `0.1.9` |
| exllamav3 | `0.0.43` @ `c5d9c657966ffeeaa9353f0cc899f18629da4a13` |
| CUDA base image | `nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` |
| Dependency-stage base | `python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea` |
| pip | `26.2.1` |
| Model (EXL3 TR3 4bpw) | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` |

`13.0.3` is chosen to match torch's own `cuda-toolkit==13.0.3` requirement.
Ubuntu 24.04 gives Python 3.12.3 — the cp312 ABI both the torch wheel and the
vendor extension target.

## Build (on an x86_64 build host)

```sh
cd <build-host staging root>
./build.sh        # cross-builds linux/arm64; every RUN step executes under qemu-user
```

`build.sh` expects, by default:

- `$SRV/wheels/` — the five pinned wheels (`WHEEL_SRC` overrides);
- `$SRV/trees/extracted-image/exllamav3_ext.cpython-312-aarch64-linux-gnu.so`
  for `EXT_MODE=prebuilt` (`MIA_SO` overrides);
- binfmt/qemu-aarch64 registered on the host.

It writes `$SRV/receipts/image-<tag>.json` and
`$SRV/images/<tag>.tar.zst` with a sha256 sidecar. `EXT_MODE=inimage` compiles
the extension inside the image instead of staging a prebuilt one; the build is
then materially longer.

## Distribute

```sh
NODES="user@192.0.2.10 user@192.0.2.11 user@192.0.2.12 user@192.0.2.13" \
  ./distribute.sh <tarball-or-tag>
```

The script verifies each node's `docker image inspect --format '{{.Id}}'` against
the pinned `IMAGE_ID` and requires all ranks to agree. **That check matters:**
the image Id does *not* survive `docker save` → `docker load` across different
docker storage backends — the containerd snapshotter store and the classic store
produce different config digests for byte-identical content (the 55 `RootFS.Layers`
digests are unchanged). Pin the **loaded** Id per fleet and re-record it; a stale
`IMAGE_ID` stops the fleet by design. See
[`../../docs/FINDINGS-FIRST-BOOT-20260910.md`](../../docs/FINDINGS-FIRST-BOOT-20260910.md) §1.1.

## What is NOT verified

Stated plainly, carried from the campaign's own build notes:

- The **whole dependency-resolve stage was validated natively** on a non-arm64
  host with `pip install --dry-run --report` against the base image, and the
  `deps` stage was built and run for real there. That validates the resolve, not
  an ARM64 image.
- The **full image build under qemu-aarch64 was not re-run** for this release.
- The x86 build historically needed an extra final-stage fix in the wrapping
  builder repository; the ARM64 path here does not.
- **No public prebuilt image is published, and local image tags are not pullable
  releases.** Treat every tag as local-only.
- Absolute paths in the scripts (`$STAGE`, `$WORK`, `pinned-vllm`,
  `trees/extracted-image`) are sanitization placeholders; set them to your own
  staging root and checkout.
- No import-level runtime check of the built image is claimed here. The in-image
  check is `patches/verify/verify_overlay_runtime.py`, run by the build when
  `RUN_IMPORT_CHECKS=1`.
