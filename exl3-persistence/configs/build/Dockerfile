# syntax=docker/dockerfile:1.7
#
# PUBLISHED FOR REVIEW. This file is reproduced from the private build tree: its
# COPY paths are relative to that tree's `image/` build context (`context/...`),
# not to this release's top-level layout. The context is reconstructed from the
# release by the mapping table in `configs/build/README.md`. Two context
# subtrees referenced below are NOT distributed -- `context/ext/` (a prebuilt
# vendor .so) and `context/wheels/` (pinned wheels staged on the build host) --
# so this file does not build unmodified.
#
# GLM-5.3-Flash EXL3 4bpw, TP4 across four DGX Spark GB10 (sm_121a), with
# rank-local disk KV persistence active from first boot.
#
# End-state ARM64 image. Three lineages are combined here, in this order,
# because the order is load-bearing:
#
#   1. Upstream vLLM @ 83252ea899c6538eaa0c1fb31f28a92c661bbffc (nightly wheel).
#   2. Our persistence patch series -- applied FIRST, on a pristine tree,
#      because its apply.py refuses on any BEFORE-hash mismatch and two of the
#      overlay patches (patch_scheduler_decode_floor, patch_adaptive_k) edit
#      v1/core/sched/scheduler.py, which is also one of its seven files.
#   3. Mia's SM121 NoPE sparse-MLA + EXL3 overlay, re-anchored onto that pin
#      (notes/OVERLAY-PORT-NOTES.md: 15/15 scripts apply; heredoc 14/17 anchors
#      intact, 1 dropped as upstreamed, 2 re-anchored).
#   4. The startup observer call sites, last, so they sit on the final source.
#
# NOTHING IN THIS FILE HAS BEEN BUILT YET. Every runtime claim is UNVERIFIED
# until build.sh runs on build-host. See README.md §"What is UNVERIFIED".
#
# Build (on build-host, x86_64, cross-building linux/arm64 under qemu):
#   ./build.sh
#
# ---------------------------------------------------------------------------
# STAGE MAP -- why this file is multi-stage (see README §"Native dependency
# stage"). The previous single-stage form is kept verbatim as
# Dockerfile.pre-native for diffing.
#
#   osbase          linux/arm64, qemu   OS + Python 3.12. Shared parent.
#   sdist-aarch64   linux/arm64, qemu   Builds the ONE dependency with no
#                                       aarch64 wheel (instanttensor) into a
#                                       wheel. Cached; ~5 min, paid once.
#   deps            $BUILDPLATFORM      NATIVE pip resolve of all 198 packages
#                   (x86_64 on build-host)   with aarch64 platform tags into
#                                       --target /out/dist-packages. No qemu.
#   <final>         linux/arm64, qemu   FROM osbase; COPYs the resolved tree in
#                                       and continues exactly as before.
#
# The two pip layers this replaces measured 2,501.6 s + 1,805.1 s = 71.8 min
# (build-host:$STAGE/receipts/build-20260910-123921.log, steps #12/#13).
#
# HONEST ACCOUNTING, because the obvious reading of that number is wrong: most
# of it was NOT emulation. Measured on head 2026-09-10, the same
# nvidia_cudnn_cu13 wheel downloads at 1.19 MB/s natively; the emulated build
# pulled it at 1.04 MB/s. Four parallel connections aggregate to 3.2 MB/s, so
# the ~1.2 MB/s ceiling is per-connection throttling on the path to
# files.pythonhosted.org, and pip uses one connection. Roughly 3,100 s of the
# 4,307 s was that download of ~3.4 GB and it does NOT get faster by going
# native. What emulation actually cost, and what this restructure recovers:
#
#   ~744 s   unpack + byte-compile of 198 packages     -> ~60 s native
#   ~128 s   metadata collection / resolution           -> ~25 s native
#    281 s   compiling the instanttensor sdist          -> own cached stage
#
# So expect the cold-cache deps chain to land around 48-53 min instead of
# 71.8 min (~20 min saved, ~28%), and the instanttensor 281 s to be paid once
# rather than on every miss of the big pip layer. The large win is on REBUILDS:
# the deps stage now carries a BuildKit pip cache mount, so a pin change that
# re-resolves against an already-populated cache skips the download entirely
# and the whole stage runs in a couple of minutes. That mount is only safe
# because the stage is native, isolated, and installs to --target -- it never
# touches the shipped image.
#
# The native resolve was validated on head 2026-09-10 to produce an identical
# package set to the emulated build: 198 packages, 0 name differences,
# 0 version differences, 0 sdists, 0 non-aarch64/any wheels. Evidence:
# build-receipts/native-deps-resolved-aarch64.txt. The stage mechanics (platform
# ladder, aarch64/any gate, dist-info freeze, bin/ move, entry-point
# regeneration) were additionally built and run for real on head against a
# 20-package subset -- aarch64 ELF confirmed by e_machine=0xb7.
# ---------------------------------------------------------------------------

# Base: nvidia/cuda 13.0.3 matches torch 2.13.0's own cuda-toolkit==13.0.3 pin
# exactly (torch METADATA, receipts/build-host/wheels/MANIFEST.md). Ubuntu 24.04
# gives Python 3.12.3 -- the same interpreter node0 runs, and the cp312 ABI the
# torch wheel and the prebuilt exllamav3_ext both target.
#
# devel (not runtime) for BOTH EXT_MODE values: inimage needs nvcc, and prebuilt
# keeps it only so the FlashInfer JIT fallback and any on-host kernel debugging
# still work. Cost: ~4 GB of image over the runtime flavour. Accepted for
# uniformity -- one image, one digest, four nodes (tony hard-won rule #3).
ARG BASE_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

# Resolver host for the native deps stage. It NEVER runs any of the wheels it
# downloads -- it only resolves and unpacks them -- so its libc and CPU are
# irrelevant. Its Python MINOR must match the runtime's, because the resolve is
# driven by --python-version 3.12/--abi cp312 and because the .pyc it emits are
# carried into the image. python:3.12-slim is 3.12.14; Ubuntu 24.04 is 3.12.3;
# both report pyc MAGIC_NUMBER cb0d0d0a (checked on head 2026-09-10), and the
# final stage re-checks it and purges bytecode if it ever diverges.
ARG DEPS_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# ===========================================================================
# STAGE osbase -- OS + Python 3.12. linux/arm64, emulated. Shared by the
# sdist-aarch64 stage and the final stage so the apt layer is paid once.
# ===========================================================================
FROM ${BASE_IMAGE} AS osbase

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=8 \
    GLM53_SITE=/usr/local/lib/python3.12/dist-packages \
    GLM53_OPT=/opt/glm53/overlay

# Every overlay script hardcodes /usr/local/lib/python3.12/dist-packages
# (14 of 17 do). GLM53_SITE = site-packages for our own patches/verify scripts;
# patch_dense_fp8 alone reads GLM53_SITE as the *vllm package dir* and the overlay
# exl3.py from GLM53_OPT, so it gets an inline override at its call site. On Ubuntu 24.04 that
# is exactly where a root `pip install --break-system-packages` lands, so the
# hardcoded paths are correct here without editing a single script. It is also
# exactly where the native stage's --target tree is COPYed.
ENV SITE=/usr/local/lib/python3.12/dist-packages

# ---------------------------------------------------------------------------
# Layer 1: OS + Python 3.12
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
        ca-certificates curl git patch zstd \
        libnuma1 libibverbs1 librdmacm1 ibverbs-providers rdma-core \
        numactl kmod pciutils procps; \
    rm -rf /var/lib/apt/lists/*; \
    python3 --version; \
    test "$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = "3.12"

# ===========================================================================
# STAGE sdist-aarch64 -- the ONE package with no aarch64 wheel.
#
# instanttensor (vLLM's `instanttensor>=0.1.9`) publishes cp310-cp314
# manylinux_2_24/2_28 **x86_64** wheels plus an sdist, and nothing else
# (PyPI JSON, checked on head 2026-09-10). aarch64 therefore has to compile it.
# In the pre-native build that compile sat inside the big emulated pip layer:
# 988.7 s -> 1269.4 s of step #13, i.e. ~281 s, and it was re-paid on every
# cache miss of that whole layer.
#
# Here it is its own stage, so:
#   * it is the ONLY thing still built under qemu in the dependency path;
#   * its cache key is osbase + INSTANTTENSOR_VERSION, so it is built once and
#     then reused across every other context change.
#
# Build isolation is left ON, exactly as pip had it before: the pre-native log
# shows its build deps installing in 16 s, i.e. it does not need torch to
# build, and giving it an isolated env keeps it from seeing (or moving) the
# pinned runtime set.
# ===========================================================================
FROM osbase AS sdist-aarch64

# Sourced: the version the pre-native build actually resolved and compiled
# ("Created wheel for instanttensor: filename=instanttensor-0.1.9-cp312-cp312-
# linux_aarch64.whl", build-20260910-123921.log:1371-1373).
ARG INSTANTTENSOR_VERSION=0.1.9

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    mkdir -p /opt/glm53/wheels-built; \
    pip wheel --no-deps \
        --wheel-dir /opt/glm53/wheels-built \
        "instanttensor==${INSTANTTENSOR_VERSION}"; \
    ls -la /opt/glm53/wheels-built; \
    ls /opt/glm53/wheels-built/instanttensor-${INSTANTTENSOR_VERSION}-*aarch64.whl

# ===========================================================================
# STAGE deps -- NATIVE dependency resolution. Runs on the BUILD platform
# (x86_64 on build-host), never under qemu.
#
# pip is told to resolve for the TARGET interpreter and platform instead of its
# own: --implementation cp --python-version 3.12 --abi cp312/abi3/none and an
# explicit aarch64 manylinux ladder. Those flags require --only-binary=:all:
# and --target, which is precisely the shape we want: nothing is built here,
# every artefact is a prebuilt aarch64 (or pure-python) wheel, and the result
# is a plain directory tree that the arm64 stage can COPY into dist-packages.
#
# THE LADDER IS ENUMERATED, NOT ABBREVIATED. pip 26 does NOT expand
# `--platform manylinux_2_39_aarch64` downward to older glibc minors (verified
# on head with TargetPython.get_sorted_tags(): the tag list contained exactly the
# platforms passed). Passing only manylinux_2_28 loses five wheels that the
# emulated build accepted -- llguidance (manylinux_2_31_aarch64),
# PyNvVideoCodec + tilelang (manylinux_2_34_aarch64) and two more -- and
# passing only manylinux2014 silently downgrades packages such as cbor2
# (6.1.4 -> 5.9.0). Ubuntu 24.04 is glibc 2.39, so the ladder runs 2_39 -> 2_17
# newest-first (= preference order), then the manylinux2014 alias, then plain
# linux_aarch64 for the locally-built instanttensor wheel.
#
# Verified on head 2026-09-10 against the pre-native emulated build:
#   198 packages resolved, 198 in the emulated build, 0 name differences,
#   0 version differences, 0 sdists, 0 wheels that are not aarch64 or `any`.
#   Full list: build-receipts/native-deps-resolved-aarch64.txt.
# ===========================================================================
FROM --platform=$BUILDPLATFORM ${DEPS_IMAGE} AS deps

# The resolver's own pip. Pinned because the resolution is its output.
# 26.2.1 is the version the head validation ran.
ARG PIP_VERSION=26.2.1
ARG INSTANTTENSOR_VERSION=0.1.9
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN set -eux; \
    test "$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = "3.12"; \
    python3 -m pip install --no-cache-dir "pip==${PIP_VERSION}"; \
    python3 -m pip --version

# ---------------------------------------------------------------------------
# The pinned wheel set (offline), the constraint file, and the one wheel the
# sdist-aarch64 stage had to compile.
#
#   flashinfer *0.6.18.post1* -- NOT bare 0.6.18. The vLLM nightly wheel's own
#   METADATA declares Requires-Dist: flashinfer-python==0.6.18.post1
#   (receipts/build-host/wheels/MANIFEST.md flags the bare-0.6.18 mismatch loudly).
#   All three post1 files were re-verified byte-complete against the upstream
#   GitHub release content-length on 2026-09-10.
# ---------------------------------------------------------------------------
COPY context/constraints.txt /opt/glm53/constraints.txt
COPY context/wheels/ /opt/wheels/
COPY --from=sdist-aarch64 /opt/glm53/wheels-built/ /opt/wheels-built/

RUN set -eux; \
    for w in \
        vllm-0.28.1rc1.dev617+g83252ea89-cp38-abi3-manylinux_2_28_aarch64.whl \
        torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
        flashinfer_python-0.6.18.post1-py3-none-any.whl \
        flashinfer_cubin-0.6.18.post1-py3-none-any.whl \
        flashinfer_jit_cache-0.6.18.post1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl \
    ; do test -f "/opt/wheels/$w" || { echo "MISSING WHEEL: $w"; ls -la /opt/wheels; exit 1; }; done; \
    ls /opt/wheels-built/instanttensor-*aarch64.whl

# ---------------------------------------------------------------------------
# The resolve. ONE pip invocation, not two. OFFLINE: every transitive wheel was
# prefetched to /opt/wheels/deps on build-host with 16 parallel connections (PyPI
# throttles ~1 MB/s per connection, which was ~50 of the 72 build minutes).
# --no-index makes a missing wheel fail loudly instead of silently downloading.
#
# The pre-native file installed torch first, alone, with the comment "Pulling
# it in the same resolve as vLLM lets pip briefly consider a different torch".
# That mitigation does not carry over and is not needed here: `pip --target`
# sets ignore_installed=True, so a second --target call would re-resolve the
# whole graph from scratch anyway (it cannot see the first call's output), and
# the thing that actually pins torch is constraints.txt (`torch==2.13.0`) plus
# handing pip the exact local wheel by path. What replaces the ordering is a
# DIRECT assertion in the next layer: torch must come out as 2.13.0+cu130 or
# the build fails. The head validation confirms the single resolve lands on the
# identical 198-package set the two-step emulated build produced.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    PLAT=""; \
    for m in $(seq 39 -1 17); do PLAT="${PLAT} --platform manylinux_2_${m}_aarch64"; done; \
    PLAT="${PLAT} --platform manylinux2014_aarch64 --platform linux_aarch64"; \
    echo "platform ladder:${PLAT}"; \
    python3 -m pip install \
        --target /out/dist-packages \
        --only-binary=:all: \
        ${PLAT} \
        --implementation cp \
        --python-version 3.12 \
        --abi cp312 --abi abi3 --abi none \
        --no-index \
        --find-links /opt/wheels \
        --find-links /opt/wheels/deps \
        --find-links /opt/wheels-built \
        --constraint /opt/glm53/constraints.txt \
        /opt/wheels/torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
        /opt/wheels/flashinfer_python-0.6.18.post1-py3-none-any.whl \
        /opt/wheels/flashinfer_cubin-0.6.18.post1-py3-none-any.whl \
        "/opt/wheels/flashinfer_jit_cache-0.6.18.post1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl" \
        "/opt/wheels/vllm-0.28.1rc1.dev617+g83252ea89-cp38-abi3-manylinux_2_28_aarch64.whl" \
        "instanttensor==${INSTANTTENSOR_VERSION}"

# ---------------------------------------------------------------------------
# Gate the resolve, then produce the receipts, then get `bin/` out of the way.
#
#   * every installed distribution must be a wheel whose WHEEL Tag platforms
#     are all `any` or aarch64 -- i.e. nothing x86 slipped through and nothing
#     was built from source here;
#   * torch must be exactly 2.13.0+cu130 and vLLM exactly the pinned nightly;
#   * pip-freeze.txt is regenerated from *.dist-info. It is a DIST-INFO
#     LISTING, not `pip freeze` output: same name==version lines, but without
#     the `pkg @ file:///opt/wheels/...` direct-URL forms pip freeze emits for
#     by-path installs, and produced natively instead of under emulation.
#   * pip --target puts console scripts in <target>/bin with THIS stage's
#     interpreter in the shebang. That directory must not land in
#     dist-packages, so it is moved aside for the arm64 stage to regenerate.
#     (Side effect, recorded not hidden: each dist-info RECORD still lists its
#     `bin/<script>` entries, which are now dangling. Nothing reads RECORD in
#     this image; `pip uninstall` would warn.)
# ---------------------------------------------------------------------------
RUN set -eux; \
    mkdir -p /out/receipts; \
    python3 - <<'PY'
import email.parser, pathlib, sys

target = pathlib.Path("/out/dist-packages")
rec = pathlib.Path("/out/receipts")
parse = email.parser.Parser().parsestr

rows, bad = [], []
for di in sorted(target.glob("*.dist-info")):
    md = parse((di / "METADATA").read_text(encoding="utf-8", errors="replace"))
    name, ver = md["Name"], md["Version"]
    rows.append((name, ver))
    wheel = di / "WHEEL"
    if not wheel.is_file():
        bad.append((name, ver, "no WHEEL file -- not installed from a wheel"))
        continue
    tags = parse(wheel.read_text(encoding="utf-8", errors="replace")).get_all("Tag") or []
    plats = {p for t in tags for p in t.rsplit("-", 1)[-1].split(".")}
    if not plats:
        bad.append((name, ver, "WHEEL has no Tag"))
    # "sbsa" (Server Base System Architecture) is NVIDIA's arm64-server tag,
    # e.g. nvidia-cusparselt-cu13's WHEEL says manylinux2014_sbsa: aarch64 by
    # another name, and exactly what a GB10 wants.
    elif not all(p == "any" or "aarch64" in p or "sbsa" in p for p in sorted(plats)):
        bad.append((name, ver, sorted(plats)))

if bad:
    for b in bad:
        print("NOT-AARCH64-OR-ANY:", b, file=sys.stderr)
    raise SystemExit("native deps stage resolved something that is not an aarch64/any wheel")

byname = {n.lower().replace("_", "-"): v for n, v in rows}
assert byname.get("torch") == "2.13.0+cu130", ("torch", byname.get("torch"))
assert byname.get("vllm") == "0.28.1rc1.dev617+g83252ea89", ("vllm", byname.get("vllm"))
assert byname.get("flashinfer-python") == "0.6.18.post1", ("flashinfer-python", byname.get("flashinfer-python"))

(rec / "pip-freeze.txt").write_text(
    "\n".join(sorted((f"{n}=={v}" for n, v in rows), key=str.lower)) + "\n"
)
(rec / "native-deps.txt").write_text(
    "resolver      = native (BUILDPLATFORM), pip --target with aarch64 platform tags\n"
    "listing_kind  = dist-info listing (name==version), NOT `pip freeze` output\n"
    f"packages      = {len(rows)}\n"
    "sdists_built  = instanttensor (aarch64 wheel built in the sdist-aarch64 stage)\n"
)
print(f"native deps stage OK: {len(rows)} distributions, all aarch64/any wheels")
PY

RUN set -eux; \
    python3 -c "import importlib.util, binascii; open('/out/receipts/pyc-magic.txt','w').write(binascii.hexlify(importlib.util.MAGIC_NUMBER).decode())"; \
    if [ -d /out/dist-packages/bin ]; then mv /out/dist-packages/bin /out/bin; else mkdir -p /out/bin; fi; \
    test ! -e /out/dist-packages/bin; \
    ls /out/bin | head -50; \
    cat /out/receipts/native-deps.txt

# ===========================================================================
# FINAL STAGE -- linux/arm64, emulated. Everything from here down is the
# pre-native file unchanged, apart from the deps landing layer that replaces
# the two emulated pip layers.
# ===========================================================================
FROM osbase

# --- pins, all sourced -----------------------------------------------------
# ARGs must be re-declared inside the stage that uses them. Declared here, at
# the top, deliberately: that is where the pre-native single-stage file had
# them, so their layer-cache blast radius (an ARG is an implicit env var for
# every later RUN) is unchanged.
#
# vLLM commit: the wheel's own version string; matches port/vllm-83252ea89.COMMIT
# and persistence-flash/manifest.json "upstream_commit". The persistence series
# itself is the fork branch pinned in that manifest ("fork.commit"); build.sh
# stages its eight files into context/fork-src/ before the build.
ARG VLLM_COMMIT=83252ea899c6538eaa0c1fb31f28a92c661bbffc
# exllamav3: Mia's Dockerfile ARG EXLLAMAV3_COMMIT, unchanged since her day-0 pin
# (notes/upstream-repo.md §2). Resolves to exllamav3 0.0.43 -- the same version
# present in her built image (build-host:$STAGE/receipts/mia-image-versions.txt).
ARG EXLLAMAV3_COMMIT=c5d9c657966ffeeaa9353f0cc899f18629da4a13
# prebuilt | inimage -- see README §"The EXT_MODE switch".
ARG EXT_MODE=prebuilt
# Import-level verification (imports torch + vllm). Minutes-slow under qemu.
ARG RUN_IMPORT_CHECKS=1
# Stamp, baked by build.sh so a context change forces one rebuild.

# ---------------------------------------------------------------------------
# Layer 2: land the natively-resolved dependency tree.
#
# This replaces the two emulated pip layers (71.8 min of qemu) with a COPY.
# `pip --target` produces exactly the dist-packages layout -- packages flat at
# the root, one *.dist-info per distribution -- so nothing is rewritten here.
#
# The .pyc were byte-compiled natively. CPython bytecode is
# architecture-independent and both interpreters report MAGIC_NUMBER cb0d0d0a
# (3.12.14 in the deps image, 3.12.3 here, checked on head 2026-09-10), but the
# check below is cheap and self-healing: on any future divergence the stale
# bytecode is purged rather than silently mis-loaded.
# ---------------------------------------------------------------------------
COPY --from=deps /out/dist-packages/ /usr/local/lib/python3.12/dist-packages/
COPY --from=deps /out/bin/ /opt/glm53/deps-bin/
COPY --from=deps /out/receipts/ /opt/receipts/

RUN set -eux; \
    test -f "${SITE}/vllm/v1/core/kv_cache_utils.py"; \
    test ! -e "${SITE}/bin"; \
    M_BUILD="$(cat /opt/receipts/pyc-magic.txt)"; \
    M_RUN="$(python3 -c 'import importlib.util,binascii;print(binascii.hexlify(importlib.util.MAGIC_NUMBER).decode())')"; \
    echo "pyc magic: deps-stage=${M_BUILD} runtime=${M_RUN}"; \
    if [ "${M_BUILD}" != "${M_RUN}" ]; then \
        echo "pyc magic mismatch -- purging natively-compiled bytecode"; \
        find "${SITE}" -type d -name __pycache__ -prune -exec rm -rf {} +; \
    fi; \
    python3 -c "import torch; print('torch', torch.__version__); assert torch.__version__.startswith('2.13.0')"

# ---------------------------------------------------------------------------
# Console-script entry points.
#
# `pip --target` wrote its scripts into <target>/bin carrying the NATIVE
# stage's interpreter path (#!/usr/local/bin/python) in the shebang, so they
# are not usable here and were moved aside. They are regenerated from each
# distribution's own entry_points.txt with this image's interpreter -- the same
# text pip itself would have emitted. This is generic on purpose: the launcher
# only needs `vllm` (context/launch/start-tp4.sh:248 `exec vllm serve ...`,
# entry point `vllm = vllm.entrypoints.cli.main:main`), but torch ships
# `torchrun`/`torchfrtrace`, flashinfer ships `flashinfer`, huggingface_hub
# ships `hf`, and a hand-written subset would silently lose the rest.
#
# Anything the deps stage put in bin/ that is NOT a console script (a packaged
# data script) is carried over with its shebang rewritten.
# ---------------------------------------------------------------------------
RUN set -eux; \
    python3 - <<'PY'
import configparser, os, pathlib

site = pathlib.Path(os.environ["SITE"])
bindir = pathlib.Path("/usr/local/bin")
staged = pathlib.Path("/opt/glm53/deps-bin")
bindir.mkdir(parents=True, exist_ok=True)

SHEBANG = "#!/usr/bin/python3"
TEMPLATE = """{shebang}
# -*- coding: utf-8 -*-
# Regenerated at build time from {dist}/entry_points.txt ({spec}).
# The native deps stage's copy of this script carried that stage's interpreter
# path in its shebang; this is the arm64 equivalent.
import re
import sys
from {module} import {first}
if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\\.pyw|\\.exe)?$", "", sys.argv[0])
    sys.exit({call}())
"""

made = {}
for di in sorted(site.glob("*.dist-info")):
    ep = di / "entry_points.txt"
    if not ep.is_file():
        continue
    cp = configparser.ConfigParser(delimiters=("=",))
    cp.optionxform = str
    cp.read_string(ep.read_text(encoding="utf-8", errors="replace"))
    if not cp.has_section("console_scripts"):
        continue
    for script, spec in cp.items("console_scripts"):
        spec = spec.strip()
        module, _, attr = spec.partition(":")
        attr = attr.split("[")[0].strip() or "main"
        module = module.strip()
        path = bindir / script
        path.write_text(TEMPLATE.format(
            shebang=SHEBANG, dist=di.name, spec=spec,
            module=module, first=attr.split(".")[0], call=attr))
        path.chmod(0o755)
        made[script] = f"{script} = {spec}   [{di.name}]"

carried = []
if staged.is_dir():
    for f in sorted(staged.iterdir()):
        if not f.is_file() or f.name in made:
            continue
        data = f.read_bytes()
        if data.startswith(b"#!"):
            data = SHEBANG.encode() + data[data.find(b"\n"):]
        dst = bindir / f.name
        dst.write_bytes(data)
        dst.chmod(0o755)
        carried.append(f.name)

assert (bindir / "vllm").is_file(), "no `vllm` console script was generated"
for required in ("vllm", "torchrun", "flashinfer"):
    assert (bindir / required).is_file(), f"missing console script: {required}"

pathlib.Path("/opt/receipts/console-scripts.txt").write_text(
    "# regenerated from dist-info entry_points.txt in the arm64 stage\n"
    + "\n".join(sorted(made.values()))
    + "\n\n# carried over from the deps stage's bin/ with a rewritten shebang\n"
    + "\n".join(carried or ["(none)"])
    + "\n"
)
print(f"console scripts: {len(made)} regenerated, {len(carried)} carried over")
PY

RUN set -eux; \
    rm -rf /opt/glm53/deps-bin; \
    head -1 /usr/local/bin/vllm; \
    command -v vllm; \
    command -v torchrun; \
    cat /opt/receipts/console-scripts.txt

# Receipt: exactly what the native stage resolved, before anything is patched.
# pip-freeze.txt itself was written in the deps stage (a dist-info listing --
# see README §"Native dependency stage").
RUN set -eux; \
    mkdir -p /opt/receipts; \
    test -s /opt/receipts/pip-freeze.txt; \
    wc -l /opt/receipts/pip-freeze.txt; \
    python3 -c "import json,sys;print(json.dumps({'python':sys.version}))" > /opt/receipts/python.json
# ---------------------------------------------------------------------------
# Layer 3: PERSISTENCE -- FIRST, on the pristine tree.
#
# The series is not carried as diffs: apply.py overlays the eight files from
# context/fork-src/ (staged by build.sh from the fork at manifest.json's pinned
# commit, hash-verified at staging) onto the wheel's tree. It hash-pins the tree
# against manifest.json's BEFORE sha256 and every staged file against AFTER, and
# refuses on any mismatch, so it must see upstream 83252ea89 untouched. It
# re-checks BEFORE hashes immediately before writing (concurrent-modification
# guard) and rolls every file back on failure. No network is used here.
#
# NOTE, UNVERIFIED: those BEFORE hashes were computed from the *git* tree
# (pinned-vllm). This layer is the first thing that proves the nightly *wheel*
# ships byte-identical .py sources. If it refuses, that is the finding -- do not
# force it; re-derive the manifest against the wheel.
# ---------------------------------------------------------------------------
COPY context/persistence-flash/ /opt/glm53/persistence-flash/
COPY context/fork-src/ /opt/glm53/fork-src/
RUN set -eux; \
    test -f /opt/glm53/fork-src/FORK.json; \
    python3 /opt/glm53/persistence-flash/apply.py check  "${SITE}/vllm" --from /opt/glm53/fork-src; \
    python3 /opt/glm53/persistence-flash/apply.py apply  "${SITE}/vllm" --from /opt/glm53/fork-src; \
    python3 /opt/glm53/persistence-flash/apply.py verify "${SITE}/vllm"; \
    cp /opt/glm53/fork-src/FORK.json /opt/receipts/persistence-fork.json

# The connector implementation itself. --no-deps: it deliberately imports
# neither torch nor vLLM at import time, and pulling its build deps would risk
# moving a pinned runtime package.
COPY context/persistence-pkg/ /opt/glm53/persistence-pkg/
RUN set -eux; \
    pip install --break-system-packages --no-cache-dir --no-deps /opt/glm53/persistence-pkg; \
    test -f "${SITE}/recipe_persistence/native.py"; \
    python3 -c "import recipe_persistence; print('recipe_persistence OK', recipe_persistence.__file__)"

# ---------------------------------------------------------------------------
# Persistence integration fix — ambient vLLM config for out-of-tree specs.
#
# The first real engine boot (2026-09-10 AEST, node0-3) died in
# Scheduler.__init__ -> OffloadingConnector.__init__ -> spec construction with
#   ValueError: persistence requires an active vLLM cache configuration
# recipe_persistence.native derives the prefix-cache hash algorithm from the
# ambient vLLM config (OffloadingConfig carries no cache_config since
# a9531edfa6 / #48150), and vLLM's scheduler path builds the spec inside
# EngineCoreProc.__init__, which has no set_current_vllm_config context -- only
# the worker/model-runner paths establish one. Wrap spec construction and the
# scheduler/worker-side object built from it in that context.
#
# Anchored, hash-pinned BEFORE/AFTER and fail-closed; see the script docstring.
# ---------------------------------------------------------------------------
COPY context/patches/patch_offloading_ambient_config.py /opt/glm53/patch_offloading_ambient_config.py
RUN set -eux; python3 /opt/glm53/patch_offloading_ambient_config.py "${SITE}/vllm"

# ---------------------------------------------------------------------------
# Layer 4: Mia's Dockerfile heredocs, ported.
#
#   patch_nope_sm120.py           = her heredoc at Dockerfile:91-318, minus the
#                                   upstreamed cuda.py indexer-alignment edit,
#                                   with both expand_pools_and_append_tail call
#                                   sites re-anchored into kpool_ops.
#   patch_register_exl3_quant.py  = her heredoc at Dockerfile:386-415, verbatim.
#
# Carried as COPY-ed scripts rather than inline heredocs so they can be
# py_compile-checked on the authoring host and diffed against Mia's originals.
# Semantics unchanged: every edit asserts count()==1 before replacing, and the
# shell runs set -e.
# ---------------------------------------------------------------------------
COPY context/patches/ /opt/glm53/patches/
RUN set -eux; python3 /opt/glm53/patches/patch_nope_sm120.py

# ---------------------------------------------------------------------------
# Layer 5: exl3 quantization registration + the EXL3/MCG trellis extension.
#
# Order matches Mia's: register the method, THEN build the ext, THEN drop the
# Python-only exl3.py overlay -- so an edit to exl3.py never invalidates the
# expensive compile layer.
# ---------------------------------------------------------------------------
RUN set -eux; python3 /opt/glm53/patches/patch_register_exl3_quant.py

# These two operate on the exllamav3 SOURCE tree, not on vLLM:
#   patch_exl3_ext_aarch64.py  stubs the AVX2/AVX512 CPU targets that do not
#                              compile on aarch64.
#   patch_exl3_fat_kernel.py   stages exl3_fat_gemm.{cu,cuh} and
#                              exl3_fat_moe.{cu,cuh} into that tree pre-build.
COPY context/overlay/patch_exl3_ext_aarch64.py /opt/glm53/patch_exl3_ext_aarch64.py
COPY context/overlay/patch_exl3_fat_kernel.py /opt/glm53/patch_exl3_fat_kernel.py
COPY context/overlay/exl3_fat_gemm.cu /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu
COPY context/overlay/exl3_fat_gemm.cuh /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cuh
COPY context/overlay/exl3_fat_moe.cu /opt/glm53/exl3-fat-kernel/exl3_fat_moe.cu
COPY context/overlay/exl3_fat_moe.cuh /opt/glm53/exl3-fat-kernel/exl3_fat_moe.cuh
COPY context/overlay/build_exl3_fat_moe_ext.py /opt/glm53/build_exl3_fat_moe_ext.py

# The prebuilt .so. In EXT_MODE=inimage this directory only needs its README, so
# the COPY is unconditional and the *use* of it is gated below.
COPY context/ext/ /opt/glm53/ext/

# Fetch + patch the exllamav3 source. Identical in both modes: the source tree is
# what `pip install` needs for the Python package, and patching it keeps the two
# modes' provenance identical.
RUN set -eux; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    python3 -c "from pathlib import Path; assert (Path('/tmp/exllamav3')/'exllamav3/modules/quant/exl3.py').is_file()"; \
    test "$(python3 -c "v={};exec(open('/tmp/exllamav3/exllamav3/version.py').read(),v);print(v['__version__'])")" = "0.0.43"; \
    python3 /opt/glm53/patch_exl3_ext_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    python3 /opt/glm53/patch_exl3_fat_kernel.py /tmp/exllamav3/exllamav3/exllamav3_ext /opt/glm53/exl3-fat-kernel

# EXT_MODE=prebuilt  -> install the Python package with EXLLAMA_NOCOMPILE (the
#                       upstream setup.py opt-out, setup.py:10) and drop in a
#                       cross-compiled exllamav3_ext.
# EXT_MODE=inimage   -> Mia's compile, verbatim, under qemu. Hours.
#
# Expected prebuilt artifact:
#   context/ext/exllamav3_ext.cpython-312-aarch64-linux-gnu.so
#   ELF 64-bit LSB aarch64, CPython 3.12 ABI, linked against torch 2.13.0+cu130,
#   fatbin containing sm_121a, exporting exl3_moe, exl3_fat_gemm,
#   exl3_fat_gemm_scatter, exl3_fat_moe_gateup, exl3_fat_moe_down,
#   exl3_fat_moe_gather.
# The symbol/arch assertions below are the gate; a wrong-ABI .so fails here.
RUN set -eux; \
    cd /tmp/exllamav3; \
    if [ "${EXT_MODE}" = "prebuilt" ]; then \
        SO=/opt/glm53/ext/exllamav3_ext.cpython-312-aarch64-linux-gnu.so; \
        test -f "$SO" || { echo "EXT_MODE=prebuilt but $SO is absent"; ls -la /opt/glm53/ext; exit 1; }; \
        EXLLAMA_NOCOMPILE=1 pip install --break-system-packages \
            --no-deps --no-build-isolation --no-cache-dir .; \
        install -m 0755 "$SO" "${SITE}/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"; \
    elif [ "${EXT_MODE}" = "inimage" ]; then \
        export CPATH="${SITE}/nvidia/cu13/include${CPATH:+:$CPATH}"; \
        export CPLUS_INCLUDE_PATH="${SITE}/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"; \
        export C_INCLUDE_PATH="${SITE}/nvidia/cu13/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
        TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS="${MAX_JOBS}" \
          pip install --break-system-packages --no-deps --no-build-isolation --no-cache-dir .; \
    else \
        echo "EXT_MODE must be prebuilt or inimage (got: ${EXT_MODE})"; exit 1; \
    fi; \
    rm -rf /tmp/exllamav3 /root/.cache/pip

# Same assertion set Mia runs after her compile. This DOES import torch, so it is
# qemu-slow; it is not optional -- a silently symbol-short .so is the exact
# failure this image exists to avoid.
RUN set -eux; \
    python3 -c "import torch, exllamav3_ext as e; \
missing=[s for s in ('exl3_moe','exl3_fat_gemm','exl3_fat_gemm_scatter','exl3_fat_moe_gateup','exl3_fat_moe_down','exl3_fat_moe_gather') if not hasattr(e,s)]; \
assert not missing, (missing, dir(e)); \
print('exllamav3_ext', e.__file__, 'all six exl3/fat symbols present')"; \
    python3 -c "import exllamav3; print('exllamav3', exllamav3.__version__ if hasattr(exllamav3,'__version__') else 'pkg ok')" || true; \
    echo "EXT_MODE=${EXT_MODE}" > /opt/receipts/ext-mode.txt

# ---------------------------------------------------------------------------
# Layer 6: the Python-only overlay. AFTER the ext layer, exactly as Mia orders
# it, so editing exl3.py never triggers a recompile.
# ---------------------------------------------------------------------------
COPY context/overlay/exl3.py ${SITE}/vllm/model_executor/layers/quantization/exl3.py
COPY context/overlay/ /opt/glm53/overlay/
# patch_ablit.py hard-codes the flat Mia layout for the runtime it installs.
COPY context/overlay/ablit_runtime.py /opt/glm53/ablit_runtime.py
COPY context/tests/ /opt/glm53/tests/
COPY context/files/chat_template.jinja /opt/glm53/chat_template.jinja
COPY context/ablit/ /opt/glm53/ablit/
COPY context/verify/ /opt/glm53/verify/

# Mia's 12 build-time scripts, in her Dockerfile order (lines 467-480), plus the
# two she applies at container start (patch_adaptive_k, patch_dense_fp8) which
# are baked here instead -- both are unconditional source edits whose behaviour
# is gated by runtime env, so build time is equivalent and keeps the launcher
# free of bind-mounts.
#
# NOT run here, deliberately:
#   patch_glm_video_placeholders.py -- Mia runs it at container start; it writes
#       a .pth shim into the site root and is re-run per boot. Kept at
#       /opt/glm53/overlay/ and invoked by launch/start-tp4.sh.
#   patch_spinwait.py (without --preflight) -- reads GLM53_SPINWAIT_MS at
#       container start; --preflight here is validate-only, exactly as Mia does.
RUN set -eux; \
    cd /opt/glm53/overlay; \
    python3 patch_model_overrides.py; \
    python3 patch_dflash2.py; \
    python3 patch_glm_eagle3.py; \
    python3 patch_glm5_drafter_group.py; \
    python3 patch_suppress_stops_in_reasoning.py; \
    python3 patch_scheduler_decode_floor.py; \
    python3 patch_hybrid_prefix_hit.py; \
    python3 patch_xgrammar_termination.py; \
    python3 patch_kpool_tail_slotmap.py; \
    python3 patch_indexer_workspace.py; \
    python3 patch_adaptive_k.py; \
    GLM53_SITE=/usr/local/lib/python3.12/dist-packages/vllm python3 patch_dense_fp8.py; \
    python3 patch_spinwait.py --preflight; \
    python3 patch_ablit.py

# ---------------------------------------------------------------------------
# Layer 7: startup observer -- LAST source edit, so its four call sites land on
# the final patched tree. patch -p1 against the installed package; --dry-run
# first so a rejected hunk fails the build instead of leaving .rej litter.
#
# The call sites do `from startup_observer import emit`, i.e. a TOP-LEVEL
# module, so the emitter installs at the site root, not inside vllm/.
# ---------------------------------------------------------------------------
COPY context/observer/ /opt/glm53/observer/
RUN set -eux; \
    install -m 0644 /opt/glm53/observer/startup_observer.py "${SITE}/startup_observer.py"; \
    cd "${SITE}"; \
    patch -p1 --dry-run --forward --batch < /opt/glm53/observer/observer-callsites-83252ea89.patch; \
    patch -p1 --forward --batch < /opt/glm53/observer/observer-callsites-83252ea89.patch; \
    python3 -m py_compile "${SITE}/vllm/v1/core/kv_cache_utils.py" "${SITE}/vllm/v1/worker/gpu/attn_utils.py"; \
    python3 -m py_compile "${SITE}/startup_observer.py"

# ---------------------------------------------------------------------------
# Layer 8: gates. Static first (import-free, always runs), then Mia's own
# import-free tests, then the import-level checks.
# ---------------------------------------------------------------------------
RUN set -eux; python3 /opt/glm53/verify/verify_overlay_static.py

RUN set -eux; \
    cd /opt/glm53/tests; \
    python3 test_suppress_stops.py; \
    python3 test_scheduler_decode_floor.py; \
    python3 test_hybrid_prefix_hit.py; \
    python3 test_xgrammar_termination.py; \
    python3 test_kpool_tail_slotmap.py; \
    python3 test_spinwait_patch.py; \
    python3 test_adaptive_k_patch.py; \
    python3 test_dense_fp8_patch.py

# These import torch/vllm (35 imports in test_exl3_overlay alone) and are the
# slow ones under emulation. RUN_IMPORT_CHECKS=0 skips them -- if you do that,
# run /opt/glm53/verify/run-import-checks.sh on a Spark before trusting the image.
RUN set -eux; \
    if [ "${RUN_IMPORT_CHECKS}" = "1" ]; then \
        python3 /opt/glm53/verify/verify_overlay_runtime.py; \
        cd /opt/glm53/tests; \
        EXL3_SELFCHECK_GPU=0 python3 test_exl3_overlay.py; \
        python3 test_indexer_workspace.py; \
        python3 test_ablit.py; \
    else \
        echo "RUN_IMPORT_CHECKS=0 -- import-level gates SKIPPED, run them on a Spark" \
            | tee /opt/receipts/IMPORT-CHECKS-SKIPPED; \
    fi

# ---------------------------------------------------------------------------
# Layer 9: receipts + launcher payload. Last, so a stamp-only rebuild is cheap.
# ---------------------------------------------------------------------------
COPY context/launch/ /opt/glm53/launch/
# Declared here, not at the top: an ARG is an implicit env var for every later RUN,
# so declaring it early would bust the layer cache whenever the stamp changes.
ARG RECIPE_STAMP=unknown
# BASE_IMAGE is declared before the first FROM, which puts it in the global
# (pre-stage) scope only -- referencing it inside a stage without this
# re-declaration yields an EMPTY string in the receipt and the LABEL. That was
# already true of the pre-native single-stage file (it printed
# `base_image=` into pins.txt); the multi-stage split makes the re-declaration
# unavoidable, so the receipt is correct from here on. Declared late, next to
# its first use, for the same layer-cache reason as RECIPE_STAMP.
ARG BASE_IMAGE
RUN set -eux; \
    pip freeze > /opt/receipts/pip-freeze-final.txt; \
    { \
      echo "vllm_commit=${VLLM_COMMIT}"; \
      echo "exllamav3_commit=${EXLLAMAV3_COMMIT}"; \
      echo "ext_mode=${EXT_MODE}"; \
      echo "recipe_stamp=${RECIPE_STAMP}"; \
      echo "base_image=${BASE_IMAGE}"; \
      echo "import_checks=${RUN_IMPORT_CHECKS}"; \
    } > /opt/receipts/pins.txt; \
    python3 - <<'PY' > /opt/receipts/patched-files.sha256
import hashlib, os
site = os.environ["SITE"]
targets = [
    "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
    "vllm/platforms/cuda.py",
    "vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
    "vllm/model_executor/layers/quantization/__init__.py",
    "vllm/model_executor/layers/quantization/exl3.py",
    "vllm/model_executor/warmup/kernel_warmup.py",
    "vllm/models/glm5next/nvidia/model.py",
    "vllm/models/glm5next/nvidia/mtp.py",
    "vllm/models/glm5next/nvidia/kda.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/worker/gpu/attn_utils.py",
    "vllm/v1/worker/block_table.py",
    "vllm/v1/attention/backends/mla/indexer.py",
    "vllm/v1/engine/detokenizer.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py",
    "vllm/v1/kv_offload/base.py",
    "vllm/v1/kv_offload/cpu/manager.py",
    "startup_observer.py",
    "exllamav3_ext.cpython-312-aarch64-linux-gnu.so",
]
for rel in targets:
    path = os.path.join(site, rel)
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            print(hashlib.sha256(handle.read()).hexdigest(), rel)
    else:
        print("MISSING".ljust(64), rel)
PY

LABEL glm53.recipe.stamp="${RECIPE_STAMP}" \
      glm53.vllm.commit="${VLLM_COMMIT}" \
      glm53.exllamav3.commit="${EXLLAMAV3_COMMIT}" \
      glm53.ext.mode="${EXT_MODE}" \
      glm53.persistence="recipe_persistence.native:NodeLocalDiskOffloadingSpec" \
      glm53.lane="flash-exl3-tp4-persist"

WORKDIR /
ENTRYPOINT ["/bin/bash"]
