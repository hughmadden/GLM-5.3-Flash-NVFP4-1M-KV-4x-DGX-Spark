#!/usr/bin/env bash
# build.sh -- build the end-state ARM64 image. RUNS ON build-host, not on head, not on
# a Spark.
#
# build-host is x86_64, so this is a cross-build: buildx targets linux/arm64 and every
# RUN step executes under qemu-user (binfmt already installed there --
# $STAGE/receipts/binfmt.txt shows the tonistiigi/binfmt arm64
# handler registered and qemu-aarch64 present in /proc/sys/fs/binfmt_misc).
#
# What this does:
#   1. sanity-checks the host, buildx, binfmt and the wheel set;
#   2. rsyncs the five pinned wheels and (for EXT_MODE=prebuilt) the exllamav3_ext
#      .so into the build context;
#   3. computes a recipe stamp over the whole context so a content change forces
#      exactly one rebuild;
#   4. builds and --loads the image;
#   5. writes $STAGE/receipts/image-<tag>.json from docker image
#      inspect;
#   6. saves + zstd-compresses to $STAGE/images/<tag>.tar.zst with
#      a sha256 sidecar.
#
# It does NOT distribute anything. That is distribute.sh, and it is S1-gated.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT="$HERE/context"

SRV=${SRV:-$STAGE}
WHEEL_SRC=${WHEEL_SRC:-$SRV/wheels}
MIA_SO=${MIA_SO:-$SRV/trees/extracted-image/exllamav3_ext.cpython-312-aarch64-linux-gnu.so}
RECEIPTS=${RECEIPTS:-$SRV/receipts}
IMAGES=${IMAGES:-$SRV/images}

EXT_MODE=${EXT_MODE:-prebuilt}
RUN_IMPORT_CHECKS=${RUN_IMPORT_CHECKS:-1}

# Native dependency stage (Dockerfile stages `deps` / `sdist-aarch64`). These
# are passed explicitly rather than left to the Dockerfile defaults so the
# receipt and the LABELs record what was actually used. Keep them in step with
# the ARG defaults in the Dockerfile.
BASE_IMAGE=${BASE_IMAGE:-nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266}
DEPS_IMAGE=${DEPS_IMAGE:-python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea}
PIP_VERSION=${PIP_VERSION:-26.2.1}
INSTANTTENSOR_VERSION=${INSTANTTENSOR_VERSION:-0.1.9}
VLLM_SHORTSHA=83252ea89
DATE_TAG=${DATE_TAG:-$(date +%Y%m%d)}
TAG=${TAG:-glm53-flash-exl3-tp4-persist:${DATE_TAG}-${VLLM_SHORTSHA}}

WHEELS=(
  "vllm-0.28.1rc1.dev617+g83252ea89-cp38-abi3-manylinux_2_28_aarch64.whl"
  "torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl"
  "flashinfer_python-0.6.18.post1-py3-none-any.whl"
  "flashinfer_cubin-0.6.18.post1-py3-none-any.whl"
  "flashinfer_jit_cache-0.6.18.post1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl"
)

log()  { printf '\033[1;36m[build]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[build]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. host sanity --------------------------------------------------------
[ "$(uname -m)" = "x86_64" ] || log "note: host is $(uname -m), not the expected x86_64 build-host"
command -v docker >/dev/null || die "docker not found"
docker buildx version >/dev/null 2>&1 || die "docker buildx not available"
grep -q . /proc/sys/fs/binfmt_misc/qemu-aarch64 2>/dev/null \
  || die "qemu-aarch64 binfmt handler not registered -- run: docker run --privileged --rm tonistiigi/binfmt --install arm64"
[ -d "$WHEEL_SRC" ] || die "wheel dir not found: $WHEEL_SRC"

# --- 2. stage the context --------------------------------------------------
log "staging wheels from $WHEEL_SRC"
for w in "${WHEELS[@]}"; do
    [ -f "$WHEEL_SRC/$w" ] || die "missing wheel: $WHEEL_SRC/$w"
    rsync -a --info=name0 "$WHEEL_SRC/$w" "$CONTEXT/wheels/$w"
done
# Prefetched transitive closure (deps/URLS.txt + SHA256-expected.txt alongside).
[ -d "$WHEEL_SRC/deps" ] || die "missing prefetched deps dir: $WHEEL_SRC/deps (see Dockerfile deps stage)"
rsync -a --delete --info=name0 "$WHEEL_SRC/deps/" "$CONTEXT/wheels/deps/"
log "staged $(ls "$CONTEXT/wheels/deps"/*.whl | wc -l) prefetched dependency wheels"

if [ "$EXT_MODE" = "prebuilt" ]; then
    SO_DST="$CONTEXT/ext/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"
    if [ ! -f "$SO_DST" ]; then
        [ -f "$MIA_SO" ] || die "EXT_MODE=prebuilt and neither $SO_DST nor $MIA_SO exists -- see context/ext/README.md"
        log "staging prebuilt exllamav3_ext from $MIA_SO"
        rsync -a "$MIA_SO" "$SO_DST"
    fi
    # Cheap provenance gate before paying for a build. The authoritative symbol
    # check is inside the image (it actually imports the module).
    for sym in exl3_moe exl3_fat_gemm exl3_fat_gemm_scatter exl3_fat_moe_gateup exl3_fat_moe_down exl3_fat_moe_gather; do
        # grep -c reads the whole stream: grep -q would SIGPIPE strings under pipefail.
        [ "$(strings "$SO_DST" | grep -cx "$sym")" -gt 0 ] || die "prebuilt .so is missing symbol $sym"
    done
    [ "$(strings "$SO_DST" | grep -cw sm_121a)" -gt 0 ] || die "prebuilt .so has no sm_121a fatbin -- wrong arch list"
    log "prebuilt .so: six exl3 symbols + sm_121a present ($(stat -c %s "$SO_DST") bytes)"
fi

# --- 2c. persistence series: stage the fork files ---------------------------
# The series is the fork branch pinned in persistence-flash/manifest.json. Stage
# its eight files (hash-verified against the manifest) into context/fork-src/ so
# the image build stays offline; the stamp below covers them, so a fork-commit
# change forces a rebuild. Set PERSIST_FORK_SOURCE to a local fork checkout (a
# root containing vllm/) to stage without network.
rm -rf "$CONTEXT/fork-src"
python3 "$CONTEXT/persistence-flash/apply.py" fetch "$CONTEXT/fork-src" \
    || die "could not stage the persistence series from the fork (see persistence-flash/README.md)"
log "staged persistence series from fork $(python3 -c "import json;d=json.load(open('$CONTEXT/fork-src/FORK.json'));print(d['repo']+'@'+d['commit'][:10])")"

# --- 3. recipe stamp -------------------------------------------------------
# Hash the context minus the giant binaries (whose identity is already pinned by
# filename + the in-image pip freeze). A change to any script, patch or launcher
# changes the stamp and forces exactly one rebuild of the final layers.
RECIPE_STAMP="$(cd "$CONTEXT" && find . -type f \
    ! -path './wheels/*.whl' ! -path './ext/*.so' \
    -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)"
log "recipe stamp ${RECIPE_STAMP}"

# --- 4. build --------------------------------------------------------------
log "building ${TAG} (EXT_MODE=${EXT_MODE} RUN_IMPORT_CHECKS=${RUN_IMPORT_CHECKS})"
log "arm64 stages run under qemu; the dependency resolve runs NATIVELY on $(uname -m)"
log "deps image ${DEPS_IMAGE}"
log "instanttensor ${INSTANTTENSOR_VERSION} is the one dep with no aarch64 wheel --"
log "  it is compiled once in the sdist-aarch64 stage and then cached"
# The Dockerfile is multi-stage: `deps` resolves every Python dependency
# NATIVELY on $BUILDPLATFORM (this host, x86_64) and only the final stage runs
# under qemu. Nothing extra is needed to enable that -- buildx sets
# BUILDPLATFORM -- but two things matter here:
#   * BuildKit is required (buildx gives it) for the `--platform=$BUILDPLATFORM`
#     stage and for the pip cache mount in the deps stage. Do NOT fall back to
#     the legacy builder (DOCKER_BUILDKIT=0) -- it understands neither.
#   * the pip cache mount lives in the builder, not the image. `docker buildx
#     prune` clears it; `docker builder prune --filter type=exec.cachemount`
#     clears just it.
docker buildx build \
    --platform linux/arm64 \
    --load \
    --progress plain \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "DEPS_IMAGE=${DEPS_IMAGE}" \
    --build-arg "PIP_VERSION=${PIP_VERSION}" \
    --build-arg "INSTANTTENSOR_VERSION=${INSTANTTENSOR_VERSION}" \
    --build-arg "EXT_MODE=${EXT_MODE}" \
    --build-arg "RUN_IMPORT_CHECKS=${RUN_IMPORT_CHECKS}" \
    --build-arg "RECIPE_STAMP=${RECIPE_STAMP}" \
    -t "${TAG}" \
    -f "$HERE/Dockerfile" \
    "$HERE"

# --- 5. receipt ------------------------------------------------------------
mkdir -p "$RECEIPTS" "$IMAGES"
SAFE_TAG="${TAG//[:\/]/-}"
RECEIPT="$RECEIPTS/image-${SAFE_TAG}.json"
docker image inspect "${TAG}" > "$RECEIPT"
IMAGE_ID="$(docker image inspect "${TAG}" --format '{{.Id}}')"
log "image id ${IMAGE_ID}"
log "receipt  ${RECEIPT}"

# Pull the in-image receipts out too -- these are the record of what pip
# actually resolved and what the patched files hash to.
CID="$(docker create "${TAG}" true)"
docker cp "${CID}:/opt/receipts" "$RECEIPTS/in-image-${SAFE_TAG}" >/dev/null
docker rm -f "${CID}" >/dev/null
log "in-image receipts ${RECEIPTS}/in-image-${SAFE_TAG}"

# --- 6. save + compress ----------------------------------------------------
TARBALL="$IMAGES/${SAFE_TAG}.tar.zst"
log "saving to ${TARBALL} (this is a large image; zstd -T0)"
docker save "${TAG}" | zstd -T0 -3 -o "${TARBALL}" -f
sha256sum "${TARBALL}" > "${TARBALL}.sha256"
log "saved $(stat -c %s "${TARBALL}") bytes"
log "sha256 $(cut -d' ' -f1 "${TARBALL}.sha256")"

cat <<EOF

Build complete.
  tag        ${TAG}
  image id   ${IMAGE_ID}
  tarball    ${TARBALL}
  receipts   ${RECEIPT}

NOT distributed. distribute.sh is S1-gated and must not run until the operator releases
the Sparks.
EOF
