#!/usr/bin/env bash
# distribute.sh -- fan the built image out to the four Sparks and prove all four
# hold the SAME image, by .Id, not by tag.
#
#   *** NOT TO BE RUN YET. S1 (fleet phase) only. ***
#
# The Sparks are gated on the operator releasing them after full-model testing. This
# script exists so the distribution step is written, reviewed and correct before
# it is ever needed; it refuses to run without an explicit acknowledgement.
#
# Method (tonyd2wild §4): ONE node holds the image, then
#   zstd -d < tarball | ssh nodeN "docker load"
# Never four concurrent registry pulls (rate limits), never a per-node build.
#
#   "Matching tags prove nothing -- four nodes that each built the tag locally
#    get four different images."
#
# so the verification step compares docker image inspect --format '{{.Id}}'
# across all four and fails on any divergence. (His own speed-run notes one real
# case where four ranks reported different Ids and still worked -- same vLLM
# hash, only .pyc caches differed -- so treat a divergence as a smell to chase,
# not an automatic abort; this script reports it and exits non-zero rather than
# pretending to know which case it is.)
#
# SSH/AES on GB10 ARM cores tops out around 1 Gbps. On a 100G rail his field
# notes prefer an unencrypted rsyncd pull per node. We keep the ssh form because
# it needs no daemon; if the transfer time is the bottleneck, switch to the
# rsyncd variant rather than parallelising registry pulls.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRV=${SRV:-$STAGE}
IMAGES=${IMAGES:-$SRV/images}

TARBALL=${1:-}
TAG=${TAG:-}
NODES=(${NODES:-user@192.0.2.10 user@192.0.2.11 user@192.0.2.12 user@192.0.2.13})

log() { printf '\033[1;36m[distribute]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[distribute]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# --- S1 gate ---------------------------------------------------------------
if [ "${I_HAVE_S1_AUTHORISATION:-no}" != "yes" ]; then
    cat >&2 <<'EOF'
REFUSING TO RUN.

This script touches all four DGX Sparks. The fleet phases of this campaign are
gated on the operator explicitly releasing the Sparks after full-model testing. Nothing
here should reach a Spark before that.

If and when that gate opens, re-run with:

    I_HAVE_S1_AUTHORISATION=yes ./distribute.sh $STAGE/images/<tag>.tar.zst

EOF
    exit 2
fi

[ -n "$TARBALL" ] || die "usage: I_HAVE_S1_AUTHORISATION=yes $0 <tarball.tar.zst>"
[ -f "$TARBALL" ] || die "no such tarball: $TARBALL"
[ -f "${TARBALL}.sha256" ] || die "no sha256 sidecar for $TARBALL"

log "verifying local tarball checksum"
( cd "$(dirname "$TARBALL")" && sha256sum -c "$(basename "$TARBALL").sha256" ) \
    || die "tarball checksum mismatch -- rebuild, do not ship"

if [ -z "$TAG" ]; then
    TAG="$(basename "$TARBALL" .tar.zst)"
    TAG="${TAG/glm53-flash-exl3-tp4-persist-/glm53-flash-exl3-tp4-persist:}"
fi
log "tag ${TAG}"

# --- load on each node, serially -------------------------------------------
for node in "${NODES[@]}"; do
    log "loading on ${node} (streaming, decompress on the far side)"
    zstd -d -c "$TARBALL" | ssh -o BatchMode=yes -o ConnectTimeout=15 "$node" "docker load" \
        || die "docker load failed on ${node}"
done

# --- verify .Id parity ------------------------------------------------------
log "verifying image identity across all four nodes"
declare -a IDS=()
ok=1
for node in "${NODES[@]}"; do
    id="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$node" \
          "docker image inspect '$TAG' --format '{{.Id}}'" 2>/dev/null || true)"
    [ -n "$id" ] || { echo "  ${node}: MISSING"; ok=0; IDS+=("missing"); continue; }
    echo "  ${node}: ${id}"
    IDS+=("$id")
done
for id in "${IDS[@]}"; do
    [ "$id" = "${IDS[0]}" ] || ok=0
done

if [ "$ok" != "1" ]; then
    die "image identity is NOT uniform across the fleet -- chase this before launching"
fi
log "all four nodes hold ${IDS[0]}"

RECEIPT="$SRV/receipts/distribute-$(date +%Y%m%dT%H%M%S).json"
python3 - "$TAG" "${IDS[0]}" "$TARBALL" "${NODES[@]}" > "$RECEIPT" <<'PY'
import json, sys
tag, image_id, tarball, *nodes = sys.argv[1:]
print(json.dumps({"tag": tag, "image_id": image_id, "tarball": tarball,
                  "nodes": nodes, "uniform": True}, indent=2))
PY
log "receipt ${RECEIPT}"
