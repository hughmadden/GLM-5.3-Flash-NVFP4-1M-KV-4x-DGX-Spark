#!/usr/bin/env bash
# Memory ritual -- run on EVERY node BEFORE start-tp4.sh, every launch.
#
# Source: tonyd2wild README.md:159-165 plus fleet_watchdog.sh:65 (mem_ritual()),
# which adds the compact_memory step the README omits and which his
# auto-recovery path uses on every recovery.
#
# This is a HOST script by design. It is not a floor guard and not kill
# machinery -- it does not watch memory, does not decide anything, and never
# stops a process. It puts the allocator in the state the recipe requires and
# exits.
#
# Two facts from his notes that make each step non-optional:
#   * swap must EXIST but not be USED. With no swap at all a worker is
#     OOM-killed outright during MoE repack. With swappiness > 0 the UVM driver
#     "can livelock unrecoverably and take the node off the network entirely".
#   * none of this survives a reboot. Put it in /etc/sysctl.d/ or you will lose
#     boots to it.
set -euo pipefail

if ! sudo -n true 2>/dev/null; then
    echo "FATAL: passwordless sudo required" >&2
    exit 1
fi

echo "mem-ritual: swappiness=0"
sudo -n sysctl -w vm.swappiness=0

if [ "$(awk 'NR==2{print $3}' /proc/swaps 2>/dev/null || echo 0)" != "" ]; then
    echo "mem-ritual: swapoff/swapon cycle"
    sudo -n swapoff -a
    sudo -n swapon -a
else
    echo "mem-ritual: WARNING no swap configured -- a worker can be OOM-killed during MoE repack" >&2
fi

echo "mem-ritual: drop_caches"
sync
echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null

echo "mem-ritual: compact_memory"
echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null

echo "mem-ritual: done ($(awk '/MemAvailable/{print $2" kB available"}' /proc/meminfo))"
