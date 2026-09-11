# `context/host/` — host-side scripts (NOT used by the image build)

These run on the Spark hosts, outside any container, before `launch/start-tp4.sh`.
They live in the context only so they ship with the recipe; nothing in the
Dockerfile consumes them.

| Script | Source | When |
|---|---|---|
| `flusher-unconditional.sh` | tonyd2wild repo root, verbatim | started on **every node** BEFORE the launcher; left running for the whole boot window; `pkill -f flusher-unconditional.sh` once serving |
| `mem-ritual.sh` | tonyd2wild README:159-165 + `fleet_watchdog.sh:65` | on every node, every launch, before the flusher |

Order on each node:

```bash
./mem-ritual.sh
nohup ./flusher-unconditional.sh 5400 >/tmp/flusher.log 2>&1 &
# ... then, on node0 only:
./start-tp4.sh up
# once /health is 200:
pkill -f flusher-unconditional.sh
```

**The flusher must stay unconditional.** A threshold-triggered flusher can sit
below its threshold and still leave the NVRM allocator short, which shows up as
the same command booting or OOMing depending on the moment — this is the single
change that made 24 GiB/rank pass where it had been dying, and it is also the
correction of the earlier (wrong) "phantom KV backing" diagnosis.

Neither script is a floor guard: neither watches a threshold, neither decides
anything, neither kills a process. Mia's own preflight is inside
`launch/start-tp4.sh` (`./start-tp4.sh preflight`), which is read-only and
starts nothing.
