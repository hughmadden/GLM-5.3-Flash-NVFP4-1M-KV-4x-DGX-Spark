#!/usr/bin/env python3
"""200k session restore cycle: seed resident -> flood (40x85k unique) forces
GPU eviction with the pressure-gated sink capturing it -> restart clears GPU
(cold) -> re-request = cold restore from disk -> re-request = hot in memory.
Phases 1-2 here; the restart is manual between phases; phase 3+4 after.
"""
import json, os, sys, threading, time, urllib.request

URL = os.environ.get("PERSIST_BENCH_URL", "http://127.0.0.1:8888")

def filler(tag, target, seed):
    n = max(1, int(target / 17))
    body = "\n".join(f"{i:06d} {(i*7919+seed)%10**9:09d} {(i*104729+seed)%10**9:09d}"
                     for i in range(n))
    return (f"Archive {tag} seed {seed}. Each line is an independent record.\n"
            f"{body}\nSummarise the archive briefly.")

def send(prompt, max_tokens=8, timeout=1800, label=""):
    body = {"model": "glm-5.3-flash-exl3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    o = json.loads(p)
                except Exception:
                    continue
                if o.get("usage"):
                    usage = o["usage"]
    except Exception as e:
        return {"label": label, "ok": False, "error": str(e)[:120],
                "wall": round(time.monotonic() - t0, 1)}
    return {"label": label, "ok": True, "wall": round(time.monotonic() - t0, 1),
            "prompt": (usage or {}).get("prompt_tokens"),
            "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)}

PHASE = sys.argv[1] if len(sys.argv) > 1 else "seed"

if PHASE == "seed":
    stop = threading.Event()
    def drive():
        n = 0
        while not stop.is_set():
            try:
                send(filler("D", 200, 990000 + n), max_tokens=1, timeout=60)
            except Exception:
                pass
            n += 1
            stop.wait(2.0)
    threading.Thread(target=drive, daemon=True).start()
    out = {}
    out["seed200"] = send(filler("S200K", 200_000, 626262), label="seed200")
    print("seed200:", json.dumps(out["seed200"]), flush=True)
    results = [None] * 40
    def flood_one(i):
        results[i] = send(filler(f"G{i}", 85_000, 660000 + i * 13),
                          timeout=1800, label=f"G{i}")
    idx = 0
    t0 = time.monotonic()
    while idx < 40:
        batch = []
        for _ in range(4):
            if idx >= 40:
                break
            t = threading.Thread(target=flood_one, args=(idx,))
            t.start()
            batch.append(t)
            idx += 1
        for t in batch:
            t.join()
        print(f"flood {idx}/40 elapsed={round(time.monotonic()-t0)}s", flush=True)
    out["flood"] = {"completed": sum(1 for r in results if r and r.get("ok")),
                    "seconds": round(time.monotonic() - t0, 1)}
    print("flood:", json.dumps(out["flood"]), flush=True)
    time.sleep(45)
    stop.set()
    print("READY-FOR-RESTART", flush=True)

elif PHASE == "restore":
    # Run AFTER a full restart (GPU cold, disk retains).
    out = {}
    out["restore200"] = send(filler("S200K", 200_000, 626262), label="restore200")
    print("restore200:", json.dumps(out["restore200"]), flush=True)
    time.sleep(8)
    out["hot200"] = send(filler("S200K", 200_000, 626262), label="hot200")
    print("hot200:", json.dumps(out["hot200"]), flush=True)
