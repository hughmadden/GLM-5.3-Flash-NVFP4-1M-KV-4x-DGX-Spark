#!/usr/bin/env python3
"""800k + 64k session restore under the flood method.

Phase 1: send the 64k and 800k sessions (resident in GPU pool).
Phase 2: flood 300 x 9k unique sessions at C=4 to force eviction of the
        seeded sessions (the write-behind sink copies them out).
Phase 3: re-request both sessions; the evicted portions restore from disk.

Deterministic seeds; same filler as the seed protocol.
"""
import json, os, subprocess, threading, time, urllib.request

URL = os.environ.get("PERSIST_BENCH_URL", "http://127.0.0.1:8888")


def filler(tag, target, seed):
    n = max(1, int(target / 17))
    body = "\n".join(f"{i:06d} {(i*7919+seed)%10**9:09d} {(i*104729+seed)%10**9:09d}"
                     for i in range(n))
    return (f"Archive {tag} seed {seed}. Each line is an independent record.\n"
            f"{body}\nSummarise the archive briefly.")


def send(prompt, max_tokens=8, timeout=2400, label=""):
    body = {"model": "glm-5.3-flash-exl3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
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
    return {"label": label, "ok": True,
            "wall": round(time.monotonic() - t0, 1),
            "prompt": (usage or {}).get("prompt_tokens"),
            "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)}


def drive(steps_event):
    n = 0
    while not steps_event.is_set():
        try:
            send(filler("DRIVER", 200, 90000 + n), max_tokens=1, timeout=60)
        except Exception:
            pass
        n += 1
        steps_event.wait(2.0)


def main():
    out = {}
    stop = threading.Event()
    driver = threading.Thread(target=drive, args=(stop,), daemon=True)
    driver.start()

    # Phase 1: seed sessions resident
    s64 = filler("S64K", 65_536, 515151)
    s800 = filler("S800K", 800_000, 424242)
    out["seed64"] = send(s64, label="seed64")
    print("seed64:", json.dumps(out["seed64"]), flush=True)
    out["seed800"] = send(s800, label="seed800")
    print("seed800:", json.dumps(out["seed800"]), flush=True)

    # Phase 2: flood to force eviction (drive steps throughout)
    t0 = time.monotonic()
    results = [None] * 300
    def flood_one(i):
        results[i] = send(filler(f"F{i}", 9000, 700000 + i),
                          max_tokens=8, timeout=900, label=f"flood-{i}")
    idx = 0
    batch = 4
    while idx < 300:
        threads = []
        for _ in range(batch):
            if idx >= 300:
                break
            t = threading.Thread(target=flood_one, args=(idx,))
            t.start()
            threads.append(t)
            idx += 1
        for t in threads:
            t.join()
        print(f"flood {idx}/300 elapsed={round(time.monotonic()-t0)}s", flush=True)
    ok = sum(1 for r in results if r and r.get("ok"))
    out["flood"] = {"completed": ok, "of": 300,
                    "seconds": round(time.monotonic() - t0, 1)}
    print("flood done:", json.dumps(out["flood"]), flush=True)
    time.sleep(45)  # let post-flood drains and evictions settle

    # Phase 3: restores
    out["restore64"] = send(s64, label="restore64")
    print("restore64:", json.dumps(out["restore64"]), flush=True)
    out["restore800"] = send(s800, label="restore800")
    print("restore800:", json.dumps(out["restore800"]), flush=True)
    stop.set()
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
