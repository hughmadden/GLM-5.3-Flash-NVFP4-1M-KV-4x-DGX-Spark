#!/usr/bin/env python3
"""Pressure-tax measurement: decode throughput while a flood keeps the
write-behind sink active, vs the no-pressure baseline.

Phase A: idle decode baseline (C=2 decode streams, unique 8k prompts).
Phase B: same decode streams while a 120 x 9k flood runs at C=4 (pool
        under pressure -> eviction copies active -> measured tax).
"""
import json, os, threading, time, urllib.request

URL = os.environ.get("PERSIST_BENCH_URL", "http://127.0.0.1:8888")

def filler(tag, target, seed):
    n = max(1, int(target / 17))
    body = "\n".join(f"{i:06d} {(i*7919+seed)%10**9:09d} {(i*104729+seed)%10**9:09d}"
                     for i in range(n))
    return (f"Bench {tag} seed {seed}.\n{body}\nReply briefly.")

def send(prompt, max_tokens, timeout=1200, label=""):
    body = {"model": "glm-5.3-flash-exl3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    usage = None
    ttft = None
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
                d = (o.get("choices") or [{}])[0].get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if piece and ttft is None:
                    ttft = time.monotonic() - t0
    except Exception as e:
        return {"label": label, "ok": False, "error": str(e)[:100]}
    wall = time.monotonic() - t0
    ct = (usage or {}).get("completion_tokens") or 0
    decode_window = max(wall - (ttft or 0), 1e-9)
    return {"label": label, "ok": True, "wall": round(wall, 1),
            "decode_tok_s": round(ct / decode_window, 1) if ct else None}

stop = threading.Event()

def decode_stream(results, tag, max_tokens=300):
    n = 0
    while not stop.is_set():
        r = send(filler(f"D{tag}", 8192, 880000 + n * 7 + tag), max_tokens,
                 label=f"{tag}-{n}")
        results.append(r)
        n += 1

def main():
    out = {}
    # Phase A: idle decode baseline, 3 minutes of C=2
    base = []
    threads = [threading.Thread(target=decode_stream, args=(base, t)) for t in (0, 1)]
    for t in threads: t.start()
    time.sleep(180)
    stop.set()
    for t in threads: t.join()
    ok = [r for r in base if r.get("decode_tok_s")]
    out["idle_decode"] = {"samples": len(ok),
                          "median_tok_s": sorted(r["decode_tok_s"] for r in ok)[len(ok)//2] if ok else None}
    print("idle_decode:", json.dumps(out["idle_decode"]), flush=True)

    # Phase B: decode streams + flood at C=4
    stop.clear()
    under = []
    threads = [threading.Thread(target=decode_stream, args=(under, t + 10)) for t in (0, 1)]
    for t in threads: t.start()
    time.sleep(5)
    flood_results = [None] * 120
    def flood_one(i):
        flood_results[i] = send(filler(f"F{i}", 9000, 770000 + i), 8,
                                timeout=900, label=f"F{i}")
    idx = 0
    t0 = time.monotonic()
    while idx < 120:
        batch = []
        for _ in range(4):
            if idx >= 120: break
            t = threading.Thread(target=flood_one, args=(idx,))
            t.start(); batch.append(t); idx += 1
        for t in batch: t.join()
    flood_s = round(time.monotonic() - t0, 1)
    time.sleep(20)
    stop.set()
    for t in threads: t.join()
    ok = [r for r in under if r.get("decode_tok_s")]
    out["under_flood_decode"] = {"samples": len(ok),
                                 "median_tok_s": sorted(r["decode_tok_s"] for r in ok)[len(ok)//2] if ok else None}
    out["flood"] = {"completed": sum(1 for r in flood_results if r and r.get("ok")),
                    "seconds": flood_s}
    print("under_flood_decode:", json.dumps(out["under_flood_decode"]), flush=True)
    print("flood:", json.dumps(out["flood"]), flush=True)
    print("RESULT " + json.dumps(out))

if __name__ == "__main__":
    main()
