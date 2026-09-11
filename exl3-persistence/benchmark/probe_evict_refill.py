#!/usr/bin/env python3
"""probe_evict_refill.py -- measure the persistence tier's eviction/refill behaviour.

The disk tier's own quota is 1 TB, so its eviction path is unreachable at test
scale. What IS reachable is the interesting direction: force the *GPU* prefix
cache to evict a session, then re-request it and see who serves it -- GPU cache,
the disk tier, or a full recompute.

Method
  phase 0  warm the target prompt once (cold prefill), record TTFT
  phase 1  confirm a repeat of the target is fast and see what served it
  phase 2  flood the engine with N unique long prompts, to push the target out
           of the GPU KV pool (~3.4M tokens at this configuration)
  phase 3  re-request the target; classify the serve path by TTFT and by the
           kv_offload load-bytes / prefix-cache counters

Usage:
  probe_evict_refill.py --url http://HEAD:8000 --model <id> [--n 400]
                        [--tokens 9000] [--conc 4] [--max-tokens 8]
"""
import argparse, json, random, threading, time, urllib.request

KEYS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_load_time_total",
    "vllm:kv_offload_lookup_sync_delay_seconds_count",
    "vllm:kv_offload_allocation_failure_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


def scrape(url):
    txt = urllib.request.urlopen(url + "/metrics", timeout=20).read().decode()
    out = {}
    for line in txt.splitlines():
        for k in KEYS:
            # skip the transfer_type-labelled legacy duplicates
            if line.startswith(k) and "transfer_type" not in line:
                out[k] = float(line.split()[-1])
    return out


def post(url, model, content, max_tokens, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return time.time() - t0, r


def filler(seed, approx_tokens):
    """A unique prompt of roughly `approx_tokens` tokens."""
    rnd = random.Random(seed)
    n = max(1, approx_tokens // 14)
    words = [
        f"ctx{seed}w{i}{rnd.randrange(10**6)}" for i in range(n)
    ]
    return (
        "Consider the following unique record identifiers: "
        + " ".join(words)
        + "\n\nReply with exactly the word DONE."
    )


def timed(url, model, prompt, max_tokens):
    t0 = time.time()
    before = scrape(url)
    try:
        dt, r = post(url, model, prompt, max_tokens)
        ok = True
        toks = r["usage"]["prompt_tokens"]
    except Exception as e:  # noqa: BLE001
        dt, ok, toks = time.time() - t0, False, -1
        print(f"    request failed: {str(e)[:90]}")
    time.sleep(2.0)
    after = scrape(url)
    return {"seconds": dt, "ok": ok, "prompt_tokens": toks,
            "before": before, "after": after}


def delta(res, key):
    return res["after"].get(key, 0.0) - res["before"].get(key, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=400, help="flood requests")
    ap.add_argument("--tokens", type=int, default=9000, help="approx prompt tokens")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8)
    a = ap.parse_args()

    target = filler("TARGET", a.tokens)

    print("=== phase 0: cold prefill of the target prompt ===")
    r0 = timed(a.url, a.model, target, a.max_tokens)
    print(f"  cold: {r0['seconds']:.2f} s  prompt_tokens={r0['prompt_tokens']}  "
          f"prefix_hits={delta(r0,'vllm:prefix_cache_hits_total'):.0f}  "
          f"store_B={delta(r0,'vllm:kv_offload_store_bytes_total'):.0f}")

    print("=== phase 1: immediate repeat (should be served from a cache) ===")
    r1 = timed(a.url, a.model, target, a.max_tokens)
    print(f"  warm: {r1['seconds']:.2f} s  "
          f"prefix_hits={delta(r1,'vllm:prefix_cache_hits_total'):.0f}  "
          f"load_B={delta(r1,'vllm:kv_offload_load_bytes_total'):.0f}")

    print(f"=== phase 2: flooding {a.n} unique ~{a.tokens}-token prompts "
          f"at concurrency {a.conc} to evict the target from GPU KV ===")
    t_flood = time.time()
    done = [0]
    lock = threading.Lock()

    def worker(i):
        try:
            post(a.url, a.model, filler(f"F{i}", a.tokens), a.max_tokens)
        except Exception:  # noqa: BLE001
            pass
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"    {done[0]}/{a.n} flooded ({time.time()-t_flood:.0f}s)")

    idx = [0]

    def next_idx():
        with lock:
            idx[0] += 1
            return idx[0]

    threads = []
    for _ in range(a.conc):
        def run():
            while True:
                i = next_idx()
                if i > a.n:
                    return
                worker(i)
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    flood_s = time.time() - t_flood
    mf = scrape(a.url)
    print(f"  flood done in {flood_s:.0f} s "
          f"({a.n} requests, {a.n * a.tokens / flood_s:.0f} prompt-tok/s)")

    print("=== phase 3: re-request the target (evicted from GPU?) ===")
    r3 = timed(a.url, a.model, target, a.max_tokens)
    d_load = delta(r3, "vllm:kv_offload_load_bytes_total")
    d_hits = delta(r3, "vllm:prefix_cache_hits_total")
    d_q = delta(r3, "vllm:prefix_cache_queries_total")
    print(f"  refill: {r3['seconds']:.2f} s  "
          f"prefix_queries={d_q:.0f}  prefix_hits={d_hits:.0f}  "
          f"disk_load_B={d_load:.0f}")

    if d_load > 0:
        served = "disk tier"
    elif d_hits > 0:
        served = "GPU prefix cache (never evicted)"
    else:
        served = "full recompute"
    print()
    print("| phase | TTFT/total s | prefix_queries | prefix_hits | disk_load_B |")
    print("|---|---:|---:|---:|---:|")
    for name, r in (("0 cold target", r0), ("1 warm repeat", r1),
                    ("3 after flood", r3)):
        print(f"| {name} | {r['seconds']:.2f} | "
              f"{delta(r,'vllm:prefix_cache_queries_total'):.0f} | "
              f"{delta(r,'vllm:prefix_cache_hits_total'):.0f} | "
              f"{delta(r,'vllm:kv_offload_load_bytes_total'):.0f} |")
    print()
    print(f"SERVE PATH AFTER EVICTION FLOOD: {served}")
    print(json.dumps({"flood_seconds": round(flood_s, 1),
                      "flood_requests": a.n,
                      "recompute_ratio": round(r1["seconds"] / max(r3["seconds"], 1e-6), 2),
                      "serve_path": served,
                      "store_bytes_total_after": mf.get("vllm:kv_offload_store_bytes_total"),
                      "load_bytes_total_after": mf.get("vllm:kv_offload_load_bytes_total"),
                      "allocation_failure_total": mf.get("vllm:kv_offload_allocation_failure_total")},
                     indent=2))


if __name__ == "__main__":
    main()
