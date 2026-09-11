#!/usr/bin/env python3
"""Does a disk-tier refill EVER complete, or does it hang indefinitely?

Stores one unique prompt, then re-requests it while sampling the engine's own
counters every 10 s. No restart, no code change: this only observes.

Usage: probe_refill_patience.py --url ... --model ... [--wait 300]
"""
import argparse, json, random, threading, time, urllib.request

KEYS = (
    "vllm:kv_offload_tiering_chunk_queries_total",
    "vllm:kv_offload_tiering_chunk_hits_total",
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:kv_offload_allocation_failure_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


def scrape(url):
    txt = urllib.request.urlopen(url + "/metrics", timeout=15).read().decode()
    out = {}
    for line in txt.splitlines():
        for k in KEYS:
            if line.startswith(k) and "transfer_type" not in line:
                out[k] = float(line.split()[-1])
        if line.startswith("vllm:num_requests_waiting_by_reason") and 'deferred' in line:
            out["deferred"] = float(line.split()[-1])
    return out


def post(url, model, prompt, max_tokens, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    r = urllib.request.Request(url + "/v1/chat/completions",
                              data=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = json.load(urllib.request.urlopen(r, timeout=timeout))
        return time.time() - t0, resp["usage"], None
    except Exception as e:  # noqa: BLE001
        return time.time() - t0, None, f"{type(e).__name__}: {str(e)[:60]}"


def short(k):
    return (k.replace("vllm:kv_offload_tiering_", "tier_")
             .replace("vllm:kv_offload_", "off_")
             .replace("vllm:external_prefix_cache_", "ext_")
             .replace("vllm:num_requests_", "req_")
             .replace("_total", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wait", type=float, default=300.0,
                    help="client timeout for the repeat request")
    ap.add_argument("--tokens", type=int, default=9000)
    ap.add_argument("--max-tokens", type=int, default=8)
    a = ap.parse_args()

    rnd = random.Random(4242)
    words = " ".join(f"tok{rnd.randrange(10**9)}x{i}" for i in range(900))
    prompt = "Identifiers: " + words + "\n\nReply with exactly the word DONE."

    print("=== pass 1: cold ===")
    b = scrape(a.url)
    dt, usage, err = post(a.url, a.model, prompt, a.max_tokens, a.wait)
    time.sleep(3)
    m = scrape(a.url)
    print(f"  {dt:.1f}s usage={usage} err={err}")
    for k in KEYS:
        d = m.get(k, 0) - b.get(k, 0)
        if d:
            print(f"    +{d:>14.0f}  {short(k)}")

    print(f"=== pass 2: repeat, client timeout {a.wait:.0f}s (sampling every 10s) ===")
    result = {}
    stop = threading.Event()

    def sampler():
        last = None
        while not stop.is_set():
            try:
                c = scrape(a.url)
            except Exception:  # noqa: BLE001
                time.sleep(10); continue
            row = (c.get("vllm:kv_offload_tiering_chunk_queries_total", 0),
                   c.get("vllm:kv_offload_tiering_chunk_hits_total", 0),
                   c.get("vllm:kv_offload_load_bytes_total", 0),
                   c.get("deferred", 0),
                   c.get("vllm:num_requests_running", 0))
            if row != last:
                print(f"    t={time.time()-t0:6.1f}s  queries={row[0]:>7.0f} hits={row[1]:>5.0f} "
                      f"load_B={row[2]:>12.0f} deferred={row[3]:>3.0f} running={row[4]:>3.0f}")
                last = row
            time.sleep(2)

    t0 = time.time()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    dt2, usage2, err2 = post(a.url, a.model, prompt, a.max_tokens, a.wait)
    stop.set()
    time.sleep(1)
    c = scrape(a.url)
    print(f"  RESULT: {dt2:.1f}s usage={usage2} err={err2}")
    for k in KEYS:
        d = c.get(k, 0) - b.get(k, 0)
        print(f"    total +{d:>14.0f}  {short(k)}")
    d = c.get("deferred", 0) - b.get("deferred", 0)
    print(f"    total +{d:>14.0f}  deferred")


if __name__ == "__main__":
    main()
