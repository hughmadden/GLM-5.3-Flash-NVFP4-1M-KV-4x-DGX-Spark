#!/usr/bin/env python3
"""Prefill-tax ladder: persistence ON vs OFF, cold (unique prompt) vs warm
repeat, at several context sizes.

For each size: a UNIQUE prompt (guaranteed cold: no GPU prefix, no disk
object), timed with max_tokens=8 to isolate prefill; then the SAME prompt
immediately re-requested (warm: whoever serves it -- GPU APC and/or the
disk tier). Cold ON measures the store tax; warm ON measures the resume
benefit; OFF arms are the baseline.

Usage: probe_prefill_tax.py --url ... --model ... [--sizes 2048,8192,...]
"""
import argparse, json, statistics, threading, time, urllib.request


def filler(tag, target, seed):
    n = max(1, int(target / 17))
    body = "\n".join(f"{i:06d} {(i*7919+seed)%10**9:09d} {(i*104729+seed)%10**9:09d}"
                     for i in range(n))
    return f"Ledger {tag} seed {seed}. Each line is an independent record.\n{body}\nDescribe the ledger briefly."


KEYS = ("vllm:kv_offload_tiering_chunk_hits_total", "vllm:kv_offload_load_bytes_total",
        "vllm:kv_offload_store_bytes_total", "vllm:external_prefix_cache_hits_total")


def scrape(url):
    out = {}
    try:
        txt = urllib.request.urlopen(url + "/metrics", timeout=20).read().decode()
        for line in txt.splitlines():
            for k in KEYS:
                if line.startswith(k) and "engine=" in line:
                    out[k] = float(line.split()[-1])
    except Exception:
        pass
    return out


def run(url, model, prompt, max_tokens=8, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    m0 = scrape(url)
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:80]}",
                "wall": round(time.monotonic() - t0, 1)}
    wall = time.monotonic() - t0
    m1 = scrape(url)
    u = resp.get("usage", {})
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in KEYS}
    return {"ok": True, "wall": round(wall, 2),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "ext_hits": d["vllm:external_prefix_cache_hits_total"],
            "load_B": d["vllm:kv_offload_load_bytes_total"],
            "store_B": d["vllm:kv_offload_store_bytes_total"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", default="2048,8192,16384,32768,65536")
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--settle", type=int, default=15,
                    help="seconds between the cold and warm request so the "
                         "finish-drain store lands before the repeat probes it")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    # one warmup to settle the engine
    w = run(args.url, args.model, filler("WARMUP", 2000, 1))
    print(f"warmup: ok={w['ok']} wall={w.get('wall')}")

    # Step driver: the finish-drain (engine patch 0004) advances only when
    # the scheduler makes steps, and on an idle engine a lone request's
    # drain may not run until a LATER request drives it -- so a repeat can
    # never hit what immediately preceded it. A tiny background request
    # every few seconds keeps steps flowing so drains land promptly. The
    # load is identical across arms.
    stop = threading.Event()

    def drive():
        n = 0
        while not stop.is_set():
            try:
                run(args.url, args.model, filler("DRIVER", 300, 70000 + n), max_tokens=1, timeout=60)
            except Exception:
                pass
            n += 1
            stop.wait(3.0)

    driver = threading.Thread(target=drive, daemon=True)
    driver.start()

    rows = []
    try:
        for size in sizes:
            prompt = filler(f"COLD{size}", size, args.seed_base + size)
            cold = run(args.url, args.model, prompt)
            if args.settle:
                time.sleep(args.settle)
            warm1 = run(args.url, args.model, prompt)   # drives the drain
            time.sleep(5)
            warm = run(args.url, args.model, prompt)    # the measurement
            for label, r in (("cold", cold), ("warm-probe", warm1), ("warm", warm)):
                pt = r.get("prompt_tokens")
                rate = round(pt / r["wall"]) if r["ok"] and pt and r["wall"] else None
                rows.append({"size": size, "kind": label, "ok": r["ok"],
                             "tokens": pt, "wall_s": r.get("wall"), "prefill_tok_s": rate,
                             "cached": r.get("cached_tokens", 0), "ext_hits": r.get("ext_hits", 0),
                             "load_MB": round(r.get("load_B", 0) / 1e6, 1),
                             "store_MB": round(r.get("store_B", 0) / 1e6, 1)})
                print(f"  {size:>6} {label}: wall={r.get('wall')}s prefill={rate} tok/s "
                      f"cached={r.get('cached_tokens', 0)} ext_hits={r.get('ext_hits', 0)} "
                      f"load={round(r.get('load_B',0)/1e6,1)}MB store={round(r.get('store_B',0)/1e6,1)}MB"
                      + ("" if r["ok"] else f" ERROR {r.get('error')}"))
    finally:
        stop.set()
    print(json.dumps({"arm": args.model, "rows": rows}))


if __name__ == "__main__":
    main()
