#!/usr/bin/env python3
"""A/B matrix driver: prefill/decode across modes x sizes x concurrency.

Modes: off (PERSISTENCE=off), eager, evict-only (write-behind).
Sizes: agentic-class ladder. Concurrency: parallel UNIQUE prompts.
Per cell: N unique prompts, per-request prefill tok/s (TTFT), decode
tok/s (200 tokens, temp 0), spec acceptance. Mode switches restart the
fleet via the launcher host (PERSIST_FLEET_HOST).

Usage: ab_matrix.py [--sizes 8192,65536,262144] [--conc 1,4] [--reps 2]
"""
import json, os, statistics, subprocess, sys, threading, time, urllib.request

# Target configuration comes from the environment so the script carries no
# internal identities: PERSIST_BENCH_URL (the serving endpoint) and
# PERSIST_FLEET_HOST (any rank's launcher host for mode switches).
URL = os.environ.get("PERSIST_BENCH_URL", "http://127.0.0.1:8888")
MODEL = "glm-5.3-flash-exl3"
FLEET_HOST = os.environ.get("PERSIST_FLEET_HOST", "localhost")
SPARK1 = ["ssh", "-o", "BatchMode=yes", FLEET_HOST]


def filler(tag, target, seed):
    n = max(1, int(target / 17))
    body = "\n".join(f"{i:06d} {(i*7919+seed)%10**9:09d} {(i*104729+seed)%10**9:09d}"
                     for i in range(n))
    return (f"Ledger {tag} seed {seed}. Each line is an independent record.\n"
            f"{body}\nDescribe the ledger briefly.")


def metrics():
    out = {}
    try:
        txt = urllib.request.urlopen(URL + "/metrics", timeout=20).read().decode()
        for line in txt.splitlines():
            if line.startswith("vllm:spec_decode_num_accepted_tokens_total") or \
               line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                out[line.split("{")[0]] = float(line.split()[-1])
    except Exception:
        pass
    return out


def one(prompt, max_tokens=200, timeout=1800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    m0 = metrics()
    ttft = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    o = json.loads(payload)
                except Exception:
                    continue
                if o.get("usage"):
                    usage = o["usage"]
                choices = o.get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                # Reasoning models stream reasoning_content long before
                # content; the FIRST delta of either kind is the TTFT.
                piece = (delta.get("content") or delta.get("reasoning_content")
                         or delta.get("reasoning"))
                if piece and ttft is None:
                    ttft = time.monotonic() - t0
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:90], "wall": round(time.monotonic()-t0, 1)}
    wall = time.monotonic() - t0
    m1 = metrics()
    acc = (m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) -
           m0.get("vllm:spec_decode_num_accepted_tokens_total", 0))
    drf = (m1.get("vllm:spec_decode_num_draft_tokens_total", 0) -
           m0.get("vllm:spec_decode_num_draft_tokens_total", 0))
    u = usage or {}
    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
    ttft = ttft or (pt / 1800 if pt else wall)
    return {"ok": True, "wall": round(wall, 2), "ttft": round(ttft, 2),
            "prompt": pt, "completion": ct,
            "prefill_tok_s": round(pt / ttft) if pt else None,
            "decode_tok_s": round(ct / max(wall - ttft, 1e-9)) if ct else None,
            "accept": round(acc / drf, 2) if drf else None}


def cell(size, conc, reps, seed_base):
    rows = []
    for rep in range(reps):
        prompts = [filler(f"S{size}R{rep}C{c}", size, seed_base + rep * 10 + c)
                   for c in range(conc)]
        results = [None] * conc
        def work(i):
            results[i] = one(prompts[i])
        threads = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
        t0 = time.monotonic()
        for t in threads: t.start()
        for t in threads: t.join()
        wall = round(time.monotonic() - t0, 1)
        ok = [r for r in results if r.get("ok")]
        rows.append({
            "wall": wall, "conc": conc,
            "prefill_tok_s": [r.get("prefill_tok_s") for r in ok],
            "decode_tok_s": [r.get("decode_tok_s") for r in ok],
            "accept": [r.get("accept") for r in ok if r.get("accept")],
            "errors": [r.get("error") for r in results if not r.get("ok")],
        })
    return rows


def current_mode():
    r = subprocess.run(SPARK1 + ["grep -E '^PERSISTENCE=|^PERSIST_STORE_MODE=' ~/glm53-launch/env.tp4.fleet"],
                       capture_output=True, text=True).stdout
    persist = "on" if "PERSISTENCE=on" in r else "off"
    if persist == "off":
        return "off"
    return "eager" if "PERSIST_STORE_MODE=eager" in r else "evict"


def set_mode(mode):
    """mode in {off, eager, evict}. Restart the fleet via the launcher host."""
    if current_mode() == mode:
        print(f"  mode {mode} already active; no restart", flush=True)
        wait_healthy(900)
        return
    def ssh(cmd):
        subprocess.run(SPARK1 + [cmd], check=True, timeout=300)
    if mode == "off":
        ssh("sed -i 's/^PERSISTENCE=on/PERSISTENCE=off/' ~/glm53-launch/env.tp4.fleet")
    else:
        ssh("sed -i 's/^PERSISTENCE=off/PERSISTENCE=on/' ~/glm53-launch/env.tp4.fleet")
        policy = "eager" if mode == "eager" else "evict_only"
        ssh(f"sed -i 's/^PERSIST_STORE_MODE=.*/PERSIST_STORE_MODE={policy}/' ~/glm53-launch/env.tp4.fleet")
    ssh("cd ~/glm53-launch && ./start-tp4.sh down >/dev/null 2>&1; "
        "for h in ${PERSIST_PEER_HOSTS:-}; do [ -n \"$h\" ] && ssh -o BatchMode=yes $h "
        "\"docker ps -q --filter name=glm53-exl3-tp4 | xargs -r docker stop >/dev/null 2>&1\"; done; "
        "docker ps -q --filter name=glm53-exl3-tp4 | xargs -r docker stop >/dev/null 2>&1; sleep 3")
    # Issue the up command without waiting on the ssh session: the
    # launcher backgrounds its own work server-side, and the health poll
    # below is the actual gate. (Waiting hangs: ssh keeps the session
    # open on a stray fd regardless of stdio redirection.)
    subprocess.Popen(SPARK1 + ["cd ~/glm53-launch && setsid nohup sh -c './start-tp4.sh up > /tmp/ab-up.log 2>&1' >/dev/null 2>&1 </dev/null &"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(15)
    wait_healthy(900)


def wait_healthy(budget):
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        try:
            code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                                   "--max-time", "6", URL + "/health"],
                                  capture_output=True, text=True).stdout.strip()
            if code == "200":
                time.sleep(20)
                return
        except Exception:
            pass
        time.sleep(20)
    raise RuntimeError("engine did not become healthy (mode budget exhausted)")


def main():
    sizes = [int(x) for x in (sys.argv[sys.argv.index("--sizes") + 1].split(",")
                              if "--sizes" in sys.argv else ["8192", "65536", "262144"])]
    concs = [int(x) for x in (sys.argv[sys.argv.index("--conc") + 1].split(",")
                              if "--conc" in sys.argv else ["1", "4"])]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2
    modes = sys.argv[sys.argv.index("--modes") + 1].split(",") if "--modes" in sys.argv else ["evict"]
    out = {"matrix": []}
    for mode in modes:
        print(f"=== MODE {mode} ===", flush=True)
        set_mode(mode)
        one(filler("WARMUP", 2000, 1), max_tokens=8)  # settle
        for size in sizes:
            for conc in concs:
                rows = cell(size, conc, reps, seed_base=7000 + hash(mode) % 1000)
                pre = [x for r in rows for x in r["prefill_tok_s"] if x]
                dec = [x for r in rows for x in r["decode_tok_s"] if x]
                acc = [x for r in rows for x in r["accept"] if x]
                rec = {"mode": mode, "size": size, "conc": conc,
                       "prefill_tok_s_median": statistics.median(pre) if pre else None,
                       "decode_tok_s_median": statistics.median(dec) if dec else None,
                       "accept_median": statistics.median(acc) if acc else None,
                       "errors": [e for r in rows for e in r["errors"]]}
                out["matrix"].append(rec)
                print(json.dumps(rec), flush=True)
    print("MATRIX-JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
