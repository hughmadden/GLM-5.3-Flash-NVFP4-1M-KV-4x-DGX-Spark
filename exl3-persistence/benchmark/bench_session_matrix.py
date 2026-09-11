#!/usr/bin/env python3
"""bench_session_matrix.py -- cold / warm / concurrent / eviction session matrix.

Purpose: give a defensible picture of how this deployment behaves for *sessions*
rather than for single requests:

  L1 cold-session ladder   one unique session per context size, nothing cached
  L2 warm-session ladder   the same session sent twice; measures the cache path
  L3 concurrent sessions   N distinct sessions in flight, 1..C
  L4 eviction pressure     fill the KV pool with unique sessions until the GPU
                           cache must evict, then re-request an early session
                           while the engine is still busy -- the busy-disk case

Every cell reports time-to-first-token (prefill), decode rate, per-step latency
and the speculative-decoding acceptance scraped from /metrics, plus that cell's
delta for every persistence counter. The point of the persistence deltas is that
"the disk tier did work here" is then a measurement rather than an assertion.

Requests are streamed so TTFT is real. Every request has a bounded timeout and a
timeout is recorded as a failure -- never retried silently, never backfilled.

Usage:
  bench_session_matrix.py --url URL --model ID --out DIR [--lanes L1,L2,L3,L4]
                          [--smoke] [--request-timeout 120]
"""
import argparse, json, os, random, statistics, threading, time, urllib.request

# ---------------------------------------------------------------- prompt build

WORDS = ("ledger", "kelvin", "atlas", "vector", "quartz", "harbour", "nimbus",
         "cobalt", "meridian", "pelican", "tundra", "zephyr", "granite", "orbit")


def make_prompt(seed: int, approx_tokens: int, question: str) -> str:
    """Unique per (seed); ~approx_tokens long. Uniqueness defeats prefix cache."""
    rnd = random.Random(seed)
    n = max(8, approx_tokens // 5)
    body = " ".join(f"{rnd.choice(WORDS)}{rnd.randrange(10**9)}" for _ in range(n))
    return f"[session {seed}] {body}\n\n{question}"


Q_LONG = "Summarise the identifiers above in one short sentence, then reply DONE."
Q_SHORT = "Reply with exactly the word DONE."

# ------------------------------------------------------------------- metrics

METRIC_KEYS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:kv_offload_tiering_chunk_queries_total",
    "vllm:kv_offload_tiering_chunk_hits_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_allocation_failure_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
)
SHORT = {k: (k.replace("vllm:kv_offload_tiering_", "tier_")
              .replace("vllm:kv_offload_", "off_")
              .replace("vllm:external_prefix_cache_", "ext_")
              .replace("vllm:prefix_cache_", "prefix_")
              .replace("_total", "")) for k in METRIC_KEYS}


def wait_idle(url, timeout=300.0, poll=3.0):
    """Block until the engine reports nothing running. Bounded, never silent."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            m = scrape(url)
        except Exception:  # noqa: BLE001
            time.sleep(poll); continue
        if m.get("running", 1) == 0:
            return True
        time.sleep(poll)
    return False


def scrape(url: str) -> dict:
    txt = urllib.request.urlopen(url + "/metrics", timeout=20).read().decode()
    out = {}
    for line in txt.splitlines():
        for k in METRIC_KEYS:
            if line.startswith(k) and "transfer_type" not in line:
                out[k] = float(line.split()[-1])
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["drafted"] = float(line.split()[-1])
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = float(line.split()[-1])
        if line.startswith("vllm:num_requests_running"):
            out["running"] = float(line.split()[-1])
    return out


# ------------------------------------------------------------------ requests


def stream_request(url, model, prompt, max_tokens, timeout):
    """One streamed completion. Returns a timing dict; never raises."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    first_t = None
    last_t = None
    events = 0
    chunks = 0
    usage = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            events += 1
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    now = time.time() - t0
                    if ttft is None:
                        ttft = now
                        first_t = now
                    last_t = now
                    chunks += 1
        total = time.time() - t0
        ct = (usage or {}).get("completion_tokens", chunks)
        return {"ok": True, "ttft": ttft, "total": total, "completion_tokens": ct,
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "events": events, "content_deltas": chunks,
                "decode_span": (last_t - first_t) if (first_t is not None and last_t is not None) else None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": type(e).__name__, "total": time.time() - t0,
                "ttft": ttft, "completion_tokens": chunks, "events": events,
                "content_deltas": chunks, "decode_span": None}


def run_wave(url, model, prompts, max_tokens, timeout):
    """Run one prompt per thread; return per-request results in order."""
    out = [None] * len(prompts)
    def worker(i):
        out[i] = stream_request(url, model, prompts[i], max_tokens, timeout)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, time.time() - t0


# -------------------------------------------------------------------- cells


def summarise(results, before, after, wall, label, running=None):
    ok = [r for r in results if r["ok"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    # A response delivered in one burst has a near-zero span, and dividing a
    # token count by it yields millions of tok/s. That is a measurement artefact,
    # not a result, so a span below MIN_DECODE_SPAN is reported as not measurable
    # rather than as a number.
    MIN_DECODE_SPAN = 0.25
    dec = []
    for r in ok:
        span = r.get("decode_span")
        if span is not None and span >= MIN_DECODE_SPAN and r["completion_tokens"] > 1:
            dec.append((r["completion_tokens"] - 1) / span)
    toks = sum(r["completion_tokens"] for r in ok)
    acc = None
    d = after.get("drafted", 0) - before.get("drafted", 0)
    a = after.get("accepted", 0) - before.get("accepted", 0)
    if d > 0:
        acc = a / d
    ms_step = None
    if acc is not None and toks and wall:
        accept_len = 1 + 7 * acc
        conc = max(1, running if running is not None else len(results))
        steps = toks / (accept_len * conc)
        if steps > 0:
            ms_step = 1000.0 / steps
    ptoks = [r.get("prompt_tokens") for r in ok if r.get("prompt_tokens")]
    row = {
        "cell": label,
        "requests": len(results),
        "ok": len(ok),
        "timeouts": sum(1 for r in results if not r["ok"]),
        "wall_s": round(wall, 2),
        "prompt_tokens": int(statistics.mean(ptoks)) if ptoks else None,
        # Always-valid aggregate: total emitted tokens over wall time.
        "agg_output_tok_s": round(toks / wall, 2) if wall > 0 and toks else None,
        "ttft_mean_s": round(statistics.mean(ttfts), 3) if ttfts else None,
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_max_s": round(max(ttfts), 3) if ttfts else None,
        # On this deployment the response is delivered in one burst, so the
        # first content delta lands at the very end and TTFT is effectively the
        # prefill latency. Reported as such, and the burst is flagged by
        # decode_measurable=False rather than hidden.
        "prefill_tok_s": round((ok[0].get("prompt_tokens") or 0) / statistics.mean(ttfts), 1)
                         if ttfts and ok and ok[0].get("prompt_tokens") else None,
        "decode_tok_s_per_stream": round(statistics.mean(dec), 1) if dec else None,
        "agg_decode_tok_s": round(sum(dec), 1) if dec else None,
        "accept_ratio": round(acc, 3) if acc is not None else None,
        "derived_ms_per_step": round(ms_step, 1) if ms_step else None,
        "out_tokens": toks,
        "sse_events": statistics.median([r.get("events") or 0 for r in ok]) if ok else None,
        "decode_span_s": round(statistics.median(
            [r["decode_span"] for r in ok if r.get("decode_span") is not None]), 3)
            if any(r.get("decode_span") is not None for r in ok) else None,
        "decode_measurable": bool(dec),
    }
    for k in METRIC_KEYS:
        delta = after.get(k, 0) - before.get(k, 0)
        if delta:
            row[SHORT[k]] = delta
    return row


COLUMNS = ["cell", "requests", "ok", "timeouts", "wall_s", "prompt_tokens", "agg_output_tok_s",
           "ttft_mean_s", "ttft_p50_s",
           "ttft_max_s", "prefill_tok_s", "decode_tok_s_per_stream", "agg_decode_tok_s",
           "accept_ratio", "derived_ms_per_step", "out_tokens",
           "sse_events", "decode_span_s", "decode_measurable",
           "tier_chunk_queries", "tier_chunk_hits", "off_store_bytes", "off_load_bytes",
           "off_allocation_failure", "ext_hits", "ext_queries", "prefix_hits", "prefix_queries"]


def emit(rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "session-matrix.tsv"), "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in COLUMNS) + "\n")
    with open(os.path.join(outdir, "session-matrix.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print()
    print("\t".join(COLUMNS))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in COLUMNS))


# --------------------------------------------------------------------- lanes


def lane_cold(url, model, ctxs, mt, timeout, rows, reps=1):
    for ctx in ctxs:
        prompts = [make_prompt(10_000 + ctx * 100 + i, ctx, Q_SHORT) for i in range(reps)]
        b = scrape(url)
        res, wall = run_wave(url, model, prompts, mt, timeout)
        a = scrape(url)
        rows.append(summarise(res, b, a, wall, f"L1-cold-ctx{ctx}"))


def lane_warm(url, model, ctxs, mt, timeout, rows):
    for ctx in ctxs:
        p = make_prompt(20_000 + ctx, ctx, Q_SHORT)
        for i in (1, 2):
            b = scrape(url)
            res, wall = run_wave(url, model, [p], mt, timeout)
            a = scrape(url)
            rows.append(summarise(res, b, a, wall, f"L2-warm-ctx{ctx}-pass{i}"))


def lane_concurrent(url, model, ctx, levels, mt, timeout, rows):
    for c in levels:
        prompts = [make_prompt(30_000 + c * 1000 + i, ctx, Q_SHORT) for i in range(c)]
        b = scrape(url)
        res, wall = run_wave(url, model, prompts, mt, timeout)
        a = scrape(url)
        rows.append(summarise(res, b, a, wall, f"L3-conc-ctx{ctx}-C{c}"))


def lane_eviction(url, model, ctx, n_sessions, conc, mt, timeout, rows,
                  fill_budget=900.0):
    """Fill with unique sessions, then re-request the first while still busy.

    Phase A runs unique sessions at `conc` to push the GPU KV pool toward
    eviction. Phase B re-requests session 0; if the disk tier can serve it, that
    shows up as load bytes and external hits.

    The fill is hard-bounded by `fill_budget` seconds. Without that bound a
    deferral storm makes each request hold its worker for the full socket
    timeout and the phase would run for tens of minutes; the budget keeps this
    lane honest about being time-boxed rather than quietly truncating a number.
    """
    sessions = [make_prompt(40_000 + i, ctx, Q_SHORT) for i in range(n_sessions)]
    results = []
    lock = threading.Lock()
    idx = [0]
    deadline = time.time() + fill_budget

    def next_i():
        with lock:
            idx[0] += 1
            return idx[0] - 1

    def worker():
        while True:
            if time.time() >= deadline:
                with lock:
                    results.append({"ok": False, "err": "fill_budget_exhausted",
                                    "ttft": None, "total": 0.0, "completion_tokens": 0})
                return
            i = next_i()
            if i >= n_sessions:
                return
            r = stream_request(url, model, sessions[i], mt, timeout)
            r["session"] = i
            with lock:
                results.append(r)

    b = scrape(url)
    t0 = time.time()
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall_a = time.time() - t0
    a = scrape(url)
    started = sum(1 for r in results if r.get("err") != "fill_budget_exhausted")
    rows.append(summarise(results, b, a, wall_a,
                          f"L4-fill-ctx{ctx}-n{n_sessions}-C{conc}", running=conc))
    print(f"    fill: started={started}/{n_sessions} wall={wall_a:.0f}s "
          f"ok={sum(1 for r in results if r['ok'])}")

    # Phase B: the re-request, against the state the fill left behind
    b2 = scrape(url)
    res, wall_b = run_wave(url, model, [sessions[0]], mt, timeout)
    a2 = scrape(url)
    rows.append(summarise(res, b2, a2, wall_b, f"L4-refill-ctx{ctx}"))
    print(f"    refill: {res[0].get('total', 0):.1f}s ok={res[0]['ok']} "
          f"err={res[0].get('err')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lanes", default="L1,L2,L3,L4")
    ap.add_argument("--request-timeout", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evict-ctx", type=int, default=32768,
                    help="target context per session in the eviction fill")
    ap.add_argument("--evict-n", type=int, default=128,
                    help="unique sessions in the eviction fill")
    ap.add_argument("--evict-conc", type=int, default=4)
    ap.add_argument("--evict-budget", type=float, default=900.0,
                    help="hard wall-clock budget for the fill phase, seconds")
    a = ap.parse_args()

    if a.smoke:
        ctxs = [4096]
        conc_levels = [1, 2]
        evict = (4096, 8, 2)
    else:
        ctxs = [4096, 16384, 32768, 65536]
        conc_levels = [1, 2, 3, 4]
        # The GPU KV pool holds ~3.4M tokens at this configuration, so the fill
        # must exceed that before eviction is even possible. Sizing is exposed on
        # the command line because the first attempt (128 x 32768) did NOT reach
        # the pool within its budget and therefore could not demonstrate
        # eviction: 37 of 128 sessions started, ~1.7M tokens.
        evict = (a.evict_ctx, a.evict_n, a.evict_conc)

    lanes = a.lanes.split(",")
    rows = []
    print(f"# session matrix  url={a.url} model={a.model} lanes={lanes} "
          f"max_tokens={a.max_tokens} request_timeout={a.request_timeout}s")
    print("# evidence lane: controlled performance. Infrastructure failures "
          "(timeouts, deferrals) are recorded in the timeouts column, never "
          "dropped and never retried into the sample.")
    if not wait_idle(a.url, timeout=300.0):
        print("# WARNING: engine was not idle within 300s; numbers may be contended")
    t0 = time.time()
    # Lanes run in the order requested. Order matters: the warm lane exercises
    # the cache-hit path, which on this build can leave the engine holding
    # deferred requests, so it is deliberately runnable last (`--lanes
    # L1,L3,L4,L2`) to keep the cold and concurrency numbers clean. Results are
    # flushed after every lane so a later stall cannot discard earlier cells.
    for lane in lanes:
        if lane == "L1":
            print("## L1 cold-session ladder")
            lane_cold(a.url, a.model, ctxs, a.max_tokens, a.request_timeout, rows)
        elif lane == "L2":
            print("## L2 warm-session ladder")
            lane_warm(a.url, a.model, ctxs, a.max_tokens, a.request_timeout, rows)
        elif lane == "L3":
            print("## L3 concurrent sessions")
            lane_concurrent(a.url, a.model, 32768 if not a.smoke else 4096,
                            conc_levels, a.max_tokens, a.request_timeout, rows)
        elif lane == "L4":
            print("## L4 eviction / busy-disk refill")
            lane_eviction(a.url, a.model, *evict, a.max_tokens, a.request_timeout, rows,
                          fill_budget=a.evict_budget)
        else:
            print(f"## unknown lane {lane!r} -- skipped")
        emit(rows, a.out)          # incremental: survive a later stall
        print(f"# lane {lane} complete, {len(rows)} cells so far, "
              f"elapsed {time.time()-t0:.0f}s")
    print(f"# total wall {time.time()-t0:.0f}s")
    emit(rows, a.out)


if __name__ == "__main__":
    main()
