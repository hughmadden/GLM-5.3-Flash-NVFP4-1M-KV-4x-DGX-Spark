#!/usr/bin/env python3
"""Zero-dependency concurrency test runner for recipe_persistence.

Rules this harness enforces:
  * STDLIB ONLY. No pytest, no torch, no vLLM, no engine, no inference. pytest is
    deliberately absent from the runtime image and installing it costs >30 s.
  * NOTHING MAY TAKE MORE THAN 30 s TO LOAD. `--load-report` times every import
    stage and fails if one exceeds LOAD_BUDGET_S.
  * Every test runs with a watchdog: a hung or deadlocked test is reported as
    HANG after `--timeout` seconds instead of stalling the run.
  * Dummy data only. Tests must never touch a real GPU, a real rank, or the network.

Usage:
  python3 conc_runner.py --load-report mod1 mod2 ...      # load budget check
  python3 conc_runner.py --json mod1 mod2 ...             # run every test_* function
  python3 conc_runner.py --json --timeout 20 mod1
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
import traceback

LOAD_BUDGET_S = 30.0
DEFAULT_TEST_TIMEOUT_S = 60.0


def _stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_modules(mods: list[str], report: bool) -> list:
    out, bad = [], []
    for name in mods:
        t0 = time.perf_counter()
        mod = importlib.import_module(name)
        dt = time.perf_counter() - t0
        out.append(mod)
        if report:
            flag = "OVER-BUDGET" if dt > LOAD_BUDGET_S else "ok"
            _stamp(f"load {name:52s} {dt:6.3f}s  {flag}")
        if dt > LOAD_BUDGET_S:
            bad.append((name, dt))
    if report and bad:
        raise SystemExit(
            f"REFUSED: {len(bad)} module(s) exceeded the {LOAD_BUDGET_S:.0f}s load budget: "
            + ", ".join(f"{n}={d:.1f}s" for n, d in bad)
        )
    return out


def discover(mod) -> list[tuple[str, object]]:
    """Every callable named test_* at module level, plus Test* classes' test_* methods."""
    found: list[tuple[str, object]] = []
    for attr in sorted(dir(mod)):
        if attr.startswith("test_") and callable(getattr(mod, attr)):
            found.append((f"{mod.__name__}::{attr}", getattr(mod, attr)))
        elif attr.startswith("Test"):
            cls = getattr(mod, attr)
            if isinstance(cls, type):
                for m in sorted(dir(cls)):
                    if m.startswith("test_") and callable(getattr(cls, m)):
                        found.append((f"{mod.__name__}::{attr}::{m}", getattr(cls(), m)))
    return found


def run_one(name: str, fn, timeout: float) -> dict:
    box: dict = {}

    def body() -> None:
        t0 = time.perf_counter()
        try:
            fn()
            box["ok"] = True
        except BaseException as exc:  # noqa: BLE001 - report anything
            box["ok"] = False
            box["error"] = f"{type(exc).__name__}: {exc}"
            box["traceback"] = traceback.format_exc()[-1500:]
        finally:
            box["seconds"] = round(time.perf_counter() - t0, 3)

    th = threading.Thread(target=body, name=f"test:{name}", daemon=True)
    t0 = time.perf_counter()
    th.start()
    th.join(timeout)
    if th.is_alive():
        return {"test": name, "ok": False, "hang": True,
                "seconds": round(time.perf_counter() - t0, 3),
                "error": f"HANG: still running after {timeout:.0f}s (deadlock or unbounded wait)"}
    box.setdefault("seconds", round(time.perf_counter() - t0, 3))
    out = {"test": name}
    out.update(box)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="+")
    ap.add_argument("--json", action="store_true", help="emit one JSON object per test")
    ap.add_argument("--load-report", action="store_true")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TEST_TIMEOUT_S)
    ap.add_argument("--filter", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    mods = load_modules(args.modules, args.load_report)
    if args.load_report:
        _stamp(f"LOAD BUDGET OK: {time.perf_counter() - t0:.3f}s total for {len(mods)} module(s) "
               f"(budget {LOAD_BUDGET_S:.0f}s per module)")
        return 0

    tests = []
    for m in mods:
        tests.extend(discover(m))
    if args.filter:
        tests = [t for t in tests if args.filter in t[0]]
    if not tests:
        _stamp("no tests discovered")
        return 2

    results = []
    for name, fn in tests:
        r = run_one(name, fn, args.timeout)
        results.append(r)
        if args.json:
            print(json.dumps(r), flush=True)
        else:
            mark = "PASS" if r.get("ok") else ("HANG" if r.get("hang") else "FAIL")
            _stamp(f"{mark} {name} ({r.get('seconds')}s)")
            if not r.get("ok"):
                print(r.get("traceback") or r.get("error"), flush=True)

    passed = sum(1 for r in results if r.get("ok"))
    hung = sum(1 for r in results if r.get("hang"))
    failed = len(results) - passed
    _stamp(f"SUMMARY total={len(results)} pass={passed} fail={failed} hang={hung} "
           f"wall={time.perf_counter() - t0:.2f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
