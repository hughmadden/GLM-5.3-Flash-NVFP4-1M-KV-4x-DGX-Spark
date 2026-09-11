#!/usr/bin/env python3
"""REPRO: the disk-tier refill never converges.

Models vLLM's real `_maximal_prefix_lookup` contract
(`.../offloading/scheduler.py`, ~line 646) against the real `_Manager`, with a
fake in-process coordinator. No engine, no torch, no network, no disk.

The contract that matters:

    hit_count = 0; defer = False
    for key in keys:
        match lookup(key):          # True=HIT, None=RETRY, False=MISS
            HIT:      hit_count += 1
            RETRY:    defer = True   # note: does NOT break
            MISS:     break
    return hit_count if not defer else None

Two consequences drive this repro:
  * a single RETRY anywhere discards the whole hit count and defers the request
  * the caller then re-runs the *entire* scan from chunk 0 on the next step

Our `_Manager.lookup` charges at most `lookup_keys_per_step` new reservations per
scheduler step and returns RETRY (None) for every other chunk, so the scan can
only ever resolve a prefix if it converges. This script shows whether it does.

Run:  cd <pkg>; PYTHONPATH=. python3 ../../<this file>
"""
import sys
from types import SimpleNamespace as NS

from recipe_persistence.coordinator import Ticket
from recipe_persistence.handlers import DiskLoadStoreSpec
from recipe_persistence.native import _Manager

SIZES = ((28,),)


def key(number):
    # Trailing 4 bytes are the cache group (must be 0 for SIZES with one group);
    # the leading bytes carry the chunk identity.
    return number.to_bytes(4, "big") + (0).to_bytes(4, "big")


def ctx(owner="req"):
    return NS(req_id=owner)


class FakeCoordinator:
    """Holds `present` chunks on the 'disk'; everything else misses.

    Mirrors `RemoteCoordinator`'s lease bookkeeping for the one detail that
    matters: `lease_deadline` reads `_valid`, and `_valid` is populated ONLY by
    `renew` -- never by `reserve_load`. So a freshly reserved ticket has no
    deadline until a renew lands.
    """

    def __init__(self, present, sizes=SIZES, async_mode=False, lease_seconds=300.0,
                 seed_valid_on_reserve=False):
        self.sizes = tuple(tuple(s) for s in sizes)
        self.present = set(present)
        self.live = {}
        self.async_mode = async_mode
        self.lease_seconds = lease_seconds
        self.seed_valid_on_reserve = seed_valid_on_reserve
        self._tickets = {}
        self._valid = {}
        self._retiring = set()
        self.reserve_load_calls = 0
        self.reserve_load_ok = 0
        self.renew_calls = 0
        self.renew_ok = 0

    def _ticket(self, namespace, k, owner):
        token = "t%d" % len(self.live)
        group = int.from_bytes(k[-4:], "big")
        t = Ticket(token, namespace, k, (token,), self.sizes[group], owner, False)
        self.live[token] = t
        self._tickets[token] = t
        return t

    def reserve_load(self, namespace, k, owner):
        self.reserve_load_calls += 1
        if k not in self.present:
            return None
        self.reserve_load_ok += 1
        t = self._ticket(namespace, k, owner)
        if self.seed_valid_on_reserve:
            # THE FIX: a granted reservation carries its lease window with it,
            # so it is valid immediately instead of only after a renew lands.
            self._valid[t.token] = time.monotonic() + self.lease_seconds
        return t

    def reserve_store(self, namespace, k, owner, size_by_rank):
        return None

    # --- async-metadata surface -------------------------------------------
    def lease_deadline(self, ticket):
        """Non-I/O snapshot, exactly as RemoteCoordinator does it."""
        if (ticket.token in self._retiring
                or self._tickets.get(ticket.token) != ticket):
            return None
        return self._valid.get(ticket.token)

    def renew(self, ticket):
        self.renew_calls += 1
        self._valid[ticket.token] = time.monotonic() + self.lease_seconds
        self.renew_ok += 1
        return True

    def release(self, ticket):
        self._retiring.add(ticket.token)
        self._valid.pop(ticket.token, None)
        self.live.pop(ticket.token, None)

    def complete_store(self, ticket, success):
        self.live.pop(ticket.token, None)

    def invalidate(self, namespace, k):
        return None


def maximal_prefix_lookup(manager, keys, context):
    """vLLM's `_maximal_prefix_lookup`, verbatim in behaviour."""
    hit_count = 0
    defer = False
    for k in keys:
        r = manager.lookup(k, context)
        if r is True:
            hit_count += 1
        elif r is None:
            defer = True
        else:
            break
    return hit_count if not defer else None


def run(chunks=2038, on_disk=76, keys_per_step=8, capacity=32768, max_steps=4000,
        async_mode=False, seed_valid=False):
    coordinator = FakeCoordinator(present={key(i) for i in range(on_disk)},
                                  async_mode=async_mode, seed_valid_on_reserve=seed_valid)
    kwargs = {"lookup_keys_per_step": keys_per_step}
    if async_mode:
        kwargs.update(metadata_workers=2, metadata_max_submitted=8,
                      metadata_shutdown_timeout=10.0)
    manager = _Manager(coordinator, "ns", SIZES, capacity, DiskLoadStoreSpec,
                       NS, NS, **kwargs)
    keys = [key(i) for i in range(chunks)]
    context = ctx()
    manager.on_new_request(context)

    resolved_at = None
    history = []
    for step in range(1, max_steps + 1):
        matched = maximal_prefix_lookup(manager, keys, context)
        manager.on_schedule_end()
        if step <= 3 or step % 50 == 0:
            history.append((step, matched, len(manager.loads),
                            coordinator.reserve_load_calls))
        if matched is not None:
            resolved_at = (step, matched)
            break
    manager.shutdown()
    return resolved_at, history, coordinator, manager


if __name__ == "__main__":
    import time
    chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 2038
    on_disk = int(sys.argv[2]) if len(sys.argv) > 2 else 76
    keys_per_step = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    async_mode = "--async" in sys.argv
    seed_valid = "--fix" in sys.argv

    print(f"chunks={chunks} on_disk={on_disk} lookup_keys_per_step={keys_per_step} "
          f"async_metadata={async_mode} seed_valid_on_reserve={seed_valid}")
    resolved, history, coordinator, manager = run(chunks, on_disk, keys_per_step,
                                                 async_mode=async_mode, seed_valid=seed_valid)

    print(f"{'step':>6} {'returned':>9} {'len(loads)':>11} {'reserve_load':>13} {'renews':>7}")
    for step, matched, loads, calls in history:
        print(f"{step:>6} {str(matched):>9} {loads:>11} {calls:>13} {coordinator.renew_calls:>7}")

    print()
    print(f"total reserve_load attempts : {coordinator.reserve_load_calls}")
    print(f"successful leases           : {coordinator.reserve_load_ok}")
    print(f"renew calls / ok            : {coordinator.renew_calls} / {coordinator.renew_ok}")
    print(f"holders of a valid deadline : {len(coordinator._valid)}")
    print(f"tier hit counter            : {manager.tier_chunk_hits}")
    print(f"tier query counter          : {manager.tier_chunk_queries}")
    print()
    if resolved:
        step, matched = resolved
        print(f"RESULT: converged at step {step} with matched={matched}")
    else:
        print("RESULT: NEVER CONVERGED -- vLLM's scan returns None on every step,"
              " so the request defers forever and the hit is never counted")
