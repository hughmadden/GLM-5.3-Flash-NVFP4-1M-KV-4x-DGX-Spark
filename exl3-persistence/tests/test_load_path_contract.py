# SPDX-License-Identifier: Apache-2.0
"""Stub reproduction of the "disk tier stores but never serves" defect.

No engine, no weights, no GPU, no inference. Milliseconds.

WHY THIS FILE EXISTS
--------------------
On the live engine the tier is reached (`lookup_disk_hit`) and `prepare_load` is
never reached, so a fully resolved disk hit is never turned into a transfer
(`kv_offload_load_bytes_total` stays 0). Finding that cost repeated engine
restarts of ~8 minutes each. This file reproduces the same decision boundary
in-process.

WHAT IT MODELS
--------------
vLLM's offloading connector has two relevant methods
(`.../kv_transfer/kv_connector/v1/offloading/scheduler.py`):

1. ``get_num_new_matched_tokens`` -> ``(_lookup(...), bool(_lookup(...)))``
   which returns ``None`` when the backend deferred.

2. ``update_state_after_alloc(request, blocks, num_external_tokens)`` which
   begins::

       if num_external_tokens == 0:
           return

   and only *after* that early return does it call
   ``self.manager.prepare_load(keys_to_load, ...)``.

So ``prepare_load`` is reached **only** when the lookup returns a positive token
count. ``_lookup_complete_chunks`` can return 0 or None even when the backend
found chunks, on any of these paths -- each is modelled below and each is
reported individually, because knowing *which* one fires is the whole point:

  A  ``max_hit_size_tokens - num_computed_tokens < tokens_per_chunk``   -> 0
  B  ``num_hit_chunks == 0``                                           -> 0
  C  ``new_num_hit_tokens < tokens_per_chunk``                         -> 0
  D  ``defer_lookup`` (any chunk returned RETRY)                       -> None

A RETRY from our manager can come from three places, and only one of them was
instrumented on the engine. Two are silent::

    if identity in self.loads:
        if self._async_metadata:
            if self._lease_valid(ticket): return True
            self._renew_requested[...] = ticket; self._dispatch_metadata()
            return None if not self.closed else False     # <-- SILENT RETRY
    ...
    self.loads[identity] = ticket
    if self._async_metadata and not self._lease_valid(ticket):
        self._renew_requested[...] = ticket; self._dispatch_metadata()
        return None if not self.closed else False         # <-- SILENT RETRY

Run directly for a per-scenario report::

    PYTHONPATH=. python3 tests/test_load_path_contract.py
"""
import sys
import time
import unittest
from types import SimpleNamespace as NS

# NOTE: deliberately no process-wide logging configuration here. An earlier
# revision called logging.getLogger("recipe_persistence").setLevel(CRITICAL) at
# import time to quieten the path traces, which suppressed warnings for every
# later test in the same process and broke three unrelated assertions in
# test_native.py. The traces are INFO now, so nothing needs silencing; if that
# ever changes, scope it to a fixture rather than mutating global state.

from recipe_persistence.coordinator import Ticket
from recipe_persistence.handlers import DiskLoadStoreSpec
from recipe_persistence.native import _Manager

SIZES = ((28,),)
TOKENS_PER_CHUNK = 73          # measured: 8,171-token prompt -> 112 chunks
REQUEST_TOKENS = 8171
NUM_CHUNKS = 112


def cdiv(a, b):
    return -(-a // b)


def safe_shutdown(manager):
    """Cleanup must never mask the behaviour under test.

    reset_cache() refuses while transfers are outstanding, which is expected
    here because a successful run deliberately leaves a load prepared."""
    try:
        safe_shutdown(manager)
    except Exception:
        pass


def key(number):
    """Trailing 4 bytes are the cache group (0); leading bytes are identity."""
    return number.to_bytes(4, "big") + (0).to_bytes(4, "big")


def ctx(owner="req"):
    return NS(req_id=owner)


class FakeCoordinator:
    """In-process disk tier.

    ``present`` is the set of chunk indices the tier holds. ``lease_none`` makes
    ``lease_deadline`` report no deadline for tickets, which is how a real
    coordinator behaves for a lease it can no longer vouch for. ``stale_from``
    makes chunks at that index and beyond be granted with an already-expired
    deadline that renew cannot resurrect (the real stale-lease trigger).
    """

    def __init__(self, present, lease_none=False, lease_seconds=300.0,
                 stale_from=None):
        self.sizes = tuple(tuple(s) for s in SIZES)
        self.present = set(present)
        self.lease_none = lease_none
        self.lease_seconds = lease_seconds
        self.stale_from = stale_from
        self._tickets = {}
        self._valid = {}
        self._retiring = set()
        self.live = {}
        self.reserve_load_calls = 0
        self.renew_calls = 0

    def _ticket(self, namespace, k, owner):
        token = "t%d" % len(self.live)
        group = int.from_bytes(k[-4:], "big")
        t = Ticket(token, namespace, k, (token,), self.sizes[group], owner, False)
        self.live[token] = t
        self._tickets[token] = t
        idx = int.from_bytes(k[:4], "big")
        if self.stale_from is not None and idx >= self.stale_from:
            # Granted already expired: renew of an expired lease cannot
            # succeed, exactly like the real rank store.
            self._valid[token] = time.monotonic() - 1.0
        else:
            self._valid[token] = time.monotonic() + self.lease_seconds
        return t

    def reserve_load(self, namespace, k, owner):
        self.reserve_load_calls += 1
        if int.from_bytes(k[:4], "big") not in self.present:
            return None
        t = self._ticket(namespace, k, owner)
        return t

    def reserve_store(self, namespace, k, owner, size_by_rank):
        return None

    # async-metadata surface
    def lease_deadline(self, ticket):
        if self.lease_none or ticket.token in self._retiring:
            return None
        return self._valid.get(ticket.token)

    def renew(self, ticket):
        self.renew_calls += 1
        if self._valid.get(ticket.token, 0.0) <= time.monotonic():
            return False  # expired beyond renew
        self._valid[ticket.token] = time.monotonic() + self.lease_seconds
        return True

    def release(self, ticket):
        self._retiring.add(ticket.token)
        self._valid.pop(ticket.token, None)
        self.live.pop(ticket.token, None)

    def complete_store(self, ticket, success):
        self.live.pop(ticket.token, None)

    def invalidate(self, namespace, k):
        return None


# --------------------------------------------------------------------------
# A faithful model of vLLM's decisive lookup path.
# --------------------------------------------------------------------------

class Deferral(Exception):
    """Raised with the name of the vLLM condition that deferred the request."""


class ZeroTokens(Exception):
    """Raised with the name of a vLLM condition that returned 0 external
    tokens. Unlike a deferral this is TERMINAL: the scheduler proceeds and
    the request runs with 0 offload tokens -- no retry, no deferral."""


def maximal_prefix_lookup(manager, keys, context):
    """vLLM `_maximal_prefix_lookup`: returns hit count, or None to defer."""
    hit_count = 0
    defer = False
    for k in keys:
        r = manager.lookup(k, context)
        if r is True:
            hit_count += 1
        elif r is None:
            defer = True          # does NOT break
        else:
            break                 # MISS
    return None if defer else hit_count


def lookup_complete_chunks(manager, keys, num_computed_tokens, request_tokens,
                           tokens_per_chunk, owner="req"):
    """vLLM `_lookup_complete_chunks`, reduced to the decisive arithmetic.

    A/B/C return 0 (terminal: the request runs with 0 offload tokens); only
    D defers. Keeping that distinction is the point: a 0 is a completed
    decision, a None is a parked request.
    """
    max_hit_size_tokens = request_tokens
    max_hit_size_tokens = min(max_hit_size_tokens,
                              len(keys) * tokens_per_chunk)
    if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
        raise ZeroTokens("A_max_hit_below_one_chunk")

    num_chunks = min(cdiv(max_hit_size_tokens, tokens_per_chunk), len(keys))
    start_chunk_idx = num_computed_tokens // tokens_per_chunk
    window = keys[start_chunk_idx:num_chunks]

    num_hit_chunks = maximal_prefix_lookup(manager, window, ctx(owner))
    if num_hit_chunks == 0:
        raise ZeroTokens("B_num_hit_chunks_zero")
    if num_hit_chunks is None:
        raise Deferral("D_backend_deferred_retry")

    max_hit_size_tokens = min(max_hit_size_tokens,
                              tokens_per_chunk * (start_chunk_idx + num_hit_chunks))
    new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
    if new_num_hit_tokens < tokens_per_chunk:
        raise ZeroTokens("C_new_hit_below_one_chunk")
    return new_num_hit_tokens


def run_scheduler(manager, steps=8, request_tokens=REQUEST_TOKENS,
                  tokens_per_chunk=TOKENS_PER_CHUNK, owner="req",
                  num_computed_tokens=0):
    """Drive the scheduler loop and report whether prepare_load is reached.

    Returns (reached, detail). ``prepare_load`` is spied rather than stubbed so
    the real validation logic still runs. A ZeroTokens outcome is reported as
    ``zero:<name>`` -- the request RUNS with 0 offload tokens; that is a
    completed decision, not a block.
    """
    keys = [key(i) for i in range(cdiv(request_tokens, tokens_per_chunk))]
    reached = {"n": 0, "keys": 0}
    real_prepare = manager.prepare_load

    def spy_prepare_load(k, req_context):
        reached["n"] += 1
        reached["keys"] += len(k)
        return real_prepare(k, req_context)

    manager.prepare_load = spy_prepare_load
    manager.on_new_request(ctx(owner))

    last = None
    for _ in range(steps):
        try:
            num = lookup_complete_chunks(manager, keys, num_computed_tokens,
                                         request_tokens, tokens_per_chunk, owner)
        except ZeroTokens as z:
            manager.on_schedule_end()
            return False, f"0 external tokens ({z}); request runs without a disk load"
        except Deferral as d:
            last = str(d)
            manager.on_schedule_end()
            continue
        # get_num_new_matched_tokens returned a positive count -> the scheduler
        # calls update_state_after_alloc with num_external_tokens > 0, which is
        # the ONLY route to prepare_load.
        manager.update_state_after_alloc_spy = True
        if num > 0:
            start = num_computed_tokens // tokens_per_chunk
            manager.prepare_load(
                tuple(keys[start:start + num // tokens_per_chunk]), ctx(owner))
        manager.on_schedule_end()
        return True, f"prepare_load reached with num_external_tokens={num}"
    return False, f"deferred {steps}x, never reached prepare_load; last: {last}"


class LoadPathContractTests(unittest.TestCase):
    """These should PASS. A failure is the bug, named."""

    def test_hits_lead_to_prepare_load(self):
        """The whole point: if the tier holds the prefix, a load must be prepared."""
        manager = _Manager(FakeCoordinator(present=range(NUM_CHUNKS)),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        reached, detail = run_scheduler(manager)
        safe_shutdown(manager)
        self.assertTrue(reached, f"prepare_load NOT reached -> {detail}")

    def test_hits_lead_to_prepare_load_without_async_metadata(self):
        manager = _Manager(FakeCoordinator(present=range(NUM_CHUNKS)),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048)
        reached, detail = run_scheduler(manager)
        safe_shutdown(manager)
        self.assertTrue(reached, f"prepare_load NOT reached -> {detail}")

    def test_hits_lead_to_prepare_load_with_tiny_budget(self):
        """The shipped value of 8 must not be able to block the load."""
        manager = _Manager(FakeCoordinator(present=range(NUM_CHUNKS)),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=8, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        reached, detail = run_scheduler(manager, steps=40)
        safe_shutdown(manager)
        self.assertTrue(reached, f"prepare_load NOT reached -> {detail}")

    def test_stale_lease_terminates_as_a_clean_miss(self):
        """A coordinator that can never vouch (lease_none) used to livelock:
        every lookup answered RETRY, vLLM discarded the scan, and the
        reservation was never released (BUG-LOAD-PATH-STALE-LEASE).

        Post-fix the RETRYs are bounded: after _STALE_LEASE_MAX_RETRY the
        reservation is released and the chunk answers a clean MISS. The
        request then computes locally instead of hanging -- and in the
        non-degenerate case (a stale TAIL) the found prefix is still loaded
        (test below).
        """
        manager = _Manager(FakeCoordinator(present=range(NUM_CHUNKS), lease_none=True),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        context = ctx("req")
        outcomes = [manager.lookup(key(0), context)
                    for _ in range(manager._STALE_LEASE_MAX_RETRY + 2)]
        safe_shutdown(manager)
        self.assertEqual(outcomes[:manager._STALE_LEASE_MAX_RETRY],
                         [None] * manager._STALE_LEASE_MAX_RETRY)
        self.assertFalse(outcomes[manager._STALE_LEASE_MAX_RETRY])

    def test_stale_tail_still_prepares_the_found_prefix(self):
        """Chunks before a stale chunk keep their hits; the stale chunk
        becomes a clean miss; prepare_load runs for the found prefix. This is
        the actual restore win: a dead lease costs ONE chunk, not the scan.
        """
        STALE = 40
        manager = _Manager(FakeCoordinator(present=range(NUM_CHUNKS),
                                           stale_from=STALE),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        reached, detail = run_scheduler(manager, steps=40)
        safe_shutdown(manager)
        self.assertTrue(reached, f"prepare_load NOT reached -> {detail}")
        self.assertIn(f"num_external_tokens={STALE * TOKENS_PER_CHUNK}", detail)

    def test_leaked_owner_from_an_aborted_request_does_not_starve_a_new_one(self):
        """A request that is aborted, not finished, must not deny a later request.

        on_request_finished() is what forgets a lookup owner, and it raises
        when the request still has prepared loads or stores. A client that
        disconnects mid-deferral therefore leaves its owner in the rotation. If
        the surviving gate then denies the next request its turn, every scan
        retries and prepare_load is never reached -- which is exactly what was
        observed on a fresh engine after a handful of timed-out probes.
        """
        manager = _Manager(FakeCoordinator(present=range(NUM_CHUNKS)),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        keys = [key(i) for i in range(NUM_CHUNKS)]
        # "aborted" request A: takes a turn gate slot and is never finished
        manager.on_new_request(ctx("A"))
        lookup_complete_chunks(manager, keys, 0, REQUEST_TOKENS,
                               TOKENS_PER_CHUNK, "A")
        manager.on_schedule_end()
        # request B must still be served
        reached, detail = run_scheduler(manager, steps=40, owner="B")
        safe_shutdown(manager)
        self.assertTrue(reached, f"prepare_load NOT reached -> {detail}")

    def test_partial_prefix_still_prepares_a_load(self):
        """A tier holding only part of the prefix must still serve that part."""
        manager = _Manager(FakeCoordinator(present=range(40)),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        reached, detail = run_scheduler(manager, steps=40)
        safe_shutdown(manager)
        self.assertTrue(reached, f"prepare_load NOT reached -> {detail}")

    # -- get_num_new_matched_tokens is called with the block-aligned LOCAL
    # -- (GPU prefix cache) hit, not zero (handover §4). These scenarios pin
    # -- how the offload window behaves around a nonzero local hit.

    def test_tier_beyond_local_hit_still_loads(self):
        """The offload window starts AFTER the locally-computed prefix.
        Tier chunks beyond the local hit must still load."""
        computed = 74 * TOKENS_PER_CHUNK
        manager = _Manager(FakeCoordinator(present=range(74, NUM_CHUNKS)),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        reached, detail = run_scheduler(manager, steps=40,
                                        num_computed_tokens=computed)
        safe_shutdown(manager)
        self.assertTrue(reached, f"prepare_load NOT reached -> {detail}")
        # The tail beyond the local hit, clamped to the request length:
        # 8171 - 74*73 = 2769 (37.9 chunks -- the engine clamps, not rounds).
        self.assertIn(f"num_external_tokens={REQUEST_TOKENS - computed}", detail)

    def test_tier_inside_local_hit_is_a_terminal_zero(self):
        """Tier holds ONLY chunks the GPU already computed: nothing NEW to
        load. 0 external tokens is a completed decision (request runs), not a
        deferral and not a defect."""
        manager = _Manager(FakeCoordinator(present=range(37)),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        reached, detail = run_scheduler(manager, steps=8,
                                        num_computed_tokens=74 * TOKENS_PER_CHUNK)
        safe_shutdown(manager)
        self.assertFalse(reached)
        self.assertIn("0 external tokens (B_num_hit_chunks_zero)", detail)

    def test_subchunk_remainder_is_a_terminal_zero(self):
        """A local hit leaving less than one chunk of NEW tokens legitimately
        yields 0 (condition A). This is the handover's sub-chunk-remainder
        hypothesis shape: terminal zero, not a block."""
        computed = REQUEST_TOKENS - TOKENS_PER_CHUNK // 2
        manager = _Manager(FakeCoordinator(present=range(NUM_CHUNKS)),
                           "ns", SIZES, 32768, DiskLoadStoreSpec, NS, NS,
                           lookup_keys_per_step=2048, metadata_workers=2,
                           metadata_max_submitted=8, metadata_shutdown_timeout=10.0)
        reached, detail = run_scheduler(manager, steps=8,
                                        num_computed_tokens=computed)
        safe_shutdown(manager)
        self.assertFalse(reached)
        self.assertIn("0 external tokens (A_max_hit_below_one_chunk)", detail)


def _report():
    """Per-scenario diagnosis, for the 'identify it quickly' workflow."""
    scenarios = [
        ("full prefix, budget 2048, async", dict(present=range(NUM_CHUNKS)),
         dict(lookup_keys_per_step=2048, metadata_workers=2,
              metadata_max_submitted=8, metadata_shutdown_timeout=10.0), 8),
        ("full prefix, budget 2048, sync", dict(present=range(NUM_CHUNKS)),
         dict(lookup_keys_per_step=2048), 8),
        ("full prefix, budget 8 (shipped)", dict(present=range(NUM_CHUNKS)),
         dict(lookup_keys_per_step=8, metadata_workers=2,
              metadata_max_submitted=8, metadata_shutdown_timeout=10.0), 40),
        ("full prefix, never-vouching coordinator (degenerate)",
         dict(present=range(NUM_CHUNKS), lease_none=True),
         dict(lookup_keys_per_step=2048, metadata_workers=2,
              metadata_max_submitted=8, metadata_shutdown_timeout=10.0), 40),
        ("stale tail from chunk 40 (prefix must still load)",
         dict(present=range(NUM_CHUNKS), stale_from=40),
         dict(lookup_keys_per_step=2048, metadata_workers=2,
              metadata_max_submitted=8, metadata_shutdown_timeout=10.0), 40),
        ("partial prefix (40/112 chunks)", dict(present=range(40)),
         dict(lookup_keys_per_step=2048, metadata_workers=2,
              metadata_max_submitted=8, metadata_shutdown_timeout=10.0), 40),
        ("leaked owner A, then request B", dict(present=range(NUM_CHUNKS)),
         dict(lookup_keys_per_step=2048, metadata_workers=2,
              metadata_max_submitted=8, metadata_shutdown_timeout=10.0), 40),
        ("empty tier (control)", dict(present=()),
         dict(lookup_keys_per_step=2048, metadata_workers=2,
              metadata_max_submitted=8, metadata_shutdown_timeout=10.0), 8),
    ]
    print(f"{'scenario':<36} {'prepare_load':<14} detail")
    print("-" * 100)
    for name, coord_kw, mgr_kw, steps in scenarios:
        manager = _Manager(FakeCoordinator(**coord_kw), "ns", SIZES, 32768,
                           DiskLoadStoreSpec, NS, NS, **mgr_kw)
        reached, detail = run_scheduler(manager, steps=steps)
        safe_shutdown(manager)
        print(f"{name:<36} {'REACHED' if reached else 'NOT REACHED':<14} {detail}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        _report()
    else:
        unittest.main(verbosity=2)
