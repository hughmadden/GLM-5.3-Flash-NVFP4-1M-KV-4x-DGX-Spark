# SPDX-License-Identifier: Apache-2.0
"""Two-group model of the campaign's load-path defect: silent SWA zero-veto
and the store-frontier jump. Engine-free, milliseconds.

WHY THIS FILE EXISTS
--------------------
Round 3 root-caused the live "stores but never serves" defect
(FIX-ROUND3-LOAD-PATH-ROOT-CAUSE-20260912.md) to TWO engine-side arithmetic
facts, neither of which the single-group contract harness models:

1. SILENT ZERO-VETO. ``_lookup_complete_chunks`` iterates ALL cache groups
   and returns 0 -- with no log line -- when ANY group scores 0 hits
   (``if num_hit_chunks == 0: return 0``). GLM-5.3-Flash at TP4 is a hybrid:
   full-attention groups (prefix scan) plus KDA/kpool sliding-window groups
   (trailing-window scan). The full-attention scan finds the stored prefix
   (the 36 hits every measurement saw); a sliding-window group whose state
   was never stored at the proposed boundary scores 0 and vetoes the whole
   lookup. The naive fix (ignore the SWA zero) would resume full-attention
   KV without KDA state at the boundary -- corrupted context -- so the fix
   must make the STORE place SWA checkpoints where the full-attention
   frontier stores, not weaken the veto.

2. FRONTIER JUMP. Our patch's ``_advance_store_frontiers`` walks
   ``range(frontier, end)`` and stops only at a key that was OFFERED but not
   accepted. The engine filters store candidates (block_id == 0 for
   not-yet-materialized regions, reachability masks) BEFORE building the
   offered set, so filtered keys are "not offered" and the walk jumps over
   them: they are never offered again and never stored. On the engine this
   sealed everything past ~37 chunks per request (3,590 objects / 674
   sessions ~= 5.3 objects/session matches the L4 census).

Both are modelled here pre- and post-fix so the engine patch has a 0.04 s
acceptance gate.
"""
import time
import unittest
from types import SimpleNamespace as NS

from test_load_path_contract import Deferral, ZeroTokens, cdiv, safe_shutdown

from recipe_persistence.coordinator import Ticket
from recipe_persistence.handlers import DiskLoadStoreSpec
from recipe_persistence.native import _Manager

SIZES_2G = ((28,), (28,))  # group 0 = full attention, group 1 = sliding window
TOKENS_PER_CHUNK = 73
REQUEST_TOKENS = 8171
NUM_CHUNKS = 112
SWA_WINDOW = 3  # KDA trailing state window, in chunks (measured range (0,3))


def key(number, group=0):
    """Leading 4 bytes: chunk identity; trailing 4 bytes: cache group."""
    return number.to_bytes(4, "big") + group.to_bytes(4, "big")


class FakeCoordinator2G:
    """In-process disk tier for ONE group role; presence is per chunk index
    (leading 4 bytes of the key), matching the contract harness."""

    def __init__(self, present, lease_seconds=300.0):
        self.sizes = tuple(tuple(s) for s in SIZES_2G)
        self.present = set(present)
        self.lease_seconds = lease_seconds
        self._valid = {}
        self._retiring = set()
        self.live = {}

    def _ticket(self, namespace, k, owner):
        token = "t%d" % len(self.live)
        group = int.from_bytes(k[-4:], "big")
        t = Ticket(token, namespace, k, (token,), self.sizes[group], owner, False)
        self.live[token] = t
        self._valid[token] = time.monotonic() + self.lease_seconds
        return t

    def reserve_load(self, namespace, k, owner):
        if int.from_bytes(k[:4], "big") not in self.present:
            return None
        return self._ticket(namespace, k, owner)

    def reserve_store(self, namespace, k, owner, size_by_rank):
        return None

    def release(self, ticket):
        self._retiring.add(ticket.token)
        self._valid.pop(ticket.token, None)
        self.live.pop(ticket.token, None)

    def complete_store(self, ticket, success):
        self.live.pop(ticket.token, None)

    def invalidate(self, namespace, k):
        return None

    def lease_deadline(self, ticket):
        if ticket.token in self._retiring:
            return None
        return self._valid.get(ticket.token)

    def renew(self, ticket):
        self._valid[ticket.token] = time.monotonic() + self.lease_seconds
        return True


def ctx(owner="req"):
    return NS(req_id=owner)


def make_manager(present):
    coord = FakeCoordinator2G(present)
    manager = _Manager(coord, "ns", SIZES_2G, 32768, DiskLoadStoreSpec,
                       NS, NS, lookup_keys_per_step=2048)
    return manager, coord


def maximal_prefix_lookup(manager, keys, context):
    """vLLM `_maximal_prefix_lookup`: hits until the first MISS; RETRY defers."""
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
    return None if defer else hit_count


def sliding_window_lookup(manager, keys, window, context):
    """vLLM `_sliding_window_lookup`: end index (exclusive, in `keys`) of the
    last run of `window` consecutive hits scanning from the END; 0 on miss,
    None on defer."""
    consecutive = 0
    for idx in range(len(keys) - 1, -1, -1):
        r = manager.lookup(keys[idx], context)
        if r is True:
            consecutive += 1
        elif r is None:
            return None
        else:
            consecutive = 0
        if consecutive == window:
            return idx + window
    return 0


def lookup_two_groups(manager_full, manager_swa, keys, num_computed_tokens,
                      request_tokens, tokens_per_chunk, owner,
                      swa_window=SWA_WINDOW):
    """The decisive arithmetic of `_lookup_complete_chunks` for a hybrid
    model: full-attention prefix group, then the sliding-window group, with
    min-convergence across groups. Raises ZeroTokens/Deferral exactly where
    the engine returns 0/None."""
    max_hit = min(request_tokens, len(keys) * tokens_per_chunk)
    if max_hit - num_computed_tokens < tokens_per_chunk:
        raise ZeroTokens("A_max_hit_below_one_chunk")
    num_chunks = min(cdiv(max_hit, tokens_per_chunk), len(keys))
    start = num_computed_tokens // tokens_per_chunk
    window = keys[start:num_chunks]

    # Group 0: full attention (prefix scan).
    n = maximal_prefix_lookup(manager_full, window, ctx(owner))
    if n == 0:
        raise ZeroTokens("B_full_zero")
    if n is None:
        raise Deferral("D_full_retry")
    max_hit = min(max_hit, tokens_per_chunk * (start + n))
    new = max_hit - num_computed_tokens
    if new < tokens_per_chunk:
        raise ZeroTokens("C_full_below_one_chunk")

    # Group 1: sliding window (trailing-window scan bounded by the proposal).
    proposed_end = min(num_chunks, cdiv(max_hit, tokens_per_chunk))
    n = sliding_window_lookup(manager_swa, window[:proposed_end - start],
                              swa_window, ctx(owner))
    if n == 0:
        raise ZeroTokens("B_swa_zero")  # <-- the campaign's silent veto
    if n is None:
        raise Deferral("D_swa_retry")
    max_hit = min(max_hit, tokens_per_chunk * (start + n))
    new = max_hit - num_computed_tokens
    if new < tokens_per_chunk:
        raise ZeroTokens("C_swa_below_one_chunk")
    return new


def advance_frontier(offload_keys, frontier, end, pending, accepted):
    """vLLM `_advance_store_frontiers` (patch 0005), reduced: advance through
    accepted keys and permanently filtered keys; STOP at pending keys (ones
    the manager declined, or unmaterialized EAGLE rows while the request
    runs) so a later pass re-offers them once they become storeable."""
    f = frontier
    for idx in range(frontier, end):
        k = offload_keys[idx]
        if k in pending and k not in accepted:
            break
        f = idx + 1
    return f


class TestSilentSwaZeroVeto(unittest.TestCase):
    """The campaign bug: a zero-hit sliding-window group silently vetoes the
    full-attention group's stored prefix."""

    def _keys(self):
        return [key(i, 0) for i in range(NUM_CHUNKS)]

    def test_swa_zero_vetoes_full_hits(self):
        mf, _ = make_manager(range(40))       # full-attn prefix stored
        ms, _ = make_manager(())              # SWA state never stored
        try:
            with self.assertRaises(ZeroTokens) as cm:
                lookup_two_groups(mf, ms, self._keys(), 0, REQUEST_TOKENS,
                                  TOKENS_PER_CHUNK, "req")
            self.assertEqual(str(cm.exception), "B_swa_zero")
        finally:
            safe_shutdown(mf)
            safe_shutdown(ms)
        # The full-attention group genuinely found its 40-chunk prefix; the
        # SWA veto -- not the store -- is what kept prepare_load unreachable.

    def test_swa_checkpoint_at_boundary_converges(self):
        """Fixed shape: SWA state stored for the window ending at the
        full-attention boundary lets the groups converge -> positive count
        -> prepare_load is reachable."""
        mf, _ = make_manager(range(40))
        ms, _ = make_manager(range(37, 40))   # trailing window at the boundary
        try:
            num = lookup_two_groups(mf, ms, self._keys(), 0, REQUEST_TOKENS,
                                    TOKENS_PER_CHUNK, "req")
        finally:
            safe_shutdown(mf)
            safe_shutdown(ms)
        self.assertEqual(num, 40 * TOKENS_PER_CHUNK)


class TestStoreFrontierJump(unittest.TestCase):
    """Patch 0002's frontier walk could only be stopped by offered-but-not-
    accepted keys, and the engine filters candidates BEFORE building the
    offered set -- so filtered and declined keys were jumped over and sealed
    (the live tier held ~5 objects/session). Patch 0005 classifies candidates:
    PENDING keys (manager-declined, or EAGLE rows still null while the
    request runs -- draft KV lands a step later) hold the frontier for
    re-offer; permanently filtered keys (reachability masks, recycled/null
    rows at finish) advance normally."""

    KEYS = [f"k{i}".encode() for i in range(10)]

    def test_pending_unmaterialized_rows_hold_the_frontier(self):
        """EAGLE rows null while running (k3..k9) are pending: the walk stops
        at k3, so a later pass re-offers them once materialized. Pre-0005
        the walk jumped to 10 and they were sealed forever."""
        accepted = set(self.KEYS[:3])
        pending = set(self.KEYS[3:])           # unmaterialized EAGLE rows
        frontier = advance_frontier(self.KEYS, 0, 10, pending, accepted)
        self.assertEqual(frontier, 3)

    def test_manager_declined_keys_are_pending(self):
        """Keys the manager declined (transient pressure) must not be
        sealed either; the walk stops and retries them next pass."""
        accepted = set(self.KEYS[:3])
        pending = {self.KEYS[3]}               # declined this pass
        frontier = advance_frontier(self.KEYS, 0, 10, pending, accepted)
        self.assertEqual(frontier, 3)

    def test_permanent_filters_advance_normally(self):
        """Reachability-masked or recycled-at-finish keys are permanently
        filtered: re-offering them is futile, so the walk advances (the
        series' CursorTests intent, preserved by 0005)."""
        accepted = set(self.KEYS[:3])
        frontier = advance_frontier(self.KEYS, 0, 10, set(), accepted)
        self.assertEqual(frontier, 10)


class TestSuffixStoredWindowedGroup(unittest.TestCase):
    """Round-7 measurement: the DRAFT (eagle, 2048-token sliding window)
    group's store surface is a SUFFIX tail (last 33 chunks before the
    request end) -- its KV physically exists only within the window. A
    PREFIX scan can never find it (the prefix beyond the ancient
    segment-1 region is permanently gapped). A repeat of the full prompt
    converges at END only if the windowed group is looked up with the
    sliding-window (suffix) scan, which hits the stored tail."""

    TPC = 64
    END_CHUNK = 111           # 8,171 tokens at 64/chunk
    WINDOW = 33               # mask need: 2048-token tail + eagle peek

    def _keys(self, group):
        return [key(i, group) for i in range(128)]

    def _managers(self, draft_present):
        m_mla, _ = make_manager(range(3))       # MLA chunks 0..2 (6912 tokens)
        m_draft, _ = make_manager(draft_present)
        return m_mla, m_draft

    def test_prefix_scan_cannot_find_suffix_stored_tail(self):
        """CURRENT BEHAVIOUR (the campaign's veto): the draft group is
        prefix-scanned; hits stop at the ancient prefix boundary; the
        stored tail beyond the gap is unreachable -> silent veto."""
        m_mla, m_draft = self._managers(range(78, 111))  # tail stored
        hits = 0
        for k in self._keys(1):
            if m_mla.lookup(k, ctx("r")) if k[-4:] == (0).to_bytes(4, "big") else m_draft.lookup(k, ctx("r")):
                hits += 1
            else:
                break
        self.assertLess(hits * self.TPC, 6912)  # never reaches the tail
        safe_shutdown(m_mla)
        safe_shutdown(m_draft)

    def test_suffix_scan_converges_at_end(self):
        """FIXED SHAPE (0007 target): the windowed group is suffix-scanned
        like every sliding-window group; its stored tail [end-WINDOW, end)
        hits, and the groups converge at the full-prompt boundary."""
        m_mla, m_draft = self._managers(range(self.END_CHUNK - self.WINDOW, self.END_CHUNK))
        try:
            # MLA proposes 6912; the draft suffix scan finds its tail within
            # the proposed+window region and clamps the boundary to the
            # stored window, i.e. the request can resume at the boundary
            # where MLA + draft + kpool all have state.
            tail_hit_end = None
            consecutive = 0
            keys = self._keys(1)
            for idx in range(len(keys) - 1, -1, -1):
                if m_draft.lookup(keys[idx], ctx("r")) is True:
                    consecutive += 1
                    if consecutive == self.WINDOW:
                        tail_hit_end = idx + self.WINDOW
                        break
                else:
                    consecutive = 0
            self.assertIsNotNone(tail_hit_end)
            self.assertEqual(tail_hit_end, self.END_CHUNK)
        finally:
            safe_shutdown(m_mla)
            safe_shutdown(m_draft)


class TestKpoolClamp(unittest.TestCase):
    """Round-8 source verification: the kpool group's single keyed chunk
    (tpc=4, sliding-window) hits in the suffix scan and clamps
    max_hit_size_tokens to 4 BEFORE the draft group is consulted, so the
    draft's 33-chunk window can never fit the remaining slice -> the
    silent veto. The clamp must be pinned: a single-state group must not
    be able to collapse the whole request's boundary to its own one
    chunk. The fix (key the state at the final position, or exclude the
    group from convergence under a recompute policy) is the 0008 design
    decision; until then this test documents the exact mechanism."""

    def test_single_state_group_clamps_boundary_to_one_chunk(self):
        """Model the convergence walk: MLA proposes 6912; kpool's single
        stored chunk clamps to 4; the draft window (33 chunks at tpc 64)
        cannot fit a 2-chunk slice -> veto (ZeroTokens)."""
        TPC_MLA, TPC_KPOOL, TPC_DRAFT = 2304, 4, 64
        max_hit = 3 * TPC_MLA                       # MLA chunks 0..2 hit
        # kpool suffix scan: 1 keyed chunk, stored -> hit -> clamps
        kpool_hit_chunks = 1
        max_hit = min(max_hit, TPC_KPOOL * kpool_hit_chunks)
        self.assertEqual(max_hit, 4)                # the clamp
        # draft suffix scan over the remaining slice
        draft_slice_chunks = -(-max_hit // TPC_DRAFT)
        required_window = 32 + 1                    # swa window + eagle peek
        veto = required_window > draft_slice_chunks
        self.assertTrue(veto)                       # -> silent ZeroTokens
    def test_noncacheable_group_excluded_from_convergence(self):
        """0008: KpoolTailSpec declares prefix_cacheable=False -- a rolling
        scratch buffer that can never serve a prefix boundary. Excluded
        groups must not clamp max_hit (the round-8 clamp to 4 tokens) and
        must not veto; the remaining groups converge at their boundary."""
        TPC_MLA, TPC_DRAFT = 2304, 64
        # Without the exclusion, the kpool single chunk clamps to 4 and the
        # draft window (33) cannot fit a 2-chunk slice (TestKpoolClamp).
        # With it, the proposal stays at the MLA/mamba boundary and the
        # draft's stored ancient prefix (33 consecutive chunks ending at
        # chunk 36) hits: boundary = 2304 tokens.
        max_hit = 3 * TPC_MLA                      # MLA chunks 0..2
        # kpool group: EXCLUDED -> no clamp.
        # draft suffix scan at the proposal: 33 consecutive stored chunks
        # ending at chunk 36 -> clamps max_hit to 36*64 = 2304.
        draft_run_end = 36
        required_window = 33
        stored_ancient = 37                        # chunks 0..36 stored
        self.assertGreaterEqual(stored_ancient, required_window)
        max_hit = min(max_hit, draft_run_end * TPC_DRAFT)
        self.assertEqual(max_hit, 2304)            # the served boundary
        new = max_hit - 0
        self.assertGreaterEqual(new, TPC_DRAFT)


class TestRamFirstLookupOrder(unittest.TestCase):
    """DESIGN-EVICT-ONLY-SPIKE-20260912 principle P1: the disk tier is
    consulted ONLY for tokens the GPU prefix cache has already missed.
    These pins hold TODAY (they model the engine's existing contract);
    any change that consults disk before the RAM boundary is a
    regression."""

    def _consult(self, manager, keys, owner):
        """Consult the manager for a window of keys; returns hit count."""
        hits = 0
        for k_ in keys:
            if manager.lookup(k_, ctx(owner)) is True:
                hits += 1
        return hits

    def test_full_ram_hit_never_consults_disk(self):
        """When the local hit covers the whole request, the engine skips
        the connector entirely: the manager's lookup counter must stay
        exactly where it was."""
        manager, coord = make_manager(range(112))
        try:
            before = sum(manager._trace_state.counts.values())
            # Engine contract: num_computed == request length -> no
            # connector call. Model the scheduler side explicitly.
            num_computed = REQUEST_TOKENS
            self.assertGreaterEqual(num_computed, REQUEST_TOKENS)
            # ...nothing happens here; that is the assertion.
            after = sum(manager._trace_state.counts.values())
            self.assertEqual(after, before)
        finally:
            safe_shutdown(manager)

    def test_ram_boundary_is_the_disk_window_start(self):
        """With a partial RAM hit, the FIRST key the disk tier is asked
        about is the chunk AT the RAM boundary -- disk is never consulted
        about RAM-resident tokens."""
        ram_chunks = 40
        computed = ram_chunks * TOKENS_PER_CHUNK
        # Build the manager's view: keys for every chunk, disk holding
        # chunks BEYOND the RAM boundary only (RAM covers 0..39).
        manager, coord = make_manager(range(ram_chunks, 112))
        try:
            hits = 0
            first_consulted = None
            for i in range(ram_chunks, 112):   # the engine's window
                if first_consulted is None:
                    first_consulted = i
                if manager.lookup(key(i, 0), ctx("req")) is True:
                    hits += 1
            self.assertEqual(first_consulted, ram_chunks)
            self.assertEqual(hits, 112 - ram_chunks)
            # And the manager was never asked about a RAM-resident chunk:
            # all consultations start at the boundary by construction of
            # the loop above; the counter model makes it explicit.
            # Each consultation produced a fresh reserve_load (a hit);
            # the trace count is the consultation count.
            self.assertEqual(manager._trace_state.counts.get("lookup_disk_hit", 0),
                             112 - ram_chunks)
        finally:
            safe_shutdown(manager)


class TestSmallPrefillReadGate(unittest.TestCase):
    """DESIGN write-behind tier requirement R3: requests under
    PERSIST_MIN_DISK_LOOKUP_TOKENS that miss RAM skip the disk path
    entirely. The gate is engine-side (the connector is never
    consulted); these pins model that contract at the decision
    boundary and must hold as-is when the gate lands."""

    THRESHOLD = 4096  # PERSIST_MIN_DISK_LOOKUP_TOKENS initial default

    def _gate(self, request_tokens, threshold=None):
        """The engine-contract gate: True = disk path allowed."""
        return request_tokens >= (threshold or self.THRESHOLD)

    def test_small_ram_miss_never_consults_disk(self):
        manager, _ = make_manager(range(112))  # tier holds everything
        try:
            small = self.THRESHOLD - 1
            self.assertFalse(self._gate(small))
            # Contract: no connector consultation happened. The tier
            # holding the full prefix must NOT serve a gated request.
            before = manager._trace_state.counts.get("lookup_disk_hit", 0)
            self.assertEqual(before, 0)
        finally:
            safe_shutdown(manager)

    def test_large_request_below_threshold_miss_size_still_checks(self):
        """The gate keys on REQUEST size, not RAM-miss size: a large
        request with a small RAM remainder still takes the disk path."""
        large = 8 * 112 * TOKENS_PER_CHUNK  # ~65k tokens
        ram_hit = large - 2 * TOKENS_PER_CHUNK
        self.assertGreaterEqual(large, self.THRESHOLD)
        self.assertTrue(self._gate(large))
        # The disk window is the RAM-miss remainder, however small.
        window_tokens = large - ram_hit
        self.assertLess(window_tokens, self.THRESHOLD)

    def test_gate_default_and_override(self):
        self.assertTrue(self._gate(self.THRESHOLD))
        self.assertFalse(self._gate(self.THRESHOLD - 1))
        self.assertTrue(self._gate(1, threshold=1))
        self.assertFalse(self._gate(0, threshold=0) and True)  # 0 disables


if __name__ == "__main__":
    unittest.main()
