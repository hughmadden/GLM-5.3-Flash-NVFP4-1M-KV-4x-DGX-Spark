# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hugh Madden and contributors
"""Failure-injection suite: real HTTP metadata plane on loopback, no engine.

The goal's stability item, exercised engine-free: a rank's metadata server
dying mid-session, a full disk, and a rank flap during renewal. Each
scenario asserts the fail-closed contract: no leaked leases on survivors,
no phantom tickets, recovery without process restarts beyond the injected
fault, and no permanent tier disable from transient faults (the R2
degrade-on-evidence semantics).
"""
import time
import unittest

import test_coordinator_http as base
from test_storage import NS

from recipe_persistence import coordinator_http as ch

GEOM2 = base.GEOM2


def k(n, group=0):
    """Distinct valid offload keys: 28-byte identity + 4-byte group index."""
    return bytes([n]) * 28 + group.to_bytes(4, "big")



def _store(coord, n, workers):
    """Store key k(n) DURABLY (reserve, write every rank, complete) so a
    later reserve_load can hit it -- mirrors Deployment.put; without the
    per-rank writes complete_store sees no durable object and invalidates."""
    t = coord.reserve_store(NS, k(n), "writer", coord.size_by_group[0])
    assert t is not None, coord._reserve_reasons
    for rank, lease in enumerate(t.leases):
        assert workers[rank].store.write(lease, (b"x" * t.sizes[rank],))
    coord.complete_store(t, True)

def lease_count(provider):
    with provider.store._lock:
        return provider.store.db.execute(
            "SELECT COUNT(*) FROM leases").fetchone()[0]


class RankDeathTests(unittest.TestCase):
    """A rank's metadata server dies mid-session."""

    def setUp(self):
        self.d = base.Deployment(world_size=2, geometry=GEOM2, rpc_timeout=10.0)

    def tearDown(self):
        self.d.close()

    def test_dead_rank_reservation_rolls_back_the_survivor_cleanly(self):
        _store(self.d.coordinator, 1, self.d.workers)
        ticket = self.d.coordinator.reserve_load(NS, k(1), "reader")
        self.assertIsNotNone(ticket)
        alive = self.d.workers[1]
        self.assertEqual(lease_count(alive), 1)

        self.d.workers[0].server.shutdown(timeout=5.0)

        # A new all-rank reservation must fail closed AND free the lease
        # the survivor granted during the rolled-back fanout.
        self.assertIsNone(self.d.coordinator.reserve_load(NS, k(2), "reader"))
        # The rolled-back fanout must free the survivor's grant, leaving
        # only the still-held k(1) read lease.
        deadline = time.monotonic() + 5.0
        while lease_count(alive) > 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(lease_count(alive), 1)
        # Releasing a ticket whose rank is dead is HONEST uncertainty, not
        # a clean ack: False, with the dead rank's cleanup retained in the
        # bounded retry queue (never a silent claim of success).
        self.assertFalse(self.d.coordinator.release(ticket))
        self.assertGreater(self.d.coordinator.pending_retries(), 0)

    def test_rank_restart_recovers_without_new_tickets_leaking(self):
        _store(self.d.coordinator, 1, self.d.workers)
        _store(self.d.coordinator, 2, self.d.workers)
        ticket = self.d.coordinator.reserve_load(NS, k(1), "reader")
        provider = self.d.workers[0]
        identity = provider.server.rpc.identity
        endpoint = provider.endpoint
        provider.server.shutdown(timeout=5.0)

        self.assertIsNone(self.d.coordinator.reserve_load(NS, k(2), "reader"))

        host, _, port = endpoint.rpartition(":")
        server = ch.MetadataServer(
            store=provider.store, identity=identity, rank=0, world_size=2,
            host=host, port=int(port), token=self.d.token)
        server.start()
        try:
            again = self.d.coordinator.reserve_load(NS, k(2), "reader")
            self.assertIsNotNone(again)
            self.assertIsNone(self.d.coordinator.release(again))
            # Release the held k(1) ticket while BOTH ranks are up: clean.
            self.assertIsNone(self.d.coordinator.release(ticket))
        finally:
            server.shutdown(timeout=5.0)


class DiskPressureTests(unittest.TestCase):
    """A full disk is pressure, not death: the tier keeps answering."""

    def test_quota_refusal_does_not_close_the_coordinator(self):
        tiny = dict(base.SMALL_LIMITS)
        tiny.update(quota=200_000, high=190_000, low=180_000, index_bytes=65536)
        d = base.Deployment(world_size=2, geometry=GEOM2, limits=tiny, rpc_timeout=10.0)
        try:
            self.assertTrue(d.coordinator.can_store())
            # Seed the read target FIRST: the fill loop below exhausts the
            # tiny quota, after which no new store can land.
            _store(d.coordinator, 1, d.workers)
            # Fill the tiny quota, then observe a clean refusal.
            stored = 0
            for i in range(16):
                t = d.coordinator.reserve_store(NS, k(50 + i), "w",
                                                d.coordinator.size_by_group[0])
                if t is None:
                    break
                stored += 1
                d.coordinator.complete_store(t, True)
            self.assertGreater(stored, 0)
            refused = d.coordinator.reserve_store(NS, k(99), "w",
                                                  d.coordinator.size_by_group[0])
            self.assertIsNone(refused)
            # Crucially: not closed, reads still answered, cleanup still works.
            self.assertTrue(d.coordinator.can_store())
            lease = d.coordinator.reserve_load(NS, k(1), "r")
            self.assertIsNotNone(lease)
            self.assertIsNone(d.coordinator.release(lease))
        finally:
            d.close()


class RankFlapTests(unittest.TestCase):
    """A rank flaps during renewal: transient, never a tier kill."""

    def setUp(self):
        self.d = base.Deployment(world_size=2, geometry=GEOM2, rpc_timeout=10.0)

    def tearDown(self):
        self.d.close()

    def test_renew_with_a_dead_rank_returns_false_without_disabling(self):
        _store(self.d.coordinator, 1, self.d.workers)
        ticket = self.d.coordinator.reserve_load(NS, k(1), "reader")
        self.assertIsNotNone(ticket)
        self.d.workers[1].server.shutdown(timeout=5.0)
        # Force the real RPC path: a fresh ticket's cached validity would
        # short-circuit renew to True without touching the network.
        self.d.coordinator._valid[ticket.token] = 0.0
        # R2 semantics: a lost renewal is not evidence; the coordinator
        # answers False and stays open.
        self.assertFalse(self.d.coordinator.renew(ticket))
        self.assertTrue(self.d.coordinator.can_store())
        # Same honest-False contract: the dead rank's release cannot be
        # acked; the live rank's is freed and the uncertainty is queued.
        self.assertFalse(self.d.coordinator.release(ticket))
        self.assertGreater(self.d.coordinator.pending_retries(), 0)


if __name__ == "__main__":
    unittest.main()
