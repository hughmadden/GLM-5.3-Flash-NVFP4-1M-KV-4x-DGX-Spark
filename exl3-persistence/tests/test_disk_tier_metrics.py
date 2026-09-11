# SPDX-License-Identifier: Apache-2.0
"""Stub tests for the disk-tier chunk query/hit counters.

Same spirit as ``tests/conc``: everything is synthesised in-process, nothing
starts an engine, imports torch/vLLM, binds a socket or performs inference.
A whole module loads and runs in milliseconds.

The counters live on the vLLM-free ``_Manager`` (native.py), so these tests
exercise the exact accounting without the native ABI. Emission into
``vllm:kv_offload_tiering_chunk_{queries,hits}{tier="disk"}`` is a thin native
wrapper around ``take_tier_stats()`` and is exercised only against a real
engine, which this suite deliberately never starts.
"""
from types import SimpleNamespace as NS
import unittest

from recipe_persistence.coordinator import Ticket
from recipe_persistence.handlers import DiskLoadStoreSpec
from recipe_persistence.native import (
    DISK_TIER_LABEL,
    _Manager,
    TIER_CHUNK_HITS_METRIC,
    TIER_CHUNK_QUERIES_METRIC,
)

SIZES = ((28,),)


def key(number=0, group=0):
    return bytes([number + 1]) + group.to_bytes(4, "big")


def ctx(owner="req"):
    return NS(req_id=owner)


class FakeCoordinator:
    """Instant in-process disk-tier stand-in: no network, files or torch.

    ``present`` selects whether the disk tier holds the queried chunk, so a
    miss is a real miss from the manager's point of view rather than an
    exception. Every call is recorded for assertions.
    """

    def __init__(self, sizes=SIZES, present=True, absent=(), refuse_store=False):
        self.sizes = tuple(tuple(s) for s in sizes)
        self.present = present
        self.absent = {bytes(a) for a in absent}
        self.refuse_store = refuse_store
        self.live = {}
        self.load_calls = []
        self.store_calls = []

    def _ticket(self, namespace, k, owner, writing):
        group = int.from_bytes(k[-4:], "big")
        token = "tok-%d" % len(self.live)
        ticket = Ticket(token, namespace, k, (token,), self.sizes[group], owner, writing)
        self.live[token] = ticket
        return ticket

    def reserve_load(self, namespace, k, owner):
        self.load_calls.append((owner, k))
        if not self.present or k in self.absent:
            return None
        return self._ticket(namespace, k, owner, False)

    def reserve_store(self, namespace, k, owner, size_by_rank):
        self.store_calls.append((owner, k))
        if self.refuse_store:
            return None
        return self._ticket(namespace, k, owner, True)

    def release(self, ticket):
        self.live.pop(ticket.token, None)

    def complete_store(self, ticket, success):
        self.live.pop(ticket.token, None)

    def invalidate(self, namespace, k):
        return None

    def renew(self, ticket):
        return ticket.token in self.live

    def lease_deadline(self, ticket):
        return None

    def can_store(self):
        return True


def make_manager(capacity=8, present=True, absent=(), lookup_keys_per_step=None,
                 refuse_store=False):
    coordinator = FakeCoordinator(present=present, absent=absent,
                                  refuse_store=refuse_store)
    manager = _Manager(coordinator, "ns", SIZES, capacity, DiskLoadStoreSpec,
                       NS, NS, lookup_keys_per_step=lookup_keys_per_step)
    return manager, coordinator


class DiskTierMetricTests(unittest.TestCase):
    def test_disk_hit_counts_one_query_and_one_hit(self):
        manager, coordinator = make_manager()
        self.assertEqual(manager.take_tier_stats(), (0, 0))
        self.assertIs(manager.lookup(key(), ctx()), True)
        self.assertEqual(coordinator.load_calls, [("req", key())])
        self.assertEqual(manager.take_tier_stats(), (1, 1))
        # Deltas only: a second read must not replay the observation.
        self.assertEqual(manager.take_tier_stats(), (0, 0))

    def test_disk_miss_counts_a_query_without_a_hit(self):
        manager, coordinator = make_manager(present=False)
        self.assertIs(manager.lookup(key(), ctx()), False)
        self.assertEqual(len(coordinator.load_calls), 1)
        self.assertEqual(manager.take_tier_stats(), (1, 0))

    def test_hit_and_miss_accumulate_without_hits_exceeding_queries(self):
        manager, _ = make_manager(absent=(key(1),))
        self.assertIs(manager.lookup(key(0), ctx("a")), True)
        self.assertIs(manager.lookup(key(1), ctx("b")), False)
        self.assertIs(manager.lookup(key(2), ctx("c")), True)
        queries, hits = manager.take_tier_stats()
        self.assertEqual((queries, hits), (3, 2))
        self.assertLessEqual(hits, queries)

    def test_repeat_lookup_of_a_reserved_key_is_not_a_new_query(self):
        manager, coordinator = make_manager()
        self.assertIs(manager.lookup(key(), ctx()), True)
        self.assertEqual(manager.take_tier_stats(), (1, 1))
        # Second observation of the same (owner, key) resolves from the held
        # lease, so it neither re-consults the tier nor double-counts.
        self.assertIs(manager.lookup(key(), ctx()), True)
        self.assertEqual(len(coordinator.load_calls), 1)
        self.assertEqual(manager.take_tier_stats(), (0, 0))

    def test_capacity_rejection_is_not_a_disk_query(self):
        manager, coordinator = make_manager(capacity=1)
        self.assertIs(manager.lookup(key(), ctx()), True)
        manager.take_tier_stats()
        # len(loads) == capacity: rejected before any tier consultation.
        self.assertIs(manager.lookup(key(1), ctx()), False)
        self.assertEqual(len(coordinator.load_calls), 1)
        self.assertEqual(manager.take_tier_stats(), (0, 0))

    def test_per_step_deferral_is_not_a_disk_query(self):
        manager, coordinator = make_manager(lookup_keys_per_step=1)
        self.assertIs(manager.lookup(key(), ctx("owner-a")), True)
        manager.take_tier_stats()
        # The step quantum is spent and the turn belongs to owner-a, so
        # owner-b is deferred (None) without touching the tier.
        self.assertIsNone(manager.lookup(key(1), ctx("owner-b")))
        self.assertEqual(len(coordinator.load_calls), 1)
        self.assertEqual(manager.take_tier_stats(), (0, 0))

    def test_invalidated_and_disabled_lookups_are_not_disk_queries(self):
        manager, coordinator = make_manager()
        manager._invalidations[key()] = True
        self.assertIs(manager.lookup(key(), ctx()), False)
        self.assertEqual(coordinator.load_calls, [])
        self.assertEqual(manager.take_tier_stats(), (0, 0))

        closed, closed_coordinator = make_manager()
        closed.closed = True
        self.assertIs(closed.lookup(key(), ctx()), False)
        self.assertEqual(closed_coordinator.load_calls, [])
        self.assertEqual(closed.take_tier_stats(), (0, 0))

    def test_store_durability_probe_is_not_a_disk_query(self):
        manager, coordinator = make_manager()
        output = manager.prepare_store([key()], ctx())
        self.assertIsNotNone(output)
        # prepare_store consults reserve_load to detect durable data, but that
        # is a store-path probe, not a prefix-lookup query.
        self.assertEqual(len(coordinator.load_calls), 1)
        self.assertEqual(manager.take_tier_stats(), (0, 0))

    def test_metric_names_and_tier_label_are_pinned(self):
        self.assertEqual(DISK_TIER_LABEL, "disk")
        self.assertEqual(TIER_CHUNK_QUERIES_METRIC,
                         "vllm:kv_offload_tiering_chunk_queries")
        self.assertEqual(TIER_CHUNK_HITS_METRIC,
                         "vllm:kv_offload_tiering_chunk_hits")


if __name__ == "__main__":
    unittest.main()
