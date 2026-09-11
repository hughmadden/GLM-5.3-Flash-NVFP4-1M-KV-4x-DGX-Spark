# SPDX-License-Identifier: Apache-2.0
"""Liveness test for the disk-tier lookup gate.

The gate in ``_Manager._admit_lookup`` admits one owner's worth of new disk
reservations per turn and returns RETRY (``None``) to everyone else. The turn was
advanced only by ``_Manager.on_schedule_end``.

That is a deadlock, and it was observed live on the engine as ``Running: 0`` with
the request parked in ``deferred``:

  * a request whose lookups are all answered RETRY produces no batch;
  * a scheduler with no batch does not call ``on_schedule_end``;
  * so the turn never advances, every later lookup is RETRY again, and the
    request can never make progress.

These tests pin the liveness property directly: with the hook deliberately never
called, the gate must still let a lookup through. They are engine-free -- no
torch, vLLM, socket, filesystem or inference.
"""
import time
import unittest
from types import SimpleNamespace as NS

from recipe_persistence.coordinator import Ticket
from recipe_persistence.handlers import DiskLoadStoreSpec
from recipe_persistence.native import _Manager

SIZES = ((28,),)


def key(number):
    return number.to_bytes(4, "big") + (0).to_bytes(4, "big")


def ctx(owner):
    return NS(req_id=owner)


class FakeCoordinator:
    """Every queried chunk is present, so a charged lookup is a hit."""

    def __init__(self, sizes=SIZES):
        self.sizes = tuple(tuple(s) for s in sizes)
        self.live = {}
        self.calls = 0

    def _ticket(self, namespace, k, owner):
        token = "t%d" % len(self.live)
        group = int.from_bytes(k[-4:], "big")
        t = Ticket(token, namespace, k, (token,), self.sizes[group], owner, False)
        self.live[token] = t
        return t

    def reserve_load(self, namespace, k, owner):
        self.calls += 1
        return self._ticket(namespace, k, owner)

    def reserve_store(self, namespace, k, owner, size_by_rank):
        return None

    def release(self, ticket):
        self.live.pop(ticket.token, None)

    def complete_store(self, ticket, success):
        self.live.pop(ticket.token, None)

    def invalidate(self, namespace, k):
        return None


def make_manager(capacity=64, keys_per_step=1):
    coordinator = FakeCoordinator()
    manager = _Manager(coordinator, "ns", SIZES, capacity, DiskLoadStoreSpec,
                       NS, NS, lookup_keys_per_step=keys_per_step)
    return manager, coordinator


class LookupGateLivenessTests(unittest.TestCase):

    def test_second_owner_is_admitted_without_the_scheduler_hook(self):
        """The core deadlock: owner B must not be starved forever.

        ``on_schedule_end`` is deliberately NEVER called, reproducing a
        scheduler that has no batch to run.
        """
        manager, _ = make_manager(keys_per_step=1)
        a, b = ctx("A"), ctx("B")
        manager.on_new_request(a)
        manager.on_new_request(b)

        # A takes the single key of the first turn.
        self.assertIs(manager.lookup(key(0), a), True)
        # B is throttled -- this is the RETRY that used to be terminal.
        self.assertIsNone(manager.lookup(key(1), b))

        # No hook call. After the turn window the gate must advance by itself.
        deadline = time.monotonic() + 2.0
        admitted = False
        while time.monotonic() < deadline:
            if manager.lookup(key(1), b) is not None:
                admitted = True
                break
            time.sleep(0.01)
        self.assertTrue(admitted, "owner B was never admitted without on_schedule_end")

    def test_repeated_retry_always_resolves(self):
        """A deferral loop must terminate, not spin: the property that failed."""
        manager, _ = make_manager(keys_per_step=1)
        a, b = ctx("A"), ctx("B")
        manager.on_new_request(a)
        manager.on_new_request(b)
        manager.lookup(key(0), a)

        retries = 0
        for _ in range(400):
            if manager.lookup(key(1), b) is not None:
                break
            retries += 1
            time.sleep(0.005)
        self.assertLess(retries, 400, "lookup returned RETRY for every attempt")

    def test_budget_still_bounds_one_turn(self):
        """Liveness must not be bought by removing the throttle entirely."""
        manager, coordinator = make_manager(keys_per_step=2)
        a = ctx("A")
        manager.on_new_request(a)
        self.assertIs(manager.lookup(key(0), a), True)
        self.assertIs(manager.lookup(key(1), a), True)
        # Third charge in the same turn is throttled.
        self.assertIsNone(manager.lookup(key(2), a))

    def test_gate_clears_on_degrade(self):
        manager, _ = make_manager(keys_per_step=1)
        a = ctx("A")
        manager.on_new_request(a)
        manager.lookup(key(0), a)
        manager._degrade("coordinator_callback_failed")
        self.assertIsNone(manager._lookup_turn_owner)
        self.assertIsNone(manager._lookup_turn_started)
        self.assertEqual(manager._lookup_owners, {})


if __name__ == "__main__":
    unittest.main()
