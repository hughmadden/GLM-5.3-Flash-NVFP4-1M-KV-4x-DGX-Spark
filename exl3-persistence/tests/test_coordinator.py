# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from recipe_persistence import DiskStore, LocalCoordinator
from test_storage import small_limits, NS, KEY


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stores = [DiskStore(Path(self.tmp.name)/str(i),small_limits()) for i in range(4)]
        self.c = LocalCoordinator(self.stores,((4,5,6,7),),max_pending_keys=2,max_pending_bytes=14)

    def tearDown(self):
        for store in self.stores:
            store.close()
        self.tmp.cleanup()

    def put(self,key=KEY):
        ticket = self.c.reserve_store(NS,key,"request",(4,5,6,7))
        self.assertIsNotNone(ticket)
        for store, token, size in zip(self.stores,ticket.leases,ticket.sizes):
            self.assertTrue(store.write(token,(b"x"*size,)))
        self.c.complete_store(ticket,True)

    def test_all_rank_reservation_then_exact_restore(self):
        self.put()
        ticket = self.c.reserve_load(NS,KEY,"reader")
        self.assertIsNotNone(ticket)
        self.assertEqual(self.c.reserve_load(NS,KEY,"reader"),ticket)
        for store, token, size in zip(self.stores,ticket.leases,ticket.sizes):
            dest = bytearray(size)
            self.assertTrue(store.read_into(token,(dest,)))
            self.assertEqual(dest,b"x"*size)
        self.c.release(ticket)
        self.c.release(ticket)
        self.assertEqual(self.c._bytes,0)

    def test_failed_rank_veto_and_partial_rollback(self):
        self.put()
        self.stores[-1].invalidate(NS,KEY)
        self.assertIsNone(self.c.reserve_load(NS,KEY,"reader"))
        for s in self.stores:
            self.assertEqual(s.db.execute("SELECT COUNT(*) FROM leases").fetchone()[0],0)

    def test_failed_store_invalidates_successful_shards(self):
        ticket = self.c.reserve_store(NS,KEY,"request",(4,5,6,7))
        for s,t,n in zip(self.stores[:3],ticket.leases,ticket.sizes):
            self.assertTrue(s.write(t,(b"x"*n,)))
        self.c.complete_store(ticket,False)
        self.assertIsNone(self.c.reserve_load(NS,KEY,"reader"))
        self.assertTrue(all(not s.exists(NS,KEY) for s in self.stores))
        self.assertEqual(self.c._bytes,0)

    def test_queue_byte_and_key_credits(self):
        self.put()
        a = self.c.reserve_load(NS,KEY,"a")
        b = self.c.reserve_load(NS,KEY,"b")
        self.assertIsNone(self.c.reserve_load(NS,KEY,"c"))
        self.c.release(a)
        self.assertIsNotNone(self.c.reserve_load(NS,KEY,"c"))
        self.c.release(b)

    def test_failed_store_cannot_renew_or_readvertise_cached_ticket(self):
        self.put()
        ticket = self.c.reserve_load(NS,KEY,"reader")
        self.stores[0].failed = True
        self.assertFalse(self.stores[0].renew(ticket.leases[0]))
        self.assertIsNone(self.c.reserve_load(NS,KEY,"reader"))
        self.assertEqual(self.c._bytes,7)
        self.assertEqual(self.c._tickets,{ticket.token:ticket})
        self.assertIsNone(self.c.lease_deadline(ticket))

    def test_negative_invalidation_is_not_an_ack_and_other_ranks_are_tried(self):
        self.put()
        with patch.object(self.stores[0],"invalidate",return_value=False):
            self.assertIs(self.c.invalidate(NS,KEY),False)
        self.assertTrue(self.stores[0].exists(NS,KEY))
        self.assertTrue(all(not s.exists(NS,KEY) for s in self.stores[1:]))

    def test_exception_invalidation_still_tries_remaining_ranks(self):
        self.put()
        with patch.object(self.stores[0],"invalidate",side_effect=OSError("unavailable")):
            self.assertIs(self.c.invalidate(NS,KEY),False)
        self.assertTrue(all(not s.exists(NS,KEY) for s in self.stores[1:]))

    def test_failed_quarantine_propagates_but_drained_leases_are_released(self):
        ticket = self.c.reserve_store(NS,KEY,"request",(4,5,6,7))
        with patch.object(self.stores[0],"invalidate",return_value=False):
            self.assertIs(self.c.complete_store(ticket,False),False)
        self.assertEqual(self.c._bytes,7)
        self.assertTrue(all(not s.renew(t) for s,t in zip(self.stores,ticket.leases)))
        self.assertIsNone(self.c.complete_store(ticket,False))
        self.assertEqual(self.c._bytes,0)

    def test_failed_release_attempts_all_ranks_and_reports_failure(self):
        self.put()
        ticket = self.c.reserve_load(NS,KEY,"reader")
        with patch.object(self.stores[0],"release",side_effect=OSError("unavailable")):
            self.assertIs(self.c.release(ticket),False)
        for s in self.stores[1:]:
            self.assertEqual(s.db.execute("SELECT COUNT(*) FROM leases").fetchone()[0],0)
        self.assertEqual(self.c._bytes,7)
        self.assertIsNone(self.c.lease_deadline(ticket))
        self.assertIsNone(self.c.reserve_load(NS,KEY,"reader"))
        self.assertIsNone(self.c.release(ticket))
        self.assertEqual(self.c._bytes,0)

    def test_geometry_and_unknown_group_fail_closed(self):
        self.assertIsNone(self.c.reserve_store(NS,KEY,"request",(4,5,6,8)))
        self.assertIsNone(self.c.reserve_load(NS,b"hash"+b"\x00\x00\x00\x09","reader"))


if __name__ == "__main__":
    unittest.main()
