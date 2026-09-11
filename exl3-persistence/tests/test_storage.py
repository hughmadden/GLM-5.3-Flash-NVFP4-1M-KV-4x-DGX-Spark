# SPDX-License-Identifier: Apache-2.0
import dataclasses
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from recipe_persistence import DiskStore, Limits, fingerprint
from recipe_persistence.storage import HEADER_BYTES


def small_limits(**kwargs):
    return dataclasses.replace(Limits(quota=2_000_000, high=1_700_000,
        low=1_400_000, index_bytes=65536, free_bytes=0, free_inodes=0,
        max_object_bytes=128_000, grace_seconds=5, lease_seconds=20,
        io_chunk_bytes=19), **kwargs)


NS = fingerprint(model="model-revision", draft="draft-revision", layout="layout-v1",
                 tenant="trusted-canary", hash="sha256-seed-fixed")
KEY = b"x" * 32 + bytes(4)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = [100.0]
        self.root = Path(self.tmp.name) / "rank"
        self.s = DiskStore(self.root, small_limits(), clock=lambda:self.clock[0])

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def put(self, key=KEY, data=b"payload", namespace=NS):
        token = self.s.reserve_write(namespace,key,len(data),"request")
        self.assertIsNotNone(token)
        self.assertTrue(self.s.write(token,(data[:2],memoryview(data)[2:])))
        self.s.release(token)  # Aggregate durable completion releases writer hold.
        return token

    def test_cpu_import_is_standalone(self):
        subprocess.run([sys.executable,"-c",
            "import sys; import recipe_persistence; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"],check=True)

    def test_invalid_numeric_limits_fail_before_resource_creation(self):
        for kwargs in ({"lease_seconds":float("inf")}, {"grace_seconds":float("nan")},
                       {"max_leases":1.5}, {"quota":2**63}, {"io_chunk_bytes":True}):
            with self.assertRaises(ValueError):
                small_limits(**kwargs)

    def test_decimal_defaults(self):
        l = Limits()
        self.assertEqual((l.quota,l.high,l.low),(10**12,9*10**11,8*10**11))

    def test_roundtrip_restart_permissions_and_accounting(self):
        before = self.s.usage()
        token = self.put(data=b"abcdefghij")
        charged = self.s.usage()
        self.assertGreater(charged,before+10)
        self.assertEqual(self.s._path(token).stat().st_mode & 0o777,0o600)
        self.assertEqual(self.root.stat().st_mode & 0o777,0o700)
        lease = self.s.reserve_read(NS,KEY,"reader")
        a,b = bytearray(3),bytearray(7)
        self.assertTrue(self.s.read_into(lease,(a,b)))
        self.assertEqual(a+b,b"abcdefghij")
        self.s.close()
        self.s = DiskStore(self.root,small_limits(),clock=lambda:self.clock[0])
        self.assertEqual(self.s.usage(),charged)
        self.assertFalse(self.s.read_into(lease,(bytearray(10),)))
        self.assertTrue(self.s.exists(NS,KEY))

    def test_namespace_and_key_isolation(self):
        self.put()
        for field in ("model","draft","layout","tenant","hash"):
            ns = fingerprint(**{field:"different"})
            self.assertIsNone(self.s.reserve_read(ns,KEY,"reader"))
        self.assertIsNone(self.s.reserve_read(NS,KEY+b"x","reader"))
        self.assertIsNone(self.s.reserve_write("../../escape",KEY,10,"request"))
        self.assertIsNone(self.s.reserve_write(NS,b"k"*129,10,"request"))
        self.assertIsNone(self.s.reserve_write(NS,KEY,10,"r"*129))

    def test_missing_short_corrupt_wrong_header(self):
        for mode in ("missing","short","corrupt","header"):
            key = mode.encode()+bytes(4)
            token = self.put(key,data=b"abcdef")
            lease = self.s.reserve_read(NS,key,"reader")
            path = self.s._path(token)
            if mode == "missing":
                path.unlink()
            else:
                with open(path,"r+b") as f:
                    if mode == "short":
                        f.truncate(HEADER_BYTES+2)
                    else:
                        f.seek(HEADER_BYTES if mode == "corrupt" else 20)
                        f.write(b"!")
            self.assertFalse(self.s.read_into(lease,(bytearray(6),)),mode)
            self.assertIsNone(self.s.reserve_read(NS,key,"next"),mode)
            self.s.release(lease)

    def test_partial_short_long_and_producer_failure(self):
        baseline = self.s.usage()
        def broken():
            yield b"ab"
            raise OSError("injected producer failure")
        for i,chunks in enumerate(((b"a",),(b"abcdefg",),broken())):
            key = bytes([i])+KEY
            token = self.s.reserve_write(NS,key,6,"request")
            self.assertFalse(self.s.write(token,chunks))
            self.assertFalse(self.s.exists(NS,key))
        self.assertGreater(self.s.usage(),baseline)  # Partials stay charged.
        self.clock[0] += 6
        self.s.collect()
        self.assertEqual(self.s.usage(),baseline)

    def test_fsync_or_rename_failure_never_commits(self):
        for func in ("os.fsync","os.rename"):
            key = func.encode()+KEY
            token = self.s.reserve_write(NS,key,6,"request")
            with patch(func,side_effect=OSError("injected disk failure")):
                self.assertFalse(self.s.write(token,(b"abcdef",)))
            self.assertFalse(self.s.exists(NS,key))

    def test_reservations_and_partials_obey_hard_quota(self):
        self.s.close()
        self.s = DiskStore(self.root,small_limits(quota=210_000,high=205_000,low=200_000),clock=lambda:self.clock[0])
        baseline = self.s.usage()
        tokens = []
        for i in range(100):
            token = self.s.reserve_write(NS,bytes([i])+KEY,5000,"request")
            if token is None:
                break
            tokens.append(token)
            self.assertLessEqual(self.s.usage(),self.s.limits.quota)
        self.assertTrue(tokens)
        self.assertLess(len(tokens),100)
        for token in tokens:
            self.s.release(token)
        self.assertGreater(self.s.usage(),baseline)  # Grace still consumes quota.
        self.clock[0] += 6
        self.s.collect()
        self.assertEqual(self.s.usage(),baseline)

    def test_tombstone_grace_and_reader_lease(self):
        baseline = self.s.usage()
        token = self.put()
        lease = self.s.reserve_read(NS,KEY,"reader")
        self.s.invalidate(NS,KEY)
        self.assertIsNone(self.s.reserve_read(NS,KEY,"new-reader"))
        self.clock[0] += 6
        self.s.collect()
        self.assertTrue(self.s._path(token).exists())
        dest = bytearray(7)
        self.assertTrue(self.s.read_into(lease,(dest,)))
        self.s.release(lease)
        self.s.collect()
        self.assertFalse(self.s._path(token).exists())
        self.assertEqual(self.s.usage(),baseline)

    def test_expired_queued_lease_is_miss_and_can_reclaim(self):
        token = self.put()
        lease = self.s.reserve_read(NS,KEY,"reader")
        self.assertTrue(self.s.renew(lease))
        self.clock[0] += 21
        self.assertFalse(self.s.renew(lease))
        self.s.invalidate(NS,KEY)
        self.clock[0] += 6
        self.s.collect()
        self.assertFalse(self.s.read_into(lease,(bytearray(7),)))
        self.assertFalse(self.s._path(token).exists())

    def test_active_io_owns_lease_even_past_expiry(self):
        self.put()
        lease = self.s.reserve_read(NS,KEY,"reader")
        entered, unblock, collected = threading.Event(), threading.Event(), threading.Event()
        result = []
        def buffers():
            entered.set()
            self.assertTrue(unblock.wait(3))
            yield bytearray(7)
        reader = threading.Thread(target=lambda:result.append(self.s.read_into(lease,buffers())))
        reader.start()
        self.assertTrue(entered.wait(3))
        self.clock[0] += 100
        def collect():
            self.s.invalidate(NS,KEY)
            self.s.collect()
            collected.set()
        janitor = threading.Thread(target=collect)
        janitor.start()
        self.assertFalse(collected.wait(0.05))
        unblock.set()
        reader.join(3); janitor.join(3)
        self.assertEqual(result,[True])
        self.assertTrue(collected.is_set())

    def test_restart_discards_orphans_pending_and_expired(self):
        self.put()
        pending = self.s.reserve_write(NS,b"pending"+KEY,7,"request")
        self.s._path(pending,True).write_bytes(b"partial")
        orphan = self.s.objects / ("a"*32+".kv")
        orphan.write_bytes(b"orphan")
        self.s.close()
        self.s = DiskStore(self.root,small_limits(),clock=lambda:self.clock[0])
        self.assertTrue(self.s.exists(NS,KEY))
        self.assertFalse(orphan.exists())
        self.assertFalse(self.s._path(pending,True).exists())
        self.assertEqual(self.s.db.execute("SELECT COUNT(*) FROM objects").fetchone()[0],1)

    def test_process_crash_between_rename_and_index_commit(self):
        self.s.close()
        script = '''
import os
from recipe_persistence import DiskStore, Limits, fingerprint
from dataclasses import replace
s = DiskStore(os.environ['FIXTURE_ROOT'], replace(Limits(), index_bytes=65536, free_bytes=0, free_inodes=0))
t = s.reserve_write(os.environ['FIXTURE_NS'], b'crash'+bytes(4), 7, 'writer')
rename = os.rename
def crash(src,dst):
    rename(src,dst)
    os._exit(23)
os.rename = crash
s.write(t,(b'payload',))
'''
        result = subprocess.run([sys.executable,"-c",script],env={**os.environ,
            "FIXTURE_ROOT":str(self.root),"FIXTURE_NS":NS})
        self.assertEqual(result.returncode,23)
        self.s = DiskStore(self.root,small_limits())
        self.assertFalse(self.s.exists(NS,b"crash"+bytes(4)))
        self.assertEqual(self.s.db.execute("SELECT COUNT(*) FROM objects").fetchone()[0],0)
        self.assertEqual(list(self.s.objects.iterdir()),[])

    def test_commit_order_is_file_fsync_rename_directory_fsync_then_index(self):
        token = self.s.reserve_write(NS,KEY,7,"writer")
        events = []
        fsync, rename = os.fsync, os.rename
        def synced(fd):
            events.append("directory" if fd == self.s._dirfd else "file")
            return fsync(fd)
        def renamed(src,dst):
            events.append("rename")
            return rename(src,dst)
        self.s.db.set_trace_callback(lambda sql:events.append("index") if "SET state='C'" in sql else None)
        with patch("os.fsync",side_effect=synced),patch("os.rename",side_effect=renamed):
            self.assertTrue(self.s.write(token,(b"payload",)))
        self.s.db.set_trace_callback(None)
        self.assertEqual(events,["file","rename","directory","index"])

    def test_sqlite_full_declines_without_unbounded_index(self):
        self.s.close()
        self.s = DiskStore(self.root,small_limits(quota=10_000_000,high=9_000_000,low=8_000_000))
        accepted = 0
        for i in range(1000):
            token = self.s.reserve_write(NS,i.to_bytes(4,"big")+KEY,1,"writer")
            if token is None:
                break
            accepted += 1
        self.assertGreater(accepted,0)
        self.assertLess(accepted,1000)
        self.assertLessEqual(self.s.db.execute("PRAGMA page_count").fetchone()[0],16)
        self.assertEqual(self.s.db.execute("PRAGMA integrity_check").fetchone()[0],"ok")
        self.assertLessEqual(self.s.usage(),self.s.limits.quota)

    def test_reduced_quota_restart_reclaims_before_serving(self):
        for i in range(10):
            self.put(bytes([i])+KEY,data=b"x"*4000)
        self.assertGreater(self.s.usage(),200_000)
        self.s.close()
        self.s = DiskStore(self.root,small_limits(quota=210_000,high=200_000,low=190_000))
        self.assertLessEqual(self.s.usage(),190_000)

    def test_committed_writer_remains_renewable_until_aggregate_release(self):
        token = self.s.reserve_write(NS,KEY,7,"writer")
        self.assertTrue(self.s.write(token,(b"payload",)))
        self.assertTrue(self.s.renew(token))
        self.s.release(token)
        self.assertFalse(self.s.renew(token))
        self.assertTrue(self.s.exists(NS,KEY))

    def test_corrupt_index_fails_closed_and_releases_process_lock(self):
        self.s.close()
        (self.root / "index.sqlite").write_bytes(b"not a sqlite database")
        for _ in range(2):
            with self.assertRaises(sqlite3.DatabaseError):
                DiskStore(self.root,small_limits())
        # A corrupt index is not silently discarded alongside sensitive payloads.
        self.assertEqual((self.root / "index.sqlite").read_bytes(),b"not a sqlite database")

    def test_unknown_root_files_are_not_omitted_from_accounting(self):
        self.s.close()
        unknown = self.root / "unaccounted.bin"
        unknown.write_bytes(b"external data")
        with self.assertRaisesRegex(ValueError,"unaccounted"):
            DiskStore(self.root,small_limits())
        self.assertTrue(unknown.exists())

    def test_gc_failure_during_admission_prevents_new_reservation(self):
        def failed_collect():
            self.s.failed = True
        with patch.object(self.s,"collect",side_effect=failed_collect):
            self.assertIsNone(self.s.reserve_write(NS,KEY,7,"writer"))
        self.assertEqual(self.s.db.execute("SELECT COUNT(*) FROM objects").fetchone()[0],0)

    def test_single_process_root_ownership(self):
        with self.assertRaises(OSError):
            DiskStore(self.root,small_limits())

    def test_bounded_metadata_and_free_space_reserve(self):
        self.s.close()
        self.s = DiskStore(self.root,small_limits(max_objects=1,max_leases=1),clock=lambda:self.clock[0])
        self.put()
        self.assertIsNone(self.s.reserve_write(NS,b"other"+KEY,10,"request"))
        self.assertIsNotNone(self.s.reserve_read(NS,KEY,"reader"))
        self.assertIsNone(self.s.reserve_read(NS,KEY,"second"))
        self.assertEqual(self.s.db.execute("PRAGMA cache_size").fetchone()[0],-2048)
        self.assertEqual(self.s.db.execute("PRAGMA mmap_size").fetchone()[0],0)
        self.assertEqual(self.s.db.execute("PRAGMA synchronous").fetchone()[0],3)
        self.assertEqual(self.s.db.execute("PRAGMA max_page_count").fetchone()[0],16)
        self.s.close()
        self.s = DiskStore(self.root,small_limits(free_bytes=10**30))
        self.assertIsNone(self.s.reserve_write(NS,b"another"+KEY,10,"request"))

    def test_watermark_eviction_excludes_reader(self):
        self.s.close()
        self.s = DiskStore(self.root,small_limits(quota=240_000,high=220_000,low=190_000),clock=lambda:self.clock[0])
        first = self.put()
        lease = self.s.reserve_read(NS,KEY,"reader")
        for i in range(6):
            self.clock[0] += 0.1
            self.put(bytes([i])+KEY,data=b"x"*4000)
        self.assertGreater(self.s.evict(),0)
        self.clock[0] += 6
        self.s.collect()
        self.assertTrue(self.s._path(first).exists())
        self.assertLessEqual(self.s.usage(),self.s.limits.low)
        self.s.release(lease)


if __name__ == "__main__":
    unittest.main()
