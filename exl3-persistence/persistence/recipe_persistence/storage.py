# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hugh Madden and contributors
"""Single-owner, rank-local immutable objects. No torch/vLLM dependency.

The index is deliberately disk-backed, with a fixed SQLite cache and file budget.
All I/O is serialized under a lock: expiry cannot unlink an active syscall's file.
Expired *queued* leases are allowed to fail cleanly. One process owns each root.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import threading
import time
import uuid

HEADER_BYTES = 4096
_HEADER = struct.Struct("!8sQ32s32s32s")
_MAGIC = b"RPKV0001"
_ERRORS = (OSError, sqlite3.Error, ValueError, BufferError)


def fingerprint(**fields: str) -> str:
    """Stable, length-delimited model/draft/layout/hash-policy/tenant identity."""
    h = hashlib.sha256()
    for key, value in sorted(fields.items()):
        for part in (key, value):
            data = part.encode("utf-8")
            if len(data) > 16384:
                raise ValueError("fingerprint field too long")
            h.update(struct.pack("!I", len(data)))
            h.update(data)
    return h.hexdigest()


@dataclass(frozen=True)
class Limits:
    quota: int = 1_000_000_000_000
    high: int = 900_000_000_000
    low: int = 800_000_000_000
    # Reserve enough for DB + rollback journal, independent of object charges.
    index_bytes: int = 64 * 1024 * 1024
    max_objects: int = 1_000_000
    max_leases: int = 32768
    max_object_bytes: int = 64 * 1024 * 1024
    free_bytes: int = 1024 * 1024 * 1024
    free_inodes: int = 128
    lease_seconds: float = 300.0
    grace_seconds: float = 60.0
    io_chunk_bytes: int = 1024 * 1024

    def __post_init__(self):
        integer_fields = (self.quota, self.high, self.low, self.index_bytes,
                          self.max_objects, self.max_leases, self.max_object_bytes,
                          self.free_bytes, self.free_inodes, self.io_chunk_bytes)
        if any(type(n) is not int for n in integer_fields):
            raise ValueError("byte, count and inode bounds must be integers")
        if self.quota > 2**63 - 1:
            raise ValueError("quota exceeds SQLite accounting range")
        for duration in (self.lease_seconds, self.grace_seconds):
            if type(duration) not in (int, float) or not math.isfinite(duration):
                raise ValueError("lease and grace durations must be finite")
        if not (0 < self.low < self.high <= self.quota):
            raise ValueError("require 0 < low < high <= quota")
        if self.index_bytes < 65536 or self.quota <= 2 * self.index_bytes:
            raise ValueError("quota must cover bounded index and rollback journal")
        if min(self.max_objects, self.max_leases, self.max_object_bytes,
               self.io_chunk_bytes, self.lease_seconds) <= 0:
            raise ValueError("positive bounds required")
        if min(self.grace_seconds, self.free_bytes, self.free_inodes) < 0:
            raise ValueError("negative reserve/grace")


class DiskStore:
    """Rank store; root must be private and exclusively owned, not a shared cache.

    Object quota charges rounded physical capacity BEFORE creating a partial.
    Metadata reserves twice index_bytes (DB plus DELETE rollback journal).
    Filesystem-free reserve is separate. No unbounded residency dictionary exists.
    """
    def __init__(self, root, limits: Limits | None = None, *, clock=time.time,
                 journal_mode="delete", synchronous="extra"):
        self._lock = threading.RLock()
        self.closed = False
        self.db = None
        self._dirfd = None
        self._lockfile = None
        try:
            self._initialize(root, limits, clock=clock,
                               journal_mode=journal_mode, synchronous=synchronous)
        except BaseException:
            self.close()
            raise

    def _initialize(self, root, limits, *, clock, journal_mode="delete",
                    synchronous="extra"):
        self.limits = limits or Limits()
        self.clock = clock
        # Durability is a POLICY, not a constant. The historical defaults
        # (DELETE + EXTRA) are maximum durability and stay the package default, so
        # a caller that says nothing keeps exactly the old contract. This store is
        # a RECONSTRUCTIBLE cache though -- the connector runs with
        # kv_load_failure_policy=recompute -- and EXTRA forces an fsync on every
        # commit, including every lease grant: measured 8.7 ms median / 11.4 ms p95
        # on the lookup-hit path, which is the decode penalty. WAL + NORMAL measured
        # 0.024 ms (360x) and also removes reader/writer serialisation. The engine
        # opts in via spec extra_config; the library default does not change.
        if journal_mode not in {"delete", "wal", "truncate", "persist", "memory"}:
            raise ValueError("unsupported journal_mode")
        if synchronous not in {"off", "normal", "full", "extra"}:
            raise ValueError("unsupported synchronous level")
        self._journal_mode = journal_mode
        self._synchronous = synchronous
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.stat().st_uid != os.getuid():
            raise ValueError("cache root must be owned and not a symlink")
        if self.root.stat().st_mode & 0o077:
            raise ValueError("cache root must have mode 0700")
        with os.scandir(self.root) as entries:
            if any(e.name not in {"owner.lock", "objects", "index.sqlite", "index.sqlite-journal",
                                  "index.sqlite-wal", "index.sqlite-shm"}
                   for e in entries):
                raise ValueError("cache root contains unaccounted files")
        lockfd = os.open(self.root / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self._lockfile = os.fdopen(lockfd, "a+b")
        os.chmod(self.root / "owner.lock", 0o600)
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lockfile.close()
            raise
        self._lock = threading.RLock()
        self.failed = False
        # Distinguishable, NON-LATCHING record of storage failures. self.failed is
        # reserved for unrecoverable faults because it refuses every operation from
        # then on with no recovery path; a full disk or an exhausted index budget
        # is recoverable (free space, invalidate()+collect()), so it must be
        # reportable without bricking the rank's store.
        self.failure_count = 0
        self.last_failure = None
        self.closed = False
        self.objects = self.root / "objects"
        self.objects.mkdir(mode=0o700, exist_ok=True)
        if self.objects.is_symlink() or self.objects.stat().st_mode & 0o077:
            self._lockfile.close()
            raise ValueError("object directory must be private and not a symlink")
        self._dirfd = os.open(self.objects, os.O_RDONLY | os.O_DIRECTORY)
        self._unit = max(4096, os.statvfs(self.root).f_frsize)
        self._metadata = 2 * self.limits.index_bytes + 4 * self._unit
        if self._metadata >= self.limits.low:
            raise ValueError("low watermark must exceed metadata reserve")
        dbpath = self.root / "index.sqlite"
        # Create restrictive permissions before SQLite ever sees the file.
        fd = os.open(dbpath, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        os.chmod(dbpath, 0o600)
        self.db = sqlite3.connect(dbpath, isolation_level=None, check_same_thread=False)
        # This store holds a RECONSTRUCTIBLE CACHE, not a journal: the connector is
        # configured with kv_load_failure_policy=recompute, so a lost object costs a
        # recompute and nothing else. synchronous=EXTRA therefore bought no real
        # guarantee while forcing an fsync on every commit -- including every lease
        # grant, which measured 8.7 ms median / 11.4 ms p95 on the lookup hit path
        # and was the dominant decode penalty (engine metric
        # vllm:kv_offload_lookup_sync_delay: ~800 lookups/10 s costing 9.3 s of the
        # 10 s window). WAL keeps readers concurrent with the writer -- which also
        # removes the read-vs-write serialisation the concurrency suite found -- and
        # NORMAL still leaves the database consistent across a process crash; only a
        # power loss can lose the most recent transactions, and those are cache
        # entries this design already tolerates losing.
        self.db.execute(f"PRAGMA journal_mode={self._journal_mode}")
        # EXTRA includes DELETE-journal directory synchronization on commit.
        self.db.execute(f"PRAGMA synchronous={self._synchronous}")
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("PRAGMA temp_store=FILE")
        page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
        page_limit = self.limits.index_bytes // page_size
        actual_limit = self.db.execute(f"PRAGMA max_page_count={page_limit}").fetchone()[0]
        if actual_limit > page_limit:
            self.close()
            raise ValueError("existing index exceeds configured metadata budget")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS objects (
            id TEXT PRIMARY KEY, ns TEXT NOT NULL, key BLOB NOT NULL,
            size INTEGER NOT NULL, charge INTEGER NOT NULL, state TEXT NOT NULL,
            touched REAL NOT NULL, expiry REAL NOT NULL, owner TEXT NOT NULL,
            UNIQUE(ns,key));
          CREATE INDEX IF NOT EXISTS victims ON objects(state,touched);
          CREATE TABLE IF NOT EXISTS leases (
            token TEXT PRIMARY KEY, object TEXT NOT NULL, owner TEXT NOT NULL,
            expiry REAL NOT NULL);
          CREATE INDEX IF NOT EXISTS readers ON leases(object);
          CREATE TABLE IF NOT EXISTS totals (id INTEGER PRIMARY KEY, bytes INTEGER);
          INSERT OR IGNORE INTO totals VALUES(1,0);
        """)
        self._recover()
        rootfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(rootfd)
        finally:
            os.close(rootfd)

    @contextmanager
    def _tx(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def _charge(self, size):
        # One extra filesystem unit per object conservatively covers directory
        # entries/inode-associated allocation; partial and final never coexist.
        return ((size + HEADER_BYTES + self._unit - 1) // self._unit + 1) * self._unit

    def _identity(self, ns, key, owner=""):
        if not isinstance(ns, str) or not re.fullmatch("[0-9a-f]{64}", ns):
            raise ValueError("namespace must be SHA256 fingerprint")
        if not isinstance(key, bytes) or not 1 <= len(key) <= 128:
            raise ValueError("key must contain 1..128 bytes")
        if not isinstance(owner, str) or len(owner.encode()) > 128:
            raise ValueError("owner too long")

    def _path(self, ident, partial=False):
        if not re.fullmatch("[0-9a-f]{32}", ident):
            raise ValueError("invalid object identifier")
        return self.objects / (ident + (".part" if partial else ".kv"))

    def _recover(self):
        # A process lock establishes that every previous lease/writer is dead.
        with self._lock, self._tx():
            self.db.execute("DELETE FROM leases")
            self.db.execute("UPDATE objects SET state='T',expiry=0 WHERE state='W'")
            self.db.execute("UPDATE objects SET expiry=0 WHERE state='C'")
            self.db.execute("UPDATE objects SET charge=((size+?+?-1)/?+1)*?",
                            (HEADER_BYTES,self._unit,self._unit,self._unit))
            self.db.execute("UPDATE totals SET bytes=(SELECT COALESCE(SUM(charge),0) FROM objects) WHERE id=1")
        self.collect(force=True)
        # Streaming scan; no list of a terabyte's worth of filenames in RAM.
        with os.scandir(self.objects) as entries:
            for entry in entries:
                stem, ext = os.path.splitext(entry.name)
                row = self.db.execute("SELECT state FROM objects WHERE id=?", (stem,)).fetchone()
                if not row or ext != ".kv":
                    if not entry.is_file(follow_symlinks=False):
                        raise ValueError("unexpected non-file in owned object directory")
                    os.unlink(entry.path)
        os.fsync(self._dirfd)
        # A missing/truncated committed file is tombstoned, never rediscovered as hit.
        for ident, size in self.db.execute("SELECT id,size FROM objects WHERE state='C'"):
            try:
                valid = self._path(ident).stat().st_size == HEADER_BYTES + size
            except OSError:
                valid = False
            if not valid:
                self.db.execute("UPDATE objects SET state='T',expiry=0 WHERE id=?", (ident,))
        self.collect(force=True)
        # A smaller configured quota after restart must not serve an oversized
        # retained namespace; no old process leases survive exclusive startup.
        self.evict()
        self.collect(force=True)
        if self.failed or self.usage() > self.limits.quota:
            self.close()
            raise ValueError("recovery could not establish the hard quota")

    def usage(self):
        with self._lock:
            try:
                charged = self.db.execute("SELECT bytes FROM totals WHERE id=1").fetchone()[0]
                return self._metadata + charged
            except _ERRORS:
                # Matches the rest of the API: after close() every accessor is
                # falsy rather than raising sqlite3.ProgrammingError at a caller
                # that is merely logging usage during shutdown. The metadata
                # floor is already charged, so report it rather than 0.
                return self._metadata

    def clear_failure(self):
        """Explicit, operator-driven recovery from a latched failure.

        ``self.failed`` is a deliberate fail-closed latch: if a durable
        invalidation cannot be proved, the store must stop rather than serve a
        stale object. But there was NO recovery path -- once a capacity error
        (a full disk, or an exhausted SQLite index budget via PRAGMA
        max_page_count) was caught by ``invalidate()``/``collect()``/``evict()``,
        every operation was refused forever and the only remedy was reopening the
        store. Recovery stays explicit and is never automatic, so a real
        durability fault is still never masked: free the space, drain, then call
        this. Returns False on a closed store.
        """
        with self._lock:
            if self.closed:
                return False
            self.failed = False
            self.last_failure = None
            self.failure_count = 0
            return True

    def _expire(self):
        now = self.clock()
        self.db.execute("DELETE FROM leases WHERE expiry<=?", (now,))
        self.db.execute("UPDATE objects SET state='T',expiry=? WHERE state='W' AND expiry<=?",
                        (now + self.limits.grace_seconds, now))

    def reserve_write(self, namespace, key, size, owner):
        try:
            self._identity(namespace, key, owner)
            if not isinstance(size, int) or not 0 < size <= self.limits.max_object_bytes:
                return None
            with self._lock:
                if self.failed or self.closed:
                    return None
                self.collect()
                self.evict()
                if self.failed:
                    return None
                with self._tx():
                    self._expire()
                    if self.db.execute("SELECT 1 FROM objects WHERE ns=? AND key=?", (namespace,key)).fetchone():
                        return None
                    charge = self._charge(size)
                    if self.usage() + charge > self.limits.quota:
                        return None
                    if self.db.execute("SELECT COUNT(*) FROM objects").fetchone()[0] >= self.limits.max_objects:
                        return None
                    fs = os.statvfs(self.root)
                    pending = self.db.execute("SELECT COALESCE(SUM(charge),0) FROM objects WHERE state='W'").fetchone()[0]
                    if fs.f_bavail * fs.f_frsize < self.limits.free_bytes + pending + charge + self._metadata:
                        return None
                    if fs.f_favail < self.limits.free_inodes + self.db.execute("SELECT COUNT(*) FROM objects WHERE state='W'").fetchone()[0] + 1:
                        return None
                    ident = uuid.uuid4().hex
                    now = self.clock()
                    self.db.execute("INSERT INTO objects VALUES(?,?,?,?,?,'W',?,?,?)",
                                    (ident,namespace,key,size,charge,now,now+self.limits.lease_seconds,owner))
                    self.db.execute("UPDATE totals SET bytes=bytes+? WHERE id=1", (charge,))
                    return ident
        except ValueError:
            # Caller error (namespace/key/owner shape, or a size that is not a
            # positive int): refuse THIS call. A bad argument must not take the
            # whole rank's store out of service.
            return None
        except _ERRORS as exc:
            # OSError / sqlite3.Error / BufferError are STORAGE failures. Record
            # them so a full disk or an exhausted index budget
            # (PRAGMA max_page_count) is distinguishable from the benign
            # "key already exists" refusal -- previously both were a bare None and
            # the store reported itself healthy while silently ceasing to persist.
            # Do NOT latch self.failed: that refuses every operation forever with
            # no recovery path, which is worse than a reported, recoverable refusal.
            self.failure_count += 1
            self.last_failure = f"{type(exc).__name__}: {exc}"
            return None

    def write(self, token, chunks):
        """Commit exactly the reserved bytes; never hold a payload-sized copy.

        The payload I/O deliberately runs OUTSIDE self._lock. The temporary file is
        created with O_EXCL under a token-derived name, so no reader can observe a
        partial object, and the row stays state='W' until the index commit below.
        The reservation is re-validated under the lock before commit, so an expired
        or failed reservation still cannot publish a payload.
        """
        path = None
        try:
            with self._lock:
                row = self.db.execute("SELECT ns,key,size FROM objects WHERE id=? AND state='W' AND expiry>?", (token,self.clock())).fetchone()
                if not row or self.failed or self.closed:
                    return False
                ns, key, size = row
                path = self._path(token, True)
            digest = hashlib.sha256()
            written = 0
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb", buffering=0) as f:
                self._write_all(f, bytes(HEADER_BYTES))
                for chunk in chunks:
                    view = memoryview(chunk).cast("B")
                    if written + len(view) > size:
                        raise ValueError("payload exceeds reservation")
                    for offset in range(0, len(view), self.limits.io_chunk_bytes):
                        part = view[offset:offset+self.limits.io_chunk_bytes]
                        self._write_all(f, part)
                        digest.update(part)
                    written += len(view)
                if written != size:
                    raise ValueError("short payload")
                header = _HEADER.pack(_MAGIC,size,bytes.fromhex(ns),hashlib.sha256(key).digest(),digest.digest())
                f.seek(0)
                self._write_all(f, header + bytes(HEADER_BYTES-len(header)))
                os.fsync(f.fileno())
                self._drop_cache(f.fileno())
            with self._lock:
                if self.failed or self.closed:
                    self._unlink_temp(path); return False
                if not self.db.execute("SELECT 1 FROM objects WHERE id=? AND state='W' AND expiry>?",
                                       (token,self.clock())).fetchone():
                    self._unlink_temp(path); return False
                os.rename(path, self._path(token))
                os.fsync(self._dirfd)
                with self._tx():
                    self.db.execute("UPDATE objects SET state='C',expiry=?,touched=? WHERE id=?",
                                    (self.clock()+self.limits.lease_seconds,self.clock(),token))
                return True
        except Exception:
            if path is not None:
                self._unlink_temp(path)
            self._retire(token)
            return False

    @staticmethod
    def _unlink_temp(path):
        try:
            os.unlink(path)
        except (FileNotFoundError, OSError):
            pass

    @staticmethod
    def _drop_cache(fd):
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            except OSError:
                pass  # Advisory only; fsync/checksum provide correctness.

    @staticmethod
    def _write_all(f, data):
        view = memoryview(data)
        while view:
            n = f.write(view)
            if not n:
                raise OSError("short write")
            view = view[n:]

    def exists(self, namespace, key):
        try:
            self._identity(namespace,key)
            with self._lock:
                row = self.db.execute("SELECT id,size FROM objects WHERE ns=? AND key=? AND state='C'", (namespace,key)).fetchone()
                return bool(row and not self.failed and self._path(row[0]).stat().st_size == HEADER_BYTES+row[1])
        except _ERRORS:
            return False

    def reserve_read(self, namespace, key, owner):
        try:
            self._identity(namespace,key,owner)
            with self._lock, self._tx():
                if self.failed or self.closed:
                    return None
                self._expire()
                if self.db.execute("SELECT COUNT(*) FROM leases").fetchone()[0] >= self.limits.max_leases:
                    return None
                row = self.db.execute("SELECT id,size FROM objects WHERE ns=? AND key=? AND state='C'", (namespace,key)).fetchone()
                if not row or self._path(row[0]).stat().st_size != HEADER_BYTES+row[1]:
                    return None
                token = uuid.uuid4().hex
                self.db.execute("INSERT INTO leases VALUES(?,?,?,?)", (token,row[0],owner,self.clock()+self.limits.lease_seconds))
                return token
        except _ERRORS:
            return None

    def read_into(self, token, buffers):
        """Verify full payload checksum; caller must not publish/copy before True.

        Buffers are owned staging, not live GPU destinations. Partial buffer writes
        on False must be discarded. Each buffer is filled without payload copies.
        """
        with self._lock:
            ident = None
            try:
                row = self.db.execute("SELECT o.id,o.ns,o.key,o.size FROM objects o JOIN leases l ON o.id=l.object WHERE l.token=? AND l.expiry>? AND o.state IN ('C','T')", (token,self.clock())).fetchone()
                if not row or self.failed or self.closed:
                    return False
                ident, ns, key, size = row
                fd = os.open(self._path(ident), os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb", buffering=0) as f:
                    header = f.read(HEADER_BYTES)
                    if len(header) != HEADER_BYTES:
                        raise ValueError("short header")
                    magic, length, ns_hash, key_hash, checksum = _HEADER.unpack_from(header)
                    if (magic,length,ns_hash,key_hash) != (_MAGIC,size,bytes.fromhex(ns),hashlib.sha256(key).digest()) or any(header[_HEADER.size:]):
                        raise ValueError("incompatible header")
                    digest = hashlib.sha256()
                    read = 0
                    for buffer in buffers:
                        view = memoryview(buffer).cast("B")
                        if read + len(view) > size:
                            raise ValueError("destination length exceeds payload")
                        start = 0
                        while start < len(view):
                            part = view[start:start+self.limits.io_chunk_bytes]
                            n = f.readinto(part)
                            if not n:
                                raise ValueError("short read")
                            digest.update(part[:n])
                            start += n
                        read += len(view)
                    if read != size or f.read(1) or digest.digest() != checksum:
                        raise ValueError("payload integrity failure")
                    self._drop_cache(f.fileno())
                self.db.execute("UPDATE objects SET touched=? WHERE id=?", (self.clock(),ident))
                return True
            except Exception:
                if ident:
                    self._retire(ident)
                return False

    def _retire(self, ident):
        try:
            with self._tx():
                self.db.execute("UPDATE objects SET state='T',expiry=? WHERE id=?", (self.clock()+self.limits.grace_seconds,ident))
        except _ERRORS:
            self.failed = True  # Fail closed if durable invalidation is unavailable.

    def invalidate(self, namespace, key):
        try:
            self._identity(namespace,key)
            with self._lock:
                row = self.db.execute("SELECT id FROM objects WHERE ns=? AND key=?", (namespace,key)).fetchone()
                if row:
                    self._retire(row[0])
                return not self.failed
        except _ERRORS:
            self.failed = True
            return False

    def release(self, token):
        try:
            with self._lock, self._tx():
                self.db.execute("DELETE FROM leases WHERE token=?", (token,))
                self.db.execute("UPDATE objects SET expiry=0 WHERE id=? AND state='C'", (token,))
                self.db.execute("UPDATE objects SET state='T',expiry=? WHERE id=? AND state='W'", (self.clock()+self.limits.grace_seconds,token))
            return not self.failed
        except _ERRORS:
            self.failed = True
            return False

    def lease_valid(self, token):
        """Check a retry receipt without extending its original lease window."""
        try:
            with self._lock:
                if self.failed or self.closed:
                    return False
                now = self.clock()
                return self.db.execute(
                    "SELECT 1 FROM leases l JOIN objects o ON l.object=o.id "
                    # Same states read_into() serves with a live lease ('C','T'): a
                    # retry receipt must not call a still-servable read dead just
                    # because invalidate() tombstoned the object underneath it.
                    "WHERE l.token=? AND l.expiry>? AND o.state IN ('C','T') "
                    "UNION ALL SELECT 1 FROM objects WHERE id=? "
                    "AND state IN ('W','C') AND expiry>? LIMIT 1",
                    (token, now, token, now)).fetchone() is not None
        except _ERRORS:
            return False

    def renew(self, token):
        try:
            with self._lock:
                if self.failed or self.closed:
                    return False
                now = self.clock()
                n = self.db.execute("UPDATE leases SET expiry=? WHERE token=? AND expiry>?", (now+self.limits.lease_seconds,token,now)).rowcount
                n += self.db.execute("UPDATE objects SET expiry=? WHERE id=? AND state IN ('W','C') AND expiry>?",  (now+self.limits.lease_seconds,token,now)).rowcount
                return bool(n)
        except _ERRORS:
            return False

    def evict(self):
        """Persist tombstones high->low. Active read leases veto victim selection."""
        if self.closed:
            return 0
        with self._lock:
            try:
                with self._tx():
                    self._expire()
                    usage = self.usage()
                    if usage < self.limits.high:
                        return 0
                    # Existing tombstones already count toward eventual reclamation.
                    marked = self.db.execute("SELECT COALESCE(SUM(charge),0) FROM objects WHERE state='T'").fetchone()[0]
                    count = 0
                    while usage-marked > self.limits.low:
                        row = self.db.execute("SELECT id,charge FROM objects WHERE state='C' AND expiry<=? AND NOT EXISTS(SELECT 1 FROM leases WHERE object=objects.id) ORDER BY touched LIMIT 1", (self.clock(),)).fetchone()
                        if not row:
                            break
                        self.db.execute("UPDATE objects SET state='T',expiry=? WHERE id=?", (self.clock()+self.limits.grace_seconds,row[0]))
                        marked += row[1]
                        count += 1
                    return count
            except _ERRORS:
                self.failed = True
                return 0

    def collect(self, *, force=False):
        """Unlink only durable tombstones, after grace and all live readers drain.

        force is used only during exclusive-owner startup; it never bypasses leases.
        """
        if self.closed:
            return 0
        with self._lock:
            try:
                with self._tx():
                    self._expire()
                while True:
                    row = self.db.execute("SELECT id,charge FROM objects WHERE state='T' AND (? OR expiry<=?) AND NOT EXISTS(SELECT 1 FROM leases WHERE object=objects.id) LIMIT 1", (force,self.clock())).fetchone()
                    if not row:
                        break
                    for partial in (False,True):
                        try:
                            self._path(row[0],partial).unlink()
                        except FileNotFoundError:
                            pass
                    os.fsync(self._dirfd)
                    with self._tx():
                        self.db.execute("DELETE FROM objects WHERE id=?", (row[0],))
                        self.db.execute("UPDATE totals SET bytes=bytes-? WHERE id=1", (row[1],))
            except _ERRORS:
                self.failed = True

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed = True
                if self.db is not None:
                    self.db.close()
                if self._dirfd is not None:
                    os.close(self._dirfd)
                if self._lockfile is not None:
                    self._lockfile.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
