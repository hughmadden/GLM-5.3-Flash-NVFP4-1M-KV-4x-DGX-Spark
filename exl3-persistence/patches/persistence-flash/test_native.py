#!/usr/bin/env python3
"""CPU tests executing actual patched AST methods with inert dependency fakes.

Pass --source-tree containing pinned vllm/, or the ``vllm`` package directory
itself (the bare overlay layout). Only manifest-listed files are copied into a
temporary tree. A pristine tree has the real patches applied there; an
already-ported tree is used as-is. --no-apply runs the suite against the tree
exactly as given, which is how the pre-fix reproduction (every contract failing
on pristine source) is demonstrated.
No torch, vLLM import, network, CUDA, deployment configuration or model needed.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict, defaultdict
from itertools import chain
from dataclasses import dataclass, field
from enum import Enum, auto
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("native_apply", HERE / "apply.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
ROOT = None
ORIGINAL = None
BARE = False
OFF = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/"
CORE = "vllm/v1/core/sched/scheduler.py"
LOG = NS(debug=lambda *a, **kw: None, warning=lambda *a, **kw: None, error=lambda *a, **kw: None)


def extract(path, name, namespace=None, *, root=None, class_name=None):
    tree = ast.parse(((root or ROOT) / path).read_text())
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    node.decorator_list = [] if isinstance(node, ast.FunctionDef) else node.decorator_list
    env = {"dataclass": dataclass, "field": field, "logger": LOG, "time": time,
           "defaultdict": defaultdict, **(namespace or {})}
    exec("from __future__ import annotations\n" + ast.unparse(node), env)
    return env[name]


def classes():
    env = {"KVConnectorWorkerMetadata": object,
           # storable_chunks ceils at finish since 0007; extracted methods
           # resolve module helpers through this env.
           "cdiv": lambda a, b: -(-a // b)}
    for name in ["DirectionalTransferStats", "TransferStats", "OffloadingWorkerMetadata"]:
        env[name] = extract(OFF + "common.py", name, env)
    env["TransferJobStatus"] = extract(OFF + "scheduler.py", "TransferJobStatus")
    # TransferResult moved worker/worker.py -> kv_offload/base.py and lost
    # its transfer_type field (upstream f237e16b41, #45053).
    env["TransferResult"] = extract("vllm/v1/kv_offload/base.py", "TransferResult")
    return env


def bind(obj, path, name, env=None):
    method = extract(path, name, env)
    setattr(obj, name, lambda *args, **kwargs: method(obj, *args, **kwargs))


def gpu_spec(block_ids=()):
    """Stand-in for GPULoadStoreSpec; only .block_ids is read by our code."""
    return NS(block_ids=list(block_ids))


class WorkerTests(unittest.TestCase):
    def worker(self, submit=True):
        env = classes()
        calls, results = [], []
        # submit_store / submit_load replaced the single transfer_async
        # (upstream f237e16b41, #45053).
        inner = NS(submit_store=lambda jid, src, dst: calls.append(("submit", jid)) or submit,
                   submit_load=lambda jid, src, dst: calls.append(("submit", jid)) or submit,
                   wait=lambda ids: calls.append(("drain", set(ids))),
                   get_finished=lambda: list(results), shutdown=lambda: calls.append(("shutdown",)))
        env["GPULoadStoreSpec"] = NS
        env["OffloadingWorkerMetadata"] = env["OffloadingWorkerMetadata"]
        cls = extract(OFF + "worker.py", "OffloadingConnectorWorker", env)
        obj = cls(NS(replicated_layout=False, config=NS(parallel=NS(rank=0))), None, None)
        obj.worker = inner
        return obj, inner, calls, results, env

    def test_failed_submit_drains_before_outcome_and_suppresses_duplicate(self):
        worker, _, calls, results, env = self.worker(False)
        worker.start_kv_transfers(NS(load_jobs={7: NS(req_id="r", src_spec=None, dst_spec=gpu_spec([0, 3, 8]))}))
        self.assertEqual(calls, [("submit", 7), ("drain", {7})])
        results.append(env["TransferResult"](7, True))
        self.assertEqual(worker.get_finished(set()), (set(), set()))
        meta = worker.build_connector_worker_meta()
        self.assertEqual(meta.completed_jobs, {7: 1})
        self.assertEqual(meta.failed_jobs, {7})
        self.assertEqual(meta.failed_load_blocks, {7: {3, 8}})
        worker.get_finished(set())
        self.assertIsNone(worker.build_connector_worker_meta())

    def test_cannot_certify_failed_submission_when_drain_fails(self):
        worker, inner, _, _, _ = self.worker(False)
        def fail(ids):
            raise RuntimeError("undrained")
        inner.wait = fail
        with self.assertRaisesRegex(RuntimeError, "undrained"):
            worker._submit_load(2, None, gpu_spec([4]))
        self.assertIn(2, worker._active_jobs)
        self.assertFalse(worker._failed_submissions)

    def test_failed_store_submission_drains_before_synthetic_outcome(self):
        # Direction-specific: the store leg is a separate upstream entry point
        # from the load leg since the transfer_async split.
        worker, _, calls, results, env = self.worker(False)
        worker.prepare_store_kv(NS(store_jobs={3: NS(src_spec=gpu_spec([5]), dst_spec=None)}))
        self.assertEqual(calls, [])
        worker.start_kv_transfers(NS(load_jobs={}))
        self.assertEqual(calls, [("submit", 3), ("drain", {3})])
        results.append(env["TransferResult"](3, True))
        self.assertEqual(worker.get_finished(set()), (set(), set()))
        meta = worker.build_connector_worker_meta()
        self.assertEqual(meta.completed_jobs, {3: 1})
        self.assertEqual(meta.failed_jobs, {3})
        self.assertFalse(meta.failed_load_blocks)

    def test_store_preemption_and_shutdown_drain_before_release(self):
        worker, _, calls, _, _ = self.worker(False)
        worker.prepare_store_kv(NS(store_jobs={4: NS(src_spec=gpu_spec(), dst_spec=None)}))
        worker.handle_preemptions(NS(jobs_to_flush={4}, store_jobs={}))
        self.assertEqual(calls, [("submit", 4), ("drain", {4}), ("drain", {4})])
        worker.prepare_store_kv(NS(store_jobs={5: NS(src_spec=gpu_spec(), dst_spec=None)}))
        worker.shutdown()
        self.assertEqual(calls[-2:], [("drain", {4, 5}), ("shutdown",)])
        self.assertFalse(worker._active_jobs)

    def test_async_failure_and_success_report_only_metadata(self):
        worker, _, _, results, env = self.worker()
        worker.start_kv_transfers(NS(load_jobs={9: NS(req_id="r", src_spec=None, dst_spec=gpu_spec([10]))}))
        self.assertEqual(worker.get_finished({"r"}), (set(), set()))
        results.append(env["TransferResult"](9, False))
        self.assertEqual(worker.get_finished({"r"}), (set(), set()))
        self.assertEqual(worker.build_connector_worker_meta().failed_load_blocks, {9: {10}})


class CompletionTests(unittest.TestCase):
    def state(self, store=False, finished=False, groups=None):
        env = classes()
        calls = []
        req = NS(request_id="r", is_finished=lambda: finished)
        state = NS(req=req, req_context="ctx", transfer_jobs={1}, group_states=groups or (), load_failed=False,
                   finished_signaled=False, finish_pending=finished, finish_store_tokens=0,
                   finish_failed_store_keys=set(), finish_store_abandoned=False,
                   storable_chunks=lambda config, state, tokens: 0)
        manager = NS(complete_store=lambda *a, **kw: calls.append(("store", a, kw)),
                     complete_load=lambda *a: calls.append(("load", a)),
                     on_load_failure=lambda *a: calls.append(("invalidate", a)),
                     on_request_finished=lambda *a: calls.append(("finish", a)),
                     has_pending_work=lambda: False, can_store=lambda: True,
                     reset_cache=lambda: calls.append(("reset",)))
        obj = NS(_jobs={1: env["TransferJobStatus"]("r", 4, {"k"}, store)},
                 _req_status={"r": state}, manager=manager, _connector_stats=None,
                 _stale_job_threshold=0, _job_counter=2, _chunks_being_loaded={"k"},
                 _block_id_to_pending_jobs={}, _current_batch_load_jobs={},
                 _current_batch_jobs_to_flush=set(), _current_batch_allocated_block_ids=set(),
                 _events_tracker=NS(reset=lambda: None),
                 config=NS(kv_group_configs=(), blocks_per_chunk=1))
        bind(obj, OFF+"scheduler.py", "update_connector_output", env)
        bind(obj, OFF+"scheduler.py", "reset_cache", env)
        bind(obj, OFF+"scheduler.py", "has_pending_push_work", env)
        bind(obj, OFF+"scheduler.py", "_has_finished_store_frontier", env)
        return obj, state, calls, env

    def output(self, env, count, failed=False, blocks=None):
        return NS(kv_connector_worker_meta=env["OffloadingWorkerMetadata"](
            completed_jobs={1: count}, failed_jobs={1} if failed else set(),
            failed_load_blocks={1: blocks} if blocks else {}),
            invalid_block_ids=set(), finished_recving=set(), finished_sending=set())

    def test_mixed_rank_load_not_visible_until_last_drain(self):
        group = NS(next_stored_chunk_idx=7)
        obj, state, calls, env = self.state(groups=(group,))
        first = self.output(env, 1, True, {3, 10})
        obj.update_connector_output(first)
        self.assertFalse(first.invalid_block_ids)
        self.assertFalse(first.finished_recving)
        self.assertFalse(calls)
        self.assertFalse(obj.reset_cache())
        obj.update_connector_output(self.output(env, 2))
        self.assertFalse(calls)
        last = self.output(env, 1)
        obj.update_connector_output(last)
        self.assertEqual(last.invalid_block_ids, {3, 10})
        self.assertEqual(last.finished_recving, {"r"})
        self.assertEqual([c[0] for c in calls], ["load", "invalidate"])
        self.assertTrue(state.load_failed)
        self.assertEqual(group.next_stored_chunk_idx, 0)
        self.assertFalse(obj._chunks_being_loaded)

    def test_failed_store_vetoes_publication_and_rewinds_safe_rows(self):
        full = NS(offload_keys=["k", "l"], block_ids=[1, 2], next_stored_chunk_idx=2)
        swa = NS(offload_keys=["k", "m"], block_ids=[7, 8], next_stored_chunk_idx=2)
        obj, _, calls, env = self.state(store=True, groups=(full, swa))
        obj.config.kv_group_configs = (NS(sliding_window_size_in_chunks=None), NS(sliding_window_size_in_chunks=1))
        obj.update_connector_output(self.output(env, 3))
        self.assertFalse(calls)
        obj.update_connector_output(self.output(env, 1, True))
        self.assertEqual(calls, [("store", ({"k"}, "ctx"), {"success": False})])
        self.assertEqual((full.next_stored_chunk_idx, swa.next_stored_chunk_idx), (0, 0))
        self.assertEqual(full.block_ids, [1, 2])
        self.assertEqual(swa.block_ids, [0, 8])

    def test_failed_store_never_becomes_native_cpu_hit(self):
        obj, _, _, env = self.state(store=True)
        entries = {"k": NS(is_ready=False, ref_cnt=1)}
        released = []
        manager = obj.manager
        manager.counts = None
        manager.events = []
        manager.medium = "cpu"
        manager._num_write_pending_chunks = 1
        manager._num_evictable_cache_chunks = 0
        manager._policy = NS(get=entries.get, remove=entries.pop,
                             mark_evictable=lambda key: None)
        manager._free_chunk = released.append
        # lookup returns LookupResult since upstream bb61177e49 (#46363).
        result = extract("vllm/v1/kv_offload/base.py", "LookupResult",
                         {"Enum": Enum, "auto": auto})
        cpu_env = {"LookupResult": result,
                   "OffloadingEvent": lambda **kw: NS(**kw)}
        bind(manager, "vllm/v1/kv_offload/cpu/manager.py", "complete_store", cpu_env)
        bind(manager, "vllm/v1/kv_offload/cpu/manager.py", "lookup", cpu_env)
        obj.update_connector_output(self.output(env, 3))
        self.assertIs(manager.lookup("k", None), result.HIT_PENDING)
        self.assertFalse(released)
        obj.update_connector_output(self.output(env, 1, True))
        self.assertIs(manager.lookup("k", None), result.MISS)
        self.assertEqual(len(released), 1)
        self.assertFalse(manager.events)

    def test_success_and_cancel_finalize_after_all_ranks(self):
        obj, _, calls, env = self.state(store=True, finished=True)
        obj.update_connector_output(self.output(env, 3))
        self.assertFalse(calls)
        obj.update_connector_output(self.output(env, 1))
        self.assertEqual([c[0] for c in calls], ["store", "finish"])
        self.assertEqual(calls[0][2], {"success": True})
        self.assertFalse(obj._req_status)
        obj.update_connector_output(self.output(env, 4))
        self.assertEqual(len(calls), 2)

    def test_reset_refuses_ownership_then_stale_completions_are_inert(self):
        obj, _, calls, env = self.state(store=True)
        self.assertFalse(obj.reset_cache())
        self.assertEqual(len(obj._jobs), 1)
        obj.update_connector_output(self.output(env, 4))
        self.assertTrue(obj.reset_cache())
        before = list(calls)
        obj.update_connector_output(self.output(env, 4, True))
        self.assertEqual(calls, before)

    def test_metadata_union_is_pure_and_failure_survives_aggregation(self):
        env = classes()
        a = env["OffloadingWorkerMetadata"](completed_jobs={1: 1}, failed_jobs={1}, failed_load_blocks={1: {8}})
        b = env["OffloadingWorkerMetadata"](completed_jobs={1: 3}, failed_load_blocks={1: {9}})
        merged = a.aggregate(b)
        self.assertEqual(merged.completed_jobs, {1: 4})
        self.assertEqual(merged.failed_jobs, {1})
        self.assertEqual(merged.failed_load_blocks, {1: {8, 9}})
        self.assertEqual(a.failed_load_blocks, {1: {8}})


class CursorTests(unittest.TestCase):
    def state(self, configs=None, select=None):
        env = classes()
        groups = configs or [(64, False, None, None)]
        # `align` drives the upstream reachability mask (SWA/retention filters);
        # `_reachable_store_block_mask` itself is upstream and is faked here.
        configs = [NS(group_idx=i, tokens_per_chunk=size, tokens_per_block=size,
                      is_eagle_group=eagle, alignment_block_count=align,
                      requires_cow_source=False, kv_cache_spec=None,
                      sliding_window_size_in_chunks=tail)
                   for i, (size, eagle, align, tail) in enumerate(groups)]
        states = tuple(NS(offload_keys=[(i, j) for j in range(24)],
                          block_ids=[i*100+j+1 for j in range(24)], next_stored_chunk_idx=0)
                       for i in range(len(configs)))
        # num_prompt_tokens=0 keeps every step on the decode side of
        # `storable_chunks`, where the EAGLE/MTP volatile tail is excluded.
        req = NS(num_computed_tokens=0, num_tokens=256, num_prompt_tokens=0,
                 status=None, shared_prefix_boundary=None, is_finished=lambda: False)
        config = NS(blocks_per_chunk=1, offload_prompt_only=False,
                    kv_group_configs=configs, num_workers=4, retention_interval=None,
                    alignment_tokens=None, dcp_world_size=1)
        status = NS(req=req, req_context=None, group_states=states, transfer_jobs=set(),
                    max_offload_tokens=None, config=config,
                    finish_pending=False, finish_store_tokens=0,
                    finish_failed_store_keys=set(), finish_store_abandoned=False)
        # storable_chunks is upstream's shared per-group bound; our frontier
        # walk must agree with it exactly, so bind the real method.
        bind(status, OFF+"scheduler.py", "storable_chunks", env)
        offered, accepted, owned = [], [], set()
        def prepare(keys, context):
            offered.append(list(keys))
            chosen, skipped = select(keys, len(offered)) if select else (list(keys), [])
            if chosen is None:
                return None
            chosen = [key for key in chosen if key not in owned]
            owned.update(chosen)
            accepted.extend(chosen)
            return NS(keys_to_store=chosen, skipped_keys=skipped, store_spec=None)
        def mask(group_config, start_chunk_idx, end_chunk_idx, **kwargs):
            if group_config.alignment_block_count is None:
                return None
            return [False] * ((end_chunk_idx - start_chunk_idx)
                              * config.blocks_per_chunk)
        obj = NS(config=config,
                 _req_status={"r": status},
                 manager=NS(prepare_store=prepare, can_store=lambda: True),
                 _touch=lambda s: None, _jobs={}, _block_id_to_pending_jobs={},
                 _generate_job_id=lambda: len(offered),
                 _reachable_store_block_mask=mask,
                 _final_swa_alignment_blocks=lambda group_config: None,
                 _events_tracker=NS(record_store=lambda *a: None, reset=lambda: None),
                 _connector_stats=NS(increase_counter=lambda *a: None),
                 _current_batch_allocated_block_ids=set(),
                 _current_batch_jobs_to_flush=set())
        env.update(GPULoadStoreSpec=lambda ids, **kw: NS(block_ids=ids, **kw),
                   TransferJob=lambda **kw: NS(**kw), chain=chain,
                   RequestStatus=NS(FINISHED_ABORTED=object()),
                   _ConnectorMetricName=NS(ALLOCATION_FAILURE="alloc"))
        bind(obj, OFF+"scheduler.py", "_advance_store_frontiers", env)
        bind(obj, OFF+"scheduler.py", "_calc_num_offloadable_tokens", env)
        bind(obj, OFF+"scheduler.py", "_has_finished_store_frontier", env)
        bind(obj, OFF+"scheduler.py", "_build_store_jobs", env)
        return obj, req, states, offered, accepted

    def step(self, obj, count=256):
        return obj._build_store_jobs(
            NS(num_scheduled_tokens={"r": count}, finished_req_ids=()))

    def test_eagle_excludes_only_volatile_tail_then_retries_it(self):
        obj, req, states, offered, _ = self.state([(64, True, None, None)])
        self.step(obj)
        self.assertEqual(states[0].next_stored_chunk_idx, 3)
        req.num_computed_tokens, req.num_tokens = 256, 384
        self.step(obj, 128)
        self.assertEqual(offered, [[(0, i) for i in range(3)], [(0, 3), (0, 4)]])
        self.assertEqual(states[0].next_stored_chunk_idx, 5)

    def test_partial_decline_retries_suffix_without_duplicate_jobs(self):
        obj, req, states, offered, accepted = self.state(select=lambda k, n: (k[:2] if n == 1 else k, []))
        self.step(obj)
        self.assertEqual(states[0].next_stored_chunk_idx, 2)
        req.num_computed_tokens, req.num_tokens = 256, 384
        self.step(obj, 128)
        self.assertEqual(offered[1], [(0, i) for i in range(2, 6)])
        self.assertEqual(len(accepted), len(set(accepted)))

    def test_unequal_group_frontiers_zero_group_and_nonprefix_admission(self):
        select = lambda keys, n: ([key for key in keys if key[0] == 1 and key[1] > 1], [])
        obj, _, states, _, _ = self.state([(64, False, None, None), (16, True, None, None)], select)
        jobs = self.step(obj)
        src = next(iter(jobs.values())).src_spec
        self.assertEqual(src.group_sizes, [0, 13])
        self.assertEqual([s.next_stored_chunk_idx for s in states], [0, 0])
        self.assertEqual(self.step(obj), {})  # manager owns earlier admissions

    def test_skip_only_never_submits_zero_key_job(self):
        obj, _, states, _, _ = self.state(select=lambda keys, n: ([], keys))
        self.assertEqual(self.step(obj), {})
        self.assertEqual(states[0].next_stored_chunk_idx, 4)

    def test_empty_and_none_declines_are_retryable(self):
        for choice in [[], None]:
            obj, _, states, offered, _ = self.state(select=lambda keys, n: (choice, []))
            self.assertEqual(self.step(obj), {})
            self.assertEqual(states[0].next_stored_chunk_idx, 0)
            self.step(obj)
            self.assertEqual(offered[0], offered[1])

    def test_permanent_swa_filters_advance_but_not_eagle_tail(self):
        obj, _, states, offered, _ = self.state([(64, True, 8, 1)])
        self.assertEqual(self.step(obj), {})
        self.assertFalse(offered)
        self.assertEqual(states[0].next_stored_chunk_idx, 3)

    def test_null_rows_are_never_read_after_recycling(self):
        obj, _, states, _, _ = self.state()
        states[0].block_ids[:4] = [0, 2, 0, 4]
        jobs = self.step(obj)
        self.assertEqual(next(iter(jobs.values())).src_spec.block_ids, [2, 4])
        self.assertEqual(states[0].next_stored_chunk_idx, 4)

    def test_evict_only_mode_offers_nothing(self):
        """0009 R1: under EVICT_ONLY the eager store path offers nothing,
        ever -- a full scheduled step produces zero store batches and the
        manager's prepare_store is never consulted."""
        obj, _, _, offered, _ = self.state()
        obj._evict_only = True
        bind(obj, OFF+"scheduler.py", "_drain_eviction_queue")
        calls = []
        real = obj.manager.prepare_store
        obj.manager.prepare_store = lambda keys, ctx: (calls.append(list(keys)),
                                                       real(keys, ctx))[1]
        self.assertEqual(self.step(obj), {})
        self.assertEqual(offered, [])
        self.assertEqual(calls, [])

    def test_evict_sink_holds_copies_and_skips_at_cap(self):
        """0009: the pool's eviction sink holds blocks for background
        copies (deferred free), declines duplicates, and past the credit
        cap declines with the skip counter -- never blocking allocation."""
        obj, _, _, _, _ = self.state()
        obj._evict_only = True
        obj._held_evictions = {}
        obj._eviction_hold_cap = 2
        obj._eviction_skips = 0
        # BlockHashWithGroupId is block_hash bytes + 4-byte big-endian
        # group id (kv_cache_utils): build it directly, and bind the sink
        # with the real leaf helpers extracted (no vllm import needed).
        # The leaf helpers are one-liners (kv_cache_utils): key[:-4] and
        # int.from_bytes(key[-4:], "big"); make_offload_key is manifest-
        # listed so it extracts for real.
        env = {
            "get_block_hash": lambda key: key[:-4],
            "get_group_id": lambda key: int.from_bytes(key[-4:], "big"),
            "make_offload_key": extract(
                "vllm/v1/kv_offload/base.py", "make_offload_key",
                {"OffloadKey": bytes}),
        }
        bind(obj, OFF+"scheduler.py", "_defer_evicted_block", env)
        held = 0
        for i in range(4):
            bh = bytes([9]) * 30 + (0).to_bytes(4, "big")
            block = NS(block_hash=bh, block_id=100 + i)
            r = obj._defer_evicted_block(block)
            if r:
                held += 1
        self.assertEqual(held, 2)                    # cap respected
        self.assertEqual(obj._eviction_skips, 2)     # two declined + counted
        # idempotent: re-offering a held block declines without counting
        bh = bytes([9]) * 30 + (0).to_bytes(4, "big")
        self.assertFalse(obj._defer_evicted_block(NS(block_hash=bh, block_id=100)))
        self.assertEqual(obj._eviction_skips, 2)
        # eager mode: the sink never holds
        obj._evict_only = False
        bh = bytes([9]) * 30 + (1).to_bytes(4, "big")
        self.assertFalse(obj._defer_evicted_block(NS(block_hash=bh, block_id=200)))

    def test_drain_eviction_queue_registers_job_and_releases_declined(self):
        """0009: the drain turns held blocks into ONE job with a properly
        registered TransferJobStatus (the completion loop keys on it), and
        releases blocks whose keys the manager declined."""
        obj, _, _, _, _ = self.state()
        obj._evict_only = True
        k1 = bytes([2]) + (0).to_bytes(4, "big")
        k2 = bytes([3]) + (0).to_bytes(4, "big")
        obj._held_evictions = {
            100: (k1, 0),
            101: (k2, 0),
        }
        obj._eviction_store_per_step = 8
        obj._eviction_job_blocks = {}
        obj._block_pool = None
        released = []
        obj._jobs = {}
        counter = {"n": 0}
        def gen_id():
            counter["n"] += 1
            return counter["n"]
        obj._generate_job_id = gen_id
        real_prepare = obj.manager.prepare_store
        def prepare(keys, ctx):
            # admit the first key, decline the second (durable)
            out = real_prepare(keys[:1], ctx)
            out.skipped_keys = list(keys[1:])
            return out
        obj.manager.prepare_store = prepare
        env = {
            "GPULoadStoreSpec": lambda ids, **kw: NS(block_ids=ids, **kw),
            "TransferJobStatus": extract(OFF+"scheduler.py", "TransferJobStatus"),
            "TransferJob": extract(OFF+"common.py", "TransferJob"),
        }
        bind(obj, OFF+"scheduler.py", "_drain_eviction_queue", env)
        jobs = obj._drain_eviction_queue()
        self.assertEqual(len(jobs), 1)
        job_id = next(iter(jobs))
        status = obj._jobs[job_id]
        self.assertTrue(status.is_store)
        self.assertEqual(status.pending_count, obj.config.num_workers)
        # the admitted block stays held; the declined one was released
        self.assertIn(100, obj._held_evictions)
        self.assertNotIn(101, obj._held_evictions)
        self.assertEqual(obj._eviction_job_blocks[job_id], [100])

    def test_native_cpu_policy_skips_not_pending_writes(self):
        prepare = extract("vllm/v1/kv_offload/cpu/manager.py", "prepare_store", {"PrepareStoreOutput": lambda **kw: NS(**kw)})
        policy = {"ready": NS(is_ready=True), "pending": NS(is_ready=False)}
        def never(*args, **kwargs):
            raise AssertionError("prepare_store must not call lookup")
        obj = NS(counts=OrderedDict(ready=2, pending=2), store_threshold=2,
                 max_tracker_size=64, stores_skipped_in_current_batch=0,
                 _policy=NS(get=policy.get), lookup=never,
                 allocation_sizes_in_current_batch=[],
                 _get_load_store_spec=lambda *a: None)
        bind(obj, "vllm/v1/kv_offload/cpu/manager.py", "_record_accesses")
        output = prepare(obj, ["ready", "pending", "policy-skip"], None)
        self.assertEqual(output.skipped_keys, ["policy-skip", "ready"])
        self.assertFalse(output.keys_to_store)
        # Upstream's own store-offer accounting bumps counts; the acknowledged
        # skips are still produced without a reuse-counter-mutating lookup.
        self.assertEqual(dict(obj.counts), {"ready": 3, "pending": 3, "policy-skip": 1})


class RecoveryTests(unittest.TestCase):
    def test_unequal_groups_conservative_recompute_preserves_tokens(self):
        update = extract(CORE, "_update_requests_with_invalid_blocks")
        req = NS(request_id="r", num_computed_tokens=144, output_token_ids=[17, 19], num_output_placeholders=2)
        clean = NS(request_id="c", num_computed_tokens=80, output_token_ids=[5])
        manager = NS(get_block_ids=lambda rid: ([1, 2], [0, 7, 8, 9, 10]) if rid == "r" else ([3], [4]))
        result = update(NS(kv_cache_manager=manager), [req, clean], {8}, {"r": 16})
        self.assertEqual(result, ({"r"}, 128, {1, 2, 7, 8, 9, 10}))
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(req.output_token_ids, [17, 19])
        self.assertEqual(req.num_output_placeholders, 2)
        self.assertEqual(clean.num_computed_tokens, 80)

    def test_null_and_empty_groups_do_not_false_positive(self):
        update = extract(CORE, "_update_requests_with_invalid_blocks")
        req = NS(request_id="r", num_computed_tokens=64)
        self.assertEqual(update(NS(kv_cache_manager=NS(get_block_ids=lambda r: ([], [0, 3]))), [req], {0}, {}), (set(), 0, set()))

    def test_single_group_retains_prefix_recovery(self):
        update = extract(CORE, "_update_requests_with_invalid_blocks")
        req = NS(request_id="r", num_computed_tokens=192)
        result = update(NS(block_size=64, kv_cache_manager=NS(get_block_ids=lambda r: ([1, 2, 3],))), [req], {2}, {})
        self.assertEqual(result, ({"r"}, 128, {2, 3}))
        self.assertEqual(req.num_computed_tokens, 64)

    def test_async_recovery_and_abort_hold_blocks_until_all_rank_drain(self):
        comp = CompletionTests()
        connector, _, calls, env = comp.state()
        status = NS(WAITING_FOR_REMOTE_KVS="wait", is_finished=lambda s: s == "aborted")
        req = NS(request_id="r", status="wait", num_computed_tokens=128)
        freed = []
        core = NS(connector=connector, requests={"r": req}, finished_recving_kv_req_ids=set(),
                  failed_recving_kv_req_ids=set(), _free_blocks=lambda r: freed.append(r.request_id),
                  kv_cache_manager=NS(get_block_ids=lambda r: ([1], [8]), free=lambda r: freed.append(r.request_id)))
        bind(core, CORE, "_update_from_kv_xfer_finished", {"RequestStatus": status})
        bind(core, CORE, "_update_waiting_for_remote_kv")
        first = comp.output(env, 1, True, {8})
        connector.update_connector_output(first)
        first.finished_sending = set()
        core._update_from_kv_xfer_finished(first)
        self.assertFalse(freed)
        self.assertFalse(core.finished_recving_kv_req_ids)
        last = comp.output(env, 3)
        connector.update_connector_output(last)
        last.finished_sending = set()
        affected, _, _ = extract(CORE, "_update_requests_with_invalid_blocks")(core, [req], last.invalid_block_ids, {}, False)
        core.failed_recving_kv_req_ids.update(affected)
        core._update_from_kv_xfer_finished(last)
        self.assertFalse(freed)
        core._update_waiting_for_remote_kv(req)
        self.assertEqual(freed, ["r"])
        # Same terminal boundary frees an already aborted request, never earlier.
        freed.clear()
        req.status = "aborted"
        core._update_from_kv_xfer_finished(first)
        self.assertFalse(freed)
        core._update_from_kv_xfer_finished(last)
        self.assertEqual(freed, ["r"])

    def test_connector_reduction_precedes_invalid_recovery_in_real_method(self):
        class StopProbe(Exception):
            pass
        seen = []
        out = NS(invalid_block_ids=set())
        def reduce(output):
            seen.append("reduce")
            output.invalid_block_ids.add(7)
        def recover(*args):
            self.assertEqual(seen, ["reduce"])
            raise StopProbe()
        core = NS(defer_block_free=False, perf_metrics=None, ec_connector=None,
                  connector=NS(update_connector_output=reduce), _handle_invalid_blocks=recover)
        runner = NS(sampled_token_ids=[], logprobs=None, prompt_logprobs_dict={}, pooler_output=None,
                    num_nans_in_logits=None, kv_connector_output=out, ec_connector_output=None,
                    cudagraph_stats=None)
        method = extract(CORE, "update_from_output")
        with self.assertRaises(StopProbe):
            method(core, NS(num_scheduled_tokens={}, total_num_scheduled_tokens=0), runner)

    def test_grouped_sync_recovery_uses_native_preemption_fence(self):
        status = NS(WAITING_FOR_REMOTE_KVS="wait", RUNNING="running", PREEMPTED="preempted")
        req = NS(request_id="r", status="running", num_computed_tokens=128,
                 spec_token_ids=[99], num_preemptions=0, output_token_ids=[3, 4],
                 drop_stale_output=False, num_stale_output_tokens=0,
                 num_in_flight_tokens=0, num_output_placeholders=0)
        calls = []
        core = NS(recompute_kv_load_failures=True, skipped_waiting=[], running=[req], requests={"r": req},
                  kv_cache_manager=NS(get_block_ids=lambda r: ([1], [2, 3]), evict_blocks=lambda ids: calls.append(("evict", ids))),
                  _free_request_blocks=lambda r: calls.append(("fenced-free", r.request_id)),
                  encoder_cache_manager=NS(free=lambda r: None), _inflight_prefills=set(), log_stats=False,
                  waiting=NS(prepend_request=lambda r: calls.append(("requeue", r.request_id))),
                  reset_preempted_req_ids=set(), failed_recving_kv_req_ids=set())
        # Use a hashable request for the real preemption method's set.discard.
        class Request:
            __hash__ = object.__hash__
        real_req = Request()
        real_req.__dict__.update(req.__dict__)
        core.running, core.requests = [real_req], {"r": real_req}
        bind(core, CORE, "_update_requests_with_invalid_blocks")
        bind(core, CORE, "_preempt_request", {"RequestStatus": status})
        handle = extract(CORE, "_handle_invalid_blocks", {"RequestStatus": status})
        self.assertEqual(handle(core, {3}, {}), {"r"})
        self.assertEqual([c[0] for c in calls], ["evict", "fenced-free", "requeue"])
        self.assertEqual(real_req.output_token_ids, [3, 4])
        self.assertEqual(real_req.spec_token_ids, [])
        self.assertEqual(real_req.status, "preempted")


class FinishedFrontierTests(unittest.TestCase):
    """Run native connector methods through the real core finish/free path."""

    def harness(self, count=24, select=None, swa=False):
        cursor = CursorTests()
        configs = [(64, False, None, 1 if swa else None)]
        obj, _, groups, offered, accepted = cursor.state(configs, select)
        env = classes()
        status = NS(RUNNING="running", WAITING_FOR_REMOTE_KVS="remote",
                    WAITING_FOR_STREAMING_REQ="stream", FINISHED_STOPPED="stop",
                    FINISHED_LENGTH_CAPPED="length", FINISHED_ABORTED="abort",
                    FINISHED_ERROR="error", FINISHED_REPETITION="repetition",
                    is_finished=lambda s: s not in ("running", "remote", "stream"))
        class Request:
            pass
        req = Request()
        req.__dict__.update(request_id="r", status="running", client_index=0,
                            num_computed_tokens=0, num_tokens=count*64,
                            num_prompt_tokens=count*64, kv_transfer_params=None,
                            has_encoder_inputs=False, pooling_params=None, sampling_params=None,
                            _output_token_ids=[], stop_reason=None, trace_headers=None,
                            num_nans_in_logits=0, last_sched_seq=0,
                            num_in_flight_tokens=0, shared_prefix_boundary=None,
                            num_stale_output_tokens=0, drop_stale_output=False,
                            num_output_placeholders=0, num_preemptions=0,
                            spec_decode_metrics=None, resumable=False,
                            spec_token_ids=[], use_structured_output=False)
        req.is_finished = lambda: status.is_finished(req.status)
        req.get_finished_reason = lambda: req.status
        req.take_events = req.take_prefill_stats = lambda: None
        state = obj._req_status["r"]
        state.req = req
        bind(state, OFF+"scheduler.py", "storable_chunks", env)
        state.req_context = "ctx"
        state.finish_pending = False
        state.finished_signaled = False
        state.finish_store_tokens = 0
        state.update_offload_keys = lambda: None
        state.load_failed = False
        groups[0].offload_keys = [(0, i) for i in range(count)]
        groups[0].block_ids = list(range(1, count+1))
        tables = [groups[0].block_ids.copy()]
        calls, freed = [], []
        obj.manager.complete_store = lambda keys, ctx, success=True: calls.append(("store", set(keys), success))
        obj.manager.on_request_finished = lambda ctx: calls.append(("finish", ctx))
        obj.manager.on_schedule_end = lambda context: None
        obj.manager.has_pending_work = lambda: False
        obj.manager.can_store = lambda: True
        obj.manager.reset_cache = lambda: None
        obj.__dict__.update(_stale_job_threshold=0,
                            _job_counter=0, _chunks_being_loaded=None,
                            _sliding_window_groups=(0,) if swa else (),
                            _current_batch_load_jobs={}, _current_batch_jobs_to_flush=set(),
                            _current_batch_allocated_block_ids=set(),
                            _build_partial_tail_store_jobs=lambda output: {},
                            _maybe_observe_lookup_async_delay=lambda status: None)
        env.update(RequestStatus=status, TransferJob=lambda **kw: NS(**kw),
                   GPULoadStoreSpec=lambda ids, **kw: NS(block_ids=ids, **kw),
                   OffloadingConnectorMetadata=lambda **kw: NS(**kw),
                   ScheduleEndContext=lambda **kw: NS(**kw), chain=chain,
                   _ConnectorMetricName=NS(ALLOCATION_FAILURE="alloc"),
                   yield_req_data=lambda output: (), _create_req_context=lambda r: "ctx")
        for name in ("_generate_job_id", "_remove_pending_job", "_update_req_states",
                     "_advance_store_frontiers", "_build_store_jobs", "build_connector_meta",
                     "update_connector_output", "request_finished", "has_pending_push_work",
                     "reset_cache"):
            bind(obj, OFF+"scheduler.py", name, env)
        # New helpers are absent in the pre-fix reproduction tree.
        tree = ast.parse((ROOT / (OFF+"scheduler.py")).read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for name in ("_calc_num_offloadable_tokens", "_has_finished_store_frontier"):
            if name in names:
                bind(obj, OFF+"scheduler.py", name, env)
        class HMA:
            pass
        wrapper = HMA()
        wrapper.connector_scheduler = obj
        wrapper.get_kv_connector_stats = lambda: None
        obj.get_stats = lambda: None
        wrapper.take_events = lambda: ()
        wrapper_path = OFF.rsplit("/", 2)[0]+"/offloading_connector.py"
        for name in ("request_finished_all_groups", "update_connector_output", "has_pending_push_work"):
            bind(wrapper, wrapper_path, name)
        manager = NS(get_block_ids=lambda rid: tuple(tables),
                     get_block_ids_for_computed_tokens=lambda **kw: tuple(tables),
                     remove_skipped_blocks=lambda **kw: None,
                     free=lambda r: freed.append(r.request_id), take_events=lambda: None)
        core = NS(connector=wrapper, requests={"r": req}, kv_cache_manager=manager,
                  vllm_config=NS(kv_transfer_config=None), ec_connector=None,
                  routed_experts_mgr=None, num_sampled_tokens_per_step=1,
                  grammar_compile_error_reqs=set(), recompute_kv_load_failures=True,
                  return_sampling_mask=False,
                  is_mm_encoder_only=False, log_stats=False,
                  processed_step_seq=0, sched_step_seq=0, deferred_frees=[],
                  _inflight_prefills=set(), encoder_cache_manager=NS(free=lambda r: None),
                  finished_req_ids=set(), finished_req_ids_dict=None, defer_block_free=False,
                  running=[req], waiting=[], skipped_waiting=[], perf_metrics=None,
                  enable_return_routed_experts=False, make_stats=lambda *a: None,
                  structured_output_manager=NS(should_advance=lambda r, **kw: False),
                  finished_recving_kv_req_ids=set(), failed_recving_kv_req_ids=set(),
                  has_unfinished_requests=lambda: bool(core.running))
        def stop(r, tokens, is_stale=False):
            r.status = status.FINISHED_STOPPED
            r._output_token_ids.extend(tokens)
            return tokens, True
        core._update_request_with_output = stop
        core._handle_stopped_request = lambda r: True
        core_env = {"RequestStatus": status, "SupportsHMA": HMA,
                    "remove_all": lambda seq, remove: [r for r in seq if r not in remove],
                    "EngineCoreOutput": lambda **kw: NS(**kw),
                    "EngineCoreOutputs": lambda **kw: NS(**kw)}
        for name in ("_connector_finished", "_free_request", "_free_blocks",
                     "_free_request_blocks", "_update_from_kv_xfer_finished",
                     "finish_requests", "update_from_output", "has_finished_requests", "has_requests"):
            bind(core, CORE, name, core_env)
        return NS(obj=obj, core=core, req=req, state=state, groups=groups,
                  offered=offered, accepted=accepted, calls=calls, freed=freed,
                  env=env, status=status, tables=tables)

    def output(self, h, jobs=None, failed=()):
        return NS(kv_connector_worker_meta=h.env["OffloadingWorkerMetadata"](
            completed_jobs=jobs or {}, failed_jobs=set(failed)), invalid_block_ids=set(),
            finished_sending=set(), finished_recving=set(), kv_connector_stats=None)

    def model_output(self, h, scheduled, output, sampled=None):
        runner = NS(sampled_token_ids=sampled or [], logprobs=None,
                    prompt_logprobs_dict={}, pooler_output=None, num_nans_in_logits=None,
                    kv_connector_output=output, ec_connector_output=None,
                    cudagraph_stats=None, routed_experts=None,
                    req_id_to_index={"r": 0})
        schedule = NS(num_scheduled_tokens=scheduled, scheduled_spec_decode_tokens={},
                      total_num_scheduled_tokens=sum(scheduled.values()),
                      finished_req_ids=())
        return h.core.update_from_output(schedule, runner)

    def finish(self, h, abort=False):
        h.req.num_computed_tokens = h.req.num_tokens
        if abort:
            h.core.finish_requests("r", h.status.FINISHED_ABORTED)
        else:
            self.model_output(h, {"r": h.req.num_tokens}, self.output(h), [[17]])
        h.core.finished_req_ids.clear()  # Next schedule consumes these IDs.

    def step(self, h):
        finished = tuple(h.core.finished_req_ids)
        h.core.finished_req_ids.clear()
        return h.obj.build_connector_meta(
            NS(num_scheduled_tokens={}, preempted_req_ids=set(),
               scheduled_new_reqs=(), finished_req_ids=finished,
               kv_cache_block_copies=None)).store_jobs

    def complete(self, h, jobs, count=4, failed=()):
        output = self.output(h, {jid: count for jid in jobs}, failed)
        self.model_output(h, {}, output)
        return output

    def test_short_generation_keeps_thousands_of_unadmitted_keys(self):
        h = self.harness(2056, lambda keys, n: (keys[:8], []))
        initial = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": h.req.num_tokens}, finished_req_ids=()))
        self.finish(h)
        self.assertFalse(h.freed)
        self.assertIn("r", h.core.requests)
        self.assertTrue(h.core.has_requests())
        self.assertFalse(h.core.running)
        self.complete(h, initial)
        self.assertIn("r", h.obj._req_status)
        self.assertEqual(h.groups[0].next_stored_chunk_idx, 8)
        self.assertNotIn(("finish", "ctx"), h.calls)
        while h.obj._req_status:
            jobs = self.step(h)
            self.assertTrue(jobs)
            self.assertLessEqual(len(h.obj._jobs[next(iter(jobs))].keys), 8)
            self.complete(h, jobs)
        self.assertEqual(len(h.accepted), 2056)
        self.assertEqual(len(set(h.accepted)), 2056)
        self.assertEqual(h.freed, ["r"])
        self.assertEqual(h.calls[-1], ("finish", "ctx"))
        self.assertFalse(h.core.has_requests())

    def test_zero_admission_at_finish_retries_pressure_on_idle_steps(self):
        for denied in (None, []):
            h = self.harness(24, lambda keys, n: (denied if n < 4 else keys, []))
            self.assertFalse(h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1536}, finished_req_ids=())))
            self.finish(h)
            self.assertFalse(h.freed)
            self.assertTrue(h.obj.has_pending_push_work())
            self.assertFalse(h.obj.reset_cache())
            self.assertFalse(self.step(h))
            self.complete(h, {})
            self.assertFalse(self.step(h))
            self.complete(h, {})
            jobs = self.step(h)
            self.assertTrue(jobs)
            self.complete(h, jobs, 3)
            self.assertFalse(h.freed)
            self.complete(h, jobs, 1)
            self.assertEqual(h.freed, ["r"])

    def test_skip_only_finished_suffix_releases_without_transfer(self):
        h = self.harness(select=lambda keys, n: (keys[:8], []) if n == 1 else ([], keys))
        initial = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1536}, finished_req_ids=()))
        self.finish(h)
        self.complete(h, initial)
        self.assertFalse(self.step(h))
        output = self.complete(h, {})
        self.assertEqual(output.finished_sending, {"r"})
        self.assertEqual(h.freed, ["r"])
        self.assertEqual(len([c for c in h.calls if c[0] == "store"]), 1)

    def test_abort_stops_admissions_but_waits_for_all_rank_store_drain(self):
        h = self.harness(select=lambda keys, n: (keys[:8], []))
        initial = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1536}, finished_req_ids=()))
        self.finish(h, abort=True)
        self.assertFalse(h.freed)
        self.assertFalse(self.step(h))
        self.complete(h, initial, 3, initial)
        self.assertFalse(h.freed)
        self.complete(h, initial, 1)
        self.assertEqual(h.freed, ["r"])
        self.assertEqual(len(h.offered), 1)
        self.assertFalse(h.calls[0][2])

    def test_closed_provider_abandons_frontier_not_active_job(self):
        h = self.harness(select=lambda keys, n: (keys[:8], []))
        initial = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1536}, finished_req_ids=()))
        self.finish(h)
        h.obj.manager.can_store = lambda: False
        self.assertFalse(self.step(h))
        self.complete(h, initial, 3, initial)
        self.assertFalse(h.freed)
        self.complete(h, initial, 1)
        self.assertEqual(h.freed, ["r"])
        self.assertFalse(h.obj.has_pending_push_work())
        self.assertEqual(h.groups[0].next_stored_chunk_idx, 0)

    def test_finished_failed_store_retries_full_rows_but_nulls_swa_rows(self):
        for swa in (False, True):
            h = self.harness(8, swa=swa)
            jobs = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 512}, finished_req_ids=()))
            self.finish(h)
            self.complete(h, jobs, 3, jobs)
            self.assertFalse(h.freed)
            self.complete(h, jobs, 1)
            self.assertFalse(h.freed)
            self.assertEqual(h.groups[0].next_stored_chunk_idx, 0)
            # A failed native admission is no longer owned, unlike the simple
            # cursor fixture's de-duplicating producer; permit its actual retry.
            h.obj.manager.prepare_store = lambda keys, ctx: NS(
                keys_to_store=keys, skipped_keys=[], store_spec=None)
            retried = self.step(h)
            if swa:
                self.assertFalse(retried)
                self.assertEqual(h.groups[0].block_ids, [0]*8)
            else:
                self.assertEqual(next(iter(retried.values())).src_spec.block_ids,
                                 list(range(1, 9)))
            self.complete(h, retried)
            self.assertEqual(h.freed, ["r"])
            self.assertFalse(h.calls[0][2])

    def test_repeated_drained_failure_abandons_without_infinite_retry_or_early_free(self):
        h = self.harness(24, lambda keys, n: (keys[:8], []))
        first = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1536}, finished_req_ids=()))
        self.finish(h)
        self.complete(h, first, failed=first)
        self.assertEqual(h.state.finish_failed_store_keys, {(0, i) for i in range(8)})
        self.assertFalse(h.state.finish_store_abandoned)
        h.obj.manager.prepare_store = lambda keys, ctx: NS(
            keys_to_store=keys[:8], skipped_keys=[], store_spec=None)
        retry = self.step(h)
        later = self.step(h)  # Other keys already issued must still drain.
        warnings = []
        previous = LOG.warning
        LOG.warning = lambda message, *a: warnings.append(message % a)
        self.addCleanup(setattr, LOG, "warning", previous)
        self.complete(h, retry, 3, retry)
        self.assertFalse(h.state.finish_store_abandoned)
        self.complete(h, retry, 1)
        self.assertTrue(h.state.finish_store_abandoned)
        self.assertFalse(h.freed)
        self.assertFalse(self.step(h))
        self.assertEqual(len(warnings), 1)
        self.assertIn("incomplete store frontier abandoned", warnings[0])
        self.complete(h, later, 3)
        self.assertFalse(h.freed)
        self.complete(h, later, 1)
        self.assertEqual(h.freed, ["r"])
        self.assertFalse(h.state.finish_failed_store_keys)
        self.assertFalse(h.obj.has_pending_push_work())
        self.assertEqual(h.groups[0].next_stored_chunk_idx, 0)
        self.assertEqual([c[2] for c in h.calls if c[0] == "store"], [False, False, True])

    def test_stalled_finish_drain_abandons_and_releases(self):
        """0006: a frontier that STOPS PROGRESSING (persistent manager
        declines, rows null past finish) must not hold the finished request --
        and the client response gated on the terminal finished_sending
        signal -- forever. Steps that advance the frontier reset the stall
        counter, so slow-but-progressing drains are never cut."""
        h = self.harness(24, lambda keys, n: (keys[:8], []))
        first = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1536}, finished_req_ids=()))
        self.finish(h)
        self.complete(h, first)
        self.assertEqual(h.groups[0].next_stored_chunk_idx, 8)
        # The manager now declines everything: no admissions and no skips is
        # retryable, so the frontier can never complete on its own.
        h.obj.manager.prepare_store = lambda keys, ctx: NS(
            keys_to_store=[], skipped_keys=[], store_spec=None)
        warnings = []
        previous = LOG.warning
        LOG.warning = lambda message, *a: warnings.append(message % a)
        self.addCleanup(setattr, LOG, "warning", previous)
        for _ in range(70):
            if not h.state.finish_store_abandoned:
                self.step(h)
        self.assertTrue(h.state.finish_store_abandoned)
        self.assertTrue(any("stalled" in w for w in warnings))
        self.assertFalse(self.step(h))
        # Abandoning must actually release: the terminal finished_sending
        # signal fires through update_connector_output (driven here by a
        # completion round, even an empty one).
        self.assertIn("r", h.obj._req_status)
        self.complete(h, {})
        self.assertNotIn("r", h.obj._req_status)
        self.assertFalse(h.obj.has_pending_push_work())

    def test_failure_retry_budget_is_per_key_not_per_job_or_pressure(self):
        h = self.harness(16, lambda keys, n: (keys[:8], []))
        first = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1024}, finished_req_ids=()))
        self.finish(h)
        second = self.step(h)
        self.complete(h, first, failed=first)
        self.complete(h, second, failed=second)
        self.assertFalse(h.state.finish_store_abandoned)
        self.assertEqual(len(h.state.finish_failed_store_keys), 16)
        h.obj.manager.prepare_store = lambda keys, ctx: None
        for _ in range(4):
            self.assertFalse(self.step(h))
            self.complete(h, {})
        self.assertFalse(h.state.finish_store_abandoned)
        h.obj.manager.prepare_store = lambda keys, ctx: NS(
            keys_to_store=keys, skipped_keys=[], store_spec=None)
        retried = self.step(h)
        self.complete(h, retried)
        self.assertEqual(h.freed, ["r"])
        self.assertFalse(h.state.finish_failed_store_keys)

    def test_finished_all_admitted_jobs_release_only_after_last_rank_of_last_job(self):
        h = self.harness(16, lambda keys, n: (keys[:8], []))
        first = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1024}, finished_req_ids=()))
        self.finish(h)
        second = self.step(h)
        self.assertEqual(h.groups[0].next_stored_chunk_idx, 16)
        self.complete(h, second)
        self.assertFalse(h.freed)
        self.assertFalse(self.step(h))
        self.complete(h, first, 3)
        self.assertFalse(h.freed)
        self.complete(h, first, 1)
        self.assertEqual(h.freed, ["r"])
        self.complete(h, first)  # Duplicate completion cannot double-free.
        self.assertEqual(h.freed, ["r"])

    def test_zero_job_abort_and_closed_provider_finish_without_false_success(self):
        h = self.harness(select=lambda keys, n: (None, []))
        self.finish(h, abort=True)
        self.assertEqual(h.freed, ["r"])
        self.assertEqual(h.calls, [("finish", "ctx")])
        h = self.harness(select=lambda keys, n: (None, []))
        self.finish(h)
        self.assertFalse(h.freed)
        h.obj.manager.can_store = lambda: False
        self.assertFalse(self.step(h))
        self.complete(h, {})
        self.assertEqual(h.freed, ["r"])
        self.assertEqual(h.groups[0].next_stored_chunk_idx, 0)
        self.assertEqual(h.calls, [("finish", "ctx")])

    def test_already_drained_generation_finishes_without_extra_job(self):
        h = self.harness(8)
        jobs = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 512}, finished_req_ids=()))
        self.complete(h, jobs)
        self.assertFalse(h.freed)
        self.finish(h)
        self.assertEqual(h.freed, ["r"])
        self.assertFalse(self.step(h))
        self.assertEqual(len(h.calls), 2)

    def test_abort_during_load_releases_once_via_receive_after_rank_drain(self):
        h = self.harness()
        h.req.status = h.status.WAITING_FOR_REMOTE_KVS
        h.core.running = []
        h.core.waiting = NS(remove_requests=lambda reqs: None)
        h.core.skipped_waiting = NS(remove_requests=lambda reqs: None)
        h.state.transfer_jobs = {0}
        h.obj._jobs[0] = h.env["TransferJobStatus"]("r", 4, {"load"}, False)
        h.obj.manager.complete_load = lambda *a: h.calls.append(("load",))
        self.finish(h, abort=True)
        self.complete(h, {0}, 3)
        self.assertFalse(h.freed)
        output = self.complete(h, {0}, 1)
        self.assertEqual(output.finished_recving, {"r"})
        self.assertFalse(output.finished_sending)
        self.assertEqual(h.freed, ["r"])
        self.assertEqual(h.calls, [("load",), ("finish", "ctx")])

    def test_finished_partial_chunk_counts_at_finish(self):
        """0007: the final partial chunk is stable at finish and must be
        storable. With 24 chunks keyed/allocated and 23.84 chunks
        offloadable, the floor stopped the frontier at 23 -- one short of
        the full-prompt boundary a same-prompt repeat resumes at."""
        h = self.harness(24, lambda keys, n: (list(keys), []))
        h.req.num_tokens = 24 * 64 - 10
        h.req.num_prompt_tokens = 24 * 64 - 10
        first = h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": h.req.num_tokens}, finished_req_ids=()))
        self.finish(h)
        self.complete(h, first)
        while h.obj._req_status:
            jobs = self.step(h)
            self.complete(h, jobs)
            if h.state.finish_store_abandoned:
                break
        self.assertEqual(h.groups[0].next_stored_chunk_idx, 24)
        self.assertFalse(h.state.finish_store_abandoned)

    def test_finished_frontier_retains_prompt_cap_and_eagle_tail(self):
        h = self.harness(24, lambda keys, n: (None, []))
        h.obj.config.offload_prompt_only = True
        h.req.num_prompt_tokens = 16*64
        h.state.max_offload_tokens = 12*64
        h.obj.config.kv_group_configs[0].is_eagle_group = True
        self.finish(h)
        self.assertEqual(h.state.finish_store_tokens, 12*64)
        h.obj.manager.prepare_store = lambda keys, ctx: NS(
            keys_to_store=keys, skipped_keys=[], store_spec=None)
        jobs = self.step(h)
        # Upstream `storable_chunks` lifts the EAGLE/MTP tail exclusion once
        # the request is finished (issue #52735): no further spec-token
        # rejection can rewrite the tail, so all 12 capped chunks are stored.
        # Our frontier walk inherits that bound rather than re-deriving it.
        self.assertEqual(next(iter(jobs.values())).src_spec.block_ids,
                         list(range(1, 13)))
        self.complete(h, jobs)
        self.assertEqual(h.freed, ["r"])

    def test_finish_uses_current_swa_table_not_released_historical_ids(self):
        h = self.harness(select=lambda keys, n: ([], []), swa=True)
        h.obj._build_store_jobs(NS(num_scheduled_tokens={"r": 1536}, finished_req_ids=()))
        h.tables[0][:23] = [0]*23  # Native remove_skipped_blocks result.
        self.finish(h)
        h.obj.manager.prepare_store = lambda keys, ctx: NS(keys_to_store=keys, skipped_keys=[], store_spec=None)
        jobs = self.step(h)
        self.assertEqual(next(iter(jobs.values())).src_spec.block_ids, [24])
        self.complete(h, jobs)
        self.assertEqual(h.freed, ["r"])


class PackagingTests(unittest.TestCase):
    def test_series_patches_never_rename_or_drop_module_classes(self):
        """Regression guard (2026-09-12): a generated patch once carried a
        class-rename hunk (-class A / +class B) from a contaminated build
        tree. apply.py validated it against the same contaminated tree, the
        AST suite passed (it binds methods, not the class), and the live
        engine died on ImportError. No series patch may add or remove a
        top-level class/def line."""
        root = Path(patcher.__file__).parent
        for entry in patcher.manifest()["patches"]:
            text = (root / entry["file"]).read_text()
            for line in text.splitlines():
                if line.startswith(("+class ", "-class ", "+    class ", "-    class ")):
                    self.fail(f"{entry['file']}: class-rename hunk: {line!r}")

    def test_pristine_reproduction_and_apply_verify_reverse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in patcher.manifest()["files"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(patcher.source_path(ORIGINAL, BARE, name), path)
            if tree_state(root) == "after":
                # --source-tree named the ported tree; rewind to the pin.
                patcher.run(root, "reverse")
            patcher.run(root, "check")
            patcher.run(root, "apply")
            patcher.run(root, "verify")
            patcher.run(root, "reverse")
            patcher.check_tree(root, "before", patcher.manifest())
            target = root / CORE
            target.write_text(target.read_text() + "\n# unexpected local edit\n")
            before = {name: (root / name).read_bytes() for name in patcher.manifest()["files"]}
            with self.assertRaisesRegex(ValueError, "SHA256"):
                patcher.run(root, "apply")
            self.assertEqual(before, {name: (root / name).read_bytes() for name in before})


def tree_state(root):
    """Return "before", "after" or None for a staged manifest-listed tree."""
    for state in ("before", "after"):
        try:
            patcher.check_tree(root, state, patcher.manifest())
        except ValueError:
            continue
        return state
    return None


def main():
    global ROOT, ORIGINAL, BARE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", required=True, type=Path)
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="run against the tree exactly as given (pre-fix reproduction)",
    )
    args, rest = parser.parse_known_args()
    ORIGINAL, bare = patcher.resolve_root(args.source_tree.resolve())
    BARE = bare
    with tempfile.TemporaryDirectory(prefix="native-method-tests-") as directory:
        ROOT = Path(directory)
        for name in patcher.manifest()["files"]:
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(patcher.source_path(ORIGINAL, bare, name), path)
        if not args.no_apply:
            state = tree_state(ROOT)
            if state == "before":
                patcher.run(ROOT, "apply")
            elif state != "after":
                parser.exit(1, "Source tree is neither the pinned pristine nor "
                               "the ported state; refusing.\n")
        result = unittest.main(argv=[sys.argv[0], *rest], exit=False)
        return not result.result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(main())
