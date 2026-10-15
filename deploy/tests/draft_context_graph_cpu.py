"""Exercise installed draft-context graph plumbing with independent CPU caches.

This checks production method control flow, not CUDA execution or model output.
Run in the runtime image, which supplies torch, without --gpus.
"""
import __future__
import ast
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest

import torch


DEPLOY = Path(__file__).resolve().parents[1]
ROOT = DEPLOY / "overrides/performance/patched/vllm/v1/worker/gpu/spec_decode/dflash"
FULL, NONE = "FULL", "NONE"
MODE = NS(FULL=FULL, NONE=NONE, PIECEWISE="PIECEWISE")
PAD = -1


def method(filename, cls, name, namespace):
    path = ROOT / filename
    nodes = next(n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class RecordingModel:
    def __init__(self):
        self.calls = []
        self.cache = [{}, {}, {}]

    def precompute_and_store_context_kv(self, states, positions, slots=None):
        self.calls.append((states.clone(), positions.clone(), slots))
        if slots is None:
            return
        rows = slots if isinstance(slots, list) else [slots] * 3
        # Independent per-layer cache reference, deliberately scalar/CPU.
        for cache, layer_slots in zip(self.cache, rows):
            if layer_slots is None:
                continue
            for slot, state, position in zip(layer_slots.tolist(), states, positions):
                if slot != PAD:
                    if slot in cache:
                        raise AssertionError("context row inserted twice in one step")
                    cache[slot] = (state.tolist(), int(position))


def fixture(group_indices=(1, 0, 1), rows=1024):
    owner = NS(num_speculative_steps=5, num_query_per_req=5, max_num_tokens=rows,
               hidden_states=torch.arange(rows * 3).view(rows, 3).float(),
               context_positions=torch.arange(rows), _layer_group_idx=group_indices,
               _context_slot_mappings=torch.arange(rows).repeat(2, 1),
               model=RecordingModel())
    owner._context_slot_mappings[1] += 10000
    namespace = dict(torch=torch, CUDAGraphMode=MODE, PAD_SLOT_ID=PAD)
    for name in ("_num_graph_context_tokens", "_precompute_context_kv",
                 "_prepare_context_kv_for_execution"):
        setattr(owner, name, MethodType(method("speculator.py", "DFlashSpeculator", name, namespace), owner))
    return owner


class DraftContextTests(unittest.TestCase):
    def test_verify_rows_include_bonus_despite_dspark_k_query_rows(self):
        owner = fixture()
        self.assertEqual(owner.num_query_per_req, 5)
        self.assertEqual(owner._num_graph_context_tokens(1), 6)
        self.assertEqual(owner._num_graph_context_tokens(32), 192)
        self.assertEqual(owner._num_graph_context_tokens(200), 1024)

    def test_idle_dp_dummy_uses_agreed_request_count_and_no_context_writes(self):
        @dataclass(frozen=True)
        class Sync:
            num_tokens_across_dp: torch.Tensor
            uniform_token_count: int
            eager: bool
            num_reqs: int
        owner = fixture()
        # An idle DP engine enters the runner's padded dummy batch while the
        # other pool verifies 32 requests. Target uses 32*(k+1), draft 32*k.
        target = Sync(torch.tensor([192, 192]), 6, False, 32)
        build = method("../speculator.py", "DraftModelSpeculator", "_build_uniform_batch_dp_sync",
                       dict(torch=torch, replace=replace))
        draft, tokens = build(owner, target, num_reqs=32, num_query_per_req=5)
        self.assertEqual(tokens, 160)
        self.assertEqual(draft.num_tokens_across_dp.tolist(), [160, 160])
        self.assertEqual(draft.num_reqs, 32)
        self.assertEqual(target.num_tokens_across_dp.tolist(), [192, 192])
        owner._prepare_context_kv_for_execution(NS(cg_mode=FULL, num_reqs=32), 192, True)
        owner._precompute_context_kv(0, owner._num_graph_context_tokens(32))
        self.assertEqual(owner.model.cache, [{}, {}, {}])
        self.assertEqual(owner.model.calls[0][0].shape[0], 192)
        self.assertLessEqual(192, owner.max_num_tokens)

    def test_full_replay_matches_eager_with_tail_padding_groups_and_mutation(self):
        owner = fixture()
        for real_rows, padded_requests in ((12, 2), (4, 2), (1024, 4), (192, 32), (1, 32)):
            with self.subTest(real_rows=real_rows, padded_requests=padded_requests):
                owner.model = RecordingModel()
                owner.hidden_states.add_(13)
                owner.context_positions.add_(17)
                owner._context_slot_mappings.add_(20000)  # simulate stale live slots
                # Eager input preparation refreshes every current row, while
                # padding retains old addresses until the production helper.
                owner._context_slot_mappings[:, :real_rows] = torch.stack([
                    torch.arange(real_rows) + 1000,
                    torch.arange(real_rows) + 11000,
                ])
                before = owner._context_slot_mappings.clone()
                expected = RecordingModel()
                expected.precompute_and_store_context_kv(
                    owner.hidden_states[:real_rows], owner.context_positions[:real_rows],
                    [before[g, :real_rows] for g in owner._layer_group_idx])
                owner._prepare_context_kv_for_execution(
                    NS(cg_mode=FULL, num_reqs=padded_requests), real_rows, False)
                graph_rows = owner._num_graph_context_tokens(padded_requests)
                if real_rows > graph_rows:
                    self.assertEqual(owner.model.calls[0][0].shape[0], real_rows - graph_rows)
                else:
                    self.assertEqual(len(owner.model.calls), 0)
                    self.assertTrue(torch.all(owner._context_slot_mappings[:, real_rows:graph_rows] == PAD))
                # The exact production callback inserted by graph capture.
                owner._precompute_context_kv(0, graph_rows)
                self.assertEqual(owner.model.cache, expected.cache)
                self.assertEqual(owner._context_slot_mappings[:, :real_rows].tolist(),
                                 before[:, :real_rows].tolist())

    def test_eager_and_dummy_never_write_placeholder_or_padding_slots(self):
        for full in (False, True):
            owner = fixture()
            desc = NS(cg_mode=FULL if full else NONE, num_reqs=4)
            owner._prepare_context_kv_for_execution(desc, 64, True)
            if full:
                self.assertTrue(torch.all(owner._context_slot_mappings[:, :24] == PAD))
                owner._precompute_context_kv(0, 24)
            self.assertEqual(owner.model.cache, [{}, {}, {}])
        owner = fixture(group_indices=None)
        owner._prepare_context_kv_for_execution(NS(cg_mode=NONE), 17, False)
        self.assertEqual([len(c) for c in owner.model.cache], [17, 17, 17])
        self.assertEqual(owner.model.calls[0][2].shape, (17,))

    def test_capture_fills_all_groups_and_passes_fixed_context_callback(self):
        owner = fixture()
        owner.sample_indices = torch.ones(160)
        owner.sample_pos = torch.ones(160)
        owner.sample_idx_mapping = torch.ones(160)
        owner.input_buffers = owner.block_tables = owner.attn_groups = owner.kv_cache_config = None
        owner.max_model_len, owner._group_causal = 512000, True
        owner._speculator_name, owner._generate_draft = "DSpark", lambda: None
        recorded = {}
        owner.query_cudagraph_manager = NS(capture=lambda *a, **kw: recorded.update(kw))
        capture = method("speculator.py", "DFlashSpeculator", "capture",
                         dict(torch=torch, PAD_SLOT_ID=PAD, logger=NS(info=lambda *a: None)))
        capture(owner)
        self.assertTrue(torch.all(owner._context_slot_mappings == PAD))
        self.assertTrue(torch.all(owner.sample_idx_mapping == -1))
        recorded["precompute_context_kv"](4)
        self.assertEqual(owner.model.calls[0][0].shape[0], 24)
        self.assertEqual(owner.model.cache, [{}, {}, {}])

    def test_propose_dispatches_context_before_query_with_prefill_tail(self):
        for mode in (FULL, NONE):
            with self.subTest(mode=mode):
                owner = fixture()
                owner.max_model_len, owner.dp_size, owner.dp_rank = 512000, 2, 1
                owner.draft_kv_cache_group_id, owner.draft_kv_cache_group_ids = 0, [0, 1]
                owner._group_causal, owner._speculator_name = True, "DSpark"
                owner.parallel_drafting_token_id = 7
                owner.max_num_reqs, owner.sample_from_anchor = 32, True
                owner.input_buffers = NS()
                owner.block_tables = NS(slot_mappings=torch.zeros(2, 512),
                                        input_block_tables=[None, None], kernel_block_sizes=[64, 64],
                                        cp_rank=0, cp_size=1, cp_interleave=1)
                owner.kv_cache_config = None
                for name in ("sample_indices", "sample_pos", "sample_idx_mapping", "temperature", "seeds"):
                    setattr(owner, name, torch.zeros(160))
                owner.draft_tokens = torch.zeros(32, 5)
                owner._build_uniform_attn_metadata = lambda **kw: None
                owner._prepare_eplb_forward = lambda rows: None
                desc = NS(cg_mode=mode, num_reqs=4, num_tokens=20)
                queried = []
                def query(*args, **kwargs):
                    queried.append(True)
                    # At query time all 64 context rows must be stored exactly
                    # once, including rows 24:64 prepared before graph replay.
                    self.assertEqual([len(c) for c in owner.model.cache], [64, 64, 64])
                owner._generate_draft = query
                def replay(batch_desc):
                    owner._precompute_context_kv(0, owner._num_graph_context_tokens(batch_desc.num_reqs))
                    query()
                owner.query_cudagraph_manager = NS(run_fullgraph=replay)
                batch = NS(num_reqs=2, num_tokens=64,
                           seq_lens_cpu_upper_bound=torch.tensor([500, 700]))
                ns = dict(torch=torch, CUDAGraphMode=MODE,
                          prepare_dflash_inputs=lambda *a: None,
                          dispatch_cg_and_sync_dp=lambda *a, **kw: (desc, None),
                          build_slot_mappings_by_layer=lambda *a: None)
                propose = method("speculator.py", "DFlashSpeculator", "propose", ns)
                propose(owner, batch, {}, {}, torch.randn(64, 3), None,
                        *([torch.zeros(32)] * 6))
                self.assertEqual(queried, [True])

    def test_graph_factory_precompute_precedes_query_on_both_dp_ranks(self):
        # Execute the real graph-manager factory with an instrumented base.
        calls = []
        class Base:
            def capture(self, factory, progress):
                for warmup in (True, False):
                    forward = factory(NS(num_tokens=20, num_reqs=4, cg_mode=FULL), warmup)
                    forward(NONE)
        namespace = dict(torch=torch, CUDAGraphMode=MODE,
                         _prepare_dflash_inputs_to_capture=lambda *a, **kw: ("meta", "slots"))
        # Rebuild the method inside a class so zero-argument super() has its cell.
        tree = ast.parse((ROOT / "cudagraph.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashCudaGraphManager")
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "capture"]
        namespace["CudaGraphManager"] = Base
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "production-cudagraph", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        manager_cls = namespace["DFlashCudaGraphManager"]
        for dp_rank in (0, 1):
            manager = manager_cls()
            manager.dp_size, manager.max_num_reqs = 2, 32
            def forward(reqs, tokens, meta, slots, across_dp, mode):
                calls.append(("query", dp_rank, reqs, tokens))
                self.assertEqual(across_dp.tolist(), [20, 20])
                self.assertEqual((meta, slots, mode), ("meta", "slots", NONE))
            manager.capture(forward, None, None, None, None, 512000, True,
                            lambda reqs: calls.append(("context", dp_rank, reqs)))
        self.assertEqual([x[0] for x in calls], ["context", "query"] * 4)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DraftContextTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(dict(status="PASS", gpu_executed=False, tests=result.testsRun,
                         scope="production draft-context graph plumbing; independent CPU cache reference")))
