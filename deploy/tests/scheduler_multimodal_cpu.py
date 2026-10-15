"""Real scheduler/cache checks for strict 1024-token atomic image spans.

Run in the original runtime image without --gpus. No image encoder or model
inference executes. Optional --model builds the actual VllmConfig using local
metadata and an explicitly simulated A100 hardware facade; it is not a GPU test.
"""
import argparse
import __future__
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import sysconfig

import numpy as np
import torch
import triton
import triton.language as tl
import vllm.triton_utils as facade
facade.triton, facade.tl = triton, tl

from vllm.config import SchedulerConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager, compute_mm_encoder_budget
from vllm.v1.core.sched.scheduler import Scheduler
import vllm.v1.worker.gpu.mm.encoder_runner as encoder_module
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

DEPLOY = Path(__file__).resolve().parents[1]
MODEL = None


def scheduler_config(**overrides):
    values = dict(max_model_len=512000, is_encoder_decoder=False,
                  max_num_batched_tokens=1024, max_num_seqs=32,
                  enable_chunked_prefill=True, is_multimodal_model=True,
                  disable_chunked_mm_input=True)
    return SchedulerConfig(**dict(values, **overrides))


def request(identifier="one-image", length=1020, offset=40):
    features = [MultiModalFeatureSpec(data=None, modality="image", identifier=identifier,
                                     mm_position=PlaceholderRange(offset=offset, length=length))]
    return NS(request_id=identifier, has_encoder_inputs=True, mm_features=features,
              get_num_encoder_embeds=lambda i: features[i].mm_position.get_num_embeds())


class SchedulerMultimodalTests(unittest.TestCase):
    def test_encoder_budget_is_independent_and_does_not_inflate_decoder_budget(self):
        config = scheduler_config()
        self.assertTrue(config.disable_chunked_mm_input)
        self.assertEqual(compute_mm_encoder_budget(config, {"image": 1020}), (1024, 1024))
        self.assertEqual(config.max_num_batched_tokens, 1024)
        self.assertEqual(config.max_num_encoder_input_tokens, 1024)
        self.assertEqual(config.max_num_seqs, 32)
        with self.assertRaisesRegex(ValueError, "Chunked MM input disabled"):
            compute_mm_encoder_budget(config, {"image": 2048})

    def test_image_span_is_atomic_and_never_exceeds_1024_decoder_tokens(self):
        config = scheduler_config()
        encoder_budget, cache_size = compute_mm_encoder_budget(config, {"image": 1020})
        manager = EncoderCacheManager(cache_size=cache_size)
        scheduler = NS(scheduler_config=config, is_encoder_decoder=False,
                       encoder_cache_manager=manager, ec_connector=None)
        req = request()
        outputs = torch.arange(1020 * 3).view(1020, 3).float()
        encoder = NS(is_realtime=False, encoder_cache=NS(
            mm_features={req.request_id: req.mm_features},
            encoder_outputs={req.request_id: outputs}))
        collected, scheduled_items = [], []
        # A text prefix is scheduled first, then the complete image span,
        # then trailing text. Bidirectional image attention is never split.
        for cursor, requested, expected_count in ((0, 1020, 40), (40, 1020, 1020), (1060, 16, 16)):
            items, scheduled, left, external = Scheduler._try_schedule_encoder_inputs(
                scheduler, req, cursor, requested, encoder_budget)
            self.assertEqual(scheduled, expected_count)
            self.assertLessEqual(scheduled, 1024)
            self.assertEqual(external, [])
            for item in items:
                manager.allocate(req, item)
                scheduled_items.append(item)
            with patch.object(encoder_module, "PIN_MEMORY", False):
                embeds, mask = EncoderRunner.gather_mm_embeddings(
                    encoder, [req.request_id], scheduled,
                    np.array([scheduled]), np.array([0, scheduled]),
                    np.array([1076]), np.array([cursor]))
            self.assertEqual(sum(t.shape[0] for t in embeds), int(mask.sum()))
            collected.extend(embeds)
            self.assertEqual(left, 4 if cursor == 40 else encoder_budget)
        self.assertEqual(scheduled_items, [0])
        torch.testing.assert_close(torch.cat(collected), outputs, atol=0, rtol=0)
        self.assertEqual(config.max_num_batched_tokens, 1024)

    def test_multiple_images_reclaim_encoder_cache_without_raising_decoder_budget(self):
        config = scheduler_config()
        manager = EncoderCacheManager(cache_size=1024)
        scheduler = NS(scheduler_config=config, is_encoder_decoder=False,
                       encoder_cache_manager=manager, ec_connector=None)
        first, second = request("first", length=1024, offset=0), request("second", length=1024, offset=0)
        items, count, _, _ = Scheduler._try_schedule_encoder_inputs(scheduler, first, 0, 1024, 1024)
        self.assertEqual((items, count), ([0], 1024))
        manager.allocate(first, 0)
        items, count, _, _ = Scheduler._try_schedule_encoder_inputs(scheduler, second, 0, 1024, 1024)
        self.assertEqual((items, count), ([], 0))  # wait for the first image span
        manager.free_encoder_input(first, 0)
        items, count, _, _ = Scheduler._try_schedule_encoder_inputs(scheduler, second, 0, 1024, 1024)
        self.assertEqual((items, count), ([0], 1024))
        manager.allocate(second, 0)
        self.assertEqual(manager.get_freed_mm_hashes(), ["first"])
        self.assertEqual(config.max_num_batched_tokens, 1024)

    def test_atomic_guard_covers_exact_start_text_prefix_cached_and_duplicate_images(self):
        config = scheduler_config()
        manager = EncoderCacheManager(cache_size=1024)
        scheduler = NS(scheduler_config=config, is_encoder_decoder=False,
                       encoder_cache_manager=manager, ec_connector=None)
        req = request(length=1024)
        # Decode consumes one token or a full32-request verify batch (192).
        # An image at its exact start must wait, then progress atomically.
        for available in (1023, 704):
            for cached in (False, True):
                manager.reset()
                if cached:
                    manager.allocate(req, 0)
                items, count, _, _ = Scheduler._try_schedule_encoder_inputs(
                    scheduler, req, 40, available, 1024)
                self.assertEqual((items, count), ([], 0))
                items, count, _, _ = Scheduler._try_schedule_encoder_inputs(
                    scheduler, req, 40, 1024, 1024)
                self.assertEqual((items, count), ([] if cached else [0], 1024))
        manager.reset()
        # A prefix-cache boundary inside an image still needs the complete
        # uncomputed remainder, not another partial image chunk.
        items, count, _, _ = Scheduler._try_schedule_encoder_inputs(scheduler, req, 50, 1012, 1024)
        self.assertEqual((items, count), ([], 0))
        items, count, _, _ = Scheduler._try_schedule_encoder_inputs(scheduler, req, 50, 1014, 1024)
        self.assertEqual((items, count), ([0], 1014))
        # One image hash appears twice at different prompt positions. An
        # encoder-cache duplicate must not bypass the second decoder boundary.
        duplicate = request("duplicate", length=256, offset=0)
        duplicate.mm_features.append(MultiModalFeatureSpec(
            data=None, modality="image", identifier="duplicate",
            mm_position=PlaceholderRange(offset=256, length=256)))
        items, count, _, _ = Scheduler._try_schedule_encoder_inputs(scheduler, duplicate, 0, 400, 1024)
        self.assertEqual((items, count), ([0], 256))

    def test_real_cli_and_full_vllm_config_with_simulated_a100_capability(self):
        if MODEL is None:
            self.skipTest("pass --model for metadata-only full VllmConfig construction")
        import vllm.platforms
        from vllm.platforms.interface import DeviceCapability, PlatformEnum
        current = vllm.platforms.current_platform
        class ConfigOnlyA100(type(current)):
            _enum = PlatformEnum.CUDA
            device_type, device_name = "cuda", "cuda"
            @classmethod
            def get_device_capability(cls, *args, **kwargs): return DeviceCapability(8, 0)
            @classmethod
            def get_device_name(cls, *args, **kwargs): return "NVIDIA A100-SXM4-80GB"
            @classmethod
            def get_device_total_memory(cls, *args, **kwargs): return 80 * 1024**3
            @classmethod
            def device_count(cls, *args, **kwargs): return 8
            @classmethod
            def support_hybrid_kv_cache(cls): return True
            @classmethod
            def support_static_graph_mode(cls): return True
        previous_class = current.__class__
        current.__class__ = ConfigOnlyA100
        try:
            from vllm.entrypoints.launchers.cli_args import make_arg_parser
            from vllm.utils.argparse_utils import FlexibleArgumentParser
            from vllm.engine.arg_utils import EngineArgs
            import vllm.model_executor.models.registry as registry
            import vllm.config.vllm as config_module
            import sys
            sys.path.insert(0, str(DEPLOY / "scripts"))
            import offline_ops
            base = json.loads((DEPLOY / "configs/runtime.json").read_text())
            resolved, argv = offline_ops.resolve(base, 64)
            self.assertIn("--disable-chunked-mm-input", argv)
            parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)
            self.assertTrue(parsed.disable_chunked_mm_input)
            self.assertEqual(parsed.hf_overrides, {"vision_max_n_token": 1020})
            self.assertEqual(parsed.served_model_name, ["DeepSeek-Flash"])
            self.assertTrue(parsed.enable_chunked_prefill)
            self.assertEqual(parsed.max_num_batched_tokens, 1024)
            parsed.model = parsed.model_tag = str(MODEL)
            # Only hardware/model-inspection transport is simulated. Inspect
            # the actual model class in-process so the CPU Triton facade is
            # retained; no GPU context or model weight is initialized.
            # Triton itself is installed, but the build host has no active GPU
            # driver. Let config validation see the restored real module; no
            # kernel compilation, launch or CUDA allocation is attempted.
            with patch.object(registry, "_run_in_subprocess", side_effect=lambda fn: fn()), \
                 patch.object(config_module, "HAS_TRITON", True):
                config = EngineArgs.from_cli_args(parsed).create_engine_config()
            self.assertEqual(config.scheduler_config.max_num_batched_tokens, 1024)
            self.assertEqual(config.scheduler_config.max_num_seqs, 32)
            self.assertTrue(config.scheduler_config.enable_chunked_prefill)
            self.assertTrue(config.scheduler_config.disable_chunked_mm_input)
            self.assertTrue(config.model_config.is_mm_prefix_lm)
            self.assertEqual(config.model_config.hf_config.vision_max_n_token, 1020)
            self.assertEqual(config.model_config.max_model_len, 512000)
            self.assertEqual(config.model_config.served_model_name, "DeepSeek-Flash")
            self.assertEqual(config.parallel_config.tensor_parallel_size, 4)
            self.assertEqual(config.parallel_config.data_parallel_size, 2)
            self.assertEqual(config.speculative_config.num_speculative_tokens, 5)
            # Execute the exact CUDA platform config rule without importing
            # its driver-linked binary on this GPU-less build host. This rule
            # must preserve, rather than bypass, atomic bidirectional images.
            cuda_source = Path(sysconfig.get_paths()["purelib"]) / "vllm/platforms/cuda.py"
            classes = ast.parse(cuda_source.read_text()).body
            cuda_class = next(n for n in classes if isinstance(n, ast.ClassDef) and n.name == "CudaPlatformBase")
            check = next(n for n in cuda_class.body if isinstance(n, ast.FunctionDef) and n.name == "check_and_update_config")
            ns = dict(in_wsl=lambda: False, logger=NS(warning_once=lambda *args: None))
            exec(compile(ast.Module(body=[check], type_ignores=[]), str(cuda_source), "exec",
                         flags=__future__.annotations.compiler_flag), ns)
            check_cuda = ns["check_and_update_config"].__func__
            check_cuda(ConfigOnlyA100, config)
            self.assertTrue(config.scheduler_config.disable_chunked_mm_input)
            # Prove that explicitly false would be coerced by the same real
            # rule; oversized image spans cannot be hidden inside a fixed batch budget.
            invalid = NS(parallel_config=NS(prefill_context_parallel_size=1, data_parallel_size=2,
                                           worker_cls="auto"), model_config=config.model_config,
                         scheduler_config=scheduler_config(disable_chunked_mm_input=False))
            check_cuda(ConfigOnlyA100, invalid)
            self.assertTrue(invalid.scheduler_config.disable_chunked_mm_input)
            with self.assertRaises(ValueError):
                compute_mm_encoder_budget(invalid.scheduler_config, {"image": 2048})
            # Process real PIL inputs through the installed production image
            # processor, including tall/wide extremes and large screenshots.
            from PIL import Image, ImageDraw, ImageOps
            from vllm.models.deepseek_v4_1.common.mm_preprocess import (
                DeepseekV4VLImageProcessor, image_token_types, num_image_tokens, safe_resize)
            processor = DeepseekV4VLImageProcessor(config.model_config.hf_config)
            normal_parity, degenerate_fixed = 0, 0
            for width, height in ((1, 1), (1, 65536), (65536, 1), (240, 320),
                                  (1920, 1080), (3840, 2160), (4000, 6000), (1024, 1024)):
                image = Image.new("RGB", (width, height), "white")
                ImageDraw.Draw(image).rectangle((width // 4, height // 4,
                                                3 * width // 4, 3 * height // 4), fill="black")
                patches, vit_h, vit_w, llm_h, llm_w = processor(image)
                span = image_token_types(llm_h, llm_w)
                self.assertGreater(span.numel(), 0)
                self.assertLessEqual(span.numel(), 1020)
                self.assertEqual(span.numel(), num_image_tokens(llm_h, llm_w))
                self.assertEqual(patches.shape[0], vit_h * vit_w)
                pixel_shape = (vit_w * processor.patch_size, vit_h * processor.patch_size)
                try:
                    legacy_pad = ImageOps.pad(image, pixel_shape, color=(127, 127, 127))
                except ValueError:
                    scale = min(pixel_shape[0] / width, pixel_shape[1] / height)
                    self.assertLess(min(round(width * scale), round(height * scale)), 1)
                    degenerate_fixed += 1
                else:
                    # Original PIL padding and pixel normalization must remain
                    # bit-for-bit identical on every non-degenerate case.
                    x = torch.from_numpy(np.asarray(legacy_pad, dtype=np.float32)).permute(2, 0, 1) / 255
                    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
                    p = processor.patch_size
                    reference = x.reshape(3, vit_h, p, vit_w, p).permute(1, 3, 0, 2, 4).reshape(vit_h * vit_w, 3, p, p)
                    torch.testing.assert_close(patches, reference, atol=0, rtol=0)
                    normal_parity += 1
            self.assertGreater(normal_parity, 0)
            self.assertGreater(degenerate_fixed, 0)
            # Exhaust varied aspect ratios through the real resize arithmetic
            # without allocating giant image arrays.
            import math
            import random
            rng = random.Random(1024)
            for _ in range(1000):
                width, height = rng.randint(1, 100000), rng.randint(1, 100000)
                p, ratio = processor.patch_size, processor.downsample_ratio
                h, w, _, _ = safe_resize(height, width, math.ceil(height / p) * p,
                                        math.ceil(width / p) * p, p, ratio, 1020)
                self.assertLessEqual(num_image_tokens(h, w), 1020)
            from vllm.multimodal import MULTIMODAL_REGISTRY
            from vllm.multimodal.encoder_budget import MultiModalBudget
            budget = MultiModalBudget(config, MULTIMODAL_REGISTRY)
            self.assertEqual(budget.mm_max_toks_per_item, {"image": 1020})
            self.assertEqual((budget.encoder_compute_budget, budget.encoder_cache_size), (1024, 1024))
            self.assertEqual(config.model_config.multimodal_config.get_limit_per_prompt("image"), 999)
            self.assertFalse(hasattr(config.model_config.hf_config, "mm_prefix_span_leading_pad_modulus"))
            # Exercise the complete production scheduler, not only its encoder
            # helper. DSpark reserves k-1 slots even during image prefill.
            from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec
            from vllm.v1.request import Request
            from vllm.sampling_params import SamplingParams
            from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
            import hashlib
            import pickle
            hash_fn = lambda value: hashlib.sha256(pickle.dumps(value)).digest()
            init_none_hash(hash_fn)
            config.cache_config.num_gpu_blocks = 128
            self.assertEqual(config.speculative_config.max_num_new_slots_for_drafting, 4)
            schedule_proof = {}
            for span_tokens in (1020, 1024):
                kv_config = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=[
                    KVCacheGroupSpec(layer_names=["test.layer"], kv_cache_spec=FullAttentionSpec(
                        block_size=64, num_kv_heads=1, head_size=8, dtype=torch.bfloat16))])
                live_scheduler = Scheduler(config, kv_config, structured_output_manager=NS(), block_size=64)
                features = request(f"image-{span_tokens}", length=span_tokens, offset=0).mm_features
                live_request = Request(request_id=f"image-{span_tokens}",
                                       prompt_token_ids=[129264] * span_tokens,
                                       sampling_params=SamplingParams(max_tokens=16), pooling_params=None,
                                       mm_features=features, block_hasher=get_request_block_hasher(64, hash_fn))
                live_scheduler.add_request(live_request)
                scheduled = live_scheduler.schedule()
                count = scheduled.num_scheduled_tokens.get(live_request.request_id, 0)
                schedule_proof[str(span_tokens)] = count
                if span_tokens == 1020:
                    self.assertEqual(count, 1020)
                    self.assertEqual(count + config.speculative_config.max_num_new_slots_for_drafting, 1024)
                    self.assertEqual(scheduled.scheduled_encoder_inputs[live_request.request_id], [0])
                else:
                    self.assertEqual(count, 0)  # why the processor cap must be1020
            print(json.dumps(dict(check="vision1024", actual_processor_cases=8,
                                  normal_pad_exact_parity=normal_parity,
                                  degenerate_original_failure_fixed=degenerate_fixed,
                                  resize_math_cases=1000, image_span_token_cap=1020, requested_image_token_ceiling=1024,
                                  real_scheduler_prefill_tokens=schedule_proof,
                                  dspark_reserved_slots=4,
                                  delimiter_tokens_included=True, atomic_bidirectional_images=True)))
        finally:
            current.__class__ = previous_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    options = parser.parse_args()
    MODEL = options.model
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SchedulerMultimodalTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    print(json.dumps(dict(status="PASS", gpu_executed=False, tests=result.testsRun,
                         skipped=len(result.skipped), simulated_hardware_config=MODEL is not None,
                         scope="real scheduler, encoder cache and embedding-slice CPU checks; no GPU inference")))
