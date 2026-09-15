# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 port of vLLM #56633 (5372e72a9884), preserving backport launch tuning.

V4 shared kernels and their call signatures remain untouched. The auxiliary
mean epilogues are kept here because this pinned backport predates their ABI.
The normalized epilogue follows the SGLang-derived implementation in vLLM
(tilelang_kernels.py, originally sglang/srt/layers/mhc.py).
GPU acceptance is required; no A100 speedup has been measured on the build host.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from vllm.platforms import current_platform
from vllm.tilelang_utils import T, tilelang_jit
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.utils.torch_utils import direct_register_custom_op

ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()


def mhc_fused_post_pre_split_config(num_tokens: int, hidden_size: int, hc_mult: int):
    """Retain the A100 backport cutoff/tuning, rejecting incomplete tiles."""
    from vllm.model_executor.kernels.mhc.tilelang import _SMALL_FMA_CONFIG
    if not 0 < num_tokens <= 16:
        return None
    tile_n, n_splits, n_thr = _SMALL_FMA_CONFIG
    mix_size = hc_mult * (hc_mult + 2)
    if mix_size % tile_n or hidden_size % (n_splits * n_thr):
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if num_tokens < 8 and hidden_size <= 4096 else 4
        n_thr = 256
    if mix_size % tile_n or hidden_size % (n_splits * n_thr):
        return None
    return tile_n, n_splits, n_thr


def mhc_fused_post_pre_splits(hidden_size: int, hc_mult: int) -> tuple[int, ...]:
    choices = (mhc_fused_post_pre_split_config(n, hidden_size, hc_mult) for n in range(1, 17))
    return tuple(sorted({c[1] for c in choices if c is not None}))

@tilelang_jit
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    pre_mix_in,
    pre_mix_out,
    aux_out,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
    use_pre_mix_in: bool = False,
    save_pre_mix: bool = False,
    rms_numel: int = 0,
    write_aux: bool = False,
):
    """Fuse coefficient generation and residual collapse after the projection.

    With save_pre_mix, store the new pre-mix and collapse with pre_mix_in,
    or select stream zero when use_pre_mix_in is false.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)
    if rms_numel == 0:
        rms_numel = hc_mult * hidden_size

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    # outputs
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    pre_mix_in: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    pre_mix_out: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    aux_out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        if ENABLE_PDL:
            T.pdl_sync()
        ##################################################################
        # _pre_norm_fn_fwd_norm
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / rms_numel + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            ##################################################################
            # _pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                if save_pre_mix:
                    pre_mix_out[i, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            ##################################################################
            # _sinkhorn_fwd
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # save comb_mix to global memory
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            ##################################################################
            # _pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                if use_pre_mix_in:
                    pre_mix_shared[j] = pre_mix_in[i, j]
                elif save_pre_mix:
                    pre_mix_shared[j] = T.if_then_else(j == 0, 1.0, 0.0)
                else:
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )
            ###################################################################
            # _pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)
                if write_aux:
                    # Aux consumers (draft models) take the plain stream mean
                    # of the same residual this collapse reads.
                    aux = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(aux)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]
                        if write_aux:
                            aux[i1_h] += xl[i_hc, i1_h]

                if write_aux:
                    for i1_h in T.Parallel(hidden_block):
                        aux_out[i, i0_h * hidden_block + i1_h] = T.bfloat16(
                            aux[i1_h] / hc_mult
                        )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()

@tilelang_jit
def mhc_pre_big_fuse_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    pre_mix_in,
    pre_mix_out,
    aux_out,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    use_pre_mix_in: bool = False,
    save_pre_mix: bool = False,
    rms_numel: int = 0,
    write_aux: bool = False,
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    if rms_numel == 0:
        rms_numel = hc_mult * hidden_size
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    pre_mix_in: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    pre_mix_out: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    aux_out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / rms_numel + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                if save_pre_mix:
                    pre_mix_out[i, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                if use_pre_mix_in:
                    pre_mix_shared[j] = pre_mix_in[i, j]
                elif save_pre_mix:
                    pre_mix_shared[j] = T.if_then_else(j == 0, 1.0, 0.0)
                else:
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )

            # Pass 1: stash unnormalized weighted-sum output in shared memory
            # as bf16 (matches the rounding that RMSNorm would see) while
            # accumulating the per-position squared sum.
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)
                if write_aux:
                    # Aux consumers (draft models) take the plain stream mean
                    # of the same residual this collapse reads.
                    aux = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(aux)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]
                        if write_aux:
                            aux[i1_h] += xl[i_hc, i1_h]

                if write_aux:
                    for i1_h in T.Parallel(hidden_block):
                        aux_out[i, i0_h * hidden_block + i1_h] = T.bfloat16(
                            aux[i1_h] / hc_mult
                        )

                if save_pre_mix:
                    # Keep the BF16 boundary before the delayed input RMSNorm.
                    rounded = T.alloc_fragment(hidden_block, T.bfloat16)
                    T.copy(ol, rounded)
                    for i1_h in T.Parallel(hidden_block):
                        value = T.float32(rounded[i1_h])
                        sumsq_per_pos[i1_h] += value * value
                        output_shared[i0_h * hidden_block + i1_h] = rounded[i1_h]
                else:
                    for i1_h in T.Parallel(hidden_block):
                        sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                        output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            # Pass 2: scale by rsqrt * norm_weight and write the result to HBM.
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()

class DSV41MHCPreNormKernel(VllmJitKernel["DSV41MHCPreNormKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        hidden_size: int
        rms_eps: float
        hc_pre_eps: float
        hc_sinkhorn_eps: float
        hc_post_mult_value: float
        sinkhorn_repeat: int
        norm_eps: float
        n_splits: int
        hc_mult: int
        use_pre_mix_in: bool
        rms_numel: int
        save_pre_mix: bool = True
        write_aux: bool = False

    kernel: Any = staticmethod(mhc_pre_big_fuse_with_norm_tilelang)

    def dispatch(self, *, n_splits, **fields):
        return self.CompileKey(n_splits=n_splits, **fields)

    def get_warmup_keys(self, *, max_tokens: int, extra_splits: Iterable[int] = (), **fields):
        from vllm.model_executor.kernels.mhc.warmup import compute_mhc_pre_num_splits
        from vllm.utils.deep_gemm import is_deep_gemm_supported
        splits = set(extra_splits) | (
            {compute_mhc_pre_num_splits(fields["rms_numel"], n) for n in range(1, max_tokens + 1, 64)}
            if is_deep_gemm_supported() else {1}
        )
        return self._trace_dispatch(self.dispatch)(n_splits=sorted(splits), **fields)

    def compile(self, compile_key):
        if compile_key not in self._compiled_cache:
            self._compiled_cache[compile_key] = self.kernel.compile(**asdict(compile_key))

    def __call__(self, *tensors, **fields):
        key = self.dispatch(n_splits=tensors[0].shape[0], **fields)
        if not current_platform.is_cuda():
            return self.kernel(*tensors, **asdict(key))
        return self._get_or_compile(key)(*tensors)


DSV41_MHC_PRE_NORM_KERNEL = DSV41MHCPreNormKernel()


def register_dsv41_warmup(*, max_tokens, hidden_size, hc_mult, capture_aux, **fields):
    for use_pre_mix in (False, True):
        for write_aux in ((False, True) if capture_aux else (False,)):
            DSV41_MHC_PRE_NORM_KERNEL.register_warmup(
                max_tokens=max_tokens, hidden_size=hidden_size, hc_mult=hc_mult,
                rms_numel=hc_mult * hidden_size, use_pre_mix_in=use_pre_mix,
                extra_splits=mhc_fused_post_pre_splits(hidden_size, hc_mult),
                write_aux=write_aux, **fields,
            )

def mhc_fused_post_pre_delayed_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Run one mHC post block followed by the next delayed mHC pre block.

    Within the fused kernel's token range the post mapping is folded into the
    pre-norm GEMM, so the updated residual streams feed the projection from
    registers instead of a second pass over global memory. Above it this runs
    the same post kernel and split-k GEMM as the unfused pair.

    Args:
        x: BF16 sublayer output of shape (tokens, hidden_size).
        residual: BF16 residual streams of shape (tokens, hc_mult, hidden_size).
        post_layer_mix: FP32 post coefficients of shape (tokens, hc_mult, 1).
        comb_res_mix: FP32 residual coefficients, (tokens, hc_mult, hc_mult).
        fn: FP32 projection of shape (hc_mult * (hc_mult + 2), input_size).
        hc_scale: FP32 scales of shape (3,).
        hc_base: FP32 bias of shape (hc_mult * (hc_mult + 2),).
        rms_eps: RMS normalization epsilon.
        hc_pre_eps: Pre-mix epsilon.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult_value: Post-mix multiplier.
        sinkhorn_repeat: Number of Sinkhorn iterations.
        pre_mix: FP32 coefficients from the previous sublayer, or None to
            select residual stream zero.
        norm_weight: Optional BF16 RMSNorm weight for the collapsed input.
        norm_eps: RMSNorm epsilon for the collapsed input.
        capture_aux: Also return the mean over the post-mapped streams, which
            draft models consume as the target's hidden state. It is folded
            into the collapse, which already reads those streams.

    Returns:
        The post-mapped residual streams, the post and residual coefficients,
        the optionally normalized BF16 layer input, the next FP32 pre-mix, and
        the BF16 stream mean (empty unless capture_aux), with shapes
        (tokens, hc_mult, hidden_size), (tokens, hc_mult, 1),
        (tokens, hc_mult, hc_mult), (tokens, hidden_size), (tokens, hc_mult),
        and (tokens, hidden_size).
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_fused_tilelang,
        mhc_post_tilelang as _post_kernel,
    )
    from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm
    from vllm.model_executor.kernels.mhc.warmup import compute_mhc_pre_num_splits
    from vllm.utils.deep_gemm import (
        is_deep_gemm_supported,
        tf32_hc_prenorm_gemm,
    )

    assert residual.ndim == 3 and residual.dtype == torch.bfloat16
    assert residual.is_contiguous()
    num_tokens, hc_mult, hidden_size = residual.shape
    input_size = hc_mult * hidden_size
    mix_size = hc_mult * (hc_mult + 2)
    assert x.shape == (num_tokens, hidden_size) and x.dtype == torch.bfloat16
    assert x.is_contiguous()
    assert post_layer_mix.shape[:2] == (num_tokens, hc_mult)
    assert post_layer_mix.dtype == torch.float32 and post_layer_mix.is_contiguous()
    assert comb_res_mix.shape == (num_tokens, hc_mult, hc_mult)
    assert comb_res_mix.dtype == torch.float32 and comb_res_mix.is_contiguous()
    assert fn.shape == (mix_size, input_size) and fn.dtype == torch.float32
    assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
    assert hc_base.shape == (mix_size,) and hc_base.dtype == torch.float32
    if pre_mix is not None:
        assert pre_mix.shape == (num_tokens, hc_mult)
        assert pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()

    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    aux = torch.empty(
        num_tokens if capture_aux else 0,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )
    if num_tokens == 0:
        return (
            torch.empty_like(residual),
            post.unsqueeze(-1),
            comb.view(num_tokens, hc_mult, hc_mult),
            layer_input,
            next_pre_mix,
            aux,
        )

    fused_config = mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult)
    if fused_config is not None:
        tile_n, n_splits, n_thr = fused_config
        mixes = torch.empty(n_splits, num_tokens, mix_size, dtype=torch.float32, device=residual.device)
        sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=residual.device)
        residual_cur = torch.empty_like(residual)
        mhc_fused_tilelang(
            comb_res_mix, residual, post_layer_mix.view(num_tokens, hc_mult), x,
            fn.view(mix_size, hc_mult, hidden_size), mixes, sqrsum, residual_cur,
            hc_mult, hidden_size, mix_size, n_thr, 256, tile_n, n_splits,
        )
    else:
        residual_cur = torch.empty_like(residual)
        _post_kernel(
            comb_res_mix,
            residual,
            post_layer_mix.view(num_tokens, hc_mult),
            x,
            residual_cur,
            hc_mult,
            hidden_size,
        )
        # The delayed epilogue is compiled per bucketed split count, so the
        # projection has to use the same bucket rather than the raw estimate.
        use_deep_gemm = is_deep_gemm_supported()
        n_splits = (
            compute_mhc_pre_num_splits(input_size, num_tokens) if use_deep_gemm else 1
        )
        mixes = torch.empty(
            n_splits, num_tokens, mix_size, dtype=torch.float32, device=residual.device
        )
        sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )
        residual_cur_2d = residual_cur.view(num_tokens, input_size)
        if use_deep_gemm:
            tf32_hc_prenorm_gemm(residual_cur_2d, fn, mixes, sqrsum, n_splits)
        else:
            _tilelang_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                mixes,
                sqrsum,
                input_size,
                1,
            )

    outputs = (
        residual_cur,
        post.unsqueeze(-1),
        comb.view(num_tokens, hc_mult, hc_mult),
        layer_input,
        next_pre_mix,
        aux,
    )
    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
        DSV41_MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix if pre_mix is not None else post,
            next_pre_mix,
            aux if capture_aux else layer_input,
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_eps=norm_eps,
            hc_mult=hc_mult,
            use_pre_mix_in=pre_mix is not None,
            save_pre_mix=True,
            rms_numel=input_size,
            write_aux=capture_aux,
        )
        return outputs
    mhc_pre_big_fuse_tilelang(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post,
        comb,
        layer_input,
        pre_mix if pre_mix is not None else post,
        next_pre_mix,
        aux if capture_aux else layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        mixes.shape[0],
        hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=input_size,
        write_aux=capture_aux,
    )
    return outputs

def _mhc_fused_post_pre_delayed_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    num_tokens, hc_mult, hidden_size = residual.shape
    return (
        torch.empty_like(residual),
        torch.empty(
            num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        ),
        torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=residual.device),
        torch.empty(
            num_tokens if capture_aux else 0,
            hidden_size,
            dtype=torch.bfloat16,
            device=residual.device,
        ),
    )

direct_register_custom_op(
    op_name="mhc_fused_post_pre_delayed_tilelang",
    op_func=mhc_fused_post_pre_delayed_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_delayed_tilelang_fake,
)
