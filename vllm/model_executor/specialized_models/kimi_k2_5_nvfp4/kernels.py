# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Kimi-K2.5 NVFP4 specialized model.

Fused kernels that eliminate intermediate tensor materialisation in the
attention and MoE shared-expert paths.
"""

import torch

from vllm.triton_utils import tl, triton

# ── Helpers ──────────────────────────────────────────────────────────────


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


# ── Fused norm + RoPE + MLA cache write ─────────────────────────────────


@triton.jit
def _fused_norm_rope_kernel(
    pos_ptr,
    # Q RMS norm
    q_c_ptr,
    q_c_stride,
    q_rms_w_ptr,
    q_rms_eps,
    q_c_out_ptr,
    q_c_out_stride,
    Q_DIM: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    # KV RMS norm
    kv_ptr,
    kv_stride,
    kv_rms_w_ptr,
    kv_rms_eps,
    KV_DIM: tl.constexpr,
    # KV normed writeback (only when prefill tokens present)
    kv_out_ptr,
    kv_out_stride,
    WRITE_KV_NORMED: tl.constexpr,
    # K_pe RoPE (interleaved)
    kpe_ptr,
    kpe_stride,
    kpe_cos_sin_ptr,
    kpe_cos_sin_stride,
    KPE_HALF_ROT: tl.constexpr,
    # K_pe writeback (for prefill forward_mha)
    kpe_out_ptr,
    kpe_out_stride,
    # MLA KV cache
    slot_mapping_ptr,
    mla_cache_ptr,
    mla_cache_block_stride,
    mla_cache_entry_stride,
    MLA_FP8: tl.constexpr,
    mla_k_scale_ptr,
):
    pid = tl.program_id(0)  # 0 = KV path, 1 = Q path
    tok = tl.program_id(1)

    if pid == 1:
        # ── Q RMS norm ──
        q_off = tl.arange(0, Q_BLOCK)
        q_mask = q_off < Q_DIM
        q_c = tl.load(q_c_ptr + tok * q_c_stride + q_off, mask=q_mask, other=0.0)
        q_w = tl.load(q_rms_w_ptr + q_off, mask=q_mask)
        q_c = _rms_norm(q_c, q_w, q_rms_eps, Q_DIM)
        tl.store(q_c_out_ptr + tok * q_c_out_stride + q_off, q_c, mask=q_mask)
        return

    # pid == 0: ── KV RMS norm + K_pe RoPE + writeback + MLA cache write ──

    # KV RMS norm (result stays in registers)
    kv_off = tl.arange(0, KV_DIM)
    kv_c = tl.load(kv_ptr + tok * kv_stride + kv_off)
    kv_w = tl.load(kv_rms_w_ptr + kv_off)
    kv_c = _rms_norm(kv_c, kv_w, kv_rms_eps, KV_DIM)

    # Write kv_c_normed back (only needed when prefill tokens exist)
    if WRITE_KV_NORMED:
        tl.store(kv_out_ptr + tok * kv_out_stride + kv_off, kv_c)

    # K_pe interleaved RoPE (result stays in registers)
    pos = tl.load(pos_ptr + tok)
    cos, sin = _get_cos_sin(kpe_cos_sin_ptr, kpe_cos_sin_stride, pos, KPE_HALF_ROT)
    dim_off = tl.arange(0, KPE_HALF_ROT)
    kpe_base = kpe_ptr + tok * kpe_stride
    x1 = tl.load(kpe_base + dim_off * 2).to(tl.float32)
    x2 = tl.load(kpe_base + dim_off * 2 + 1).to(tl.float32)
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin

    # Write roped K_pe back (needed by prefill forward_mha)
    kpe_out_base = kpe_out_ptr + tok * kpe_out_stride
    tl.store(kpe_out_base + dim_off * 2, r1)
    tl.store(kpe_out_base + dim_off * 2 + 1, r2)

    # MLA concat_and_cache: [kv_c_normed, k_pe_roped] → cache
    if slot_mapping_ptr is None:
        return
    slot = tl.load(slot_mapping_ptr + tok)
    if slot < 0:
        return
    if mla_cache_entry_stride == 0:
        return

    mla_block_size = mla_cache_block_stride // mla_cache_entry_stride
    blk_idx = slot // mla_block_size
    blk_off = slot % mla_block_size
    dst = (
        mla_cache_ptr
        + blk_idx * mla_cache_block_stride
        + blk_off * mla_cache_entry_stride
    )

    if MLA_FP8:
        scale = tl.load(mla_k_scale_ptr)
        kv_fp8 = (kv_c.to(tl.float32) / scale).to(tl.float8e4nv)
        tl.store(dst + kv_off, kv_fp8)
        tl.store(dst + KV_DIM + dim_off * 2, (r1 / scale).to(tl.float8e4nv))
        tl.store(dst + KV_DIM + dim_off * 2 + 1, (r2 / scale).to(tl.float8e4nv))
    else:
        tl.store(dst + kv_off, kv_c)
        tl.store(dst + KV_DIM + dim_off * 2, r1)
        tl.store(dst + KV_DIM + dim_off * 2 + 1, r2)


def fused_norm_rope(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    q_rms_w: torch.Tensor,
    q_rms_eps: float,
    kv_c: torch.Tensor,
    kv_rms_w: torch.Tensor,
    kv_rms_eps: float,
    k_pe: torch.Tensor,
    k_rope_cos_sin_cache: torch.Tensor,
    *,
    slot_mapping: torch.Tensor | None = None,
    mla_kv_cache: torch.Tensor | None = None,
    mla_kv_cache_dtype: str = "auto",
    mla_k_scale: torch.Tensor | None = None,
    has_prefill: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Fused Q/KV RMS norm + K_pe RoPE + MLA cache write.

    Returns (q_c_normed, kv_c_normed, k_pe_roped).
    kv_c_normed is only written back when *has_prefill* is True (decode-only
    batches skip the writeback, saving global-memory bandwidth).
    k_pe_roped is always written back (prefill forward_mha needs it).
    """
    num_tokens = positions.shape[0]
    q_dim = q_c.shape[-1]
    kv_dim = kv_c.shape[-1]
    device = positions.device

    mla_fp8 = mla_kv_cache_dtype not in ("auto", "bfloat16", "float16")
    if mla_kv_cache is not None:
        blk_stride = mla_kv_cache.stride(0)
        entry_stride = mla_kv_cache.stride(1)
        if mla_fp8 and mla_kv_cache.dtype == torch.uint8:
            mla_kv_cache = mla_kv_cache.view(torch.float8_e4m3fn)
        if mla_k_scale is None:
            mla_k_scale = torch.ones(1, dtype=torch.float32, device=device)
    else:
        mla_kv_cache = torch.empty(0, dtype=torch.bfloat16, device=device)
        blk_stride = 0
        entry_stride = 0
        mla_k_scale = torch.ones(1, dtype=torch.float32, device=device)

    q_c_out = torch.empty_like(q_c)
    kv_c_out = torch.empty_like(kv_c) if has_prefill else kv_c
    k_pe_out = torch.empty_like(k_pe)

    _fused_norm_rope_kernel[(2, num_tokens)](
        positions,
        # Q
        q_c,
        q_c.stride(0),
        q_rms_w,
        q_rms_eps,
        q_c_out,
        q_c_out.stride(0),
        q_dim,
        triton.next_power_of_2(q_dim),
        # KV
        kv_c,
        kv_c.stride(0),
        kv_rms_w,
        kv_rms_eps,
        kv_dim,
        # KV writeback
        kv_c_out,
        kv_c_out.stride(0),
        has_prefill,
        # K_pe RoPE
        k_pe,
        k_pe.stride(0),
        k_rope_cos_sin_cache,
        k_rope_cos_sin_cache.stride(0),
        k_rope_cos_sin_cache.shape[-1] // 2,
        # K_pe writeback
        k_pe_out,
        k_pe_out.stride(0),
        # MLA cache
        slot_mapping,
        mla_kv_cache,
        blk_stride,
        entry_stride,
        mla_fp8,
        mla_k_scale,
    )
    return q_c_out, kv_c_out if has_prefill else None, k_pe_out


# ── Q-only RoPE (interleaved) ───────────────────────────────────────────
# Avoids the wasted K RoPE computation that happens when calling the
# standard rotary_emb module with an already-roped K_pe.


@triton.jit
def _q_rope_kernel(
    pos_ptr,
    q_ptr,
    q_stride0,  # stride across tokens (flat, before reshape)
    cos_sin_ptr,
    cos_sin_stride,
    nope_dim,  # skip this many values per head
    HALF_ROT: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Apply interleaved RoPE to the pe portion of every head, in-place."""
    tok = tl.program_id(0)
    pos = tl.load(pos_ptr + tok)
    cos, sin = _get_cos_sin(cos_sin_ptr, cos_sin_stride, pos, HALF_ROT)

    dim_off = tl.arange(0, HALF_ROT)
    head_off = tl.arange(0, NUM_HEADS)

    base = q_ptr + tok * q_stride0
    # pe starts at offset nope_dim within each head
    ptrs_even = base + head_off[:, None] * HEAD_DIM + nope_dim + dim_off[None, :] * 2
    ptrs_odd = ptrs_even + 1

    x1 = tl.load(ptrs_even).to(tl.float32)
    x2 = tl.load(ptrs_odd).to(tl.float32)
    tl.store(ptrs_even, x1 * cos[None, :] - x2 * sin[None, :])
    tl.store(ptrs_odd, x2 * cos[None, :] + x1 * sin[None, :])


def q_rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> None:
    """In-place interleaved Q RoPE.  K is not touched."""
    num_tokens = positions.shape[0]
    head_dim = qk_nope_head_dim + qk_rope_head_dim
    _q_rope_kernel[(num_tokens,)](
        positions,
        q,
        q.stride(0),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        qk_nope_head_dim,
        qk_rope_head_dim // 2,
        num_heads,
        head_dim,
    )


# ── Fused SiLU-and-Mul + NVFP4 quantization ─────────────────────────────
#
# PTX inline assembly for exact bitwise match with the CUDA reference:
#   rcp.approx.ftz.f32, ex2.approx.ftz.f32, cvt.rn.satfinite.e2m1x2.f32


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        asm="rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _ex2_approx_ftz(x):
    return tl.inline_asm_elementwise(
        asm="ex2.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_e2m1x2(even, odd):
    return tl.inline_asm_elementwise(
        asm=(
            "{ .reg .b8 tmp;"
            "  cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
            "  cvt.u32.u8 $0, tmp; }"
        ),
        constraints="=r,f,f",
        args=[even, odd],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _silu_mul_nvfp4_quant_kernel(
    inp_ptr,
    out_ptr,
    sf_out_ptr,
    sf_scale_ptr,
    M,
    N,
    num_groups,
    stride_in_m,
    stride_out_m,
    num_k_tiles,
    PAIRS: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Swizzled scale-factor offset
    sf_col = pid_g
    m_tile = pid_m // 128
    outer_m = pid_m % 32
    inner_m = (pid_m // 32) % 4
    k_tile = sf_col // 4
    inner_k = sf_col % 4
    sf_offset = (
        (m_tile * num_k_tiles + k_tile) * 512 + outer_m * 16 + inner_m * 4 + inner_k
    )

    if pid_m >= M or pid_g >= num_groups:
        tl.store(sf_out_ptr + sf_offset, tl.cast(0, tl.uint8))
        return

    col_base = pid_g * (PAIRS * 2)
    pair_off = tl.arange(0, PAIRS)
    even_cols = col_base + 2 * pair_off
    odd_cols = even_cols + 1
    row_off = pid_m * stride_in_m

    # Load gate / up
    gate_e = tl.load(inp_ptr + row_off + even_cols).to(tl.float32)
    up_e = tl.load(inp_ptr + row_off + N + even_cols).to(tl.float32)
    gate_o = tl.load(inp_ptr + row_off + odd_cols).to(tl.float32)
    up_o = tl.load(inp_ptr + row_off + N + odd_cols).to(tl.float32)

    # SiLU
    exp_e = _ex2_approx_ftz((-gate_e) * LOG2E)
    exp_o = _ex2_approx_ftz((-gate_o) * LOG2E)
    silu_e = gate_e * _rcp_approx_ftz(1.0 + exp_e)
    silu_o = gate_o * _rcp_approx_ftz(1.0 + exp_o)

    res_e = silu_e * up_e
    res_o = silu_o * up_o

    # BF16 round-trip
    res_e = res_e.to(tl.bfloat16).to(tl.float32)
    res_o = res_o.to(tl.bfloat16).to(tl.float32)

    # Per-group amax → scale factor
    amax = tl.maximum(tl.max(tl.abs(res_e)), tl.max(tl.abs(res_o)))
    sf_scale_val = tl.load(sf_scale_ptr).to(tl.float32)
    sf_raw = sf_scale_val * (amax * _rcp_approx_ftz(6.0))

    sf_fp8 = sf_raw.to(tl.float8e4nv)
    sf_rounded = sf_fp8.to(tl.float32)
    sf_byte = sf_fp8.to(tl.uint8, bitcast=True)
    tl.store(sf_out_ptr + sf_offset, sf_byte)

    rcp_sf_scale = _rcp_approx_ftz(sf_scale_val)
    out_scale = tl.where(
        sf_rounded != 0.0,
        _rcp_approx_ftz(sf_rounded * rcp_sf_scale),
        0.0,
    )

    scaled_e = res_e * out_scale
    scaled_o = res_o * out_scale

    packed = _cvt_e2m1x2(scaled_e, scaled_o).to(tl.uint8)

    out_off = pid_m * stride_out_m + col_base // 2 + pair_off
    tl.store(out_ptr + out_off, packed)


def silu_and_mul_nvfp4_quant(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU-and-Mul + NVFP4 quantization with swizzled scale layout.

    Returns (fp4_packed_uint8, fp8_block_scales).
    """
    from vllm.utils.math_utils import round_up

    if input.ndim == 1:
        input = input.unsqueeze(0)
    else:
        input = input.reshape(-1, input.shape[-1])

    M, two_N = input.shape
    N = two_N // 2
    assert N % 16 == 0, f"N must be a multiple of 16, got {N}"

    num_groups = N // 16
    num_k_tiles = (num_groups + 3) // 4

    output = torch.empty((M, N // 2), device=input.device, dtype=torch.uint8)
    rounded_m = round_up(M, 128)
    rounded_num_groups = round_up(num_groups, 4)
    output_scale = torch.empty(
        (rounded_m, rounded_num_groups // 4),
        device=input.device,
        dtype=torch.int32,
    )

    _silu_mul_nvfp4_quant_kernel[(rounded_m, rounded_num_groups)](
        input,
        output,
        output_scale.view(torch.uint8),
        input_global_scale,
        M,
        N,
        num_groups,
        input.stride(0),
        output.stride(0),
        num_k_tiles,
        PAIRS=8,
    )

    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale
