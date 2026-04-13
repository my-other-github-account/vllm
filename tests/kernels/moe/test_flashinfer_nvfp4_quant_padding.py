# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for FlashInfer NVFP4 quantization cross-row scale corruption.

FlashInfer's silu_and_mul_scaled_nvfp4_experts_quantize and
scaled_fp4_grouped_quantize kernels corrupt real scales when
CUDA Graph padding rows contain NaN or garbage. The scale
computation leaks across rows likely via warp-level reduction, corrupting
real token output.

Fix: zero-fill padding rows in flashinfer_cutedsl_moe_masked before
calling the FlashInfer quantization kernels.
"""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "num_experts,m,num_real",
    [
        (4, 8, 4),
        (8, 16, 8),
        (8, 32, 16),
        (16, 64, 32),
        (32, 128, 64),
        (8, 64, 8),  # sparse — Can happen due to with DP padding
        (4, 16, 1),  # extreme: 1 real token
    ],
    ids=[
        "E4_m8",
        "E8_m16",
        "E8_m32",
        "E16_m64",
        "E32_m128",
        "E8_m64_sparse",
        "E4_m16_1tok",
    ],
)
@pytest.mark.xfail(
    reason="FlashInfer kernel bug: cross-row scale corruption when "
    "padding rows contain NaN. Fix applied in wrapper.",
    strict=False,
)
def test_silu_quant_cross_row_corruption(num_experts, m, num_real):
    """FlashInfer silu_and_mul_scaled_nvfp4_experts_quantize must not
    corrupt real token output when padding rows contain NaN.

    This is the production kernel used between gate_up_proj and down_proj
    in the FlashInferCuteDSLBatchedExperts MoE path.
    """
    from flashinfer import silu_and_mul_scaled_nvfp4_experts_quantize

    device = "cuda"
    n = 512

    base = num_real // num_experts
    remainder = num_real % num_experts
    masked_m = torch.tensor(
        [base + (1 if i < remainder else 0) for i in range(num_experts)],
        dtype=torch.int32,
        device=device,
    )
    a2_global_scale = torch.ones(num_experts, dtype=torch.float32, device=device)

    workspace = (
        torch.randn(num_experts, m, 2 * n, dtype=torch.bfloat16, device=device) * 0.1
    )

    # Clean reference
    workspace_clean = workspace.clone()
    for e in range(num_experts):
        real = masked_m[e].item()
        if real < m:
            workspace_clean[e, real:] = 0.0
    diq_clean, sf_clean = silu_and_mul_scaled_nvfp4_experts_quantize(
        workspace_clean, masked_m, a2_global_scale
    )

    # Dirty: NaN in padding rows
    workspace_dirty = workspace.clone()
    for e in range(num_experts):
        real = masked_m[e].item()
        if real < m:
            workspace_dirty[e, real:] = float("nan")

    # Capture in CUDA graph to match production
    graph = torch.cuda.CUDAGraph()
    workspace_input = workspace_dirty.clone()
    with torch.cuda.graph(graph):
        diq, sf = silu_and_mul_scaled_nvfp4_experts_quantize(
            workspace_input, masked_m, a2_global_scale
        )

    # Pollution replay then real replay
    workspace_input.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    workspace_input.copy_(workspace_dirty)
    graph.replay()
    torch.accelerator.synchronize()

    corrupted = False
    for e in range(num_experts):
        real = masked_m[e].item()
        if real == 0:
            continue
        if not torch.equal(diq_clean[e, :real], diq[e, :real]):
            corrupted = True
            break
        if not torch.equal(sf_clean[e, :real], sf[e, :real]):
            corrupted = True
            break

    assert not corrupted, (
        f"silu_and_mul_scaled_nvfp4_experts_quantize: NaN in padding "
        f"rows corrupted real token output "
        f"(E={num_experts}, m={m}, real={num_real})"
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "num_experts,m,num_real",
    [
        (4, 8, 4),
        (8, 16, 8),
        (8, 32, 16),
        (16, 64, 32),
        (32, 128, 64),
        (8, 64, 8),
        (4, 16, 1),
    ],
    ids=[
        "E4_m8",
        "E8_m16",
        "E8_m32",
        "E16_m64",
        "E32_m128",
        "E8_m64_sparse",
        "E4_m16_1tok",
    ],
)
@pytest.mark.xfail(
    reason="FlashInfer kernel bug: cross-row scale corruption when "
    "padding rows contain NaN. Fix applied in wrapper.",
    strict=False,
)
def test_grouped_quant_cross_row_corruption(num_experts, m, num_real):
    """FlashInfer scaled_fp4_grouped_quantize must not corrupt real token
    output when padding rows contain NaN.

    This is the kernel for the first quantization step (hidden_states -> NVFP4)
    in the FlashInferCuteDSLBatchedExperts path when NVFP4 dispatch is NOT used.
    """
    from flashinfer import scaled_fp4_grouped_quantize

    device = "cuda"
    k = 1024

    base = num_real // num_experts
    remainder = num_real % num_experts
    masked_m = torch.tensor(
        [base + (1 if i < remainder else 0) for i in range(num_experts)],
        dtype=torch.int32,
        device=device,
    )
    input_global_scale = torch.ones(num_experts, dtype=torch.float32, device=device)

    hidden = torch.randn(num_experts, m, k, dtype=torch.bfloat16, device=device) * 0.1

    # Clean reference
    hidden_clean = hidden.clone()
    for e in range(num_experts):
        real = masked_m[e].item()
        if real < m:
            hidden_clean[e, real:] = 0.0
    aq_clean, sf_clean = scaled_fp4_grouped_quantize(
        hidden_clean, masked_m, input_global_scale
    )

    # Dirty
    hidden_dirty = hidden.clone()
    for e in range(num_experts):
        real = masked_m[e].item()
        if real < m:
            hidden_dirty[e, real:] = float("nan")

    # CUDA graph
    graph = torch.cuda.CUDAGraph()
    hidden_input = hidden_dirty.clone()
    with torch.cuda.graph(graph):
        aq, sf = scaled_fp4_grouped_quantize(hidden_input, masked_m, input_global_scale)

    hidden_input.fill_(float("nan"))
    graph.replay()
    torch.accelerator.synchronize()
    hidden_input.copy_(hidden_dirty)
    graph.replay()
    torch.accelerator.synchronize()

    corrupted = False
    for e in range(num_experts):
        real = masked_m[e].item()
        if real == 0:
            continue
        if not torch.equal(aq_clean[e, :real], aq[e, :real]):
            corrupted = True
            break
        if not torch.equal(sf_clean[e, :real], sf[e, :real]):
            corrupted = True
            break

    assert not corrupted, (
        f"scaled_fp4_grouped_quantize: NaN in padding rows "
        f"corrupted real token output "
        f"(E={num_experts}, m={m}, real={num_real})"
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "num_experts,m,num_real",
    [
        (4, 8, 4),
        (8, 64, 8),
        (4, 16, 1),
        (8, 32, 16),
    ],
    ids=["E4_m8", "E8_m64_sparse", "E4_m16_1tok", "E8_m32"],
)
def test_cutedsl_wrapper_nan_padding(workspace_init, num_experts, m, num_real):
    """Test the full flashinfer_cutedsl_moe_masked wrapper with NaN padding.

    Verifies that the zero-fill fix in the wrapper prevents cross-row
    scale corruption from reaching the GEMM output.
    """
    from tests.kernels.moe.utils import make_test_weights
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe import (  # noqa: E501
        flashinfer_cutedsl_moe_masked,
    )

    device = "cuda"
    K, N = 1024, 512

    (_, w1_q, w1_bs, w1_gs), (_, w2_q, w2_bs, w2_gs) = make_test_weights(
        num_experts, N, K, in_dtype=torch.bfloat16, quant_dtype="nvfp4"
    )
    a1_gs = torch.ones(num_experts, dtype=torch.float32, device=device)
    a2_gs = torch.ones(num_experts, dtype=torch.float32, device=device)

    base = num_real // num_experts
    remainder = num_real % num_experts
    masked_m = torch.tensor(
        [base + (1 if i < remainder else 0) for i in range(num_experts)],
        dtype=torch.int32,
        device=device,
    )

    hidden = torch.randn(num_experts, m, K, dtype=torch.bfloat16, device=device) * 0.1
    for e in range(num_experts):
        real = masked_m[e].item()
        if real < m:
            hidden[e, real:] = float("nan")

    # Clean reference
    hidden_clean = hidden.clone()
    for e in range(num_experts):
        real = masked_m[e].item()
        if real < m:
            hidden_clean[e, real:] = 0.0

    call_kwargs = dict(
        input_global_scale=a1_gs,
        w1=w1_q,
        w1_blockscale=w1_bs,
        w1_alpha=(1 / w1_gs),
        w2=w2_q,
        a2_global_scale=a2_gs,
        w2_blockscale=w2_bs,
        w2_alpha=(1 / w2_gs),
        masked_m=masked_m,
    )

    workspace_c = torch.zeros(
        num_experts, m, 2 * N, dtype=torch.bfloat16, device=device
    )
    out_c = torch.zeros(num_experts, m, K, dtype=torch.bfloat16, device=device)
    flashinfer_cutedsl_moe_masked(
        hidden_states=hidden_clean, workspace=workspace_c, out=out_c, **call_kwargs
    )

    workspace_d = torch.zeros(
        num_experts, m, 2 * N, dtype=torch.bfloat16, device=device
    )
    out_d = torch.zeros(num_experts, m, K, dtype=torch.bfloat16, device=device)
    flashinfer_cutedsl_moe_masked(
        hidden_states=hidden.clone(), workspace=workspace_d, out=out_d, **call_kwargs
    )

    torch.accelerator.synchronize()

    corrupted = False
    for e in range(num_experts):
        real = masked_m[e].item()
        if real == 0:
            continue
        if not torch.equal(out_c[e, :real], out_d[e, :real]):
            corrupted = True
            break

    assert not corrupted, (
        f"flashinfer_cutedsl_moe_masked: NaN in padding rows "
        f"corrupted real token output despite zero-fill fix "
        f"(E={num_experts}, m={m}, real={num_real})"
    )
