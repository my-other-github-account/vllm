# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test DeepEP v2 (ElasticBuffer) dispatch-combine logic.
Compares against a pure-PyTorch reference MoE implementation.
"""

import dataclasses

import pytest
import torch.distributed
from torch.distributed import ProcessGroup

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.utils.import_utils import has_deep_ep_v2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch

if has_deep_ep_v2():
    from .parallel_utils import DeepEPV2Args, make_deepep_v2_a2a

requires_deep_ep_v2 = pytest.mark.skipif(
    not has_deep_ep_v2(),
    reason="Requires DeepEP v2 (ElasticBuffer)",
)


def make_weights(
    e, n, k, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype in [torch.float16, torch.bfloat16]:
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        return w1, w2, None, None

    assert dtype == torch.float8_e4m3fn
    w1 = torch.empty((e, 2 * n, k), device="cuda", dtype=torch.float16)
    w2 = torch.empty((e, k, n), device="cuda", dtype=torch.float16)

    n_b_scales = 2 * n
    k_b_scales = k
    w1_q = torch.empty_like(w1, dtype=dtype)
    w2_q = torch.empty_like(w2, dtype=dtype)
    w1_scale = torch.empty((e, n_b_scales, 1), device="cuda", dtype=torch.float32)
    w2_scale = torch.empty((e, k_b_scales, 1), device="cuda", dtype=torch.float32)
    for expert in range(e):
        w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
            w1[expert], use_per_token_if_dynamic=True
        )
        w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
            w2[expert], use_per_token_if_dynamic=True
        )
    return w1_q, w2_q, w1_scale, w2_scale


@dataclasses.dataclass
class TestConfig:
    dtype: torch.dtype
    topk: int
    m: int
    k: int
    n: int
    num_experts: int


@dataclasses.dataclass
class TestTensors:
    rank_tokens: torch.Tensor
    rank_token_scales: torch.Tensor | None
    topk: torch.Tensor
    topk_weights: torch.Tensor
    config: TestConfig

    @staticmethod
    def make(config: TestConfig) -> "TestTensors":
        assert config.dtype in [torch.bfloat16, torch.float8_e4m3fn]
        token_dtype = (
            torch.bfloat16 if config.dtype == torch.float8_e4m3fn else config.dtype
        )
        rank_tokens = (
            torch.randn((config.m, config.k), device="cuda", dtype=token_dtype) / 10
        )

        topk = torch.randint(
            low=0, high=config.num_experts, size=(config.m, config.topk), device="cuda"
        ).to(dtype=torch.int64)
        topk_weights = torch.randn(topk.shape, dtype=torch.float32, device="cuda")
        return TestTensors(
            rank_tokens=rank_tokens,
            rank_token_scales=None,
            topk=topk,
            topk_weights=topk_weights,
            config=config,
        )


def make_modular_kernel(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    dp_size: int,
    hidden_size: int,
    num_experts: int,
    num_local_experts: int,
    topk: int,
    q_dtype: torch.dtype | None,
    use_fp8_dispatch: bool,
    quant_config: FusedMoEQuantConfig,
) -> FusedMoEKernel:
    assert not use_fp8_dispatch, "FP8 dispatch for v2 not yet validated"

    v2_args = DeepEPV2Args(
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        num_topk=topk,
        hidden_size=hidden_size,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    a2a = make_deepep_v2_a2a(
        pg=pg,
        pgi=pgi,
        dp_size=dp_size,
        v2_args=v2_args,
    )

    moe_config = make_dummy_moe_config()

    fused_experts = TritonExperts(
        moe_config=moe_config,
        quant_config=quant_config,
    )

    mk = FusedMoEKernel(
        prepare_finalize=a2a,
        fused_experts=fused_experts,
        inplace=False,
    )
    return mk


def deepep_v2_moe_impl(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    dp_size: int,
    test_tensors: TestTensors,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    num_experts: int,
    topk: int,
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
) -> torch.Tensor:
    num_local_experts = w1.size(0)

    def build_expert_map():
        expert_map = torch.full((num_experts,), fill_value=-1, dtype=torch.int32)
        s = pgi.rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)))
        device = torch.accelerator.current_device_index()
        return expert_map.to(device=device, dtype=torch.int32)

    is_quantized = w1.dtype == torch.float8_e4m3fn
    q_dtype = torch.float8_e4m3fn if is_quantized else None

    quant_config = FusedMoEQuantConfig.make(
        q_dtype,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        per_act_token_quant=per_act_token_quant,
        a1_scale=test_tensors.rank_token_scales,
    )

    hidden_size = test_tensors.rank_tokens.size(1)

    mk: FusedMoEKernel = make_modular_kernel(
        pg,
        pgi,
        dp_size,
        hidden_size,
        num_experts,
        num_local_experts,
        topk,
        q_dtype,
        use_fp8_dispatch,
        quant_config,
    )

    out = mk.apply(
        hidden_states=test_tensors.rank_tokens,
        w1=w1,
        w2=w2,
        topk_weights=test_tensors.topk_weights,
        topk_ids=test_tensors.topk,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=build_expert_map(),
        apply_router_weight_on_input=False,
    )

    return out


def torch_moe_impl(
    test_tensors: TestTensors,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    per_act_token_quant: bool,
):
    a, topk_ids, topk_weights = (
        test_tensors.rank_tokens,
        test_tensors.topk,
        test_tensors.topk_weights,
    )

    is_quantized = w1.dtype == torch.float8_e4m3fn
    a_dtype = a.dtype
    if is_quantized:
        w1 = w1.to(dtype=torch.float32) * w1_scale
        w2 = w2.to(dtype=torch.float32) * w2_scale
        a = a.to(dtype=torch.float32)

    m, _ = a.shape
    topk = topk_ids.size(1)
    out = torch.zeros_like(a)

    for i in range(m):
        a_i = a[i]
        o_i = out[i]
        for j in range(topk):
            e = topk_ids[i][j]
            e_w = topk_weights[i][j]
            w1_e = w1[e]
            w2_e = w2[e]
            o_i += (
                SiluAndMul()(a_i @ w1_e.transpose(0, 1)) @ w2_e.transpose(0, 1)
            ) * e_w

    if is_quantized:
        out = out.to(dtype=a_dtype)

    return out


def _deep_ep_v2_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
):
    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    is_quantized = w1.dtype == torch.float8_e4m3fn
    device_idx = torch.accelerator.current_device_index()
    w1 = w1.to(device=device_idx)
    w2 = w2.to(device=device_idx)
    if is_quantized:
        assert w1_scale is not None and w2_scale is not None
        w1_scale = w1_scale.to(device=device_idx)
        w2_scale = w2_scale.to(device=device_idx)

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(config)

    with set_current_vllm_config(VllmConfig()):
        # Reference
        torch_combined = torch_moe_impl(
            test_tensors,
            w1,
            w2,
            w1_scale,
            w2_scale,
            per_act_token_quant,
        )

        # Splice experts for this rank
        num_local_experts = config.num_experts // pgi.world_size
        e_start = num_local_experts * pgi.rank
        e_end = e_start + num_local_experts
        w1_ep = w1[e_start:e_end]
        w2_ep = w2[e_start:e_end]

        w1_scale_ep, w2_scale_ep = None, None
        if is_quantized:
            w1_scale_ep = w1_scale[e_start:e_end]  # type: ignore
            w2_scale_ep = w2_scale[e_start:e_end]  # type: ignore

        deepep_combined = deepep_v2_moe_impl(
            pg,
            pgi,
            dp_size,
            test_tensors,
            w1_ep,
            w2_ep,
            w1_scale_ep,
            w2_scale_ep,
            config.num_experts,
            config.topk,
            use_fp8_dispatch,
            per_act_token_quant,
        )

    torch.testing.assert_close(
        torch_combined,
        deepep_combined,
        atol=6e-2,
        rtol=6e-2,
    )


MNKs = [
    (1, 128, 128),
    (2, 128, 512),
    (3, 1024, 2048),
    (32, 128, 1024),
    (45, 512, 2048),
    (64, 1024, 1024),
    (222, 1024, 2048),
]

DTYPES = [torch.bfloat16, torch.float8_e4m3fn]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n,k", MNKs)
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@pytest.mark.parametrize("per_act_token_quant", [False, True])
@multi_gpu_test(num_gpus=2)
@requires_deep_ep_v2
def test_deep_ep_v2_moe(
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    world_dp_size: tuple[int, int],
    per_act_token_quant: bool,
    workspace_init,
):
    use_fp8_dispatch = False

    set_random_seed(7)
    world_size, dp_size = world_dp_size
    config = TestConfig(dtype=dtype, topk=topk, m=m, k=k, n=n, num_experts=num_experts)

    w1, w2, w1_scale, w2_scale = make_weights(num_experts, n, k, dtype)

    parallel_launch(
        world_size,
        _deep_ep_v2_moe,
        dp_size,
        config,
        w1,
        w2,
        w1_scale,
        w2_scale,
        use_fp8_dispatch,
        per_act_token_quant,
    )
