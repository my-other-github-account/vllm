# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided import (  # noqa: E501
    _group_rank_batched_inputs_by_local_expert,
    _reduce_local_expert_outputs_to_rank_batched_payload,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)


class _StandardNvFp4Kernel:
    @staticmethod
    def is_supported_config(
        _cls,
        moe_config,
        weight_key,
        activation_key,
        activation_format,
    ):
        if activation_format == mk.FusedMoEActivationFormat.Standard:
            return True, None
        return False, f"{activation_format.value} activation format"


class _BatchedNvFp4Kernel:
    @staticmethod
    def is_supported_config(
        _cls,
        moe_config,
        weight_key,
        activation_key,
        activation_format,
    ):
        if activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
            return True, None
        return False, f"{activation_format.value} activation format"


class _UnsupportedNvFp4Kernel:
    @staticmethod
    def is_supported_config(
        _cls,
        moe_config,
        weight_key,
        activation_key,
        activation_format,
    ):
        return False, "unsupported"


def _make_nvfp4_config(all2all_backend: str):
    moe_config = make_dummy_moe_config(num_experts=8, hidden_dim=16)
    moe_config.moe_backend = "flashinfer_cutedsl"
    moe_config.moe_parallel_config.dp_size = 2
    moe_config.moe_parallel_config.use_ep = True
    moe_config.moe_parallel_config.all2all_backend = all2all_backend
    return moe_config


def _fake_backend_to_kernel_cls(backend: NvFp4MoeBackend):
    if backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL:
        return [_StandardNvFp4Kernel]
    if backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED:
        return [_BatchedNvFp4Kernel]
    return [_UnsupportedNvFp4Kernel]


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.nvfp4.backend_to_kernel_cls",
    side_effect=_fake_backend_to_kernel_cls,
)
def test_select_nvfp4_backend_uses_standard_cutedsl_for_standard_all2all(
    mock_backend_to_kernel_cls,
):
    moe_config = _make_nvfp4_config("allgather_reducescatter")

    backend, experts_cls = select_nvfp4_moe_backend(
        moe_config,
        kNvfp4Static,
        kNvfp4Dynamic,
    )

    assert backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL
    assert experts_cls is _StandardNvFp4Kernel


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.nvfp4.backend_to_kernel_cls",
    side_effect=_fake_backend_to_kernel_cls,
)
def test_select_nvfp4_backend_promotes_cutedsl_for_batched_all2all(
    mock_backend_to_kernel_cls,
):
    moe_config = _make_nvfp4_config("flashinfer_nvlink_one_sided")

    backend, experts_cls = select_nvfp4_moe_backend(
        moe_config,
        kNvfp4Static,
        kNvfp4Dynamic,
    )

    assert moe_config.moe_parallel_config.use_batched_activation_format
    assert backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED
    assert experts_cls is _BatchedNvFp4Kernel


def test_flashinfer_one_sided_rank_batched_regroup_and_reduce():
    hidden_states = torch.tensor(
        [
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [0.0, 0.0]],
            [[40.0, 41.0], [50.0, 51.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    hidden_scales = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [0.0]],
            [[4.0], [5.0], [0.0], [0.0]],
        ]
    )
    topk_ids = torch.tensor(
        [
            [[2, 0], [3, 2], [1, 0], [0, 0]],
            [[3, 1], [2, 3], [0, 0], [0, 0]],
        ],
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [
            [[0.70, 0.30], [0.40, 0.60], [0.50, 0.50], [0.0, 0.0]],
            [[0.80, 0.20], [0.25, 0.75], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    (
        batched_hidden_states,
        batched_hidden_scales,
        expert_tokens_meta,
        dispatch_metadata,
    ) = _group_rank_batched_inputs_by_local_expert(
        hidden_states,
        hidden_scales,
        topk_ids,
        topk_weights,
        num_local_experts=2,
        first_local_expert=2,
        num_dispatchers=2,
        runtime_max_tokens_per_rank=4,
        source_num_tokens=[3, 2],
    )

    assert expert_tokens_meta.expert_num_tokens.tolist() == [3, 3]
    torch.testing.assert_close(
        batched_hidden_states[0, :3],
        torch.tensor([[10.0, 11.0], [20.0, 21.0], [50.0, 51.0]]),
    )
    torch.testing.assert_close(
        batched_hidden_states[1, :3],
        torch.tensor([[20.0, 21.0], [40.0, 41.0], [50.0, 51.0]]),
    )
    assert batched_hidden_scales is not None
    torch.testing.assert_close(
        batched_hidden_scales[0, :3],
        torch.tensor([[1.0], [2.0], [5.0]]),
    )
    torch.testing.assert_close(
        batched_hidden_scales[1, :3],
        torch.tensor([[2.0], [4.0], [5.0]]),
    )

    fused_expert_output = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.0, 0.0]],
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [0.0, 0.0]],
        ]
    )
    combine_payload = _reduce_local_expert_outputs_to_rank_batched_payload(
        fused_expert_output,
        dispatch_metadata,
        num_dispatchers=2,
        runtime_max_tokens_per_rank=4,
        apply_router_weight_on_input=False,
    )

    expected = torch.zeros((2, 4, 2))
    expected[0, 0] = torch.tensor([0.70, 0.70])
    expected[0, 1] = torch.tensor([5.20, 5.20])
    expected[1, 0] = torch.tensor([16.0, 16.0])
    expected[1, 1] = torch.tensor([23.25, 23.25])
    torch.testing.assert_close(combine_payload, expected)
