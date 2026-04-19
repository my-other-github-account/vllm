# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def get_local_sizes():
    return get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()


@dataclass
class _OneSidedDispatchMetadata:
    combine_indices: list[torch.Tensor]
    combine_weights: list[torch.Tensor]


def _normalize_source_num_tokens(
    source_num_tokens: Sequence[int] | None,
    num_dispatchers: int,
    runtime_max_tokens_per_rank: int,
) -> list[int]:
    if source_num_tokens is None:
        return [runtime_max_tokens_per_rank] * num_dispatchers

    normalized = [int(x) for x in source_num_tokens[:num_dispatchers]]
    if len(normalized) < num_dispatchers:
        normalized.extend(
            [runtime_max_tokens_per_rank] * (num_dispatchers - len(normalized))
        )
    return [min(max(x, 0), runtime_max_tokens_per_rank) for x in normalized]


def _group_rank_batched_inputs_by_local_expert(
    hidden_states: torch.Tensor,
    hidden_scales: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
    first_local_expert: int,
    num_dispatchers: int,
    runtime_max_tokens_per_rank: int,
    source_num_tokens: Sequence[int] | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    mk.ExpertTokensMetadata,
    _OneSidedDispatchMetadata,
]:
    hidden_dim = hidden_states.shape[-1]
    batched_hidden_states = hidden_states.new_empty(
        (num_local_experts, runtime_max_tokens_per_rank, hidden_dim)
    )
    batched_hidden_scales = (
        None
        if hidden_scales is None
        else hidden_scales.new_empty(
            (num_local_experts, runtime_max_tokens_per_rank, hidden_scales.shape[-1])
        )
    )
    tokens_per_expert = torch.zeros(
        num_local_experts, dtype=torch.int32, device=hidden_states.device
    )

    combine_indices: list[torch.Tensor] = []
    combine_weights: list[torch.Tensor] = []
    valid_source_num_tokens = _normalize_source_num_tokens(
        source_num_tokens, num_dispatchers, runtime_max_tokens_per_rank
    )

    for local_expert in range(num_local_experts):
        global_expert = first_local_expert + local_expert
        expert_indices: list[torch.Tensor] = []
        expert_weights: list[torch.Tensor] = []
        cursor = 0

        for dispatcher, num_tokens in enumerate(valid_source_num_tokens):
            if num_tokens == 0:
                continue

            token_idx, topk_slot_idx = torch.where(
                topk_ids[dispatcher, :num_tokens] == global_expert
            )
            rows = token_idx.numel()
            if rows == 0:
                continue

            batched_hidden_states[local_expert, cursor : cursor + rows] = hidden_states[
                dispatcher, token_idx
            ]
            if batched_hidden_scales is not None:
                assert hidden_scales is not None
                batched_hidden_scales[local_expert, cursor : cursor + rows] = (
                    hidden_scales[dispatcher, token_idx]
                )

            expert_indices.append(
                dispatcher * runtime_max_tokens_per_rank + token_idx.to(torch.int64)
            )
            expert_weights.append(
                topk_weights[dispatcher, token_idx, topk_slot_idx].contiguous()
            )
            cursor += rows

        tokens_per_expert[local_expert] = cursor
        combine_indices.append(
            torch.cat(expert_indices)
            if expert_indices
            else torch.empty(0, dtype=torch.int64, device=hidden_states.device)
        )
        combine_weights.append(
            torch.cat(expert_weights)
            if expert_weights
            else torch.empty(0, dtype=topk_weights.dtype, device=topk_weights.device)
        )

    expert_tokens_meta = mk.ExpertTokensMetadata(
        expert_num_tokens=tokens_per_expert, expert_num_tokens_cpu=None
    )
    dispatch_metadata = _OneSidedDispatchMetadata(
        combine_indices=combine_indices,
        combine_weights=combine_weights,
    )
    return (
        batched_hidden_states,
        batched_hidden_scales,
        expert_tokens_meta,
        dispatch_metadata,
    )


def _reduce_local_expert_outputs_to_rank_batched_payload(
    fused_expert_output: torch.Tensor,
    dispatch_metadata: _OneSidedDispatchMetadata,
    *,
    num_dispatchers: int,
    runtime_max_tokens_per_rank: int,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    hidden_dim = fused_expert_output.shape[-1]
    combine_payload = fused_expert_output.new_zeros(
        (num_dispatchers, runtime_max_tokens_per_rank, hidden_dim)
    )
    flat_payload = combine_payload.view(-1, hidden_dim)

    for local_expert, linear_indices in enumerate(dispatch_metadata.combine_indices):
        rows = linear_indices.numel()
        if rows == 0:
            continue

        expert_output = fused_expert_output[local_expert, :rows]
        if not apply_router_weight_on_input:
            expert_output = expert_output * dispatch_metadata.combine_weights[
                local_expert
            ].to(expert_output.dtype).unsqueeze(-1)
        flat_payload.index_add_(0, linear_indices, expert_output)

    return combine_payload


class FlashInferNVLinkOneSidedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """FlashInfer implementation using the Moe AlltoAll kernel."""

    def __init__(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_dispatchers_ = num_dispatchers
        self.all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert self.num_experts % self.num_dispatchers_ == 0, (
            "flashinfer_nvlink_one_sided requires evenly sharded local experts."
        )
        self.num_local_experts = self.num_experts // self.num_dispatchers_
        self.first_local_expert = self.all2all_manager.rank * self.num_local_experts
        self.dispatch_metadata: _OneSidedDispatchMetadata | None = None

        self.all2all_manager.initialize(
            max_num_tokens=self.max_num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
        )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return False

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True."
            )
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

        global_num_tokens_cpu = get_local_sizes()
        self.runtime_max_tokens_per_rank = (
            max(global_num_tokens_cpu)
            if global_num_tokens_cpu is not None
            else a1.shape[0]
        )

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            quant_config.a1_gscale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
            is_fp4_scale_swizzled=False,  # delay swizzle to after comm
        )

        payloads = []
        payloads.append(a1q)
        if a1q_scale is not None:
            payloads.append(a1q_scale)
        payloads.append(topk_ids)
        payloads.append(topk_weights)

        recv_payloads = self.all2all_manager.moe_alltoall.dispatch(
            token_selected_experts=topk_ids,
            input_payloads=payloads,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
        )
        if a1q_scale is not None:
            a1q_recv, a1q_scale_recv, topk_ids_recv, topk_weights_recv = recv_payloads
            # Swizzle after dispatch when the selected MoE kernel expects it.
            if (
                quant_config.quant_dtype == "nvfp4"
                and quant_config.is_nvfp4_scale_swizzled
            ):
                a1q_scale_recv = a1q_scale_recv.view(-1, a1q_scale_recv.shape[-1])
                a1q_scale_recv = a1q_scale_recv.view(torch.uint8)
                a1q_scale_recv = nvfp4_block_scale_interleave(a1q_scale_recv)
            a1q_scale_recv = a1q_scale_recv.view(
                self.num_dispatchers_,
                self.runtime_max_tokens_per_rank,
                self.hidden_size // 16,
            )
        else:
            a1q_recv, topk_ids_recv, topk_weights_recv = recv_payloads
            a1q_scale_recv = None
        (
            a1q_recv,
            a1q_scale_recv,
            expert_tokens_meta,
            self.dispatch_metadata,
        ) = _group_rank_batched_inputs_by_local_expert(
            a1q_recv,
            a1q_scale_recv,
            topk_ids_recv,
            topk_weights_recv,
            num_local_experts=self.num_local_experts,
            first_local_expert=self.first_local_expert,
            num_dispatchers=self.num_dispatchers_,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
            source_num_tokens=global_num_tokens_cpu,
        )

        return a1q_recv, a1q_scale_recv, expert_tokens_meta, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert self.all2all_manager.moe_alltoall is not None
        assert self.dispatch_metadata is not None, (
            "flashinfer_nvlink_one_sided finalize called before prepare"
        )
        combine_payload = _reduce_local_expert_outputs_to_rank_batched_payload(
            fused_expert_output,
            self.dispatch_metadata,
            num_dispatchers=self.num_dispatchers_,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        combined_output = self.all2all_manager.moe_alltoall.combine(
            payload=combine_payload,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
        )
        self.dispatch_metadata = None
        output.copy_(combined_output)
