# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_get_previous_event,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)


class DeepEPV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using DeepEP v2 ElasticBuffer (unified API).

    Uses do_expand=True so dispatch produces per-expert-contiguous layout:
    each (token, expert) pair gets its own row, with expert regions contiguous.
    Combine handles weighted reduction internally.
    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int, dtype: torch.dtype) -> int:
        hidden_size_bytes = hidden_size * dtype.itemsize
        xfer_atom_size = 512  # 32 * 16 (size(int4))
        if hidden_size_bytes % xfer_atom_size == 0:
            return hidden_size

        hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
        return hidden_size_bytes // dtype.itemsize

    def __init__(
        self,
        buffer: deep_ep.ElasticBuffer,
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
        num_experts: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.use_fp8_dispatch = use_fp8_dispatch
        self.async_prepare = True

        # DBO microbatching: one handle slot per micro-batch.
        self.handles: list[deep_ep.EPHandle | None] = [None, None]

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
        has_scales = token_scales is not None

        previous_event = dbo_get_previous_event(self.buffer.capture)

        dbo_yield_and_switch_from_compute_to_comm()

        token_data = tokens
        if has_scales:
            token_data = (tokens, token_scales)

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
            event,
        ) = self.buffer.dispatch(
            x=token_data,
            topk_idx=rank_topk_ids,
            topk_weights=rank_topk_weights,
            num_experts=num_experts,
            do_expand=True,
            previous_event=previous_event,
            async_with_compute_stream=(self.async_prepare and not dbo_enabled()),
        )

        a2a_idx = dbo_current_ubatch_id()
        self.handles[a2a_idx] = handle

        dbo_switch_to_compute_sync()

        return lambda: self._receiver(
            event,
            has_scales,
            recv_x,
            recv_topk_idx,
            num_experts,
            handle.num_recv_tokens_per_expert_list,
            recv_topk_weights,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
        )

    def _receiver(
        self,
        event: deep_ep.EventOverlap,
        has_scales: bool,
        recv_x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        recv_topk_idx: torch.Tensor | None,
        num_experts: int,
        recv_expert_num_tokens: list[int],
        recv_topk_weights: torch.Tensor | None,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        if event.event is not None:
            event.current_stream_wait()

        if has_scales:
            expert_x, expert_x_scale = recv_x
        else:
            expert_x, expert_x_scale = recv_x, None

        # With do_expand=True, topk_ids refer to local experts.
        # Remap -1 entries and offset to global expert space, same as v1 HT.
        assert recv_topk_idx is not None
        recv_topk_idx = torch.where(
            recv_topk_idx == -1,
            num_experts - 1 if self.rank_expert_offset == 0 else 0,
            recv_topk_idx + self.rank_expert_offset,
        )

        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            recv_expert_num_tokens, device=expert_x.device
        )

        if not quant_config.is_block_quantized and not defer_input_quant:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape,
                    is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,
                )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            recv_topk_idx,
            recv_topk_weights,
        )

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if quant_config.is_block_quantized and not defer_input_quant:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = None
            a1_post_scale = (
                quant_config.a1_gscale
                if quant_config.quant_dtype == "nvfp4"
                else quant_config.a1_scale
            )

        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            rank_topk_ids=topk_ids,
            rank_topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
        )

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
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        a2a_idx = dbo_current_ubatch_id()
        handle = self.handles[a2a_idx]
        assert handle is not None

        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        previous_event = dbo_get_previous_event(self.buffer.capture)
        dbo_yield_and_switch_from_compute_to_comm()
        assert fused_expert_output.dtype == torch.bfloat16, (
            f"Expected fused_expert_output bfloat16, got {fused_expert_output.dtype}"
        )

        combined_x, _, event = self.buffer.combine(
            x=fused_expert_output,
            handle=handle,
            topk_weights=None,
            previous_event=previous_event,
            async_with_compute_stream=do_async and not dbo_enabled(),
        )

        dbo_switch_to_compute()

        if do_async:

            def _receiver():
                if event.event is not None:
                    event.current_stream_wait()
                dbo_switch_to_comm()
                output.copy_(combined_x, non_blocking=True)
                dbo_yield_and_switch_from_comm_to_compute()

            return _receiver
        else:
            assert not dbo_enabled()
            output.copy_(combined_x, non_blocking=True)
            return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        receiver = self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            True,
        )
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
