# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only specialized Kimi-K2.5 NVFP4 implementation.

Aggressive op fusion following the DeepSeek V3.2 NVFP4 pattern:
  - A-projection as raw BF16 matmul (attention weights are unquantised)
  - Triton fused_norm_rope: Q/KV RMS-norm + K_pe RoPE + MLA cache write
  - Q B-projection as raw BF16 matmul
  - Monolithic custom op covering the full attention block
  - Triton fused SiLU+Mul+NVFP4-quant for shared experts
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4LinearMethod
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV2MLP,
    DeepseekV2MoE,
    yarn_get_mscale,
)
from vllm.model_executor.models.interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    is_quantized_kv_cache,
)

from .kernels import fused_norm_rope, q_rope, silu_and_mul_nvfp4_quant

_TARGET_MODEL_NAMES = {"nvidia/Kimi-K2.5-NVFP4"}


def _is_target_kimi_nvfp4(vllm_config: VllmConfig) -> bool:
    model_name = vllm_config.model_config.model
    if model_name in _TARGET_MODEL_NAMES:
        return True

    hf_config = vllm_config.model_config.hf_config
    if getattr(hf_config, "model_type", None) != "kimi_k25":
        return False

    text_config = getattr(hf_config, "text_config", None)
    if text_config is None:
        return False

    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config is None:
        quantization_config = getattr(text_config, "quantization_config", None)

    return (
        getattr(text_config, "model_type", None) == "deepseek_v3"
        and getattr(quantization_config, "get", lambda *_: None)("quant_algo")
        == "NVFP4"
    )


# ---------------------------------------------------------------------------
# Monolithic MLA attention custom op
# ---------------------------------------------------------------------------


def _kimi_mla_attn(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Monolithic MLA: norm + RoPE + cache + Q-proj + attention + V-proj."""
    layer = get_forward_context().no_compile_layers[layer_name]
    attn = layer.attn
    mla = attn.mla_attn

    fwd_ctx = get_forward_context()
    attn_metadata = fwd_ctx.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata.get(mla.layer_name)

    if attn_metadata is None:
        output.zero_()
        return output

    num_actual_toks = attn_metadata.num_actual_tokens
    if num_actual_toks == 0:
        output.zero_()
        return output

    # ---- FP8 scale calculation (one-time) ----
    if mla.calculate_kv_scales:
        _w_q = attn.q_a_layernorm.weight.to(torch.float32)
        _x_q = q_c.to(torch.float32)
        _rms_q = (_x_q * _x_q).mean(-1, keepdim=True).add_(layer.rms_norm_eps).rsqrt_()
        _q_tmp = (_x_q * _rms_q * _w_q).to(q_c.dtype)
        _w_kv = attn.kv_a_layernorm.weight.to(torch.float32)
        _x_kv = kv_c.to(torch.float32)
        _var = (_x_kv * _x_kv).mean(-1, keepdim=True)
        _rms_kv = _var.add_(layer.rms_norm_eps).rsqrt_()
        _kv_tmp = (_x_kv * _rms_kv * _w_kv).to(kv_c.dtype)
        mla.calc_kv_scales(_q_tmp, _kv_tmp, k_pe)
        del _q_tmp, _kv_tmp

    # ---- Step 2: fused norm + RoPE + MLA cache write ----
    slot_mapping = None
    mla_kv_cache = None
    kv_cache = mla.kv_cache
    if kv_cache.numel() > 0:
        slot_mapping = fwd_ctx.slot_mapping
        if isinstance(slot_mapping, dict):
            slot_mapping = slot_mapping.get(mla.layer_name)
        mla_kv_cache = kv_cache

    # Peek ahead to determine if prefill tokens are present.
    # This lets the Triton kernel skip the kv_c_normed writeback
    # for decode-only batches (saves global-memory bandwidth).
    has_prefill = (
        attn_metadata.num_prefills is not None and attn_metadata.num_prefills > 0
    )

    # Returns (q_c_normed, kv_c_normed | None, k_pe_roped).
    # kv_c_normed + k_pe_roped are also written to the MLA cache
    # inside the kernel.  kv_c_normed is only returned when there
    # are prefill tokens (forward_mha needs it for kv_b_proj).
    q_c, kv_c_normed, k_pe = fused_norm_rope(
        positions,
        q_c,
        attn.q_a_layernorm.weight,
        layer.rms_norm_eps,
        kv_c,
        attn.kv_a_layernorm.weight,
        layer.rms_norm_eps,
        k_pe,
        attn.rotary_emb.cos_sin_cache,
        slot_mapping=slot_mapping,
        mla_kv_cache=mla_kv_cache,
        mla_kv_cache_dtype=mla.kv_cache_dtype,
        mla_k_scale=mla._k_scale,
        has_prefill=has_prefill,
    )

    # ---- Step 3: Q B-projection (BF16 matmul) ----
    q = torch.mm(q_c, layer._q_b_proj_w.T)

    # ---- Lazy-init DCP ----
    if mla.impl.dcp_world_size == -1:
        from vllm.distributed.parallel_state import get_dcp_group

        mla.impl.dcp_world_size = get_dcp_group().world_size

    # ---- Trim to actual tokens ----
    output_padded = output
    output = output[:num_actual_toks]
    positions = positions[:num_actual_toks]
    q = q[:num_actual_toks]
    k_pe = k_pe[:num_actual_toks]

    fp8_attn = is_quantized_kv_cache(mla.kv_cache_dtype)
    if fp8_attn and mla.kv_cache_dtype != "fp8_ds_mla":
        kv_cache = kv_cache.view(current_platform.fp8_dtype())

    num_mqa_tokens = attn_metadata.num_decode_tokens
    num_mha_tokens = q.size(0) - num_mqa_tokens

    num_heads = mla.num_heads
    qk_head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim

    # ---- Q RoPE (Q-only Triton kernel, no wasted K RoPE) ----
    q_rope(
        positions,
        q,
        attn.rotary_emb.cos_sin_cache,
        num_heads,
        mla.qk_nope_head_dim,
        mla.qk_rope_head_dim,
    )
    q = q.view(-1, num_heads, qk_head_dim)

    # ---- Prefill (MHA) ----
    if num_mha_tokens > 0:
        assert kv_c_normed is not None
        mla.impl.forward_mha(
            q[num_mqa_tokens:],
            kv_c_normed[num_mqa_tokens:num_actual_toks],
            k_pe[num_mqa_tokens:].unsqueeze(1),
            kv_cache,
            attn_metadata,
            mla._k_scale,
            output=output[num_mqa_tokens:],
        )

    # ---- Decode (MQA) ----
    if num_mqa_tokens > 0:
        mqa_q = q[:num_mqa_tokens]
        mqa_q_nope, mqa_q_pe = mqa_q.split(
            [mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1
        )

        mqa_q_nope = mqa_q_nope.transpose(0, 1)
        N, B, P = mqa_q_nope.shape
        _, _, L = mla.W_UK_T.shape

        q_pad = mla.q_pad_num_heads
        if q_pad is not None:
            ql_nope = mqa_q_nope.new_empty((q_pad, B, L))
            ql_nope.resize_((N, B, L))
        else:
            ql_nope = mqa_q_nope.new_empty((N, B, L))

        torch.bmm(mqa_q_nope, mla.W_UK_T, out=ql_nope)
        ql_nope = ql_nope.transpose(0, 1)

        if q_pad is not None:
            B_pe, N_pe, L_pe = mqa_q_pe.shape
            pe_padded = mqa_q_pe.new_empty((B_pe, q_pad, L_pe))
            pe_padded.resize_((B_pe, N_pe, L_pe))
            pe_padded.copy_(mqa_q_pe)
            mqa_q_pe = pe_padded

        if fp8_attn and mla.impl.supports_quant_query_input:
            mqa_q_final = mla._decode_concat_quant_fp8_op(
                ql_nope, mqa_q_pe, mla._q_scale
            )
        else:
            mqa_q_final = (ql_nope, mqa_q_pe)

        attn_out, _ = mla.impl.forward_mqa(mqa_q_final, kv_cache, attn_metadata, mla)

        # W_UV up-projection
        x = attn_out.view(-1, num_heads, mla.kv_lora_rank).transpose(0, 1)
        out = (
            output[:num_mqa_tokens].view(-1, num_heads, mla.v_head_dim).transpose(0, 1)
        )
        torch.bmm(x, mla.W_UV, out=out)

    return output_padded


def _kimi_mla_attn_fake(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del positions, q_c, kv_c, k_pe, layer_name
    return output


direct_register_custom_op(
    op_name="monolithic_attn",
    op_func=_kimi_mla_attn,
    fake_impl=_kimi_mla_attn_fake,
    mutates_args=["output"],
    dispatch_key=current_platform.dispatch_key,
)


# ---------------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------------


class KimiK25Nvfp4MLAAttention(nn.Module):
    """MLA attention for Kimi-K2.5 NVFP4.

    MLAAttention is kept only for KV cache registration and backend init.
    The actual forward is fully inlined in the monolithic custom op.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        cache_config,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5

        # Q path
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )

        # KV path
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # Output projection (TP sync point)
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # MLAAttention stub for KV cache + backend init
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=False,
            indexer=None,
        )


class KimiK25Nvfp4DecoderLayer(nn.Module):
    """Single decoder layer with aggressive op fusion."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()

        # Register in static_forward_context for the custom op.
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.layer_name = prefix
        self.layer_idx = layer_idx
        self.rms_norm_eps = config.rms_norm_eps
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Fused A-projection lives inside self_attn namespace for
        # weight-loading compatibility with checkpoint paths.
        self.self_attn = nn.Module()
        self.self_attn.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.fused_qkv_a_proj",
        )

        # MLA attention (weights + KV cache registration)
        self.attn = KimiK25Nvfp4MLAAttention(
            vllm_config=vllm_config,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # MoE or Dense MLP
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        if self.is_moe:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Step 1: fused A-projection (BF16 matmul)
        step1 = torch.mm(hidden_states, self._fused_a_proj_w.T)
        q_c, kv_c, k_pe = step1.split(self._step1_splits, dim=-1)

        # Steps 2-4: monolithic attention
        mla = self.attn.mla_attn
        attn_out = torch.empty(
            (hidden_states.shape[0], mla.num_heads * mla.v_head_dim),
            dtype=mla.W_UV.dtype,
            device=hidden_states.device,
        )
        attn_out = torch.ops.vllm.monolithic_attn(
            positions,
            q_c,
            kv_c,
            k_pe,
            attn_out,
            self.layer_name,
        )

        hidden_states, _ = self.attn.o_proj(attn_out)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def fuse_weights(self) -> None:
        """Fuse BF16 weights into raw parameters for torch.mm paths."""
        a_proj_w = self.self_attn.fused_qkv_a_proj.weight.data
        assert a_proj_w.dtype in (torch.bfloat16, torch.float16, torch.float32), (
            f"Expected BF16 A-proj weight, got {a_proj_w.dtype}"
        )
        self._fused_a_proj_w = nn.Parameter(a_proj_w, requires_grad=False)
        self._step1_splits = [
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        ]

        q_b_w = self.attn.q_b_proj.weight.data
        assert q_b_w.dtype in (torch.bfloat16, torch.float16, torch.float32), (
            f"Expected BF16 q_b_proj weight, got {q_b_w.dtype}"
        )
        self._q_b_proj_w = nn.Parameter(q_b_w, requires_grad=False)

    def fuse_shared_expert_act_quant(self) -> None:
        """Fuse SiLU-and-Mul + NVFP4 quantize in the shared expert MLP."""
        if not self.is_moe:
            return

        shared_experts = self.mlp.shared_experts
        if shared_experts is None:
            return
        if not isinstance(
            shared_experts.down_proj.quant_method, ModelOptNvFp4LinearMethod
        ):
            return

        dp = shared_experts.down_proj

        def _fused_forward(x: torch.Tensor) -> torch.Tensor:
            gate_up, _ = shared_experts.gate_up_proj(x)
            x_fp4, x_bs = silu_and_mul_nvfp4_quant(gate_up, dp.input_global_scale_inv)
            return flashinfer_scaled_fp4_mm(
                x_fp4,
                dp.weight,
                x_bs,
                dp.weight_scale,
                dp.alpha,
                gate_up.dtype,
                backend="cutlass",
            )

        shared_experts.forward = _fused_forward  # type: ignore[method-assign]


class KimiK25Nvfp4TextModel(nn.Module):
    """Text-only model body for Kimi-K2.5 NVFP4."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                KimiK25Nvfp4DecoderLayer(
                    vllm_config=vllm_config,
                    config=config,
                    layer_idx=i,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"],
            config.hidden_size,
        )
        self.aux_hidden_state_layers = tuple[int, ...]()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError(
                "Kimi-K2.5 NVFP4 specialized text model does not support PP."
            )
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            hidden_states = self.embed_input_ids(input_ids)

        residual = None
        aux_hidden_states = []
        for idx, layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


def _remap_weight_name(name: str) -> str:
    """Remap checkpoint names to match the restructured module tree."""
    replacements = [
        ("self_attn.q_a_layernorm.", "attn.q_a_layernorm."),
        ("self_attn.kv_a_layernorm.", "attn.kv_a_layernorm."),
        ("self_attn.q_b_proj.", "attn.q_b_proj."),
        ("self_attn.kv_b_proj.", "attn.kv_b_proj."),
        ("self_attn.o_proj.", "attn.o_proj."),
    ]
    for old, new in replacements:
        if old in name:
            return name.replace(old, new)
    return name


class KimiK25Nvfp4TextForCausalLM(DeepseekV2ForCausalLM):
    model_cls = KimiK25Nvfp4TextModel

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, KimiK25Nvfp4DecoderLayer) and isinstance(
                layer.mlp, DeepseekV2MoE
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _remap(weights_iter):
            for name, tensor in weights_iter:
                yield _remap_weight_name(name), tensor

        loaded = super().load_weights(_remap(weights))
        for layer in self.model.layers:
            layer.fuse_weights()
            layer.fuse_shared_expert_act_quant()
        return loaded


class KimiK25ForConditionalGeneration(
    nn.Module,
    SupportsPP,
    SupportsQuant,
    SupportsEagle,
    SupportsEagle3,
):
    """Text-only Kimi-K2.5 wrapper for the NVFP4 checkpoint."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
        }
    )

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        text_vllm_config = vllm_config.with_hf_config(config.text_config)

        if not _is_target_kimi_nvfp4(vllm_config):
            raise ValueError(
                "The Kimi-K2.5 specialized NVFP4 model only supports "
                "`nvidia/Kimi-K2.5-NVFP4`."
            )

        self.language_model = KimiK25Nvfp4TextForCausalLM(
            vllm_config=text_vllm_config,
            prefix=f"{prefix}.language_model" if prefix else "language_model",
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        del kwargs
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["vision_tower.", "mm_projector."],
            ignore_unexpected_prefixes=["vision_tower.", "mm_projector."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
