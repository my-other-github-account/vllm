# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch


def _normalize_lora_config_keys(
    config: dict[str, int | None],
) -> dict[str, int | None]:
    """Normalize Triton config dict keys to uppercase BLOCK_SIZE_* format."""
    out: dict[str, int | None] = {}
    for key, val in config.items():
        if key.islower():
            if key.startswith("block_"):
                nk = "BLOCK_SIZE_" + key.split("_")[-1].upper()
            else:
                nk = key.upper()
        else:
            nk = key
        out[nk] = val
    return out


@dataclass
class MoELoRAContext:
    """
    Carries all LoRA state for one MoE forward pass.

    Built by FusedMoEWithLoRA.forward() and propagated explicitly through the
    modular kernel path (FusedMoEKernel -> FusedMoEExpertsModular.apply) so
    that TritonExperts.apply() can compute the LoRA contribution inline,
    replacing the decorator-based monkey-patch approach.

    Typed as Any for punica_wrapper to avoid a circular import at module load
    time: vllm.lora imports vllm.model_executor.layers.fused_moe, so the
    reverse at module level would be circular.  The actual type is
    PunicaWrapperBase from vllm.lora.punica_wrapper.
    """

    # LoRA weight tensors (same shapes as FusedMoEWithLoRA attributes)
    w13_lora_a_stacked: tuple[torch.Tensor, ...]
    w13_lora_b_stacked: tuple[torch.Tensor, ...]
    w2_lora_a_stacked: tuple[torch.Tensor, ...]
    w2_lora_b_stacked: tuple[torch.Tensor, ...]

    # (max_loras + 1,) int32; slot 0 is the "no-adapter" sentinel
    adapter_enabled: torch.Tensor

    # Metadata
    max_loras: int
    top_k: int
    w13_num_slices: int  # 2 = gated (gate + up), 1 = non-gated or 3D-fused
    fully_sharded: bool
    tp_rank: int
    tp_size: int
    local_num_experts: int

    # PunicaWrapperBase instance (typed Any to avoid circular import)
    punica_wrapper: Any

    # Whether VLLM_TUNED_CONFIG_FOLDER is set; selects get_lora_op_configs vs
    # try_get_optimal_moe_lora_config for Triton kernel tile configs.
    use_tuned_config: bool
