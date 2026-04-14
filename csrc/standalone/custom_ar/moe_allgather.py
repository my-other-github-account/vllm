# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone fused MoE all-gather for EP dispatch.

JIT-compiles the CUDA kernel on first use (cached afterwards).
The kernel gathers contiguously from all peers (NVLink-optimal),
then scatters to per-tensor outputs inside the kernel (local L2).
Zero Python-side post-processing.

Usage:
    ag = MoeAllGather(custom_allreduce)
    ids_g, wt_g, hs_g, sc_g = ag.gather(topk_ids, topk_weights, hidden, scales)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib

    from torch.utils.cpp_extension import load

    src = str(Path(__file__).with_name("moe_allgather.cu"))
    _lib = load(
        name="moe_allgather_kernel",
        sources=[src],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=os.environ.get("MOE_AG_VERBOSE", "") == "1",
    )
    return _lib


class MoeAllGather:
    """Fused MoE dispatch all-gather backed by a one-shot flag-barrier kernel."""

    def __init__(self, ca_comm):
        self.rank = ca_comm.rank
        self.world_size = ca_comm.world_size
        self.device = ca_comm.device
        self.meta_ptrs = ca_comm.meta_ptrs
        self.buffer_ptrs = ca_comm.buffer_ptrs
        self.max_size = ca_comm.max_size

        self._rank_signals = torch.zeros(8, dtype=torch.int64)
        for i in range(self.world_size):
            self._rank_signals[i] = self.meta_ptrs[i]
        self._rank_signals_ptr = self._rank_signals.data_ptr()
        self._self_signal_ptr = self.meta_ptrs[self.rank]

        self._rank_data = torch.zeros(
            8, dtype=torch.int64, device=f"cuda:{self.device.index}"
        )
        for i in range(self.world_size):
            self._rank_data[i] = self.buffer_ptrs[i]
        self._rank_data_ptr = self._rank_data.data_ptr()

    def gather(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        lib = _load_lib()
        ws = self.world_size

        inputs = [topk_ids, topk_weights, hidden_states]
        if scales is not None:
            inputs.append(scales)

        outputs = [
            torch.empty((t.shape[0] * ws, *t.shape[1:]), dtype=t.dtype, device=t.device)
            for t in inputs
        ]

        lib.moe_all_gather(
            self._rank_data_ptr,
            self._rank_signals_ptr,
            self._self_signal_ptr,
            self.rank,
            self.world_size,
            inputs,
            outputs,
        )

        if scales is not None:
            return outputs[0], outputs[1], outputs[2], outputs[3]
        return outputs[0], outputs[1], outputs[2], None
