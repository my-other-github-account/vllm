# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lamport-based fused MoE all-gather for EP dispatch.

JIT-compiles the CUDA kernel on first use (cached afterwards).
Uses a Lamport sentinel protocol (push writes + per-element sync)
with triple-buffered IPC regions — no explicit barriers.

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
    """Lamport-based MoE dispatch all-gather with triple buffering."""

    def __init__(self, ca_comm):
        self.rank = ca_comm.rank
        self.world_size = ca_comm.world_size
        self.device = ca_comm.device
        self.buffer_ptrs = ca_comm.buffer_ptrs
        self.max_size = ca_comm.max_size

        ws = self.world_size
        # Double-buffer layout: 2 segments, each with ws rank-slots.
        # Safe because kernels in the same stream are serialized, and the
        # Lamport poll ensures all cross-GPU pushes complete before the
        # kernel returns.
        # seg_capacity and rank_stride are 16-byte aligned.
        self.seg_capacity = (self.max_size // 2) & ~15
        self.rank_stride = (self.seg_capacity // ws) & ~15
        self.max_per_rank = self.rank_stride  # max packed bytes per rank

        # Buffer pointer array on device.
        self._buf_ptrs = torch.zeros(
            8, dtype=torch.int64, device=f"cuda:{self.device.index}"
        )
        for i in range(ws):
            self._buf_ptrs[i] = self.buffer_ptrs[i]
        self._buf_ptrs_ptr = self._buf_ptrs.data_ptr()

        # Counters on device: [0]=unused, [1]=ring index (0/1/2), [2]=prev_total_sz.
        self._counters = torch.zeros(
            3, dtype=torch.int32, device=f"cuda:{self.device.index}"
        )
        self._counters_ptr = self._counters.data_ptr()

        # Initialize ALL segments with sentinel values.
        lib = _load_lib()
        lib.lamport_init(self.buffer_ptrs[self.rank], self.max_size)
        torch.accelerator.synchronize(self.device)

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
            self._buf_ptrs_ptr,
            self._counters_ptr,
            self.rank,
            self.world_size,
            self.seg_capacity,
            self.rank_stride,
            inputs,
            outputs,
        )

        if scales is not None:
            return outputs[0], outputs[1], outputs[2], outputs[3]
        return outputs[0], outputs[1], outputs[2], None
