# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lamport-based MoE reduce-scatter for EP combine.

JIT-compiles the CUDA kernel on first use (cached afterwards).
Uses the same Lamport sentinel protocol as the all-gather kernel.
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

    src = str(Path(__file__).with_name("moe_reduce_scatter.cu"))
    _lib = load(
        name="moe_reduce_scatter_kernel",
        sources=[src],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=os.environ.get("MOE_RS_VERBOSE", "") == "1",
    )
    return _lib


class MoeReduceScatter:
    """Lamport-based MoE combine reduce-scatter with double buffering."""

    def __init__(self, ca_comm):
        self.rank = ca_comm.rank
        self.world_size = ca_comm.world_size
        self.device = ca_comm.device
        self.buffer_ptrs = ca_comm.buffer_ptrs
        self.max_size = ca_comm.max_size

        ws = self.world_size
        # Use the SECOND half of the IPC buffer (first half reserved for
        # MoeAllGather) to avoid overlapping writes.
        half_size = (self.max_size // 2) & ~15
        self.buffer_offset = half_size

        # Double-buffer layout within our half: 2 segments, each ws rank-slots.
        self.seg_capacity = (half_size // 2) & ~15
        self.rank_stride = (self.seg_capacity // ws) & ~15
        self.max_per_rank = self.rank_stride

        # Buffer pointer array on device — offset to our half.
        self._buf_ptrs = torch.zeros(
            8, dtype=torch.int64, device=f"cuda:{self.device.index}"
        )
        for i in range(ws):
            self._buf_ptrs[i] = self.buffer_ptrs[i] + self.buffer_offset
        self._buf_ptrs_ptr = self._buf_ptrs.data_ptr()

        # Counters: [0]=unused, [1]=seg (0/1), [2]=prev_total_sz.
        self._counters = torch.zeros(
            3, dtype=torch.int32, device=f"cuda:{self.device.index}"
        )
        self._counters_ptr = self._counters.data_ptr()

        # Initialize our half with sentinels.
        lib = _load_lib()
        lib.lamport_init(self.buffer_ptrs[self.rank] + self.buffer_offset, half_size)
        torch.accelerator.synchronize(self.device)

    def reduce_scatter(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce-scatter input [N_total, D] bf16 → output [N_per_rank, D] bf16."""
        lib = _load_lib()
        ws = self.world_size

        assert input.dim() == 2
        N_total, D = input.shape
        assert N_total % ws == 0
        N_per_rank = N_total // ws

        output = torch.empty((N_per_rank, D), dtype=input.dtype, device=input.device)

        lib.moe_reduce_scatter(
            self._buf_ptrs_ptr,
            self._counters_ptr,
            self.rank,
            self.world_size,
            self.seg_capacity,
            self.rank_stride,
            input,
            output,
        )
        return output
