# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: Lamport all-gather vs NCCL."""

import ctypes
import os
import sys

import torch
import torch.distributed as dist

_cudart = ctypes.CDLL("libcudart.so")
IPC = 64


def _cc(r):
    if r:
        raise RuntimeError(f"err {r}")


def ipc_buf(sz, rank, ws):
    p = ctypes.c_void_p()
    _cc(_cudart.cudaMalloc(ctypes.byref(p), sz))
    _cc(_cudart.cudaMemset(p, 0, sz))
    _cc(_cudart.cudaDeviceSynchronize())
    h = (ctypes.c_byte * IPC)()
    _cc(_cudart.cudaIpcGetMemHandle(ctypes.byref(h), p))
    ah = [None] * ws
    dist.all_gather_object(ah, bytes(h))
    ptrs = []
    for i in range(ws):
        if i == rank:
            ptrs.append(p.value)
        else:
            hh = (ctypes.c_byte * IPC)(*ah[i])
            pp = ctypes.c_void_p()
            _cc(_cudart.cudaIpcOpenMemHandle(ctypes.byref(pp), hh, ctypes.c_uint(1)))
            ptrs.append(pp.value)
    return ptrs


def gpu_timer(fn, warmup=20, repeats=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeats * 1000


def gpu_timer_graph(fn, warmup=20, repeats=200):
    """Time with CUDA graph to exclude CPU overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeats * 1000


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if rank == 0:
        from moe_allgather import _load_lib

        lib = _load_lib()
    dist.barrier()
    if rank != 0:
        from moe_allgather import _load_lib

        lib = _load_lib()
    dist.barrier()
    from moe_allgather import MoeAllGather

    max_size = 8 * 1024 * 1024
    bp = ipc_buf(max_size, rank, ws)
    dist.barrier()

    class FakeCA:
        pass

    ca = FakeCA()
    ca.rank = rank
    ca.world_size = ws
    ca.device = torch.device(dev)
    ca.buffer_ptrs = bp
    ca.max_size = max_size
    ag = MoeAllGather(ca)
    dist.barrier()

    configs = [
        ("1tok", 1),
        ("2tok", 2),
        ("4tok", 4),
        ("8tok", 8),
        ("16tok", 16),
        ("32tok", 32),
        ("64tok", 64),
        ("128tok", 128),
        ("256tok", 256),
    ]
    topk = 8
    hd = 3584
    sd = 448

    if rank == 0:
        print(f"world_size={ws}, max_per_rank={ag.max_per_rank} bytes")
        print(f"{'config':<12} {'lamport_graph':>10} {'nccl_graph':>10} {'speedup':>8}")
        print("-" * 65)

    for name, N in configs:
        # Check if data fits in buffer.
        cursor = 0
        per_tok = topk * 4 + topk * 4 + hd + sd
        cursor = N * per_tok
        cursor = (cursor + 15) & ~15
        if cursor > ag.max_per_rank:
            if rank == 0:
                print(f"{name:<12} {'skip (too large)':>40}")
            continue

        ids = torch.randint(0, 256, (N, topk), dtype=torch.int32, device=dev)
        wt = torch.randn(N, topk, dtype=torch.float32, device=dev).abs()
        hs = torch.randint(0, 255, (N, hd), dtype=torch.uint8, device=dev)
        sc = torch.randint(0, 255, (N, sd), dtype=torch.uint8, device=dev)
        inputs = [ids, wt, hs, sc]

        # Custom Lamport kernel.
        c_outs = [
            torch.empty(N * ws, *t.shape[1:], dtype=t.dtype, device=dev) for t in inputs
        ]

        def run_lamport():
            lib.moe_all_gather(
                ag._buf_ptrs_ptr,
                ag._counters_ptr,
                rank,
                ws,
                ag.seg_capacity,
                ag.rank_stride,
                inputs,
                c_outs,
            )

        # Lamport with CUDA graph.
        try:
            lam_g_us = gpu_timer_graph(run_lamport)
        except Exception as ex:
            lam_g_us = float("nan")
            if rank == 0:
                print(f"  [graph capture failed: {ex}]")

        # NCCL 1×AG (concat into one tensor).
        cat_inp = torch.cat(
            [t.reshape(N, -1).contiguous().view(torch.uint8) for t in inputs],
            dim=1,
        ).contiguous()
        cat_out = torch.empty(N * ws, cat_inp.shape[1], dtype=torch.uint8, device=dev)

        def run_nccl():
            dist.all_gather_into_tensor(cat_out, cat_inp)

        # NCCL with CUDA graph.
        try:
            nccl_g_us = gpu_timer_graph(run_nccl)
        except Exception:
            nccl_g_us = float("nan")

        if rank == 0:
            speedup = nccl_g_us / lam_g_us if lam_g_us > 0 else float("nan")
            print(f"{name:<12} {lam_g_us:>9.1f}µ {nccl_g_us:>9.1f}µ {speedup:>7.2f}x")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
