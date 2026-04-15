# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test + benchmark for Lamport MoE reduce-scatter."""

import ctypes
import os
import sys

import torch
import torch.distributed as dist

_cudart = ctypes.CDLL("libcudart.so")
IPC = 64


def _cc(r):
    if r:
        raise RuntimeError(f"CUDA err {r}")


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


def gpu_timer_graph(fn, warmup=20, repeats=200):
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
        from moe_reduce_scatter import _load_lib

        _load_lib()
    dist.barrier()
    if rank != 0:
        from moe_reduce_scatter import _load_lib

        _load_lib()
    dist.barrier()
    from moe_reduce_scatter import MoeReduceScatter

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
    rs = MoeReduceScatter(ca)
    dist.barrier()

    D = 7168  # DeepSeek V3 hidden_dim
    errors = 0

    # ---- Correctness tests ----
    for N_per_rank in [1, 4, 16]:
        N_total = N_per_rank * ws

        # Each rank gets a deterministic input.
        torch.manual_seed(42)
        # All ranks create the SAME "ground truth" inputs for each rank.
        all_inputs = [
            torch.randn(N_total, D, dtype=torch.bfloat16, device=dev) for _ in range(ws)
        ]
        # This rank's input is all_inputs[rank].
        my_input = all_inputs[rank]

        # Custom reduce-scatter.
        custom_out = rs.reduce_scatter(my_input)

        # NCCL reference: reduce_scatter_tensor.
        nccl_out = torch.empty(N_per_rank, D, dtype=torch.bfloat16, device=dev)
        dist.reduce_scatter_tensor(nccl_out, my_input)

        # bf16 summation order differs between our kernel and NCCL,
        # giving ~1-2 ULP differences. Use generous tolerance.
        max_diff = (custom_out.float() - nccl_out.float()).abs().max().item()
        if not torch.allclose(custom_out, nccl_out, atol=0.125, rtol=0.01):
            mismatches = (
                ((custom_out.float() - nccl_out.float()).abs() > 0.125).sum().item()
            )
            print(
                f"[{rank}] N_per_rank={N_per_rank} MISMATCH: "
                f"max_diff={max_diff:.6f}, mismatches={mismatches}"
            )
            errors += 1
        else:
            if rank == 0:
                print(f"  N_per_rank={N_per_rank}: PASS (max_diff={max_diff:.6f})")

    # ---- Benchmark ----
    if rank == 0:
        print(f"\nworld_size={ws}, max_per_rank={rs.max_per_rank} bytes")
        print(
            f"{'config':<12} {'lamport':>10} {'lamp_graph':>10} "
            f"{'nccl':>10} {'nccl_graph':>10} {'speedup':>8}"
        )
        print("-" * 65)

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

    for name, N_per_rank in configs:
        N_total = N_per_rank * ws
        input_bytes = N_total * D * 2  # bf16
        if input_bytes > rs.max_per_rank:
            if rank == 0:
                print(f"{name:<12} {'skip (too large)':>40}")
            continue

        inp = torch.randn(N_total, D, dtype=torch.bfloat16, device=dev)
        c_out = torch.empty(N_per_rank, D, dtype=torch.bfloat16, device=dev)
        n_out = torch.empty(N_per_rank, D, dtype=torch.bfloat16, device=dev)

        def run_lamport():
            rs.reduce_scatter(inp)

        def run_nccl():
            dist.reduce_scatter_tensor(n_out, inp)

        from bench_moe_allgather import gpu_timer

        lam_us = gpu_timer(run_lamport)
        try:
            lam_g_us = gpu_timer_graph(run_lamport)
        except Exception:
            lam_g_us = float("nan")

        nccl_us = gpu_timer(run_nccl)
        try:
            nccl_g_us = gpu_timer_graph(run_nccl)
        except Exception:
            nccl_g_us = float("nan")

        if rank == 0:
            speedup = nccl_g_us / lam_g_us if lam_g_us > 0 else float("nan")
            print(
                f"{name:<12} {lam_us:>9.1f}µ {lam_g_us:>9.1f}µ "
                f"{nccl_us:>9.1f}µ {nccl_g_us:>9.1f}µ {speedup:>7.2f}x"
            )

    dist.barrier()
    print(f"[rank {rank}] {'PASSED' if errors == 0 else f'FAILED ({errors})'}")
    dist.destroy_process_group()
    return errors


if __name__ == "__main__":
    sys.exit(main())
