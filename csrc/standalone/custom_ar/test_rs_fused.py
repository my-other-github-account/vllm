# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test + benchmark: fused RS + residual + RMSNorm vs separate kernels."""

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


def rms_norm_ref(x, gamma, eps):
    """Reference RMSNorm in fp32."""
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rms * gamma.float()).to(x.dtype)


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

    # Compile fused kernel
    if rank == 0:
        from torch.utils.cpp_extension import load

        load(
            name="moe_rs_fused_kernel",
            sources=[os.path.join(os.path.dirname(__file__), "moe_rs_fused.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    dist.barrier()
    from torch.utils.cpp_extension import load

    fused_lib = load(
        name="moe_rs_fused_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "moe_rs_fused.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

    # Also compile separate RS kernel for comparison
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

    rs_separate = MoeReduceScatter(ca)

    # Fused kernel setup (uses same buffer layout as MoeReduceScatter)
    half_size = (max_size // 2) & ~15
    buf_offset = half_size
    fused_seg_cap = (half_size // 2) & ~15
    fused_rank_stride = (fused_seg_cap // ws) & ~15

    fused_buf_ptrs = torch.zeros(8, dtype=torch.int64, device=dev)
    for i in range(ws):
        fused_buf_ptrs[i] = bp[i] + buf_offset
    fused_counters = torch.zeros(3, dtype=torch.int32, device=dev)

    # Init sentinels for fused kernel's buffer region
    fused_lib.lamport_init(bp[rank] + buf_offset, half_size)
    torch.cuda.synchronize()
    dist.barrier()

    D = 7168
    eps = 1e-6
    gamma = torch.randn(D, dtype=torch.bfloat16, device=dev).abs() + 0.5
    errors = 0

    # ---- Correctness ----
    for N_per_rank in [1, 4]:
        N_total = N_per_rank * ws
        torch.manual_seed(42 + rank)
        moe_out = torch.randn(N_total, D, dtype=torch.bfloat16, device=dev)
        residual = torch.randn(N_per_rank, D, dtype=torch.bfloat16, device=dev)

        # Reference: NCCL RS + add + norm
        rs_ref = torch.empty(N_per_rank, D, dtype=torch.bfloat16, device=dev)
        dist.reduce_scatter_tensor(rs_ref, moe_out)
        ref_residual = residual + rs_ref
        ref_normed = rms_norm_ref(ref_residual, gamma, eps)

        # Fused kernel
        normed_out = torch.empty(N_per_rank, D, dtype=torch.bfloat16, device=dev)
        residual_out = torch.empty(N_per_rank, D, dtype=torch.bfloat16, device=dev)
        fused_lib.moe_rs_fused(
            fused_buf_ptrs.data_ptr(),
            fused_counters.data_ptr(),
            rank,
            ws,
            fused_seg_cap,
            fused_rank_stride,
            moe_out,
            residual,
            gamma,
            normed_out,
            residual_out,
            eps,
        )
        torch.cuda.synchronize()

        # Compare
        max_diff_res = (residual_out.float() - ref_residual.float()).abs().max().item()
        max_diff_norm = (normed_out.float() - ref_normed.float()).abs().max().item()

        ok = max_diff_res < 0.125 and max_diff_norm < 0.125
        if rank == 0:
            print(
                f"  N_per_rank={N_per_rank}: {'PASS' if ok else 'FAIL'} "
                f"(res_diff={max_diff_res:.4f}, norm_diff={max_diff_norm:.4f})"
            )
        if not ok:
            errors += 1

    # ---- Benchmark ----
    if rank == 0:
        print(f"\nBenchmark: D={D}, world_size={ws}")
        print(
            f"{'config':<10} {'fused':>10} {'fused_g':>10} "
            f"{'RS+norm':>10} {'RS+norm_g':>10} {'speedup':>8}"
        )
        print("-" * 58)

    for N_per_rank in [1, 2, 4, 8]:
        N_total = N_per_rank * ws
        input_bytes = N_total * D * 2
        if input_bytes > fused_rank_stride:
            if rank == 0:
                print(f"{N_per_rank}tok      skip (too large)")
            continue

        moe_out = torch.randn(N_total, D, dtype=torch.bfloat16, device=dev)
        residual = torch.randn(N_per_rank, D, dtype=torch.bfloat16, device=dev)
        normed_out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        rs_out = torch.empty_like(residual)

        # Fused
        def run_fused():
            fused_lib.moe_rs_fused(
                fused_buf_ptrs.data_ptr(),
                fused_counters.data_ptr(),
                rank,
                ws,
                fused_seg_cap,
                fused_rank_stride,
                moe_out,
                residual,
                gamma,
                normed_out,
                residual_out,
                eps,
            )

        # Separate: RS + add + norm (our Lamport RS + triton-like ops)
        def run_separate():
            rs_separate.reduce_scatter(moe_out)
            # Simulate add + RMSNorm (in practice this is a fused triton kernel)
            tmp = residual + rs_out
            torch.rsqrt(tmp.float().pow(2).mean(-1, keepdim=True) + eps)

        fused_us = gpu_timer(run_fused)
        try:
            fused_g = gpu_timer_graph(run_fused)
        except Exception:
            fused_g = float("nan")

        sep_us = gpu_timer(run_separate)
        try:
            sep_g = gpu_timer_graph(run_separate)
        except Exception:
            sep_g = float("nan")

        if rank == 0:
            speedup = sep_g / fused_g if fused_g > 0 else float("nan")
            print(
                f"{N_per_rank}tok     {fused_us:>9.1f}µ {fused_g:>9.1f}µ "
                f"{sep_us:>9.1f}µ {sep_g:>9.1f}µ {speedup:>7.2f}x"
            )

    dist.barrier()
    print(f"[rank {rank}] {'PASSED' if errors == 0 else f'FAILED ({errors})'}")
    dist.destroy_process_group()
    return errors


if __name__ == "__main__":
    sys.exit(main())
