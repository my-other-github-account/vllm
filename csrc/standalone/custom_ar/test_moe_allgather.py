# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test for the Lamport-based MoE all-gather kernel."""

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


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if rank == 0:
        from moe_allgather import MoeAllGather, _load_lib

        _load_lib()
    dist.barrier()
    from moe_allgather import MoeAllGather, _load_lib

    _load_lib()
    dist.barrier()

    max_size = 8 * 1024 * 1024
    bp = ipc_buf(max_size, rank, ws)
    dist.barrier()

    # Build a fake ca_comm-like object.
    class FakeCA:
        pass

    ca = FakeCA()
    ca.rank = rank
    ca.world_size = ws
    ca.device = torch.device(dev)
    ca.buffer_ptrs = bp
    ca.max_size = max_size
    # meta_ptrs not needed for Lamport approach
    ag = MoeAllGather(ca)
    dist.barrier()

    errors = 0

    # Test with various token counts.
    for N in [1, 4, 16, 64]:
        topk = 8
        hd = 3584
        sd = 448
        ids = (
            torch.arange(N * topk, dtype=torch.int32, device=dev) + rank * 1000
        ).reshape(N, topk)
        wt = torch.ones(N, topk, dtype=torch.float32, device=dev) * (rank + 1) * 0.1
        hs = torch.full((N, hd), rank + 1, dtype=torch.uint8, device=dev)
        sc = torch.full((N, sd), rank + 1, dtype=torch.uint8, device=dev)

        ids_g, wt_g, hs_g, sc_g = ag.gather(ids, wt, hs, sc)

        for src in range(ws):
            s, e = src * N, (src + 1) * N
            exp_ids = (
                torch.arange(N * topk, dtype=torch.int32, device=dev) + src * 1000
            ).reshape(N, topk)
            if not torch.equal(ids_g[s:e], exp_ids):
                print(f"[{rank}] FAIL ids src={src} N={N}")
                errors += 1
            exp_wt = torch.full(
                (N, topk), (src + 1) * 0.1, dtype=torch.float32, device=dev
            )
            if not torch.allclose(wt_g[s:e], exp_wt):
                print(f"[{rank}] FAIL wt src={src} N={N}")
                errors += 1
            exp_hs = torch.full((N, hd), src + 1, dtype=torch.uint8, device=dev)
            if not torch.equal(hs_g[s:e], exp_hs):
                print(f"[{rank}] FAIL hs src={src} N={N}")
                errors += 1
            exp_sc = torch.full((N, sd), src + 1, dtype=torch.uint8, device=dev)
            if not torch.equal(sc_g[s:e], exp_sc):
                print(f"[{rank}] FAIL sc src={src} N={N}")
                errors += 1

        # Without scales.
        ids_g2, wt_g2, hs_g2, _ = ag.gather(ids, wt, hs)
        for src in range(ws):
            s, e = src * N, (src + 1) * N
            exp_ids = (
                torch.arange(N * topk, dtype=torch.int32, device=dev) + src * 1000
            ).reshape(N, topk)
            if not torch.equal(ids_g2[s:e], exp_ids):
                print(f"[{rank}] FAIL no-sc ids src={src} N={N}")
                errors += 1

    dist.barrier()
    print(
        f"[rank {rank}] {'PASSED' if errors == 0 else f'FAILED ({errors})'} (ws={ws})"
    )
    dist.destroy_process_group()
    return errors


if __name__ == "__main__":
    sys.exit(main())
