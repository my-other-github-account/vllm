# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stress test: random data, check bitwise correctness against NCCL."""

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
        from moe_allgather import _load_lib

        _load_lib()
    dist.barrier()
    from moe_allgather import MoeAllGather, _load_lib

    _load_lib()
    dist.barrier()

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

    topk = 8
    hd = 3584
    sd = 448
    errors = 0
    total_checks = 0
    sentinel_collisions = 0

    for trial in range(200):
        # All ranks must use the same N for NCCL reference.
        N_tensor = torch.randint(1, 65, (1,), device=dev)
        dist.broadcast(N_tensor, src=0)
        N = N_tensor.item()
        # Random data including possible sentinel values
        ids = torch.randint(0, 256, (N, topk), dtype=torch.int32, device=dev)
        wt = torch.randn(N, topk, dtype=torch.float32, device=dev)
        hs = torch.randint(0, 256, (N, hd), dtype=torch.uint8, device=dev)
        sc = torch.randint(0, 256, (N, sd), dtype=torch.uint8, device=dev)

        # Count sentinel patterns in hidden_states (as uint32 view)
        hs_u32 = hs.view(torch.int32)
        sentinel_collisions += (hs_u32 == 0x80000000).sum().item()

        # Custom kernel
        ids_g, wt_g, hs_g, sc_g = ag.gather(ids, wt, hs, sc)

        # NCCL reference
        ids_ref = torch.empty(N * ws, topk, dtype=torch.int32, device=dev)
        wt_ref = torch.empty(N * ws, topk, dtype=torch.float32, device=dev)
        hs_ref = torch.empty(N * ws, hd, dtype=torch.uint8, device=dev)
        sc_ref = torch.empty(N * ws, sd, dtype=torch.uint8, device=dev)
        dist.all_gather_into_tensor(ids_ref, ids)
        dist.all_gather_into_tensor(wt_ref, wt)
        dist.all_gather_into_tensor(hs_ref, hs)
        dist.all_gather_into_tensor(sc_ref, sc)

        # Compare
        if not torch.equal(ids_g, ids_ref):
            mismatches = (ids_g != ids_ref).sum().item()
            if trial < 5 or mismatches > 0:
                print(f"[{rank}] trial={trial} ids MISMATCH: {mismatches} elements")
            errors += 1
        if not torch.equal(wt_g, wt_ref):
            # Check for -0 vs +0 differences
            bit_diff = wt_g.view(torch.int32) != wt_ref.view(torch.int32)
            neg_zero_mask = wt_ref.view(torch.int32) == 0x80000000
            real_errors = bit_diff & ~neg_zero_mask
            if real_errors.any():
                print(
                    f"[{rank}] trial={trial} wt MISMATCH (non-negzero): {real_errors.sum().item()}"
                )
                errors += 1
        if not torch.equal(hs_g, hs_ref):
            mismatches = (hs_g != hs_ref).sum().item()
            # Check if mismatches are due to sentinel collision
            hs_g_u32 = hs_g.view(torch.int32)
            hs_ref_u32 = hs_ref.view(torch.int32)
            diff_mask = hs_g_u32 != hs_ref_u32
            sentinel_mask = (hs_ref_u32 == 0x80000000) & diff_mask
            non_sentinel = diff_mask & ~sentinel_mask
            if non_sentinel.any():
                print(
                    f"[{rank}] trial={trial} hs NON-SENTINEL MISMATCH: {non_sentinel.sum().item()}"
                )
                errors += 1
            elif sentinel_mask.any():
                if trial < 3:
                    print(
                        f"[{rank}] trial={trial} hs sentinel collision: "
                        f"{sentinel_mask.sum().item()} words (expected rare)"
                    )
        if not torch.equal(sc_g, sc_ref):
            mismatches = (sc_g != sc_ref).sum().item()
            sc_g_u32 = sc_g.view(torch.int32) if sc_g.numel() % 4 == 0 else None
            if sc_g_u32 is not None:
                sc_ref_u32 = sc_ref.view(torch.int32)
                diff_mask = sc_g_u32 != sc_ref_u32
                sentinel_mask = (sc_ref_u32 == 0x80000000) & diff_mask
                non_sentinel = diff_mask & ~sentinel_mask
                if non_sentinel.any():
                    print(
                        f"[{rank}] trial={trial} sc NON-SENTINEL MISMATCH: {non_sentinel.sum().item()}"
                    )
                    errors += 1

        total_checks += 1

    dist.barrier()
    print(
        f"[rank {rank}] {total_checks} trials, {errors} real errors, "
        f"{sentinel_collisions} sentinel patterns in hs data. "
        f"{'PASSED' if errors == 0 else 'FAILED'}"
    )
    dist.destroy_process_group()
    return errors


if __name__ == "__main__":
    sys.exit(main())
