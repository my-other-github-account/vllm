// Standalone fused MoE all-gather kernel for EP dispatch.
//
// JIT-compilable via torch.utils.cpp_extension — no vLLM build required.
//
// Gathers the MoE dispatch tensors from all EP ranks in a single kernel:
//   - topk_ids      [N, topk]         int32
//   - topk_weights  [N, topk]         float32 / bfloat16
//   - hidden_states [N, D_h]          uint8 (NVFP4) / bfloat16
//   - quant_scales  [N, D_s]          (optional) any dtype
//
// Each tensor is packed into a pre-registered IPC buffer at 16-byte-aligned
// offsets.  The kernel:
//   1. Copies inputs into the IPC buffer (local SM write).
//   2. Barrier — all ranks' writes become visible via NVLink.
//   3. Gathers from all peers' buffers into separate output tensors.
//   4. Barrier — done.
//
// Under CUDA graphs, the input tensor addresses are fixed and the buffer
// copies are captured.  The total overhead is dominated by the two barrier
// round-trips (~5µs each on NVLink).
//
// All data movement uses 128-bit (int4) loads/stores.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// ---------------------------------------------------------------------------
// Flag-based barrier (from custom_all_reduce.cuh, standalone)
// ---------------------------------------------------------------------------

constexpr int kMaxBlocks = 36;

using FlagType = uint32_t;

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];
  alignas(128) FlagType end[kMaxBlocks][8];
  alignas(128) FlagType _flag[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

#define DINLINE __device__ __forceinline__

static DINLINE void st_flag_volatile(FlagType* addr, FlagType val) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(val), "l"(addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* addr) {
  FlagType v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(addr));
  return v;
}

template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  FlagType flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    st_flag_volatile(&sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag);
    while (ld_flag_volatile(&self_sg->start[blockIdx.x][threadIdx.x]) != flag);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  FlagType flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    st_flag_volatile(&sg.signals[threadIdx.x]->end[blockIdx.x][rank], flag);
    while (ld_flag_volatile(&self_sg->end[blockIdx.x][threadIdx.x]) != flag);
  }
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// ---------------------------------------------------------------------------
// Fused MoE dispatch all-gather kernel
// ---------------------------------------------------------------------------

// The kernel has 4 phases:
//   1. Pack local inputs into IPC buffer.
//   2. Barrier (all ranks' data visible).
//   3. Gather: one tight contiguous read from each peer → flat staging buffer.
//   4. Barrier (gather done, safe to overwrite IPC buffer next iteration).
//   5. Scatter: redistribute the flat staging into per-tensor outputs (local
//   L2).
//
// Phase 3 is NVLink-critical: one loop, no conditionals, all peers pipelined.
// Phase 5 is local-memory only (L2 speed), runs AFTER the end barrier.

// has_scales: compile-time flag for the optional 4th tensor (quant_scales).
//
// Future optimization (TODO): register hidden_states via IPC (like custom
// allreduce does in graph mode) to skip its Phase 1 copy.  This would save
// ~3µs by eliminating the hidden_states copy + reducing the scatter.
// Requires CUDA graph integration to register the hidden_states tensor
// address during capture.

template <int ngpus, int nbufs>
__global__ void __launch_bounds__(512, 1)
    moe_allgather_kernel(RankData* _dp, RankSignals sg, Signal* self_sg,
                         int rank,
                         // up to 4 inputs
                         const void* inp0, const void* inp1, const void* inp2,
                         const void* inp3, int off0, int sz0, int off1, int sz1,
                         int off2, int sz2, int off3, int sz3, void* out0,
                         void* out1, void* out2, void* out3, void* staging,
                         int total_sz) {
  using V = int4;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto dp = *_dp;
  char* my_buf = (char*)dp.ptrs[rank];

  // Phase 1: pack local inputs into IPC buffer.
  if constexpr (nbufs > 0) {
    const V* s = (const V*)inp0;
    V* d = (V*)(my_buf + off0);
    for (int i = tid; i < sz0; i += stride) d[i] = s[i];
  }
  if constexpr (nbufs > 1) {
    const V* s = (const V*)inp1;
    V* d = (V*)(my_buf + off1);
    for (int i = tid; i < sz1; i += stride) d[i] = s[i];
  }
  if constexpr (nbufs > 2) {
    const V* s = (const V*)inp2;
    V* d = (V*)(my_buf + off2);
    for (int i = tid; i < sz2; i += stride) d[i] = s[i];
  }
  if constexpr (nbufs > 3) {
    const V* s = (const V*)inp3;
    V* d = (V*)(my_buf + off3);
    for (int i = tid; i < sz3; i += stride) d[i] = s[i];
  }

  __threadfence_system();

  // Phase 2: barrier.
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // Phase 3: single contiguous gather into staging buffer.
  {
    const V* peers[ngpus];
#pragma unroll
    for (int s = 0; s < ngpus; s++) peers[s] = (const V*)dp.ptrs[s];

    for (int i = tid; i < total_sz; i += stride) {
#pragma unroll
      for (int s = 0; s < ngpus; s++)
        ((V*)staging)[s * total_sz + i] = peers[s][i];
    }
  }

  // Phase 4: end barrier.
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // Phase 5: scatter from staging to per-tensor outputs (local L2).
  {
    const V* stg = (const V*)staging;

    if constexpr (nbufs > 0) {
      const int b0 = off0 / (int)sizeof(V);
      for (int i = tid; i < sz0; i += stride) {
#pragma unroll
        for (int s = 0; s < ngpus; s++)
          ((V*)out0)[s * sz0 + i] = stg[s * total_sz + b0 + i];
      }
    }
    if constexpr (nbufs > 1) {
      const int b1 = off1 / (int)sizeof(V);
      for (int i = tid; i < sz1; i += stride) {
#pragma unroll
        for (int s = 0; s < ngpus; s++)
          ((V*)out1)[s * sz1 + i] = stg[s * total_sz + b1 + i];
      }
    }
    if constexpr (nbufs > 2) {
      const int b2 = off2 / (int)sizeof(V);
      for (int i = tid; i < sz2; i += stride) {
#pragma unroll
        for (int s = 0; s < ngpus; s++)
          ((V*)out2)[s * sz2 + i] = stg[s * total_sz + b2 + i];
      }
    }
    if constexpr (nbufs > 3) {
      const int b3 = off3 / (int)sizeof(V);
      for (int i = tid; i < sz3; i += stride) {
#pragma unroll
        for (int s = 0; s < ngpus; s++)
          ((V*)out3)[s * sz3 + i] = stg[s * total_sz + b3 + i];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

// Compute 16-byte-aligned offset and int4-unit size for one tensor.
struct TensorDesc {
  void* inp;
  int off;  // byte offset in IPC buffer
  int sz;   // size in int4 (16-byte) units
  int64_t nbytes;
};

static TensorDesc make_desc(torch::Tensor& inp, int64_t& cursor) {
  TORCH_CHECK(inp.is_contiguous(), "input must be contiguous");
  int64_t nbytes = inp.numel() * inp.element_size();
  TORCH_CHECK(nbytes % 16 == 0, "tensor byte size must be multiple of 16, got ",
              nbytes);
  cursor = (cursor + 15) & ~15;
  TensorDesc d;
  d.inp = inp.data_ptr();
  d.off = static_cast<int>(cursor);
  d.sz = static_cast<int>(nbytes / 16);
  d.nbytes = nbytes;
  cursor += nbytes;
  return d;
}

void moe_all_gather(int64_t rank_data_ptr, int64_t signals_ptr,
                    int64_t self_signal_ptr, int64_t rank, int64_t world_size,
                    std::vector<torch::Tensor>& inputs,
                    std::vector<torch::Tensor>& outputs) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  int n = static_cast<int>(inputs.size());
  TORCH_CHECK(n >= 2 && n <= 4, "2-4 input tensors required");
  TORCH_CHECK(inputs.size() == outputs.size());

  int64_t cursor = 0;
  TensorDesc descs[4] = {};
  for (int i = 0; i < n; i++) descs[i] = make_desc(inputs[i], cursor);
  TORCH_CHECK(cursor % 16 == 0);
  int total_sz = static_cast<int>(cursor / 16);

  for (int i = 0; i < n; i++) {
    TORCH_CHECK(outputs[i].is_contiguous());
    TORCH_CHECK(outputs[i].numel() == inputs[i].numel() * world_size);
  }

  auto* ptrs = reinterpret_cast<RankData*>(rank_data_ptr);
  RankSignals sg = *reinterpret_cast<RankSignals*>(signals_ptr);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  int r = static_cast<int>(rank);

  int threads = 512;
  int blocks =
      std::max(1, std::min(kMaxBlocks, (total_sz + threads - 1) / threads));

  void *inps[4] = {}, *outs[4] = {};
  int offs[4] = {}, szs[4] = {};
  for (int i = 0; i < n; i++) {
    inps[i] = descs[i].inp;
    offs[i] = descs[i].off;
    szs[i] = descs[i].sz;
    outs[i] = outputs[i].data_ptr();
  }

  auto staging = torch::empty(
      {(int64_t)world_size * total_sz * (int64_t)sizeof(int4)},
      torch::TensorOptions().dtype(torch::kUInt8).device(inputs[0].device()));

#define KL(ngpus, nb)                                                     \
  moe_allgather_kernel<ngpus, nb><<<blocks, threads, 0, stream>>>(        \
      ptrs, sg, self_sg, r, inps[0], inps[1], inps[2], inps[3], offs[0],  \
      szs[0], offs[1], szs[1], offs[2], szs[2], offs[3], szs[3], outs[0], \
      outs[1], outs[2], outs[3], staging.data_ptr(), total_sz);

#define GPU_CASE(ngpus) \
  case ngpus:           \
    switch (n) {        \
      case 2:           \
        KL(ngpus, 2);   \
        break;          \
      case 3:           \
        KL(ngpus, 3);   \
        break;          \
      case 4:           \
        KL(ngpus, 4);   \
        break;          \
    }                   \
    break;

  switch (world_size) {
    GPU_CASE(2)
    GPU_CASE(4)
    GPU_CASE(6)
    GPU_CASE(8)
    default:
      TORCH_CHECK(false, "world_size must be 2, 4, 6, or 8");
  }
#undef GPU_CASE
#undef KL
}

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_all_gather", &moe_all_gather,
        "Fused MoE dispatch all-gather with in-kernel scatter");
}
