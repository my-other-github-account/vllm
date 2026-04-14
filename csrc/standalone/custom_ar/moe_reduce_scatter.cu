// Lamport-based MoE reduce-scatter kernel for EP combine.
//
// JIT-compilable via torch.utils.cpp_extension — no vLLM build required.
//
// Reduce-scatters a bf16 tensor [N_total, D] across EP ranks.  Each rank
// contributes its partial MoE output; the kernel sums all contributions
// and each rank receives its own slice of the result.
//
// Protocol (same as the all-gather variant):
//   1. PUSH: write own data to all peers' Lamport buffers (NVLink push).
//   2. CLEAR: write sentinels to old segment of own buffer.
//   3. POLL + REDUCE: volatile-load all peers' data for own slice,
//      accumulate in fp32, convert back to bf16, store to output.
//   4. ADVANCE: toggle double-buffer index.
//
// Sentinel: 0x80000000 (two bf16 negative-zeros packed in uint32).
// For bf16 reduce, replacing -0 with +0 is lossless.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define DINLINE __device__ __forceinline__

constexpr uint32_t SENTINEL = 0x80000000u;
constexpr int kMaxBlocks = 36;

// ---------------------------------------------------------------------------
// Volatile 128-bit load and sentinel helpers
// ---------------------------------------------------------------------------

static DINLINE int4 ld128v(const void* addr) {
  int4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr));
  return v;
}

static DINLINE bool has_sentinel(int4 v) {
  return reinterpret_cast<uint32_t&>(v.x) == SENTINEL |
         reinterpret_cast<uint32_t&>(v.y) == SENTINEL |
         reinterpret_cast<uint32_t&>(v.z) == SENTINEL |
         reinterpret_cast<uint32_t&>(v.w) == SENTINEL;
}

static DINLINE int4 remove_sentinel(int4 v) {
  if (reinterpret_cast<uint32_t&>(v.x) == SENTINEL) v.x = 0;
  if (reinterpret_cast<uint32_t&>(v.y) == SENTINEL) v.y = 0;
  if (reinterpret_cast<uint32_t&>(v.z) == SENTINEL) v.z = 0;
  if (reinterpret_cast<uint32_t&>(v.w) == SENTINEL) v.w = 0;
  return v;
}

// ---------------------------------------------------------------------------
// bf16 ↔ fp32 helpers for int4 (8 bf16 values = 16 bytes)
// ---------------------------------------------------------------------------

// Accumulate 8 bf16 values from an int4 into 8 fp32 accumulators.
static DINLINE void accumulate_bf16(float* acc, int4 v) {
  const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
  for (int k = 0; k < 8; k++) acc[k] += __bfloat162float(bp[k]);
}

// Convert 8 fp32 accumulators to bf16 and pack into int4.
static DINLINE int4 fp32_to_bf16_int4(const float* acc) {
  int4 out;
  __nv_bfloat16* bp = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (int k = 0; k < 8; k++) bp[k] = __float2bfloat16(acc[k]);
  return out;
}

// ---------------------------------------------------------------------------
// Lamport reduce-scatter kernel
// ---------------------------------------------------------------------------

template <int ngpus>
__global__ void __launch_bounds__(512, 1) moe_rs_lamport_kernel(
    int64_t* buf_ptrs,  // [ngpus] IPC buffer base addresses (device)
    int* counters,      // [0] = unused, [1] = seg (0/1), [2] = prev total_sz
    int rank,
    int seg_capacity,   // bytes per segment
    int rank_stride,    // bytes per rank-slot within a segment
    const void* input,  // [N_total, D] bf16 — full input
    void* output,       // [N_per_rank, D] bf16 — this rank's reduced slice
    int total_sz,       // int4 units of full input per rank
    int slice_off,      // int4 offset to this rank's slice within packed data
    int slice_sz) {     // int4 units of this rank's slice
  using V = int4;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  // Read segment index and previous clear size.
  const int seg = counters[1];
  const int prev_total_sz = counters[2];
  const int cur_seg = seg;
  const int old_seg = 1 - seg;

  char* bufs[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; r++)
    bufs[r] = reinterpret_cast<char*>(buf_ptrs[r]) + cur_seg * seg_capacity;

  V sent;
  sent.x = sent.y = sent.z = sent.w = static_cast<int>(SENTINEL);

  // ---- Phase 1: PUSH full input to ALL peers ----
  {
    const V* src = reinterpret_cast<const V*>(input);
    for (int i = tid; i < total_sz; i += stride) {
      V val = remove_sentinel(src[i]);
#pragma unroll
      for (int r = 0; r < ngpus; r++)
        reinterpret_cast<V*>(bufs[r] + rank * rank_stride)[i] = val;
    }
  }

  // ---- Phase 2: CLEAR old segment ----
  if (prev_total_sz > 0) {
    char* clr_base =
        reinterpret_cast<char*>(buf_ptrs[rank]) + old_seg * seg_capacity;
#pragma unroll
    for (int r = 0; r < ngpus; r++) {
      V* clr = reinterpret_cast<V*>(clr_base + r * rank_stride);
      for (int i = tid; i < prev_total_sz; i += stride) clr[i] = sent;
    }
  }

  // ---- Phase 3: POLL + REDUCE for own slice ----
  // Read all ranks' data at [slice_off, slice_off + slice_sz) from own buffer,
  // sum in fp32, store bf16 result.
  {
    char* my = bufs[rank];
    V* dst = reinterpret_cast<V*>(output);

    for (int i = tid; i < slice_sz; i += stride) {
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
      for (int s = 0; s < ngpus; s++) {
        V val;
        do {
          val = ld128v(reinterpret_cast<V*>(my + s * rank_stride) + slice_off +
                       i);
        } while (has_sentinel(val));
        accumulate_bf16(acc, val);
      }

      dst[i] = fp32_to_bf16_int4(acc);
    }
  }

  // ---- Phase 4: ADVANCE ----
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counters[1] = 1 - seg;
    counters[2] = total_sz;
  }
}

// ---------------------------------------------------------------------------
// Sentinel initialization
// ---------------------------------------------------------------------------

__global__ void lamport_init_kernel(uint32_t* buf, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) buf[i] = SENTINEL;
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void lamport_init(int64_t buf_ptr, int64_t nbytes) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  int n = static_cast<int>(nbytes / 4);
  lamport_init_kernel<<<256, 256, 0, stream>>>(
      reinterpret_cast<uint32_t*>(buf_ptr), n);
}

void moe_reduce_scatter(int64_t buf_ptrs_ptr, int64_t counters_ptr,
                        int64_t rank, int64_t world_size, int64_t seg_capacity,
                        int64_t rank_stride, torch::Tensor input,
                        torch::Tensor output) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input must be bf16, got ", input.scalar_type());
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be bf16");

  int ws = static_cast<int>(world_size);
  int r = static_cast<int>(rank);

  // Input: [N_total, D], Output: [N_per_rank, D]
  int64_t N_total = input.size(0);
  int64_t D = input.size(1);
  TORCH_CHECK(N_total % ws == 0, "N_total must be divisible by world_size");
  int64_t N_per_rank = N_total / ws;
  TORCH_CHECK(output.size(0) == N_per_rank);
  TORCH_CHECK(output.size(1) == D);

  int64_t input_bytes = input.numel() * input.element_size();
  TORCH_CHECK(input_bytes % 16 == 0,
              "input byte size must be multiple of 16, got ", input_bytes);
  TORCH_CHECK(input_bytes <= rank_stride, "input (", input_bytes,
              " bytes) exceeds rank_stride (", rank_stride, " bytes)");

  int total_sz = static_cast<int>(input_bytes / 16);
  int slice_sz = total_sz / ws;
  int slice_off = r * slice_sz;

  int threads = 512;
  int blocks =
      std::max(1, std::min(kMaxBlocks, (total_sz + threads - 1) / threads));

  auto* bp = reinterpret_cast<int64_t*>(buf_ptrs_ptr);
  auto* ct = reinterpret_cast<int*>(counters_ptr);
  int sc = static_cast<int>(seg_capacity);
  int rs = static_cast<int>(rank_stride);

#define LAUNCH(ng)                                                      \
  moe_rs_lamport_kernel<ng><<<blocks, threads, 0, stream>>>(            \
      bp, ct, r, sc, rs, input.data_ptr(), output.data_ptr(), total_sz, \
      slice_off, slice_sz);

  switch (ws) {
    case 2:
      LAUNCH(2);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 6:
      LAUNCH(6);
      break;
    case 8:
      LAUNCH(8);
      break;
    default:
      TORCH_CHECK(false, "world_size must be 2, 4, 6, or 8");
  }
#undef LAUNCH
}

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_reduce_scatter", &moe_reduce_scatter,
        "Lamport MoE reduce-scatter");
  m.def("lamport_init", &lamport_init,
        "Initialize Lamport buffer with sentinels");
}
