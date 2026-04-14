// Lamport-based MoE all-gather kernel for EP dispatch.
//
// Replaces the flag-barrier approach with a Lamport sentinel protocol
// (inspired by FlashInfer's trtllm_allreduce_fusion).
//
// Key advantages over the flag-barrier approach:
//   - No explicit barriers (sentinels provide per-element synchronization).
//   - Push model: NVLink writes (fire-and-forget) instead of NVLink reads.
//   - Triple buffering: no end barrier needed.
//
// Gathers the MoE dispatch tensors from all EP ranks:
//   - topk_ids      [N, topk]     int32
//   - topk_weights  [N, topk]     float32 / bfloat16
//   - hidden_states [N, D_h]      uint8 (NVFP4) / bfloat16
//   - quant_scales  [N, D_s]      (optional)
//
// Double-buffer layout in each rank's IPC buffer:
//   [Segment 0][Segment 1]
//   Each segment: [Rank 0 slot][Rank 1 slot]...[Rank N-1 slot]
//   Each rank slot: packed tensors at 16-byte aligned offsets.
//
// Sentinel: 0x80000000 (negative-zero in float32).  The writer replaces
// any data word matching the sentinel with 0 before pushing.  The reader
// spin-loads (volatile) until no sentinel words remain in the vector.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define DINLINE __device__ __forceinline__

constexpr uint32_t SENTINEL = 0x80000000u;
constexpr int kMaxBlocks = 36;

// ---------------------------------------------------------------------------
// Volatile 128-bit load/store and sentinel helpers
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
// Lamport all-gather kernel
// ---------------------------------------------------------------------------
//
// Phase 1 — PUSH: each rank writes its packed data to ALL peers' current
//           segment via regular stores (NVLink push, fire-and-forget).
// Phase 2 — CLEAR: each rank writes sentinels to the OLDEST segment of
//           its own buffer, preparing it for reuse.
// Phase 3 — POLL + SCATTER: each rank volatile-loads from its own current
//           segment, spinning until sentinels disappear, then scatters
//           directly to per-tensor output arrays.
// Phase 4 — ADVANCE: one thread advances the triple-buffer ring counter.

template <int ngpus, int nbufs>
__global__ void __launch_bounds__(512, 1) moe_allgather_lamport_kernel(
    int64_t* buf_ptrs,  // [ngpus] IPC buffer base addresses (device)
    int* counters,      // [0] = unused, [1] = ring (0/1/2), [2] = prev total_sz
    int rank,
    int seg_capacity,  // bytes per segment
    int rank_stride,   // bytes per rank-slot within a segment
    int total_sz,      // int4 units of actual packed data per rank
    // inputs (up to 4)
    const void* inp0, const void* inp1, const void* inp2, const void* inp3,
    int off0, int sz0, int off1, int sz1, int off2, int sz2, int off3, int sz3,
    // outputs (up to 4)
    void* out0, void* out1, void* out2, void* out3) {
  using V = int4;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  // Read segment index and previous clear size.
  const int seg = counters[1];            // 0 or 1
  const int prev_total_sz = counters[2];  // set by previous invocation
  const int cur_seg = seg;
  const int old_seg = 1 - seg;

  char* bufs[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; r++)
    bufs[r] = reinterpret_cast<char*>(buf_ptrs[r]) + cur_seg * seg_capacity;

  // Sentinel vector for clearing.
  V sent;
  sent.x = sent.y = sent.z = sent.w = static_cast<int>(SENTINEL);

  // ---- Phase 1: PUSH local data to ALL peers ----
  // Write to peer_r's buffer at [rank * rank_stride + off_i].

#define PUSH(idx, inp_ptr, off_val, sz_val)                                 \
  if constexpr (nbufs > (idx)) {                                            \
    const V* src = reinterpret_cast<const V*>(inp_ptr);                     \
    for (int i = tid; i < (sz_val); i += stride) {                          \
      V val = remove_sentinel(src[i]);                                      \
      _Pragma("unroll") for (int r = 0; r < ngpus; r++) {                   \
        reinterpret_cast<V*>(bufs[r] + rank * rank_stride + (off_val))[i] = \
            val;                                                            \
      }                                                                     \
    }                                                                       \
  }

  PUSH(0, inp0, off0, sz0)
  PUSH(1, inp1, off1, sz1)
  PUSH(2, inp2, off2, sz2)
  PUSH(3, inp3, off3, sz3)
#undef PUSH

  // ---- Phase 2: CLEAR only the previously-written data in oldest segment ----
  // Only clear what the previous invocation actually wrote (per rank-slot).
  if (prev_total_sz > 0) {
    char* clr_base =
        reinterpret_cast<char*>(buf_ptrs[rank]) + old_seg * seg_capacity;
#pragma unroll
    for (int r = 0; r < ngpus; r++) {
      V* clr = reinterpret_cast<V*>(clr_base + r * rank_stride);
      for (int i = tid; i < prev_total_sz; i += stride) clr[i] = sent;
    }
  }

  // ---- Phase 3: POLL + SCATTER ----
  // Volatile-load from own buffer; spin until sentinel gone; scatter to output.
  char* my = bufs[rank];

#define POLL(idx, out_ptr, off_val, sz_val)                                \
  if constexpr (nbufs > (idx)) {                                           \
    for (int i = tid; i < (sz_val); i += stride) {                         \
      _Pragma("unroll") for (int s = 0; s < ngpus; s++) {                  \
        V val;                                                             \
        do {                                                               \
          val = ld128v(                                                    \
              reinterpret_cast<V*>(my + s * rank_stride + (off_val)) + i); \
        } while (has_sentinel(val));                                       \
        reinterpret_cast<V*>(out_ptr)[s * (sz_val) + i] = val;             \
      }                                                                    \
    }                                                                      \
  }

  POLL(0, out0, off0, sz0)
  POLL(1, out1, off1, sz1)
  POLL(2, out2, off2, sz2)
  POLL(3, out3, off3, sz3)
#undef POLL

  // ---- Phase 4: ADVANCE ring counter + store clear size for next call ----
  // Stream serialization ensures the next kernel sees these updates.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counters[1] = 1 - seg;
    counters[2] = total_sz;
  }
}

// ---------------------------------------------------------------------------
// Sentinel initialization kernel
// ---------------------------------------------------------------------------

__global__ void lamport_init_kernel(uint32_t* buf, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) buf[i] = SENTINEL;
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

struct TensorDesc {
  void* inp;
  int off;
  int sz;
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

void lamport_init(int64_t buf_ptr, int64_t nbytes) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  int n = static_cast<int>(nbytes / 4);
  lamport_init_kernel<<<256, 256, 0, stream>>>(
      reinterpret_cast<uint32_t*>(buf_ptr), n);
}

void moe_all_gather(int64_t buf_ptrs_ptr, int64_t counters_ptr, int64_t rank,
                    int64_t world_size, int64_t seg_capacity,
                    int64_t rank_stride, std::vector<torch::Tensor>& inputs,
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
  TORCH_CHECK(cursor <= rank_stride, "packed data (", cursor,
              " bytes) exceeds rank_stride (", rank_stride, " bytes)");

  int ws = static_cast<int>(world_size);
  for (int i = 0; i < n; i++) {
    TORCH_CHECK(outputs[i].is_contiguous());
    TORCH_CHECK(outputs[i].numel() == inputs[i].numel() * ws);
  }

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

  auto* bp = reinterpret_cast<int64_t*>(buf_ptrs_ptr);
  auto* ct = reinterpret_cast<int*>(counters_ptr);
  int sc = static_cast<int>(seg_capacity);
  int rs = static_cast<int>(rank_stride);

#define KL(ng, nb)                                                        \
  moe_allgather_lamport_kernel<ng, nb><<<blocks, threads, 0, stream>>>(   \
      bp, ct, r, sc, rs, total_sz, inps[0], inps[1], inps[2], inps[3],    \
      offs[0], szs[0], offs[1], szs[1], offs[2], szs[2], offs[3], szs[3], \
      outs[0], outs[1], outs[2], outs[3]);

#define GPU_CASE(ng) \
  case ng:           \
    switch (n) {     \
      case 2:        \
        KL(ng, 2);   \
        break;       \
      case 3:        \
        KL(ng, 3);   \
        break;       \
      case 4:        \
        KL(ng, 4);   \
        break;       \
    }                \
    break;

  switch (ws) {
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
  m.def("moe_all_gather", &moe_all_gather, "Lamport MoE all-gather");
  m.def("lamport_init", &lamport_init,
        "Initialize Lamport buffer with sentinels");
}
