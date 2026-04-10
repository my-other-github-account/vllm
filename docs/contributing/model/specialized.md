# Specialized Model

Specialized models are hand-optimized implementations that override a generic model
when `VLLM_USE_SPECIALIZED_MODELS=1`. They trade generality for performance via
aggressive op fusion, inlined attention, and custom kernels.

All specialized models live in `vllm/model_executor/specialized_models/`.

## When to write a specialized model

Write a specialized model when:

- You need to fuse multiple ops (e.g. KV-cache update + attention + projection) into a
  single custom op for torch.compile and CUDA graph compatibility.
- The generic model leaves significant performance on the table for a specific
  checkpoint/hardware combination (e.g. NVFP4 on Blackwell).
- You want to inline the MLA attention path to eliminate abstraction overhead.

Do **not** share code with other specialized models. Each implementation is independent
and optimized for its own target.

## Directory layout

```
vllm/model_executor/specialized_models/
  __init__.py                    # Registry (_MODELS dict)
  <model_name>/
    __init__.py                  # Exports top-level model class
    model.py                     # Model, layers, attention, custom ops
    kernels.py                   # (optional) Triton kernels
    README.md                    # Usage docs with example commands
```

## Step 1: Register in the registry

Add your architecture to `_MODELS` in `specialized_models/__init__.py`:

```python
_MODELS["MyModelForCausalLM"] = (
    "vllm.model_executor.specialized_models.<model_name>",
    "MyModelForCausalLM",
)
```

## Step 2: Structure your model

A typical hierarchy:

```
TopLevelWrapper (SupportsPP, SupportsQuant, SupportsEagle, ...)
  └─ TextForCausalLM (extends DeepseekV2ForCausalLM or similar)
       └─ TextModel
            └─ DecoderLayer[]
                 ├─ input_layernorm
                 ├─ CustomAttention (with monolithic_attn op)
                 ├─ post_attention_layernorm
                 └─ MLP (reuse generic MoE/MLP modules)
```

The top-level wrapper handles:

- Creating text-only config via `vllm_config.with_hf_config(config.text_config)`
- Weight prefix mapping via `WeightsMapper`
- Skipping non-text weights (vision tower, etc.) during loading

## Step 3: Implement the monolithic attention op

The key optimization is registering a single custom op that covers the entire
attention block. This serves two purposes:

1. torch.compile treats it as one opaque node (no graph breaks inside).
2. It acts as a **splitting point** for piecewise CUDA graph capture.

### Registration

```python
from vllm.utils.torch_utils import direct_register_custom_op

def _my_attn(q, kv_c_normed, k_pe, output, layer_name):
    """Monolithic MLA: cache update + attention + projections."""
    mla = get_forward_context().no_compile_layers[layer_name]
    ...
    return output

def _my_attn_fake(q, kv_c_normed, k_pe, output, layer_name):
    return output

direct_register_custom_op(
    op_name="monolithic_attn",
    op_func=_my_attn,
    fake_impl=_my_attn_fake,
    mutates_args=["output"],
    dispatch_key=current_platform.dispatch_key,
)
```

!!! important
    The op **must** be named `monolithic_attn` — this name is in the compilation
    config's `splitting_ops` list for piecewise CUDA graph compatibility. Only one
    specialized model is loaded at a time, so there is no name collision.

### Accessing layer state

`MLAAttention` registers itself in `static_forward_context` during `__init__`.
The custom op retrieves it at runtime:

```python
mla = get_forward_context().no_compile_layers[layer_name]
# layer_name is mla.layer_name (the prefix passed to MLAAttention)
```

### What to inline

The custom op replaces `mla.impl.do_kv_cache_update()` + `mla.forward_impl()`:

1. **FP8 scale calculation** — `mla.calc_kv_scales(q, kv_c_normed, k_pe)` when
   `mla.calculate_kv_scales` is True.
2. **DCP world size init** — `mla.impl.dcp_world_size` starts at -1 and must be set
   before `forward_mha`.
3. **KV cache update** — `ops.concat_and_cache_mla(...)`.
4. **Prefill path (MHA)** — `mla.impl.forward_mha(...)`.
5. **Decode path (MQA)** — W_UK_T absorption, `mla.impl.forward_mqa(...)`, W_UV
   up-projection.

### MLA prefill vs decode

The two paths work fundamentally differently:

**Prefill (MHA)** — compute-bound, expand to full dimensions *before* attention:

```
kv_c_normed → kv_b_proj → [K_nope, V]       # expand latent → full
Q, K (= K_nope || K_pe), V → standard MHA → output (N*V)
```

`forward_mha` calls `kv_b_proj` internally and writes output in V-head-dim space.
No W_UV needed.

**Decode (MQA)** — memory-bound, stay in latent space:

```
Q_nope @ W_UK_T → QL_nope                    # absorb K proj into Q
(QL_nope || Q_pe) → MQA (latent space) → attn_out (kv_lora_rank)
attn_out @ W_UV → output (N*V)               # project back to V space
```

Both paths write into non-overlapping slices of the same output tensor:

```python
output[num_mqa_tokens:]   # prefill (MHA)
output[:num_mqa_tokens]   # decode  (MQA)
```

### Handling metadata and padding

```python
# Per-layer attention metadata
attn_metadata = fwd_ctx.attn_metadata
if isinstance(attn_metadata, dict):
    attn_metadata = attn_metadata.get(layer_name)
if attn_metadata is None:
    output.zero_()     # warmup / profile run
    return output

# Inputs may be padded for CUDA graphs — trim
output_padded = output
output = output[:num_actual_toks]
q = q[:num_actual_toks]
...
return output_padded   # return the original (padded) tensor
```

## Step 4: Implement weight loading

Extend `DeepseekV2ForCausalLM` (or the appropriate base) and set `model_cls`:

```python
class MyTextForCausalLM(DeepseekV2ForCausalLM):
    model_cls = MyTextModel

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # Post-load fusions here (e.g. fuse_shared_expert_act_quant)
        return loaded
```

The base `load_weights` handles `stacked_params_mapping` for fused QKV A-projections
and expert weight sharding. After all weights are loaded, the framework calls
`process_weights_after_loading` on every module, which creates `W_UK_T` and `W_UV`
from `kv_b_proj` inside `MLAAttention`.

## Step 5: Test

```bash
VLLM_USE_SPECIALIZED_MODELS=1 \
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA \
python examples/basic/offline_inference/basic.py
```

Compare output against the generic model (without `VLLM_USE_SPECIALIZED_MODELS=1`).
Outputs should be semantically equivalent; small divergences are expected from FP8
precision differences and stochastic sampling (`temperature > 0`).

## Common pitfalls

### Never use raw `torch.mm` with quantized weights

NVFP4 weights are uint8-packed. `torch.mm(x, layer.weight.t())` bypasses
dequantization and silently produces garbage. Always call the layer's forward:

```python
# Wrong — bypasses quantized matmul
q = torch.mm(q_c, self.q_b_proj.weight.t())

# Right — uses proper NVFP4 forward
q, _ = self.q_b_proj(q_c)
```

### Initialize `dcp_world_size` before calling `forward_mha`

`forward_mha` asserts `self.dcp_world_size != -1`. The standard `forward_impl` sets
it; inlined code must do so too:

```python
if mla.impl.dcp_world_size == -1:
    from vllm.distributed.parallel_state import get_dcp_group
    mla.impl.dcp_world_size = get_dcp_group().world_size
```

### Calculate FP8 KV cache scales

When `kv_cache_dtype="fp8"`, scales must be computed from actual tensor data before
the first KV cache write. The standard `MLAAttention.forward()` handles this; inlined
code must replicate it:

```python
if mla.calculate_kv_scales:
    mla.calc_kv_scales(q, kv_c_normed, k_pe)
```

### Match scale layouts for fused SiLU+FP4 quantization

The vLLM C++ kernel `silu_and_mul_nvfp4_quant` produces block scales in a layout that
is **not compatible** with `flashinfer_scaled_fp4_mm`. If you need fused
SiLU-and-mul + FP4 quantization for shared experts, write a Triton kernel that
produces swizzled scales (see `deepseek_v3_2_nvfp4/kernels.py` for reference).

### Handle FP8 KV cache dtype

```python
fp8_attn = is_quantized_kv_cache(mla.kv_cache_dtype)
if fp8_attn and mla.kv_cache_dtype != "fp8_ds_mla":
    kv_cache = kv_cache.view(current_platform.fp8_dtype())
```

### Respect backend head-padding requirements

Some attention backends require the head count to be padded:

```python
q_pad = mla.q_pad_num_heads
if q_pad is not None:
    buf = tensor.new_empty((q_pad, B, L))
    buf.resize_((N, B, L))  # logical size
```
