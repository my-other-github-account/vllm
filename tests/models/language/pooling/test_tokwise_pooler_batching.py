# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import vllm.model_executor.layers.pooler.tokwise.methods as tokwise_methods
from vllm.model_executor.layers.pooler.tokwise.heads import (
    TokenEmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates


class CountingLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.call_count = 0
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.input_shapes.append(tuple(x.shape))
        return self.linear(x)


def _patch_chunked_prefill(monkeypatch, enabled: bool) -> None:
    monkeypatch.setattr(
        tokwise_methods,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(enable_chunked_prefill=enabled)
        ),
    )


def _build_pooling_metadata(
    *,
    prompt_lens: list[int],
    pooling_params: list[PoolingParams],
    seq_lens: list[int] | None = None,
    scheduled_lens: list[int] | None = None,
) -> PoolingMetadata:
    prompt_lens_cpu = torch.tensor(prompt_lens, dtype=torch.int64)
    if seq_lens is None:
        seq_lens = prompt_lens
    if scheduled_lens is None:
        scheduled_lens = seq_lens

    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    query_start_loc_cpu = torch.tensor(
        [0, *np.cumsum(scheduled_lens, dtype=np.int64)],
        dtype=torch.int64,
    )

    metadata = PoolingMetadata(
        prompt_lens=prompt_lens_cpu,
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=pooling_params,
        pooling_states=[PoolingStates() for _ in pooling_params],
    )
    metadata.build_pooling_cursor(
        num_scheduled_tokens_np=np.asarray(scheduled_lens, dtype=np.int64),
        seq_lens_cpu=seq_lens_cpu,
        device=torch.device("cpu"),
        query_start_loc_gpu=query_start_loc_cpu,
    )
    return metadata


def test_token_embed_pooler_projects_flat_batch_once(monkeypatch):
    _patch_chunked_prefill(monkeypatch, enabled=False)

    hidden_size = 4
    lengths = [2, 3, 1]
    hidden_states = torch.randn(sum(lengths), hidden_size)
    projector = CountingLinear(hidden_size, 5)

    pooling_params = [
        PoolingParams(task="token_embed", dimensions=5, use_activation=False),
        PoolingParams(task="token_embed", dimensions=3, use_activation=True),
        PoolingParams(task="token_embed", dimensions=4, use_activation=False),
    ]
    pooling_metadata = _build_pooling_metadata(
        prompt_lens=lengths,
        pooling_params=pooling_params,
    )
    pooler = TokenPooler(
        pooling=AllPool(),
        head=TokenEmbeddingPoolerHead(
            projector=projector,
            activation=torch.tanh,
        ),
    )

    outputs = pooler(hidden_states, pooling_metadata)

    assert projector.call_count == 1
    assert projector.input_shapes == [(sum(lengths), hidden_size)]

    expected_outputs = []
    offset = 0
    for length, pooling_param in zip(lengths, pooling_params):
        chunk = hidden_states[offset : offset + length]
        embeddings = projector.linear(chunk)
        embeddings = embeddings[..., : pooling_param.dimensions]
        if pooling_param.use_activation:
            embeddings = torch.tanh(embeddings)
        expected_outputs.append(embeddings)
        offset += length

    assert len(outputs) == len(expected_outputs)
    for output, expected in zip(outputs, expected_outputs):
        assert output is not None
        torch.testing.assert_close(output, expected)


@torch.inference_mode()
def test_token_embed_pooler_projects_uniform_postprocess_once(monkeypatch):
    _patch_chunked_prefill(monkeypatch, enabled=False)

    hidden_size = 4
    lengths = [2, 2]
    hidden_states = torch.randn(sum(lengths), hidden_size)
    projector = CountingLinear(hidden_size, 6)

    pooling_params = [
        PoolingParams(task="token_embed", dimensions=4, use_activation=True),
        PoolingParams(task="token_embed", dimensions=4, use_activation=True),
    ]
    pooling_metadata = _build_pooling_metadata(
        prompt_lens=lengths,
        pooling_params=pooling_params,
    )
    pooler = TokenPooler(
        pooling=AllPool(),
        head=TokenEmbeddingPoolerHead(
            projector=projector,
            activation=torch.tanh,
        ),
    )

    outputs = pooler(hidden_states, pooling_metadata)

    assert projector.call_count == 1
    assert projector.input_shapes == [(sum(lengths), hidden_size)]

    projected = torch.tanh(projector.linear(hidden_states)[..., :4])
    expected_outputs = [projected[:2], projected[2:]]
    assert len(outputs) == len(expected_outputs)
    for output, expected in zip(outputs, expected_outputs):
        assert output is not None
        torch.testing.assert_close(output, expected)


@torch.inference_mode()
def test_token_embed_pooler_batches_finished_chunked_outputs_once(monkeypatch):
    _patch_chunked_prefill(monkeypatch, enabled=True)

    hidden_size = 4
    current_chunk_lens = [2, 2, 3]
    prompt_lens = [4, 5, 3]
    seq_lens = [4, 3, 3]
    hidden_states = torch.randn(sum(current_chunk_lens), hidden_size)
    projector = CountingLinear(hidden_size, 6)

    pooling_params = [
        PoolingParams(task="token_embed", dimensions=4, use_activation=True),
        PoolingParams(task="token_embed", dimensions=3, use_activation=False),
        PoolingParams(task="token_embed", dimensions=4, use_activation=True),
    ]
    pooling_metadata = _build_pooling_metadata(
        prompt_lens=prompt_lens,
        pooling_params=pooling_params,
        seq_lens=seq_lens,
        scheduled_lens=current_chunk_lens,
    )

    prev_req0 = torch.randn(2, hidden_size)
    prev_req1 = torch.randn(1, hidden_size)
    pooling_metadata.pooling_states[0].hidden_states_cache.append(prev_req0)
    pooling_metadata.pooling_states[1].hidden_states_cache.append(prev_req1)

    pooler = TokenPooler(
        pooling=AllPool(),
        head=TokenEmbeddingPoolerHead(
            projector=projector,
            activation=torch.tanh,
        ),
    )

    outputs = pooler(hidden_states, pooling_metadata)

    req0 = torch.concat([prev_req0, hidden_states[:2]], dim=0)
    req2 = hidden_states[4:]

    assert projector.call_count == 1
    assert projector.input_shapes == [(req0.shape[0] + req2.shape[0], hidden_size)]

    expected0 = torch.tanh(projector.linear(req0)[..., :4])
    expected2 = torch.tanh(projector.linear(req2)[..., :4])

    assert outputs[0] is not None
    torch.testing.assert_close(outputs[0], expected0)
    assert outputs[1] is None
    assert outputs[2] is not None
    torch.testing.assert_close(outputs[2], expected2)
