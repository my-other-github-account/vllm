# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.config.pooler import TokenPoolingType
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

TokenPoolingMethodOutputItem: TypeAlias = torch.Tensor | None


@dataclass
class RaggedTokenBatch:
    values: torch.Tensor
    cu_lengths_cpu: torch.Tensor

    @classmethod
    def from_lengths(
        cls,
        values: torch.Tensor,
        lengths_cpu: torch.Tensor,
    ) -> "RaggedTokenBatch":
        return cls(
            values=values,
            cu_lengths_cpu=_make_cu_lengths_cpu(lengths_cpu),
        )

    @property
    def num_items(self) -> int:
        return self.cu_lengths_cpu.shape[0] - 1

    def with_values(self, values: torch.Tensor) -> "RaggedTokenBatch":
        return RaggedTokenBatch(
            values=values,
            cu_lengths_cpu=self.cu_lengths_cpu,
        )

    def split(self) -> list[TokenPoolingMethodOutputItem]:
        outputs = list[TokenPoolingMethodOutputItem]()
        cu_lengths_cpu = self.cu_lengths_cpu

        for i in range(self.num_items):
            start = int(cu_lengths_cpu[i])
            end = int(cu_lengths_cpu[i + 1])
            outputs.append(self.values[start:end])

        return outputs


def _make_cu_lengths_cpu(lengths_cpu: torch.Tensor) -> torch.Tensor:
    # [1, 2, 3, 4] -> [0, 1, 3, 6, 10]
    lengths_cpu = lengths_cpu.to(device="cpu", dtype=torch.int64)
    cu_lengths_cpu = torch.zeros(
        lengths_cpu.shape[0] + 1, dtype=torch.int64, device="cpu"
    )
    torch.cumsum(lengths_cpu, dim=0, out=cu_lengths_cpu[1:])
    return cu_lengths_cpu


TokenPoolingMethodOutput: TypeAlias = (
    RaggedTokenBatch | list[TokenPoolingMethodOutputItem]
)


class TokenPoolingMethod(nn.Module, ABC):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolingMethodOutput:
        raise NotImplementedError


class AllPool(TokenPoolingMethod):
    def __init__(self):
        super().__init__()

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config

        self.enable_chunked_prefill = scheduler_config.enable_chunked_prefill

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        if not self.enable_chunked_prefill:
            return RaggedTokenBatch.from_lengths(
                values=hidden_states,
                lengths_cpu=pooling_cursor.num_scheduled_tokens_cpu,
            )

        hidden_states_lst = [
            hidden_states[first : last + 1]
            for first, last in zip(
                pooling_cursor.first_token_indices_gpu.tolist(),
                pooling_cursor.last_token_indices_gpu.tolist(),
            )
        ]

        pooling_states = pooling_metadata.pooling_states

        # If chunked_prefill is enabled
        # 1. first store the chunked hidden_states in pooling_states.hidden_states_cache
        for p, hs_chunk in zip(pooling_states, hidden_states_lst):
            p.hidden_states_cache.append(hs_chunk)

        # 2. Once prefill is finished, send hidden_states_cache to PoolerHead
        output_list = list[TokenPoolingMethodOutputItem]()
        for p, finished in zip(pooling_states, pooling_cursor.is_finished()):
            if finished:
                hidden_states_cache = p.hidden_states_cache
                if len(hidden_states_cache) == 1:
                    output_list.append(hidden_states_cache[0])
                else:
                    output_list.append(torch.concat(hidden_states_cache, dim=0))
                p.clean()
            else:
                output_list.append(None)

        return output_list


class StepPool(AllPool):
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooled_data = super().forward(hidden_states, pooling_metadata)
        pooled_data_lst = (
            pooled_data.split()
            if isinstance(pooled_data, RaggedTokenBatch)
            else pooled_data
        )
        prompt_token_ids = pooling_metadata.get_prompt_token_ids()
        pooling_params = pooling_metadata.pooling_params

        pooled_data = list[torch.Tensor | None]()
        for data, token_id, pooling_param in zip(
            pooled_data_lst, prompt_token_ids, pooling_params
        ):
            # for unfinished chunked prefill
            if data is None:
                pass
            else:
                step_tag_id = pooling_param.step_tag_id
                returned_token_ids = pooling_param.returned_token_ids

                if returned_token_ids is not None and len(returned_token_ids) > 0:
                    data = data[:, returned_token_ids]

                if step_tag_id is not None:
                    data = data[token_id == step_tag_id]

            pooled_data.append(data)

        return pooled_data


def get_tok_pooling_method(pooling_type: TokenPoolingType | str):
    if pooling_type == "ALL":
        return AllPool()
    if pooling_type == "STEP":
        return StepPool()

    raise NotImplementedError(f"Unknown tokenwise pooling type: {pooling_type!r}")
