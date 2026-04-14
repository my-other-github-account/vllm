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
    is_none_cpu: torch.Tensor | None = None

    @classmethod
    def from_lengths(
        cls,
        values: torch.Tensor,
        lengths_cpu: torch.Tensor,
        is_none_cpu: torch.Tensor | None = None,
    ) -> "RaggedTokenBatch":
        if is_none_cpu is not None:
            assert is_none_cpu.shape == lengths_cpu.shape, (
                "is_none_cpu must match lengths_cpu shape: "
                f"{tuple(is_none_cpu.shape)} != {tuple(lengths_cpu.shape)}."
            )
        return cls(
            values=values,
            cu_lengths_cpu=_make_cu_lengths_cpu(lengths_cpu),
            is_none_cpu=is_none_cpu,
        )

    @property
    def num_items(self) -> int:
        return self.cu_lengths_cpu.shape[0] - 1

    def with_values(self, values: torch.Tensor) -> "RaggedTokenBatch":
        expected_num_values = int(self.cu_lengths_cpu[-1])
        if values.ndim == 0 or values.shape[0] != expected_num_values:
            raise ValueError(
                "values must preserve the flattened token dimension: "
                f"{values.shape[0] if values.ndim > 0 else 0} "
                f"!= {expected_num_values}."
            )
        return RaggedTokenBatch(
            values=values,
            cu_lengths_cpu=self.cu_lengths_cpu,
            is_none_cpu=self.is_none_cpu,
        )

    def split(self) -> list[TokenPoolingMethodOutputItem]:
        outputs = list[TokenPoolingMethodOutputItem]()
        cu_lengths_cpu = self.cu_lengths_cpu
        is_none_cpu = self.is_none_cpu

        for i in range(self.num_items):
            start = int(cu_lengths_cpu[i])
            end = int(cu_lengths_cpu[i + 1])
            if is_none_cpu is not None and bool(is_none_cpu[i]):
                if start != end:
                    raise ValueError(
                        "Items materialized as None must have zero length: "
                        f"{start} != {end}."
                    )
                outputs.append(None)
                continue
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
        if self.enable_chunked_prefill:
            hidden_states_lst = RaggedTokenBatch.from_lengths(
                values=hidden_states,
                lengths_cpu=pooling_cursor.num_scheduled_tokens_cpu,
            ).split()

            pooling_states = pooling_metadata.pooling_states

            # If chunked_prefill is enabled
            # 1. first store the chunked hidden_states in
            # pooling_states.hidden_states_cache
            for p, hs_chunk in zip(pooling_states, hidden_states_lst):
                p.hidden_states_cache.append(hs_chunk)

            # 2. once prefill is finished, flatten the finished requests into a
            # ragged batch while preserving unfinished slots as None-equivalents.
            lengths_cpu = torch.zeros(
                len(pooling_states), dtype=torch.int64, device="cpu"
            )
            is_none_cpu = torch.ones(
                len(pooling_states), dtype=torch.bool, device="cpu"
            )
            finished_values = list[torch.Tensor]()
            for i, (p, finished) in enumerate(
                zip(pooling_states, pooling_cursor.is_finished())
            ):
                if not finished:
                    continue

                hidden_states_cache = p.hidden_states_cache
                lengths_cpu[i] = sum(chunk.shape[0] for chunk in hidden_states_cache)
                is_none_cpu[i] = False
                finished_values.extend(hidden_states_cache)
                p.clean()

            values = (
                torch.concat(finished_values, dim=0)
                if finished_values
                else hidden_states[:0]
            )
        else:
            values = hidden_states
            lengths_cpu = pooling_cursor.num_scheduled_tokens_cpu
            is_none_cpu = None

        return RaggedTokenBatch.from_lengths(
            values=values,
            lengths_cpu=lengths_cpu,
            is_none_cpu=is_none_cpu,
        )


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
                pooled_data.append(None)
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
