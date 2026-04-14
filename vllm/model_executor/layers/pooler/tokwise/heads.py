# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.model_executor.layers.pooler import ActivationFn, ClassifierFn, ProjectorFn
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .methods import RaggedTokenBatch, TokenPoolingMethodOutputItem

TokenPoolerHeadOutputItem: TypeAlias = torch.Tensor | None


class TokenPoolerHead(nn.Module, ABC):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    @abstractmethod
    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        raise NotImplementedError

    def forward(
        self,
        pooled_data: list[TokenPoolingMethodOutputItem],
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolerHeadOutputItem]:
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.forward_chunk(d, p) for d, p in zip(pooled_data, pooling_params)]


class TokenEmbeddingPoolerHead(TokenPoolerHead):
    def __init__(
        self,
        head_dtype: torch.dtype | str | None = None,
        projector: ProjectorFn | None = None,
        activation: ActivationFn | None = None,
    ) -> None:
        super().__init__()

        self.head_dtype = head_dtype
        self.projector = projector
        self.activation = activation

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed"}

    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        embeddings = self._project_batch(pooled_data)
        return self._postprocess_embeddings(embeddings, pooling_param)

    def forward_ragged(
        self,
        pooled_data: RaggedTokenBatch,
        pooling_params: list[PoolingParams],
    ) -> list[TokenPoolerHeadOutputItem]:
        if pooled_data.num_items != len(pooling_params):
            raise ValueError(
                "pooled_data and pooling_params must have the same length: "
                f"{pooled_data.num_items} != {len(pooling_params)}."
            )

        # doing projection for all tokens in the batch
        embeddings = self._project_batch(pooled_data.values)
        active_pooling_params = self._get_present_pooling_params(
            pooled_data, pooling_params
        )
        if self._has_uniform_postprocess(active_pooling_params):
            if active_pooling_params:
                embeddings = self._postprocess_embeddings(
                    embeddings, active_pooling_params[0]
                )
            return pooled_data.with_values(embeddings).split()

        # can't apply the same postprocess, doing it separately
        pooled_outputs = pooled_data.with_values(embeddings).split()
        return [
            None
            if output is None
            else self._postprocess_embeddings(output, pooling_param)
            for output, pooling_param in zip(pooled_outputs, pooling_params)
        ]

    def _project_batch(self, pooled_data: torch.Tensor) -> torch.Tensor:
        if self.head_dtype is not None:
            pooled_data = pooled_data.to(self.head_dtype)
        # pooled_data shape: [n_tokens, hidden_size]

        # Apply ST projector
        if self.projector is not None:
            return self.projector(pooled_data)
        return pooled_data

    def _postprocess_embeddings(
        self,
        embeddings: torch.Tensor,
        pooling_param: PoolingParams,
    ) -> torch.Tensor:
        # for matryoshka representation
        embeddings = embeddings[..., : pooling_param.dimensions]

        # for normalize
        if self.activation is not None and pooling_param.use_activation:
            embeddings = self.activation(embeddings)

        # embeddings shape: [n_tokens, embedding_size]
        return embeddings

    def _has_uniform_postprocess(self, pooling_params: list[PoolingParams]) -> bool:
        """Return whether all pooling params share the same postprocess."""
        if not pooling_params:
            return True

        first_param = pooling_params[0]
        first_dimensions = first_param.dimensions
        first_use_activation = bool(first_param.use_activation)
        return all(
            param.dimensions == first_dimensions
            and bool(param.use_activation) == first_use_activation
            for param in pooling_params[1:]
        )

    def _get_present_pooling_params(
        self,
        pooled_data: RaggedTokenBatch,
        pooling_params: list[PoolingParams],
    ) -> list[PoolingParams]:
        if pooled_data.is_none_cpu is None:
            return pooling_params
        return [
            pooling_param
            for pooling_param, is_none in zip(pooling_params, pooled_data.is_none_cpu)
            if not bool(is_none)
        ]


class TokenClassifierPoolerHead(TokenPoolerHead):
    def __init__(
        self,
        classifier: ClassifierFn | None = None,
        logit_bias: float | None = None,
        logit_scale: float | None = None,
        head_dtype: torch.dtype | str | None = None,
        activation: ActivationFn | None = None,
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.logit_bias = logit_bias
        self.logit_scale = logit_scale
        self.head_dtype = head_dtype
        self.activation = activation

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_classify"}

    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        if self.head_dtype is not None:
            pooled_data = pooled_data.to(self.head_dtype)
        # hidden_states shape: [n_token, hidden_size]

        if self.classifier is not None:
            logits = self.classifier(pooled_data)
        else:
            logits = pooled_data
        # logits shape: [n_token, num_labels]

        if self.logit_bias is not None:
            logits -= self.logit_bias
        if self.logit_scale is not None:
            logits *= self.logit_scale

        if self.activation is not None and pooling_param.use_activation:
            logits = self.activation(logits)

        # logits shape: [n_token, num_labels]
        return logits
