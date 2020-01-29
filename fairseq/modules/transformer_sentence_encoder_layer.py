# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)


class Adapter(nn.Module):
    # see https://arxiv.org/pdf/1902.00751.pdf
    def __init__(self, embedding_dim, adapter_dim, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, adapter_dim)
        self.fc2 = nn.Linear(adapter_dim, embedding_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        return self.fc2(self.activation_fn(self.fc1(x))) + x


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        export: bool = False,
        self_attn_adapter_dim: int = None,
        ffn_adapter_dim: int = None
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self_attn_adapter_dim:
            self.self_attn_adapter = Adapter(self.embedding_dim, self_attn_adapter_dim, activation_fn)

        if ffn_adapter_dim:
            self.ff_adapter = Adapter(self.embedding_dim, ffn_adapter_dim, activation_fn)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        extra_attention_mask: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        assert self_attn_mask is None
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=extra_attention_mask,
        )
        if hasattr(self, 'self_attn_adapter'):
            x = self.self_attn_adapter(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if hasattr(self, 'ff_adapter'):
            x = self.ff_adapter(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
