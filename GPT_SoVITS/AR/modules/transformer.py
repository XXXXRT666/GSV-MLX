# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/transformer.py
import copy
import numbers
from functools import partial
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from AR.modules.activation import MultiheadAttention
from AR.modules.scaling import BalancedDoubleSwish
from torch import Tensor
from torch.nn import functional as F
import mlx.core as mx
import mlx.nn as nn

_shape_t = Union[int, List[int], torch.Size]


class LayerNorm(nn.Module):
    eps: float
    affine: bool
    bias:bool
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool =True,
    ) -> None:
        super(LayerNorm, self).__init__()
        self._normalized_shape = normalized_shape
        self._eps = eps
        self._affine = elementwise_affine
        self.bias = bias
        self.LayerNorm = nn.LayerNorm(normalized_shape,eps,elementwise_affine,bias)


    def __call__(self, input:mx.array, embedding: Any = None) -> mx.array:
        if isinstance(input, tuple):
            input, embedding = input
            return (self.LayerNorm(input),embedding)

        assert embedding is None
        return self.LayerNorm(input)


class IdentityNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ) -> None:
        super(IdentityNorm, self).__init__()

    def _call__(self, input: mx.array, embedding: Any = None) -> mx.array:
        if isinstance(input, tuple):
            return input

        assert embedding is None
        return input


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.freeze(keys=self.__constants__)

    def __call__(
        self,
        src: mx.array,
        mask: Optional[mx.array] = None,
        src_key_padding_mask: Optional[mx.array] = None,
        return_layer_states: bool = False,
        cache=None,
    ) -> mx.array:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """
        if return_layer_states:
            layer_states = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    cache=cache,
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output

        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                cache=cache,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[mx.array], mx.array]] = nn.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        dtype=None,
        linear1_self_attention_cls: nn.Module = nn.Linear,
        linear2_self_attention_cls: nn.Module = nn.Linear,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        layer_norm_cls: nn.Module = LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,  # 512 16
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
        )

        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.freeze(keys=self.__constants__)
        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps)
        norm2 = layer_norm_cls(d_model, eps=layer_norm_eps)

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2
            

    def __call__(
        self,
        src: mx.array,
        src_mask: Optional[mx.array] = None,
        src_key_padding_mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != mx.bool_ and not mx.issubdtype(src_key_padding_mask.dtype,mx.floating):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding),
                src_mask,
                src_key_padding_mask,
                cache=cache,
            )
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, cache=cache),
                stage_embedding,
            )
            x = self.norm2(x + self._ff_block(x), stage_embedding)

        if is_src_tuple:
            return (x, stage_embedding)
        return x

    # self-attention block
    def _sa_block(
        self,
        x: mx.array,
        attn_mask: Optional[mx.array],
        key_padding_mask: Optional[mx.array],
        cache=None,
    ) -> mx.array:
        # print(x.shape,attn_mask.shape,key_padding_mask)
        # torch.Size([1, 188, 512]) torch.Size([188, 188]) None
        # import os
        # os._exit(23333)
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            cache=cache,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: mx.array) -> mx.array:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def __call__(self, input: mx.array, embedding: mx.array = None) -> mx.array:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = mx.split(
                self.project_layer(embedding),
                indices_or_sections=self.d_model,
                axis=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = mx.split(
            self.project_layer(embedding),
            indices_or_sections=self.d_model,
            axis=-1,
        )
        return weight * self.norm(input) + bias


def _get_clones(module, N):
    return mx.array([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.relu
    if activation == "gelu":
        return nn.gelu
    if activation == "glu":
        return nn.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")