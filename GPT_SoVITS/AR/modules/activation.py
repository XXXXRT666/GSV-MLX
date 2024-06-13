# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/activation.py
from typing import Optional
from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import Module
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
import math
from torch.nn import functional as F
from AR.modules.patched_mha_with_cache import multi_head_attention_forward_patched

multi_head_attention_forward = multi_head_attention_forward_patched
import mlx.core as mx
import mlx.nn as nn


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use a special optimized implementation if all of the following
    conditions are met:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - dropout is 0
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - at most one of ``key_padding_mask`` or ``attn_mask`` is passed
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed

    If the optimized implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ["batch_first"]
    bias_k: Optional[mx.array]
    bias_v: Optional[mx.array]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=True,
        linear1_cls=nn.Linear,
        linear2_cls=nn.Linear,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if add_bias_kv:
            self.bias_k = mx.zeros((1, 1, embed_dim))
            self.bias_v = mx.zeros((1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        if linear1_cls == nn.Linear:
            if not self._qkv_same_embed_dim:
                self.q_proj_weight = mx.zeros((embed_dim, embed_dim))
                self.k_proj_weight = mx.zeros((embed_dim, self.kdim))
                self.v_proj_weight = mx.zeros((embed_dim, self.vdim))

                self.in_proj_weight = None
            else:
                self.in_proj_weight = mx.zeros((3 * embed_dim, embed_dim))

                self.q_proj_weight = None
                self.k_proj_weight = None
                self.v_proj_weight = None

            if bias:
                self.in_proj_bias = mx.zeros(3 * embed_dim)

            else:
                self.in_proj_bias = None
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

            self._reset_parameters()
        else:
            if not self._qkv_same_embed_dim:
                raise NotImplementedError
            else:
                self.in_proj_linear = linear1_cls(
                    embed_dim, 3 * embed_dim, bias=bias, 
                )
                self.in_proj_weight = self.in_proj_linear.weight

                self.q_proj_weight = None
                self.k_proj_weight = None
                self.v_proj_weight = None

                if bias:
                    self.in_proj_bias = self.in_proj_linear.bias
                else:
                    self.in_proj_bias = None

            self.out_proj = linear2_cls(
                embed_dim, embed_dim, bias=bias, 
            )

            if self.bias_k is not None:
                self.bias_k = nn.init.glorot_normal()(self.bias_k, gain=1.0)
            if self.bias_v is not None:
                self.bias_v =nn.init.glorot_normal()(self.bias_v, gain=1.0)

        self.add_zero_attn = add_zero_attn

    def _reset_parameters(self):
        uniform = nn.init.glorot_uniform()
        constant = nn.init.constant(0.0)
        normal = nn.init.glorot_normal()
        if self._qkv_same_embed_dim:
            self.in_proj_weight = uniform(self.in_proj_weight, gain=1.0)
        else:
            self.q_proj_weight = uniform(self.q_proj_weight, gain=1.0)
            self.k_proj_weight = uniform(self.k_proj_weight, gain=1.0)
            self.v_proj_weight = uniform(self.v_proj_weight, gain=1.0)

        if self.in_proj_bias is not None:
            self.in_proj_bias = constant(self.in_proj_bias)
            self.out_proj.bias = constant(self.out_proj.bias)

        if self.bias_k is not None:
            normal(self.bias_k, gain=1.0)
        if self.bias_v is not None:
            normal(self.bias_v, gain=1.0)


    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        key_padding_mask: Optional[mx.array] = None,
        need_weights: bool = True,
        attn_mask: Optional[mx.array] = None,
        average_attn_weights: bool = True,
        cache=None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.ndim == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != mx.bool_ and not mx.issubdtype(key_padding_mask, mx.floating):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        why_not_fast_path = ""
        if not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.ndim of 3 but got {query.ndim}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same mx.array)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif (
            self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype
        ):
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"

            

        # if why_not_fast_path:
        #     raise RuntimeError(why_not_fast_path)
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.swapaxes(1, 0)
                else:
                    query, key = [x.swapaxes(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.swapaxes(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                cache=cache,
            )
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                cache=cache,
            )
        if self.batch_first and is_batched:
            return attn_output.swapaxes(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
