from typing import Callable, List, Optional, Tuple, Union
import math
import mlx.nn as nn
import mlx.core as mx
import warnings


def scaled_dot_product_attention(query:mx.array, key:mx.array, value:mx.array,
                                attn_mask:mx.array=None, dropout_p=0.0, is_causal=False, scale=None) -> mx.array:
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = mx.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = mx.tril(mx.ones((L, S), dtype=mx.bool_),0)
        attn_bias = mx.where(~temp_mask, float('-inf'), attn_bias)
        attn_bias.astype(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == mx.bool_:
            attn_bias = mx.where(~temp_mask, float('-inf'), attn_bias)
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.swapaxes(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = mx.softmax(attn_weight, dim=-1)
    attn_weight = Dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def Dropout(x:mx.array,p,training=True)->mx.array:
    p=1-p
    if p < 0 or p >= 1:
        raise ValueError(f"The dropout probability {p} is not in [0, 1)")
    if p == 1 or not training:
        return x
    mask = mx.random.bernoulli(p, x.shape)
    return (1 / p) * mask * x


def linear(input,weight,bias=None)->mx.array:
    if bias:
        return mx.addmm(bias, input, weight.T)
    else:
        return input @ weight.T


def _mha_shape_check(query: mx.array, key: mx.array, value: mx.array,
                     key_padding_mask: Optional[mx.array], attn_mask: Optional[mx.array], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.ndim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.ndim() == 3 and value.ndim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.ndim()}-D and {value.ndim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.ndim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.ndim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.ndim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.ndim()}-D tensor instead")
    elif query.ndim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.ndim() == 2 and value.ndim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.ndim()}-D and {value.ndim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.ndim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.ndim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.ndim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.ndim()}-D tensor instead")
            if attn_mask.ndim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.ndim()}-D query tensor")

    return is_batched


def _canonical_mask(
        mask: Optional[mx.array],
        mask_name: str,
        other_type: Optional[mx.Dtype],
        other_name: str,
        target_type: mx.Dtype,
        check_other: bool = True,
) -> Optional[mx.array]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = mx.issubdtype(_mask_dtype,mx.floating)
        if _mask_dtype != mx.bool_ and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = mx.where(mask, float("-inf"), mask)
    return mask


def _none_or_dtype(input: Optional[mx.array]) -> Optional[mx.Dtype]:
    if input is None:
        return None
    elif isinstance(input, mx.array):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


def _in_projection_packed(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    w: mx.array,
    b: Optional[mx.array] = None,
) -> List[mx.array]:
    r"""Perform the in-projection step of the attention operation, using packed weights.

    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = mx.expand_dims(proj.reshape(proj.shape[0:-1]+(3, E)),0).swapaxes(0, -2).squeeze(-2)
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = mx.expand_dims(kv_proj.reshape(kv_proj.shape[0:-1](2, E)),0).swapaxes(0, -2).squeeze(-2)
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.split(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.split(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    w_q: mx.array,
    w_k: mx.array,
    w_v: mx.array,
    b_q: Optional[mx.array] = None,
    b_k: Optional[mx.array] = None,
    b_v: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    r"""Perform the in-projection step of the attention operation.

    This is simply a triple of linear projections,
    with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def multi_head_attention_forward_patched(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[mx.array],
    in_proj_bias: Optional[mx.array],
    bias_k: Optional[mx.array],
    bias_v: Optional[mx.array],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: mx.array,
    out_proj_bias: Optional[mx.array],
    training: bool = True,
    key_padding_mask: Optional[mx.array] = None,
    need_weights: bool = True,
    attn_mask: Optional[mx.array] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[mx.array] = None,
    k_proj_weight: Optional[mx.array] = None,
    v_proj_weight: Optional[mx.array] = None,
    static_k: Optional[mx.array] = None,
    static_v: Optional[mx.array] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    cache=None,
) -> Tuple[mx.array, Optional[mx.array]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not nedeeded.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """

    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = mx.expand_dims(query,1)
        key = mx.expand_dims(key,1)
        value = mx.expand_dims(value,1)
        if key_padding_mask is not None:
            key_padding_mask = mx.expand_dims(key_padding_mask,0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, mx.array):
        # embed_dim can be a tensor when JIT tracing
        head_dim = mx.divide(embed_dim,num_heads)
        head_dim = mx.where(head_dim > 0, mx.floor(head_dim), mx.ceil(head_dim))
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.split(3)
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )
    if cache != None:
        if cache["first_infer"] == 1:
            cache["k"][cache["stage"]] = k
            # print(0,cache["k"].shape)
            cache["v"][cache["stage"]] = v
        else:  ###12个layer每个都要留自己的cache_kv
            # print(1,cache["k"].shape)
            cache["k"][cache["stage"]] = mx.concatenate(
                [cache["k"][cache["stage"]], k], 0
            )  ##本来时序是1，但是proj的时候可能transpose了所以时序到0维了
            cache["v"][cache["stage"]] = mx.concatenate([cache["v"][cache["stage"]], v], 0)
            # print(2, cache["k"].shape)
            src_len = cache["k"][cache["stage"]].shape[0]
            k = cache["k"][cache["stage"]]
            v = cache["v"][cache["stage"]]
            # if attn_mask is not None:
            #     attn_mask=attn_mask[-1:,]
            # print(attn_mask.shape,attn_mask)
        cache["stage"] = (cache["stage"] + 1) % cache["all_stage"]
    # print(2333,cache)
    # prep attention mask

    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.ndim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = mx.expand_dims(attn_mask, 0)
        elif attn_mask.ndim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = mx.concatenate([k, mx.tile(bias_k,(1, bsz, 1))])
        v = mx.concatenate([v, mx.tile(bias_v,(1, bsz, 1))])
        if attn_mask is not None:
            attn_mask = mx.pad(attn_mask, [(0, 0)] * (attn_mask.ndim - 1) + [(0, 1)])
        if key_padding_mask is not None:
            key_padding_mask = mx.pad(key_padding_mask, [(0, 0)] * (key_padding_mask.ndim - 1) + [(0, 1)])
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.reshape(tgt_len, bsz * num_heads, head_dim).swapaxes(0, 1)
    if static_k is None:
        k = k.reshape(k.shape[0], bsz * num_heads, head_dim).swapaxes(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_k.shape[0] == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size[0]}"
        assert (
            static_k.shape[2] == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size[2]}"
        k = static_k
    if static_v is None:
        v = v.reshape(v.shape[0], bsz * num_heads, head_dim).swapaxes(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_v.shape[0] == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size[0]}"
        assert (
            static_v.shape[2] == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size[2]}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = mx.concatenate(
            [k, mx.zeros(zero_attn_shape, dtype=k.dtype)], dim=1
        )
        v = mx.concatenate(
            [v, mx.zeros(zero_attn_shape, dtype=v.dtype)], dim=1
        )
        if attn_mask is not None:
            attn_mask = mx.pad(attn_mask, [(0, 0)] * (attn_mask.ndim - 1) + [(0, 1)])
        if key_padding_mask is not None:
            key_padding_mask = mx.pad(key_padding_mask, [(0, 0)] * (key_padding_mask.ndim - 1) + [(0, 1)])

    # update source sequence length after adjustments
    src_len = k.shape[1]

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            mx.repeat(key_padding_mask.reshape(bsz, 1, 1, src_len),num_heads,1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        assert not (
            is_causal and attn_mask is None
        ), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            attn_output_weights = mx.addmm(attn_mask,q_scaled, k.swapaxes(-2, -1))
    
        else:
            attn_output_weights = mx.matmul(q_scaled, k.swapaxes(-2, -1))
        attn_output_weights = mx.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = Dropout(attn_output_weights, p=dropout_p)

        attn_output = mx.matmul(attn_output_weights, v)

        attn_output = attn_output.swapaxes(0, 1).reshape(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.reshape(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.reshape(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.shape[0] == 1 and attn_mask.ndim() == 3:
                attn_mask = mx.expand_dims(attn_mask,0)
            else:
                attn_mask = attn_mask.reshape(bsz, num_heads, -1, src_len)

        q = q.reshape(bsz, num_heads, tgt_len, head_dim)
        k = k.reshape(bsz, num_heads, src_len, head_dim)
        v = v.reshape(bsz, num_heads, src_len, head_dim)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p
        )

        attn_output = (
            attn_output.transpose(2, 0, 1, 3).reshape(bsz * tgt_len, embed_dim)
        )

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.reshape(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
