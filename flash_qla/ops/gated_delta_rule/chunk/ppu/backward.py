# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Decomposed PPU operators used by the official FlashQLA backward path."""

import os

import torch

from flash_qla.utils import (
    pad_and_reshape,
    pack,
    unpack,
    fill_last_chunk_of_g,
    prepare_chunk_offsets,
)

def torch_cumsum(
    x: torch.Tensor,  # [B, T, H]
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 64,
    reverse: bool = False,
):
    native_reverse = (
        reverse
        and cu_seqlens is None
        and x.is_cuda
        and x.dtype == torch.float32
        and x.is_contiguous()
        and x.shape[1] % chunk_size == 0
        and chunk_size == 64
        and os.getenv("FLASHQLA_PPU_NATIVE_REVERSE_CUMSUM", "1") != "0"
    )
    if native_reverse:
        from . import native

        if hasattr(native, "reverse_chunk_cumsum") and native.is_available():
            return native.reverse_chunk_cumsum(x)

    if cu_seqlens is not None:
        x = unpack(x, cu_seqlens)

    raw_shape = x.shape

    x = pad_and_reshape(x, dim=1, chunk_size=chunk_size)

    if reverse:
        x = torch.flip(x, dims=(2,))
        x = x.cumsum(dim=2)
        x = torch.flip(x, dims=(2,))
    else:
        x = x.cumsum(dim=2)
    x = x.reshape(raw_shape[0], -1, *raw_shape[2:])
    x = x[:, :raw_shape[1]]

    if cu_seqlens is not None:
        x = pack(x, cu_seqlens)
    return x


def _masked_chunk_decay(g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Form exp(g[row] - g[col]) without evaluating masked overflow lanes."""
    difference = g[:, :, :, None, :] - g[:, :, None, :, :]
    difference = difference.masked_fill(
        mask[None, None, :, :, None], -torch.inf
    )
    return torch.exp(difference)


def torch_w_u_fwd(
    k: torch.Tensor,  # [B, T, Hk, K]
    v: torch.Tensor,  # [B, T, Hv, V]
    g: torch.Tensor,  # [B, T, Hv]
    beta: torch.Tensor,  # [B, T, Hv]
    A: torch.Tensor,  # [B, T, Hv, D]
    cu_seqlens: torch.Tensor = None,
):
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        A = unpack(A, cu_seqlens)
        beta = unpack(beta, cu_seqlens)
        g = unpack(g, cu_seqlens)

    batch_size, num_tokens, _, chunk_size = A.shape
    _, _, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = v.shape

    native_wu = (
        cu_seqlens is None
        and k.is_cuda
        and chunk_size == 64
        and head_dim_k == head_dim_v == 128
        and num_tokens % chunk_size == 0
        and os.getenv("FLASHQLA_PPU_NATIVE_BWD_WU", "1") != "0"
        and all(
            tensor.dtype == torch.float32 and tensor.is_contiguous()
            for tensor in (k, v, A, g, beta)
        )
    )
    if native_wu:
        from . import native

        if native.is_flash_qla_wu_available():
            return native.flash_qla_wu_forward_bf16_128(k, v, A, g, beta)

    head_major = (
        cu_seqlens is None
        and k.is_cuda
        and chunk_size == 64
        and head_dim_k == head_dim_v == 128
        and num_tokens % chunk_size == 0
        and os.getenv("FLASHQLA_PPU_HEAD_MAJOR_WU", "1") != "0"
    )
    if head_major:
        chunks = num_tokens // chunk_size

        def head_vector(x):
            return x.reshape(
                batch_size, chunks, chunk_size, *x.shape[2:]
            ).permute(0, 1, 3, 2, 4).contiguous()

        def head_scalar(x):
            return x.reshape(
                batch_size, chunks, chunk_size, num_v_heads
            ).permute(0, 1, 3, 2).contiguous()

        k_head = k.reshape(
            batch_size, chunks, chunk_size, num_k_heads, head_dim_k
        ).permute(0, 1, 3, 2, 4)
        if num_k_heads != num_v_heads:
            k_head = k_head.repeat_interleave(
                num_v_heads // num_k_heads, dim=2
            )
        else:
            k_head = k_head.contiguous()
        v_head = head_vector(v)
        A_head = head_vector(A)
        beta_head = head_scalar(beta)
        g_head = head_scalar(g)
        groups = batch_size * chunks * num_v_heads
        A_matrix = A_head.reshape(groups, chunk_size, chunk_size)
        w_head = torch.bmm(
            A_matrix,
            k_head.reshape(groups, chunk_size, head_dim_k)
            * (beta_head * g_head.exp()).reshape(groups, chunk_size, 1),
        ).reshape(
            batch_size, chunks, num_v_heads, chunk_size, head_dim_k
        )
        u_head = torch.bmm(
            A_matrix,
            v_head.reshape(groups, chunk_size, head_dim_v)
            * beta_head.reshape(groups, chunk_size, 1),
        ).reshape(
            batch_size, chunks, num_v_heads, chunk_size, head_dim_v
        )

        def token_vector(x):
            return x.permute(0, 1, 3, 2, 4).contiguous().reshape(
                batch_size, num_tokens, num_v_heads, x.shape[-1]
            )

        return token_vector(w_head), token_vector(u_head)

    if num_k_heads != num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k_beta = pad_and_reshape(
        k * beta.unsqueeze(-1) * g.exp().unsqueeze(-1), dim=1, chunk_size=chunk_size
    )  # [B, N, C, Hv, K]
    v_beta = pad_and_reshape(
        v * beta.unsqueeze(-1), dim=1, chunk_size=chunk_size
    )  # [B, N, C, Hv, V]
    A = pad_and_reshape(A, dim=1,chunk_size=chunk_size)

    w = torch.einsum("bnchd, bndhk -> bnchk", A, k_beta).reshape(
        (batch_size, -1, num_v_heads, head_dim_k)
    )[:, :num_tokens]
    u = torch.einsum("bnchd, bndhk -> bnchk", A, v_beta).reshape(
        (batch_size, -1, num_v_heads, head_dim_v)
    )[:, :num_tokens]

    if cu_seqlens is not None:
        w = pack(w, cu_seqlens)
        u = pack(u, cu_seqlens)
    return w, u


def torch_chunk_gdr_fwd(
    k: torch.Tensor,  # [B, T, Hk, K]
    w: torch.Tensor,  # [B, T, Hv, K]
    u: torch.Tensor,  # [B, T, Hv, V]
    g: torch.Tensor,  # [B, T, Hv]
    initial_state: torch.Tensor = None,  # [B, Hv, K, V]
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        w = unpack(w, cu_seqlens)
        u = unpack(u, cu_seqlens)
        g = unpack(g, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = u.shape

    if num_k_heads != num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    u = pad_and_reshape(u, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    if initial_state is None:
        last_state = torch.zeros(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            dtype=g.dtype,
            device=g.device,
        )
    else:
        last_state = initial_state.to(g.dtype, copy=True)

    h, vn = [], []
    for i in range(k.shape[1]):
        h.append(last_state)
        v_new = u[:, i] - torch.einsum("bchk, bhkv -> bchv", w[:, i], last_state)
        vn.append(v_new)
        last_state = last_state * g[:, i, -1, :, None, None].exp()
        last_state = last_state + torch.einsum(
            "bchk, bchv -> bhkv",
            k[:, i] * (g[:, i, -1:, :, None] - g[:, i, :, :, None]).exp(),
            v_new,
        )
    h = torch.stack(h, dim=1).contiguous()
    vn = (
        torch.stack(vn, dim=1)
        .reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
        .contiguous()
    )

    if cu_seqlens is not None:
        vn = pack(vn, cu_seqlens)
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        h = pack(h, chunk_offsets)

    return h, vn, last_state


def torch_chunk_dv_bwd(
    q: torch.Tensor,  # [B, T, Hk, K]
    k: torch.Tensor,  # [B, T, Hk, K]
    g: torch.Tensor,  # [B, T, Hv]
    do: torch.Tensor,  # [B, T, Hv, V]
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        g = unpack(g, cu_seqlens)
        do = unpack(do, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = do.shape

    if num_k_heads != num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    scale = scale or head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    q = q * scale

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device),
        diagonal=1,
    )
    decay_mask = _masked_chunk_decay(g, mask)  # [B, N, C, D, Hv]

    attn = torch.einsum("bnchk, bndhk -> bncdh", q, k) * decay_mask
    dv = torch.einsum("bncdh, bnchv -> bndhv", attn, do)

    dv = dv.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
    if cu_seqlens is not None:
        dv = pack(dv, cu_seqlens)
    return dv


def torch_chunk_gdr_bwd(
    q: torch.Tensor,  # [B, T, Hk, K]
    k: torch.Tensor,  # [B, T, Hk, K]
    w: torch.Tensor,  # [B, T, Hv, K]
    g: torch.Tensor,  # [B, T, Hv]
    do: torch.Tensor,  # [B, T, Hv, V]
    dv: torch.Tensor,  # [B, T, Hv, V]
    h0: torch.Tensor = None,  # [B, Hv, K, V]
    dht: torch.Tensor = None,  # [B, Hv, K, V]
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    chunk_size: int = 64,
    return_head_major_intermediates: bool = False,
):
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        w = unpack(w, cu_seqlens)
        g = unpack(g, cu_seqlens)
        do = unpack(do, cu_seqlens)
        dv = unpack(dv, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = do.shape

    head_repeat = num_v_heads // num_k_heads
    grouped_gqa = (
        head_repeat > 1
        and cu_seqlens is None
        and not return_head_major_intermediates
        and os.getenv("FLASHQLA_PPU_HEAD_MAJOR_STATE_BWD", "0") == "0"
        and os.getenv("FLASHQLA_PPU_GROUPED_STATE_BWD", "1") != "0"
    )
    if num_k_heads != num_v_heads and not grouped_gqa:
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    scale = scale or head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    dv = pad_and_reshape(dv, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    q = q * scale
    head_major = (
        cu_seqlens is None
        and chunk_size == 64
        and head_dim_k == head_dim_v == 128
        and num_tokens % chunk_size == 0
        and not grouped_gqa
        and (
            return_head_major_intermediates
            or os.getenv("FLASHQLA_PPU_HEAD_MAJOR_STATE_BWD", "0") != "0"
        )
    )
    if head_major:
        def head_vector(x):
            return x.permute(0, 1, 3, 2, 4).contiguous()

        chunks = q.shape[1]
        batch_heads = batch_size * num_v_heads
        q_head = head_vector(q)
        k_head = head_vector(k)
        w_head = head_vector(w)
        do_head = head_vector(do)
        dv_head = head_vector(dv)
        g_head = g.permute(0, 1, 3, 2).contiguous()
        exp_g = g_head.exp()
        exp_g_to_last = (g_head[..., -1:] - g_head).exp()

        q_matrix = q_head.reshape(-1, chunk_size, head_dim_k)
        do_matrix = do_head.reshape(-1, chunk_size, head_dim_v)
        dstate_inter = torch.bmm(
            (q_matrix * exp_g.reshape(-1, chunk_size, 1)).transpose(-1, -2),
            do_matrix,
        ).reshape(
            batch_size, chunks, num_v_heads, head_dim_k, head_dim_v
        )
        if dht is None:
            dstate = torch.zeros(
                batch_size, num_v_heads, head_dim_k, head_dim_v,
                dtype=g.dtype, device=g.device,
            )
        else:
            dstate = dht.to(g.dtype, copy=True)

        dh = []
        for chunk in reversed(range(chunks)):
            dh.insert(0, dstate)
            k_matrix = k_head[:, chunk].reshape(
                batch_heads, chunk_size, head_dim_k
            )
            dstate_matrix = dstate.reshape(
                batch_heads, head_dim_k, head_dim_v
            )
            dv_chunk = dv_head[:, chunk].reshape(
                batch_heads, chunk_size, head_dim_v
            )
            dv_chunk = dv_chunk + torch.bmm(
                k_matrix * exp_g_to_last[:, chunk].reshape(
                    batch_heads, chunk_size, 1
                ),
                dstate_matrix,
            )
            dv_head[:, chunk] = dv_chunk.reshape(
                batch_size, num_v_heads, chunk_size, head_dim_v
            )
            w_matrix = w_head[:, chunk].reshape(
                batch_heads, chunk_size, head_dim_k
            )
            dstate = (
                dstate * exp_g[:, chunk, :, -1, None, None]
                + dstate_inter[:, chunk]
                - torch.bmm(
                    w_matrix.transpose(-1, -2), dv_chunk
                ).reshape(
                    batch_size, num_v_heads, head_dim_k, head_dim_v
                )
            )
        dh = torch.stack(dh, dim=1).contiguous()
        if return_head_major_intermediates:
            bridge = (q_head, k_head, do_head, dv_head, g_head)
            return dh, None if h0 is None else dstate, dv_head, bridge
        dv = dv_head.permute(0, 1, 3, 2, 4).contiguous().reshape(
            batch_size, num_tokens, num_v_heads, head_dim_v
        )
        return dh, None if h0 is None else dstate, dv

    exp_g = g.exp()
    exp_g_last = exp_g[:, :, -1].contiguous()
    exp_g_to_last = (
        g[:, :, -1:, :, None] - g[:, :, :, :, None]
    ).exp()

    if dht is None:
        dstate = torch.zeros(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            dtype=g.dtype,
            device=g.device,
        )
    else:
        dstate = dht.to(g.dtype, copy=True)
    if grouped_gqa:
        chunks = q.shape[1]
        q_group = q.permute(0, 1, 3, 2, 4).unsqueeze(3)
        do_group = do.reshape(
            batch_size, chunks, chunk_size,
            num_k_heads, head_repeat, head_dim_v,
        ).permute(0, 1, 3, 4, 2, 5)
        exp_group = exp_g.reshape(
            batch_size, chunks, chunk_size, num_k_heads, head_repeat
        ).permute(0, 1, 3, 4, 2)
        dstate_inter = torch.matmul(
            q_group.transpose(-1, -2),
            do_group * exp_group.unsqueeze(-1),
        ).reshape(
            batch_size, chunks, num_v_heads, head_dim_k, head_dim_v
        )
    else:
        dstate_inter = torch.einsum(
            "bnchk, bnchv -> bnhkv", q * exp_g.unsqueeze(-1), do
        )
    fused_reverse_update = (
        cu_seqlens is None
        and os.getenv("FLASHQLA_PPU_FUSED_REVERSE_UPDATE", "1") != "0"
    )
    if fused_reverse_update:
        from . import native

        fused_reverse_update = (
            native.is_available()
            and hasattr(native, "reverse_state_update")
            and dstate_inter.is_contiguous()
            and exp_g_last.is_contiguous()
        )

    fused_reverse_store = (
        fused_reverse_update
        and os.getenv("FLASHQLA_PPU_FUSED_REVERSE_STORE", "1") != "0"
        and hasattr(native, "reverse_state_update_store_inplace")
    )
    native_grouped_dv_update = False
    if grouped_gqa and os.getenv(
        "FLASHQLA_PPU_NATIVE_GROUPED_DV_UPDATE", "1"
    ) == "1":
        from . import native

        native_grouped_dv_update = (
            native.is_grouped_state_dv_update_available()
            and dv.is_contiguous()
            and exp_g_to_last.is_contiguous()
        )
    dh = torch.empty_like(dstate_inter) if fused_reverse_store else []
    for i in reversed(range(k.shape[1])):
        if not fused_reverse_store:
            dh.insert(0, dstate)
        if grouped_gqa:
            k_group = k[:, i].permute(0, 2, 1, 3).unsqueeze(2)
            dstate_group = dstate.reshape(
                batch_size, num_k_heads, head_repeat,
                head_dim_k, head_dim_v,
            )
            dv_update = torch.matmul(k_group, dstate_group)
            if native_grouped_dv_update:
                native.grouped_state_dv_update(
                    dv, dv_update.contiguous(), exp_g_to_last, i
                )
            else:
                decay_group = exp_g_to_last[:, i].reshape(
                    batch_size, chunk_size,
                    num_k_heads, head_repeat, 1,
                ).permute(0, 2, 3, 1, 4)
                dv_update *= decay_group
                dv[:, i] += dv_update.permute(0, 3, 1, 2, 4).reshape(
                    batch_size, chunk_size, num_v_heads, head_dim_v
                )
        else:
            dv[:, i] += torch.einsum(
                "bchk, bhkv -> bchv",
                k[:, i] * exp_g_to_last[:, i],
                dstate,
            )
        product = torch.einsum("bchk, bchv -> bhkv", w[:, i], dv[:, i])
        if fused_reverse_store:
            dstate = native.reverse_state_update_store_inplace(
                dstate, dstate_inter, product.contiguous(),
                exp_g_last, dh, i,
            )
        elif fused_reverse_update:
            dstate = native.reverse_state_update(
                dstate.contiguous(), dstate_inter, product.contiguous(),
                exp_g_last, i,
            )
        else:
            dstate = dstate * exp_g_last[:, i, :, None, None]
            dstate = dstate + dstate_inter[:, i] - product
    if not fused_reverse_store:
        dh = torch.stack(dh, dim=1).contiguous()

    dh0 = None if h0 is None else dstate
    dv = dv.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
    if cu_seqlens is not None:
        dv = pack(dv, cu_seqlens)
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        dh = pack(dh, chunk_offsets)
    return dh, dh0, dv


def torch_chunk_dqkwg_bwd(
    q: torch.Tensor,  # [B, T, Hk, K]
    k: torch.Tensor,  # [B, T, Hk, K]
    v: torch.Tensor,  # [B, T, Hv, V]
    w: torch.Tensor,  # [B, T, Hv, K]
    g: torch.Tensor,  # [B, T, Hv]
    h: torch.Tensor,  # [B, N, Hv, K, V]
    dv: torch.Tensor,  # [B, T, Hv, V]
    do: torch.Tensor,  # [B, T, Hv, V]
    dh: torch.Tensor,  # [B, N, Hv, K, V]
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    chunk_size: int = 64,
    precomputed_dqkw: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    precomputed_dqkwg: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None,
    return_head_major_intermediates: bool = False,
    head_major_inputs: tuple[torch.Tensor, ...] | None = None,
    reduce_dq_gqa: bool = False,
):
    if head_major_inputs is not None and cu_seqlens is not None:
        raise ValueError("head-major dQKWg bridge does not support variable lengths")
    if precomputed_dqkwg is not None:
        if cu_seqlens is not None:
            raise ValueError("native dQKWg does not support variable lengths")
        return precomputed_dqkwg
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        w = unpack(w, cu_seqlens)
        g = unpack(g, cu_seqlens)
        do = unpack(do, cu_seqlens)
        dv = unpack(dv, cu_seqlens)
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        h = unpack(h, chunk_offsets)
        dh = unpack(dh, chunk_offsets)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = do.shape
    head_repeat = num_v_heads // num_k_heads
    scale = scale or head_dim_k ** (-0.5)

    head_major = (
        precomputed_dqkw is None
        and cu_seqlens is None
        and chunk_size == 64
        and head_dim_k == head_dim_v == 128
        and num_tokens % chunk_size == 0
        and os.getenv("FLASHQLA_PPU_HEAD_MAJOR_DQKWG", "1") != "0"
    )
    q_head = None
    k_head = None
    v_head = None
    do_head = None
    dv_head = None
    g_head = None
    q_group = None
    k_group = None
    grouped_head_major = (
        head_major
        and head_repeat > 1
        and head_major_inputs is None
        and os.getenv("FLASHQLA_PPU_GROUPED_DQKWG", "0") != "0"
    )
    direct_grouped_qk = (
        head_major
        and head_major_inputs is None
        and not grouped_head_major
        and os.getenv("FLASHQLA_PPU_DIRECT_GQA_HEAD", "1") != "0"
    )
    native_head_layout_mode = os.getenv(
        "FLASHQLA_PPU_NATIVE_DQKWG_LAYOUT", "auto"
    )
    native_head_layout = (
        direct_grouped_qk
        and native_head_layout_mode != "0"
        and (
            native_head_layout_mode == "1"
            or num_tokens >= 1024
        )
    )
    if native_head_layout:
        from . import native

        native_head_layout = native.is_flash_qla_dqkwg_layout_available()
    if native_head_layout:
        (
            q_head,
            k_head,
            v_head,
            do_head,
            dv_head,
            g_head,
        ) = native.prepare_dqkwg_head_major(
            q.contiguous(), k.contiguous(), v.contiguous(), do.contiguous(),
            dv.contiguous(), g.contiguous(),
        )
    if grouped_head_major:
        def grouped_qk(x):
            return x.reshape(
                batch_size, num_tokens // chunk_size,
                chunk_size, num_k_heads, head_dim_k,
            ).permute(0, 1, 3, 2, 4).contiguous()

        q_group = grouped_qk(q)
        k_group = grouped_qk(k)
    elif direct_grouped_qk and not native_head_layout:
        def grouped_qk_head(x):
            x = x.reshape(
                batch_size, num_tokens // chunk_size,
                chunk_size, num_k_heads, head_dim_k,
            ).permute(0, 1, 3, 2, 4)
            if head_repeat > 1:
                return x.repeat_interleave(head_repeat, dim=2)
            return x.contiguous()

        q_head = grouped_qk_head(q)
        k_head = grouped_qk_head(k)
    elif head_major_inputs is None:
        if head_repeat > 1:
            q = q.repeat_interleave(head_repeat, dim=2)
            k = k.repeat_interleave(head_repeat, dim=2)
        q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
        k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    if head_major_inputs is None:
        g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
        do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
        dv = pad_and_reshape(dv, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
        g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    if head_major:
        def head_vector(x):
            return x.permute(0, 1, 3, 2, 4).contiguous()

        def token_vector(x):
            return x.permute(0, 1, 3, 2, 4).contiguous().reshape(
                batch_size, num_tokens, num_v_heads, head_dim_k
            )

        def token_q_vector(x):
            if head_repeat == 1 or not reduce_dq_gqa:
                return token_vector(x)
            # dQ is not consumed by WY.  Reduce its GVA replicas while it is
            # still chunk/head-major, then materialize the smaller Hq output
            # instead of copying an Hv tensor only to reduce it afterwards.
            return x.reshape(
                batch_size,
                chunks,
                num_k_heads,
                head_repeat,
                chunk_size,
                head_dim_k,
            ).sum(dim=3).permute(0, 1, 3, 2, 4).contiguous().reshape(
                batch_size, num_tokens, num_k_heads, head_dim_k
            )

        chunks = num_tokens // chunk_size
        if grouped_head_major:
            head_shape = (
                batch_size, chunks, num_v_heads, chunk_size, head_dim_k
            )
            do_head = head_vector(do)
            dv_head = head_vector(dv)
            v_head = head_vector(v)
            g_head = g.permute(0, 1, 3, 2).contiguous()
            do_group = do_head.reshape(
                batch_size, chunks, num_k_heads, head_repeat,
                chunk_size, head_dim_v,
            )
            dv_group = dv_head.reshape_as(do_group)
            v_group = v_head.reshape_as(do_group)
            g_group = g_head.reshape(
                batch_size, chunks, num_k_heads, head_repeat, chunk_size
            )
            h_group = h.reshape(
                batch_size, chunks, num_k_heads, head_repeat,
                head_dim_k, head_dim_v,
            )
            dh_group = dh.reshape_as(h_group)
            exp_g = g_group.exp()
            g_last = g_group[..., -1:]

            dg_last = (h_group * dh_group).sum(dim=(-1, -2))
            dg_last *= exp_g[..., -1]
            ds = torch.matmul(do_group, v_group.transpose(-1, -2))
            dq = torch.matmul(do_group, h_group.transpose(-1, -2))
            dk = torch.matmul(v_group, dh_group.transpose(-1, -2))
            dw = -torch.matmul(dv_group, h_group.transpose(-1, -2))
            dq *= exp_g.unsqueeze(-1) * scale
            dk *= (g_last - g_group).exp().unsqueeze(-1)

            q_dq = (q_group.unsqueeze(3) * dq).sum(dim=-1)
            k_dk = (k_group.unsqueeze(3) * dk).sum(dim=-1)
            dg = q_dq - k_dk
            dg_last += k_dk.sum(dim=-1)
            native_dp_decay = (
                os.getenv("FLASHQLA_PPU_NATIVE_DQKWG_DECAY", "1") != "0"
            )
            if native_dp_decay:
                from . import native

                native_dp_decay = (
                    hasattr(native, "dqkwg_apply_decay_64")
                    and native.is_available()
                )
            if native_dp_decay:
                ds = native.dqkwg_apply_decay_64(
                    ds.reshape(-1, chunk_size, chunk_size).contiguous(),
                    g_head,
                    scale,
                ).reshape_as(ds)
            else:
                lower = torch.tril(
                    torch.ones(
                        chunk_size, chunk_size,
                        dtype=torch.bool, device=k.device,
                    )
                )
                decay = (
                    g_group[..., :, None] - g_group[..., None, :]
                ).masked_fill(~lower, -torch.inf).exp()
                ds *= decay * scale
            p = torch.matmul(q_group, k_group.transpose(-1, -2))
            native_gate_product = (
                native_dp_decay
                and hasattr(native, "dqkwg_accumulate_gate_product_64")
                and os.getenv("FLASHQLA_PPU_NATIVE_DQKWG_GATE", "0") != "0"
            )
            if native_gate_product:
                ds_flat = ds.reshape(-1, chunk_size, chunk_size)
                p_flat = p.unsqueeze(3).expand_as(ds).reshape_as(ds_flat)
                dg_flat = dg.reshape(-1, chunk_size)
                dg = native.dqkwg_accumulate_gate_product_64(
                    ds_flat.contiguous(), p_flat.contiguous(),
                    dg_flat.contiguous(),
                ).reshape_as(dg)
            else:
                ds2 = ds * p.unsqueeze(3)
                dg += ds2.sum(dim=-1) - ds2.sum(dim=-2)
            dq += torch.matmul(ds, k_group.unsqueeze(3))
            dk += torch.matmul(ds.transpose(-1, -2), q_group.unsqueeze(3))
            dg[..., -1] += dg_last

            dq_head = dq.reshape(head_shape)
            dk_head = dk.reshape(head_shape)
            dw_head = dw.reshape(head_shape)
            dg_head = dg.reshape(
                batch_size, chunks, num_v_heads, chunk_size
            )
            dq_token = token_q_vector(dq_head)
            if return_head_major_intermediates:
                k_bridge = k_group.unsqueeze(3).expand(
                    -1, -1, -1, head_repeat, -1, -1
                ).reshape(head_shape).contiguous()
                bridge = (
                    k_bridge,
                    dv_group.reshape(head_shape),
                    dk_head,
                    dw_head,
                    g_head,
                    dg_head,
                )
                return dq_token, dk_head, dw_head, dg_head, bridge
            return (
                dq_token,
                token_vector(dk_head),
                token_vector(dw_head),
                dg_head.permute(0, 1, 3, 2).contiguous().reshape(
                    batch_size, num_tokens, num_v_heads
                ),
            )

        groups = batch_size * chunks * num_v_heads
        if head_major_inputs is None:
            if q_head is None:
                q_head = head_vector(q)
                k_head = head_vector(k)
            if do_head is None:
                do_head = head_vector(do)
                dv_head = head_vector(dv)
                g_head = g.permute(0, 1, 3, 2).contiguous()
        else:
            q_head, k_head, do_head, dv_head, g_head = head_major_inputs
        q_matrix = q_head.reshape(groups, chunk_size, head_dim_k)
        k_matrix = k_head.reshape(groups, chunk_size, head_dim_k)
        if v_head is None:
            v_head = head_vector(v)
        v_matrix = v_head.reshape(groups, chunk_size, head_dim_v)
        do_matrix = do_head.reshape(groups, chunk_size, head_dim_v)
        dv_matrix = dv_head.reshape(groups, chunk_size, head_dim_v)
        g_matrix = g_head.reshape(groups, chunk_size)
        h_matrix = h.reshape(groups, head_dim_k, head_dim_v)
        dh_matrix = dh.reshape(groups, head_dim_k, head_dim_v)
        exp_g = g_matrix.exp()
        g_last = g_matrix[:, -1:]

        dg_last = (h_matrix * dh_matrix).sum(dim=(-1, -2))
        dg_last *= exp_g[:, -1]
        ds = torch.bmm(do_matrix, v_matrix.transpose(-1, -2))
        dq = torch.bmm(do_matrix, h_matrix.transpose(-1, -2))
        dk = torch.bmm(v_matrix, dh_matrix.transpose(-1, -2))
        dw = -torch.bmm(dv_matrix, h_matrix.transpose(-1, -2))
        dq *= exp_g.unsqueeze(-1) * scale
        dk *= (g_last - g_matrix).exp().unsqueeze(-1)

        q_dq = (q_matrix * dq).sum(dim=-1)
        k_dk = (k_matrix * dk).sum(dim=-1)
        dg = q_dq - k_dk
        dg_last += k_dk.sum(dim=-1)
        native_dp_decay = (
            os.getenv("FLASHQLA_PPU_NATIVE_DQKWG_DECAY", "1") != "0"
        )
        if native_dp_decay:
            from . import native

            native_dp_decay = (
                hasattr(native, "dqkwg_apply_decay_64")
                and native.is_available()
            )
        if native_dp_decay:
            ds = native.dqkwg_apply_decay_64(ds, g_head, scale)
        else:
            lower = torch.tril(
                torch.ones(
                    chunk_size, chunk_size, dtype=torch.bool, device=k.device
                )
            )
            decay = (
                g_matrix[:, :, None] - g_matrix[:, None, :]
            ).masked_fill(~lower, -torch.inf).exp()
            ds *= decay * scale
        p = torch.bmm(q_matrix, k_matrix.transpose(-1, -2))
        native_gate_product = (
            native_dp_decay
            and hasattr(native, "dqkwg_accumulate_gate_product_64")
            and os.getenv("FLASHQLA_PPU_NATIVE_DQKWG_GATE", "0") != "0"
        )
        if native_gate_product:
            dg = native.dqkwg_accumulate_gate_product_64(ds, p, dg)
        else:
            ds2 = ds * p
            dg += ds2.sum(dim=-1) - ds2.sum(dim=-2)
        dq += torch.bmm(ds, k_matrix)
        dk += torch.bmm(ds.transpose(-1, -2), q_matrix)
        dg[:, -1] += dg_last

        head_shape = (
            batch_size, chunks, num_v_heads, chunk_size, head_dim_k
        )
        dq = token_q_vector(dq.reshape(head_shape))
        dk_head = dk.reshape(head_shape)
        dw_head = dw.reshape(head_shape)
        dg_head = dg.reshape(
            batch_size, chunks, num_v_heads, chunk_size
        )
        if return_head_major_intermediates:
            bridge = (
                k_matrix.reshape(head_shape),
                dv_matrix.reshape(head_shape),
                dk_head,
                dw_head,
                g_matrix.reshape(
                    batch_size, chunks, num_v_heads, chunk_size
                ),
                dg_head,
            )
            return dq, dk_head, dw_head, dg_head, bridge
        dk = token_vector(dk_head)
        dw = token_vector(dw_head)
        dg = dg_head.permute(0, 1, 3, 2).contiguous().reshape(
            batch_size, num_tokens, num_v_heads
        )
        return dq, dk, dw, dg

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device),
        diagonal=1,
    )
    decay_mask = _masked_chunk_decay(g, mask)  # [B, N, C, D, Hv]

    exp_g = g.exp()
    dg_last = (h * dh).sum(dim=-1).sum(dim=-1)  # [B, N, Hv]
    ds = torch.einsum("bnchv, bndhv -> bncdh", do, v)

    g_last = g[:, :, -1]
    dg_last *= exp_g[:, :, -1]
    if precomputed_dqkw is None:
        dq = torch.einsum("bnchv, bnhkv -> bnchk", do, h)
        dk = torch.einsum("bnchv, bnhkv -> bnchk", v, dh)
        dw = -torch.einsum("bnchv, bnhkv -> bnchk", dv, h)
        dq = dq * exp_g.unsqueeze(-1) * scale
        dk = dk * (g_last.unsqueeze(-2) - g).unsqueeze(-1).exp()
    else:
        dq, dk, dw = (
            pad_and_reshape(x, dim=1, chunk_size=chunk_size)
            for x in precomputed_dqkw
        )
    q_dq = (q * dq).sum(dim=-1)
    k_dk = (k * dk).sum(dim=-1)
    dg = q_dq - k_dk  # [B, N, C, Hv]
    dg_last += k_dk.sum(dim=-2)
    ds *= decay_mask * scale
    ds2 = ds * torch.einsum("bnchk, bndhk -> bncdh", q, k)
    dg += ds2.sum(dim=-2)
    dg -= ds2.sum(dim=-3)
    dq += torch.einsum("bncdh, bndhk -> bnchk", ds, k)
    dk += torch.einsum("bncdh, bnchk -> bndhk", ds, q)
    dg[:, :, -1] += dg_last

    dg = fill_last_chunk_of_g(
        dg, num_tokens, cu_seqlens, chunk_size=chunk_size, reverse=True
    )
    dq = dq.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    dk = dk.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    dw = dw.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    dg = dg.reshape((batch_size, -1, num_v_heads))[:, :num_tokens]
    if cu_seqlens is not None:
        dq = pack(dq, cu_seqlens)
        dk = pack(dk, cu_seqlens)
        dw = pack(dw, cu_seqlens)
        dg = pack(dg, cu_seqlens)
    return dq, dk, dw, dg


def torch_chunk_wy_bwd(
    k: torch.Tensor,  # [B, T, Hk, K]
    v: torch.Tensor,  # [B, T, Hv, V]
    beta: torch.Tensor,  # [B, T, Hv]
    A: torch.Tensor,  # [B, T, Hv, D]
    g: torch.Tensor,  # [B, T, Hv]
    dw: torch.Tensor,  # [B, T, Hv, K]
    du: torch.Tensor,  # [B, T, Hv, V]
    dk1: torch.Tensor,  # [B, T, Hv, K]
    dg1: torch.Tensor,  # [B, T, Hv]
    cu_seqlens: torch.Tensor = None,
    head_major_inputs: tuple[torch.Tensor, ...] | None = None,
):
    if head_major_inputs is not None and cu_seqlens is not None:
        raise ValueError("head-major WY bridge does not support variable lengths")
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        beta = unpack(beta, cu_seqlens)
        A = unpack(A, cu_seqlens)
        g = unpack(g, cu_seqlens)
        dw = unpack(dw, cu_seqlens)
        du = unpack(du, cu_seqlens)
        dk1 = unpack(dk1, cu_seqlens)
        dg1 = unpack(dg1, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = v.shape
    chunk_size = A.shape[-1]

    fused_wy = False
    if (
        head_major_inputs is None
        and cu_seqlens is None
        and chunk_size == 64
        and head_dim_k == head_dim_v == 128
        and num_tokens % chunk_size == 0
        and os.getenv("FLASHQLA_PPU_FUSED_WY_BWD", "0") == "1"
    ):
        from . import native

        fused_wy = native.is_flash_qla_fused_wy_backward_available()
    if fused_wy:
        return native.flash_qla_fused_wy_backward_128(
            k.contiguous(), v.contiguous(), beta.contiguous(), A.contiguous(),
            g.contiguous(), dw.contiguous(), du.contiguous(), dk1.contiguous(),
            dg1.contiguous(),
        )

    native_wy = False
    if (
        cu_seqlens is None
        and chunk_size == 64
        and head_dim_k == head_dim_v == 128
        and num_tokens % chunk_size == 0
        and os.getenv("FLASHQLA_PPU_NATIVE_BWD_WY", "1") != "0"
    ):
        from . import native

        native_wy = native.is_flash_qla_wy_backward_available()
    bridge_available = head_major_inputs is not None and native_wy
    if head_major_inputs is not None and not bridge_available:
        raise ValueError("head-major WY bridge requires native WY backward")
    skip_bridge_prep = (
        bridge_available
        and os.getenv("FLASHQLA_PPU_SKIP_WY_BRIDGE_PREP", "1") != "0"
    )

    if not skip_bridge_prep:
        if num_k_heads != num_v_heads:
            k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    beta = pad_and_reshape(beta, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    A = pad_and_reshape(A, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, D]
    if head_major_inputs is None:
        g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
        dw = pad_and_reshape(dw, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
        du = pad_and_reshape(du, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
        dk1 = pad_and_reshape(dk1, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
        dg1 = pad_and_reshape(dg1, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
        g = fill_last_chunk_of_g(
            g, num_tokens, cu_seqlens, chunk_size=chunk_size
        )

    if native_wy:
        def head_vector(x):
            return x.permute(0, 1, 3, 2, 4).contiguous()

        def head_scalar(x):
            return x.permute(0, 1, 3, 2).contiguous()

        v_head = head_vector(v)
        A_head = head_vector(A)
        beta_head = head_scalar(beta)
        if head_major_inputs is None:
            k_head = head_vector(k)
            dw_head = head_vector(dw)
            du_head = head_vector(du)
            dk1_head = head_vector(dk1)
            g_head = head_scalar(g)
            dg1_head = head_scalar(dg1)
        else:
            if not skip_bridge_prep:
                k_head = head_vector(k)
            (
                bridged_k,
                du_head,
                dk1_head,
                dw_head,
                g_head,
                dg1_head,
            ) = head_major_inputs
            k_head = bridged_k
        chunks = num_tokens // chunk_size
        groups = batch_size * chunks * num_v_heads

        k_matrix = k_head.reshape(groups, chunk_size, head_dim_k)
        v_matrix = v_head.reshape(groups, chunk_size, head_dim_v)
        A_matrix = A_head.reshape(groups, chunk_size, chunk_size)
        dw_matrix = dw_head.reshape(groups, chunk_size, head_dim_k)
        du_matrix = du_head.reshape(groups, chunk_size, head_dim_v)
        beta_matrix = beta_head.reshape(groups, chunk_size)
        g_matrix = g_head.reshape(groups, chunk_size)
        beta_exp_g = beta_matrix * g_matrix.exp()
        k_beta = k_matrix * beta_matrix.unsqueeze(-1)

        dA = torch.bmm(
            dw_matrix,
            (k_matrix * beta_exp_g.unsqueeze(-1)).transpose(-1, -2),
        )
        dk_beta_g = torch.bmm(A_matrix.transpose(-1, -2), dw_matrix)
        dA += torch.bmm(
            du_matrix,
            (v_matrix * beta_matrix.unsqueeze(-1)).transpose(-1, -2),
        )
        dv_beta = torch.bmm(A_matrix.transpose(-1, -2), du_matrix)
        head_shape = (
            batch_size, chunks, num_v_heads, chunk_size, head_dim_k
        )
        dk, dv, db, dg = native.flash_qla_wy_backward_preprocess_128(
            dk_beta_g.reshape(head_shape),
            dv_beta.reshape(head_shape),
            k_head,
            v_head,
            beta_head,
            g_head,
        )

        dA = torch.tril(dA, diagonal=-1)
        dA = torch.bmm(A_matrix.transpose(-1, -2), dA)
        dA = torch.bmm(dA, A_matrix.transpose(-1, -2))
        native_decay = (
            hasattr(native, "wy_apply_decay_64")
            and os.getenv("FLASHQLA_PPU_NATIVE_WY_DECAY", "1") != "0"
        )
        if native_decay:
            dA = native.wy_apply_decay_64(
                dA.reshape(
                    batch_size, chunks, num_v_heads,
                    chunk_size, chunk_size,
                ).contiguous(),
                g_head,
            ).reshape(groups, chunk_size, chunk_size)
        else:
            lower = torch.tril(
                torch.ones(
                    chunk_size, chunk_size,
                    dtype=torch.bool, device=k.device,
                ),
                diagonal=-1,
            )
            decay = (
                g_matrix[:, :, None] - g_matrix[:, None, :]
            ).masked_fill(~lower, -torch.inf).exp()
            dA = -dA * decay

        A_gram = torch.bmm(k_beta, k_matrix.transpose(-1, -2))
        dk_beta = torch.bmm(dA, k_matrix)
        dk_da = torch.bmm(dA.transpose(-1, -2), k_beta)
        matrix_shape = (
            batch_size, chunks, num_v_heads, chunk_size, chunk_size
        )
        dk, db, dg = native.flash_qla_wy_backward_postprocess_128(
            dk_da.reshape(head_shape),
            dk_beta.reshape(head_shape),
            dk1_head,
            k_head,
            beta_head,
            dA.reshape(matrix_shape),
            A_gram.reshape(matrix_shape),
            dg1_head,
            dk,
            db,
            dg,
        )
    else:
        exp_g = g.exp()
        beta_exp_g = beta * exp_g
        k_beta = k * beta.unsqueeze(-1)
        dA = torch.einsum(
            "bnchk, bndhk -> bnchd", dw,
            k * beta_exp_g.unsqueeze(-1),
        )
        dk_beta_g = torch.einsum("bnchd, bnchk -> bndhk", A, dw)
        dk = dk_beta_g * beta_exp_g.unsqueeze(-1)
        dk_beta_g_k = dk_beta_g * k
        db = (dk_beta_g_k * exp_g.unsqueeze(-1)).sum(dim=-1)
        dg = (dk_beta_g_k * beta_exp_g.unsqueeze(-1)).sum(dim=-1)

        dA += torch.einsum(
            "bnchv, bndhv -> bnchd", du, v * beta.unsqueeze(-1)
        )
        dv_beta = torch.einsum("bnchd, bnchv -> bndhv", A, du)
        dv = dv_beta * beta.unsqueeze(-1)
        db += (dv_beta * v).sum(dim=-1)

        mask = torch.triu(
            torch.ones(
                chunk_size, chunk_size, dtype=torch.bool, device=k.device
            )
        )
        decay_mask = _masked_chunk_decay(g, mask).swapaxes(-2, -1)
        dA = torch.tril(dA.swapaxes(2, 3), diagonal=-1).swapaxes(2, 3)
        dA = torch.einsum("bndhc, bndhe -> bnche", A, dA)
        dA = torch.einsum("bnchd, bnehd -> bnche", dA, A)
        dA = -dA * decay_mask

        A = torch.einsum("bnchk, bndhk -> bnchd", k_beta, k)
        dk_beta = torch.einsum("bnchd, bndhk -> bnchk", dA, k)
        dk_da = torch.einsum("bnchd, bnchk -> bndhk", dA, k_beta)
        db += (dk_beta * k).sum(dim=-1)
        dk += dk_da
        dk += dk_beta * beta.unsqueeze(-1)
        dk += dk1

        dA_A = dA * A
        dg += dA_A.sum(dim=-1) - dA_A.sum(dim=-3).swapaxes(-1, -2)
        dg += dg1

    # TODO: NOTE: GVA
    dk = dk.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    dv = dv.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    db = db.reshape((batch_size, -1, num_v_heads))[:, :num_tokens]
    dg = dg.reshape((batch_size, -1, num_v_heads))[:, :num_tokens]
    if cu_seqlens is not None:
        dk = pack(dk, cu_seqlens)
        dv = pack(dv, cu_seqlens)
        db = pack(db, cu_seqlens)
        dg = pack(dg, cu_seqlens)
    return dk, dv, db, dg


def decomposed_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 64,
):
    native_candidate = (
        cu_seqlens is None
        and chunk_size == 64
        and k.shape[-1] == v.shape[-1] == 128
        and k.shape[1] % chunk_size == 0
    )
    if native_candidate:
        from . import native

        batch, tokens, value_heads = g.shape
        g_chunks = g.view(batch, tokens // chunk_size, chunk_size, value_heads)
        g_chunks = g_chunks.permute(0, 3, 1, 2).contiguous()
    fused_prepare_h = (
        native_candidate
        and os.getenv("FLASHQLA_PPU_FUSED_PREPARE_H", "0") == "1"
        and native.is_flash_qla_fused_prepare_h_available()
    )
    if fused_prepare_h:
        w, h, vn, _ = native.flash_qla_fused_prepare_h_bf16_128(
            k.contiguous(), v.contiguous(), A.contiguous(), g_chunks,
            beta.contiguous(), initial_state,
        )
        native_state = True
    else:
        w, u = torch_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
        )
        native_state = (
            native_candidate
            and os.getenv("FLASHQLA_PPU_NATIVE_BWD_STATE", "1") != "0"
            and native.is_flash_qla_chunk_state_available()
        )
        if native_state:
            h, vn, _ = native.flash_qla_chunk_state_forward_bf16_128(
                k.contiguous(), w.contiguous(), u.contiguous(),
                g_chunks, initial_state,
            )
        else:
            h, vn, _ = torch_chunk_gdr_fwd(
                k=k,
                w=w,
                u=u,
                g=g,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size,
            )
    fused_ska_mode = os.getenv("FLASHQLA_PPU_FUSED_SKA_BWD", "0")
    fused_ska = (
        fused_ska_mode != "0"
        and native_state
        and native.is_flash_qla_fused_state_dqkwg_backward_available()
    )
    if not fused_ska:
        native_dv_mode = os.getenv("FLASHQLA_PPU_NATIVE_BWD_DV", "auto")
        native_dv = (
            native_dv_mode != "0"
            and (native_dv_mode == "1" or q.shape[1] <= 2048)
            and native_candidate
            and native.is_flash_qla_chunk_dv_backward_available()
        )
        if native_dv:
            dv = native.flash_qla_chunk_dv_backward_bf16_128(
                q.contiguous(), k.contiguous(), g_chunks,
                do.contiguous(), scale,
            )
        else:
            dv = torch_chunk_dv_bwd(
                q=q,
                k=k,
                g=g,
                do=do,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size,
            )
    precomputed_dqkwg = None
    if fused_ska:
        (
            fused_dq,
            fused_dk,
            fused_dw,
            fused_dg,
            dv,
            dh0_native,
        ) = native.flash_qla_fused_state_dqkwg_backward_bf16_128(
            q.contiguous(), k.contiguous(), w.contiguous(), g_chunks,
            do.contiguous(), dht,
            vn.contiguous(), h.contiguous(), scale,
        )
        precomputed_dqkwg = (fused_dq, fused_dk, fused_dw, fused_dg)
        dh0 = None if initial_state is None else dh0_native
        dh = None
        state_head_major_inputs = None
    native_reverse_state = (
        not fused_ska
        and os.getenv("FLASHQLA_PPU_NATIVE_BWD_REVERSE", "0") == "1"
        and native_state
        and native.is_flash_qla_chunk_state_backward_available()
    )
    native_reverse_step = (
        not fused_ska
        and not native_reverse_state
        and os.getenv("FLASHQLA_PPU_NATIVE_BWD_STEP", "0") == "1"
        and native_candidate
        and native.is_flash_qla_chunk_state_backward_step_available()
    )
    if fused_ska:
        pass
    elif native_reverse_state:
        dh, dh0_native, dv = native.flash_qla_chunk_state_backward_bf16_128(
            q.contiguous(), k.contiguous(), w.contiguous(), g_chunks,
            do.contiguous(), dv.contiguous(), dht, scale,
        )
        dh0 = None if initial_state is None else dh0_native
        state_head_major_inputs = None
    elif native_reverse_step:
        def run_native_reverse_steps():
            batch, tokens, value_heads = g.shape
            chunks = tokens // chunk_size
            if dht is None:
                dstate = torch.zeros(
                    batch, value_heads, 128, 128,
                    dtype=torch.float32, device=q.device,
                )
            else:
                dstate = dht.to(torch.float32, copy=True).contiguous()
            dh_output = torch.empty(
                batch, chunks, value_heads, 128, 128,
                dtype=torch.float32, device=q.device,
            )
            q_native = q.contiguous()
            k_native = k.contiguous()
            w_native = w.contiguous()
            do_native = do.contiguous()
            dv_output = dv.contiguous()
            for chunk in reversed(range(chunks)):
                native.flash_qla_chunk_state_backward_step_bf16_128(
                    q_native, k_native, w_native, g_chunks, do_native,
                    dv_output, dstate, dh_output, chunk, scale,
                )
            return dh_output, dstate, dv_output

        dh, dh0_native, dv = run_native_reverse_steps()
        dh0 = None if initial_state is None else dh0_native
        state_head_major_inputs = None
    else:
        state_head_major_bridge = (
            native_candidate
            and os.getenv("FLASHQLA_PPU_STATE_DQKWG_BRIDGE", "0") != "0"
            and os.getenv("FLASHQLA_PPU_HEAD_MAJOR_DQKWG", "1") != "0"
            and os.getenv("FLASHQLA_PPU_NATIVE_BWD_WY", "1") != "0"
            and native.is_flash_qla_wy_backward_available()
            and not (
                os.getenv("FLASHQLA_PPU_NATIVE_BWD_DQKWG", "auto") != "0"
                and (
                    os.getenv("FLASHQLA_PPU_NATIVE_BWD_DQKWG", "auto") == "1"
                    or q.shape[1] <= 512
                )
            )
            and not (
                os.getenv("FLASHQLA_PPU_NATIVE_BWD_DQKW", "auto") != "0"
                and (
                    os.getenv("FLASHQLA_PPU_NATIVE_BWD_DQKW", "auto") == "1"
                    or q.shape[1] <= 512
                )
            )
        )
        state_result = torch_chunk_gdr_bwd(
            q=q,
            k=k,
            w=w,
            g=g,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            return_head_major_intermediates=state_head_major_bridge,
        )
        if state_head_major_bridge:
            dh, dh0, dv, state_head_major_inputs = state_result
        else:
            dh, dh0, dv = state_result
            state_head_major_inputs = None
    native_dqkwg_mode = os.getenv("FLASHQLA_PPU_NATIVE_BWD_DQKWG", "auto")
    native_dqkwg = (
        native_dqkwg_mode != "0"
        and (native_dqkwg_mode == "1" or q.shape[1] <= 512)
        and native_candidate
        and native.is_flash_qla_chunk_dqkwg_backward_available()
    )
    if native_dqkwg and precomputed_dqkwg is None:
        precomputed_dqkwg = native.flash_qla_chunk_dqkwg_backward_bf16_128(
            q.contiguous(), k.contiguous(), do.contiguous(), vn.contiguous(),
            dv.contiguous(), h.contiguous(), dh.contiguous(), g_chunks, scale,
        )
    native_dqkw_mode = os.getenv("FLASHQLA_PPU_NATIVE_BWD_DQKW", "auto")
    native_dqkw = (
        precomputed_dqkwg is None
        and not native_dqkwg
        and native_dqkw_mode != "0"
        and (native_dqkw_mode == "1" or q.shape[1] <= 512)
        and native_candidate
        and native.is_flash_qla_chunk_dqkw_backward_available()
    )
    precomputed_dqkw = None
    if native_dqkw:
        precomputed_dqkw = native.flash_qla_chunk_dqkw_backward_bf16_128(
            do.contiguous(), vn.contiguous(), dv.contiguous(),
            h.contiguous(), dh.contiguous(), g_chunks, scale,
        )
    head_major_bridge = (
        precomputed_dqkw is None
        and precomputed_dqkwg is None
        and native_candidate
        and os.getenv("FLASHQLA_PPU_HEAD_MAJOR_BWD_BRIDGE", "1") != "0"
        and os.getenv("FLASHQLA_PPU_NATIVE_BWD_WY", "1") != "0"
        and native.is_flash_qla_wy_backward_available()
    )
    dqkwg_result = torch_chunk_dqkwg_bwd(
        q=q,
        k=k,
        v=vn,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        precomputed_dqkw=precomputed_dqkw,
        precomputed_dqkwg=precomputed_dqkwg,
        return_head_major_intermediates=head_major_bridge,
        head_major_inputs=state_head_major_inputs,
        reduce_dq_gqa=True,
    )
    if head_major_bridge:
        dq, dk1, dw, dg1, head_major_inputs = dqkwg_result
    else:
        dq, dk1, dw, dg1 = dqkwg_result
        head_major_inputs = None
    dk, dv, db, dg = torch_chunk_wy_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        dk1=dk1,
        dg1=dg1,
        cu_seqlens=cu_seqlens,
        head_major_inputs=head_major_inputs,
    )
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        def reduce_grouped_heads(dq_value, dk_value):
            B, T, _, K = dk_value.shape
            return (
                dq_value
                if dq_value.shape[2] == Hg
                else torch.sum(
                    dq_value.reshape(B, T, Hg, -1, K), dim=3
                ),
                torch.sum(dk_value.reshape(B, T, Hg, -1, K), dim=3),
            )

        dq, dk = reduce_grouped_heads(dq, dk)
    dg = torch_cumsum(
        dg,
        chunk_size=64,
        reverse=True,
        cu_seqlens=cu_seqlens,
    )
    return dq, dk, dv, db, dg, dh0
