# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import os

import torch

from flash_qla.utils import (
    fill_last_chunk_of_g,
    pack,
    pad_and_reshape,
    prepare_chunk_offsets,
    unpack,
)

from . import native


CHUNK_SIZE = 64


def _expand_gva(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    num_qk_heads = x.shape[2]
    if num_qk_heads == num_v_heads:
        return x
    if num_v_heads % num_qk_heads:
        raise ValueError("num_v_heads must be divisible by num_qk_heads")
    return x.repeat_interleave(num_v_heads // num_qk_heads, dim=2)


def _kkt_inverse(k: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """A=(I+StrictLower(diag(beta) K K^T))^-1 in FP32."""
    length = k.shape[-2]
    if (
        length == 64
        and k.shape[-1] == 128
        and k.is_cuda
        and k.dtype == torch.bfloat16
        and not (k.requires_grad or beta.requires_grad)
        and native.is_aiu_available()
        and hasattr(native, "aiu_kkt_inverse_bf16_64x128")
    ):
        return native.aiu_kkt_inverse_bf16_64x128(
            k.contiguous(), beta.contiguous()
        )
    if k.dtype in (torch.float16, torch.bfloat16):
        batch_shape = k.shape[:-2]
        flat_k = k.reshape(-1, length, k.shape[-1])
        if k.is_cuda and k.dtype == torch.bfloat16:
            gram = torch.bmm(
                flat_k,
                flat_k.transpose(-1, -2),
                out_dtype=torch.float32,
            )
        else:
            flat_k = flat_k.float()
            gram = torch.bmm(flat_k, flat_k.transpose(-1, -2))
        gram = gram.reshape(*batch_shape, length, length)
    else:
        gram = k @ k.transpose(-1, -2)
    if (
        native.is_available()
        and k.is_cuda
        and not (gram.requires_grad or beta.requires_grad)
    ):
        groups = gram.numel() // (length * length)
        if length == 64 and groups >= 384:
            return native.kkt_inverse(gram.contiguous(), beta.contiguous())
        lower, eye = native.kkt_system(gram.contiguous(), beta.contiguous())
    else:
        eye = torch.eye(length, dtype=torch.float32, device=k.device)
        eye = eye.expand(*k.shape[:-2], length, length)
        lower = eye + torch.tril(beta[..., :, None] * gram, diagonal=-1)
    return torch.linalg.solve_triangular(
        lower, eye, upper=False, unitriangular=True
    )


def _auto_cp_max_local_chunks(num_chunks: int, num_heads: int, device) -> int:
    """Return the PPU-tuned CP segment size, with an override for profiling."""
    del num_heads, device
    override = os.getenv("FLASHQLA_PPU_CP_LOCAL_CHUNKS")
    if override is not None:
        local_chunks = int(override)
        if local_chunks < 1:
            raise ValueError("FLASHQLA_PPU_CP_LOCAL_CHUNKS must be positive")
        return local_chunks
    if num_chunks <= 28:
        return num_chunks
    if num_chunks <= 56:
        return 4
    return (num_chunks + 7) // 8


def _auto_cp_backward_local_chunks(
    num_chunks: int, num_heads: int, device
) -> int:
    """Return the PPU-tuned backward CP segment size."""
    del device
    override = os.getenv("FLASHQLA_PPU_BWD_CP_LOCAL_CHUNKS")
    if override is None:
        override = os.getenv("FLASHQLA_PPU_CP_LOCAL_CHUNKS")
    if override is not None:
        local_chunks = int(override)
        if local_chunks < 1:
            raise ValueError(
                "FLASHQLA_PPU_BWD_CP_LOCAL_CHUNKS must be positive"
            )
        return local_chunks
    if num_chunks <= 28:
        return num_chunks
    crossover = 64 if num_heads >= 8 else 80
    if crossover <= num_chunks <= 256:
        return 16
    return max(4, (num_chunks + 15) // 16)


def _uncumsum_gate(
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    def uncumsum_sequence(sequence: torch.Tensor) -> torch.Tensor:
        raw = torch.empty_like(sequence)
        raw[:, 1:] = sequence[:, 1:] - sequence[:, :-1]
        raw[:, ::CHUNK_SIZE] = sequence[:, ::CHUNK_SIZE]
        return raw

    if cu_seqlens is None:
        return uncumsum_sequence(g_cumsum)

    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    sequences = [
        uncumsum_sequence(g_cumsum[:, left:right])
        for left, right in zip(bounds[:-1], bounds[1:])
    ]
    return torch.cat(sequences, dim=1)


def _official_cp_local_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference form of the official cp_bwd P/R and X/Y dataflow."""
    batch_size, num_tokens, _, key_dim = k.shape
    num_v_heads, value_dim = do.shape[2:]
    native_candidate = (
        native.is_flash_qla_cp_dh_backward_available()
        and os.getenv("FLASHQLA_PPU_NATIVE_CP_DH", "1") != "0"
        and q.is_cuda
        and key_dim == value_dim == 128
        and num_tokens % CHUNK_SIZE == 0
        and all(
            tensor.dtype == torch.float32 and tensor.is_contiguous()
            for tensor in (q, k, A, g, beta, do)
        )
    )
    precompute_candidate = (
        native_candidate
        and os.getenv("FLASHQLA_PPU_PRECOMPUTE_CP_DH", "1") != "0"
        and batch_size * num_v_heads * (num_tokens // CHUNK_SIZE) >= 256
    )
    if precompute_candidate:
        num_chunks = num_tokens // CHUNK_SIZE

        def chunk_heads(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(
                batch_size, num_chunks, CHUNK_SIZE, *x.shape[2:]
            ).permute(0, 3, 1, 2, *range(4, x.ndim + 1)).contiguous()

        qh = chunk_heads(_expand_gva(q, num_v_heads))
        kh = chunk_heads(_expand_gva(k, num_v_heads))
        doh = chunk_heads(do)
        gh = g.reshape(
            batch_size, num_chunks, CHUNK_SIZE, num_v_heads
        ).permute(0, 3, 1, 2).contiguous()
        bh = beta.reshape(
            batch_size, num_chunks, CHUNK_SIZE, num_v_heads
        ).permute(0, 3, 1, 2).contiguous()
        ah = A.reshape(
            batch_size, num_chunks, CHUNK_SIZE, num_v_heads, CHUNK_SIZE
        ).permute(0, 3, 1, 2, 4).contiguous()

        groups = batch_size * num_v_heads * num_chunks
        q_matrix = qh.reshape(groups, CHUNK_SIZE, key_dim)
        k_matrix = kh.reshape(groups, CHUNK_SIZE, key_dim)
        x_matrix = torch.bmm(
            (ah * bh[..., None, :]).reshape(
                groups, CHUNK_SIZE, CHUNK_SIZE
            ),
            k_matrix,
        )
        p_matrix = torch.bmm(q_matrix, k_matrix.transpose(-1, -2))
        r_matrix = q_matrix - torch.bmm(torch.tril(p_matrix), x_matrix)
        x = x_matrix.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE, key_dim
        )
        r = r_matrix.reshape_as(x)

        state_groups = batch_size * num_v_heads
        dh = torch.zeros(
            state_groups, key_dim, value_dim,
            dtype=torch.float32, device=q.device,
        )
        for chunk in range(num_chunks - 1, -1, -1):
            k_chunk = kh[:, :, chunk].reshape(
                state_groups, CHUNK_SIZE, key_dim
            )
            x_chunk = x[:, :, chunk].reshape(
                state_groups, CHUNK_SIZE, key_dim
            )
            r_chunk = r[:, :, chunk].reshape(
                state_groups, CHUNK_SIZE, key_dim
            )
            do_chunk = doh[:, :, chunk].reshape(
                state_groups, CHUNK_SIZE, value_dim
            )
            exp_g = gh[:, :, chunk].exp().reshape(
                state_groups, CHUNK_SIZE
            )
            exp_last = exp_g[:, -1:, None]
            y = torch.bmm(k_chunk, dh)
            dh = (
                exp_last * dh
                + torch.bmm(
                    (scale * exp_g[..., None] * r_chunk).transpose(-1, -2),
                    do_chunk,
                )
                + torch.bmm(
                    x_chunk.transpose(-1, -2), -exp_last * y
                )
            )
        return dh.reshape(
            batch_size, num_v_heads, key_dim, value_dim
        )
    if native_candidate:
        num_chunks = num_tokens // CHUNK_SIZE
        g_chunks = g.reshape(
            batch_size, num_chunks, CHUNK_SIZE, num_v_heads
        ).permute(0, 3, 1, 2).contiguous()
        return native.flash_qla_cp_dh_backward_bf16_128(
            q, k, A, g_chunks, beta, do, scale
        )
    if num_tokens % CHUNK_SIZE:
        raise ValueError("CP backward segments must contain complete chunks")
    num_chunks = num_tokens // CHUNK_SIZE

    def chunk_heads(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(
            batch_size, num_chunks, CHUNK_SIZE, *x.shape[2:]
        ).permute(0, 3, 1, 2, *range(4, x.ndim + 1))

    qh = chunk_heads(_expand_gva(q, num_v_heads).float())
    kh = chunk_heads(_expand_gva(k, num_v_heads).float())
    gh = g.float().reshape(
        batch_size, num_chunks, CHUNK_SIZE, num_v_heads
    ).permute(0, 3, 1, 2)
    bh = beta.float().reshape(
        batch_size, num_chunks, CHUNK_SIZE, num_v_heads
    ).permute(0, 3, 1, 2)
    ah = A.float().reshape(
        batch_size, num_chunks, CHUNK_SIZE, num_v_heads, CHUNK_SIZE
    ).permute(0, 3, 1, 2, 4)
    doh = chunk_heads(do.float())

    dh = torch.zeros(
        batch_size,
        num_v_heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=q.device,
    )
    for chunk in range(num_chunks - 1, -1, -1):
        old_dh = dh
        x = (ah[:, :, chunk] * bh[:, :, chunk, None, :]) @ kh[:, :, chunk]
        p = qh[:, :, chunk] @ kh[:, :, chunk].transpose(-1, -2)
        r = qh[:, :, chunk] - torch.tril(p) @ x
        exp_g = torch.exp(gh[:, :, chunk])
        y = kh[:, :, chunk] @ old_dh
        dh = (
            exp_g[:, :, -1, None, None] * old_dh
            + (scale * exp_g[..., None] * r).transpose(-1, -2)
            @ doh[:, :, chunk]
            + x.transpose(-1, -2)
            @ (-exp_g[:, :, -1, None, None] * y)
        )
    return dh


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


def _group_reduce_vector(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.shape[2] == num_heads:
        return x
    if x.shape[2] % num_heads:
        raise ValueError("input heads must be divisible by num_heads")
    shape = (
        x.shape[0],
        x.shape[1],
        num_heads,
        x.shape[2] // num_heads,
        x.shape[3],
    )
    return x.reshape(shape).sum(dim=3, dtype=torch.float32).to(x.dtype)


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
    output_h: bool = True,
    output_vn: bool = True,
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
        if output_h:
            h.append(last_state)
        v_new = u[:, i] - torch.einsum("bchk, bhkv -> bchv", w[:, i], last_state)
        if output_vn:
            vn.append(v_new)
        last_state = last_state * g[:, i, -1, :, None, None].exp()
        last_state = last_state + torch.einsum(
            "bchk, bchv -> bhkv",
            k[:, i] * (g[:, i, -1:, :, None] - g[:, i, :, :, None]).exp(),
            v_new,
        )
    h = torch.stack(h, dim=1).contiguous() if output_h else None
    vn = (
        torch.stack(vn, dim=1)
        .reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
        .contiguous()
        if output_vn
        else None
    )

    if cu_seqlens is not None:
        if vn is not None:
            vn = pack(vn, cu_seqlens)
        if h is not None:
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

    scale = head_dim_k ** (-0.5) if scale is None else scale

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
    output_dh: bool = True,
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

    scale = head_dim_k ** (-0.5) if scale is None else scale

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

        store_dh = output_dh or return_head_major_intermediates
        dh = []
        for chunk in reversed(range(chunks)):
            if store_dh:
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
        dh = torch.stack(dh, dim=1).contiguous() if store_dh else None
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

        fused_reverse_update = (
            dstate_inter.is_cuda
            and native.is_available()
            and hasattr(native, "reverse_state_update")
            and dstate_inter.is_contiguous()
            and exp_g_last.is_contiguous()
        )

    fused_reverse_store = (
        output_dh
        and fused_reverse_update
        and os.getenv("FLASHQLA_PPU_FUSED_REVERSE_STORE", "1") != "0"
        and hasattr(native, "reverse_state_update_store_inplace")
    )
    native_grouped_dv_update = False
    if grouped_gqa and os.getenv(
        "FLASHQLA_PPU_NATIVE_GROUPED_DV_UPDATE", "1"
    ) == "1":

        native_grouped_dv_update = (
            dv.is_cuda
            and native.is_grouped_state_dv_update_available()
            and dv.is_contiguous()
            and exp_g_to_last.is_contiguous()
        )
    dh = torch.empty_like(dstate_inter) if fused_reverse_store else []
    for i in reversed(range(k.shape[1])):
        if output_dh and not fused_reverse_store:
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
    if output_dh and not fused_reverse_store:
        dh = torch.stack(dh, dim=1).contiguous()
    elif not output_dh:
        dh = None

    dh0 = None if h0 is None else dstate
    dv = dv.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
    if cu_seqlens is not None:
        dv = pack(dv, cu_seqlens)
        if dh is not None:
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
    scale = head_dim_k ** (-0.5) if scale is None else scale

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
    dv = dv.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
    db = db.reshape((batch_size, -1, num_v_heads))[:, :num_tokens]
    dg = dg.reshape((batch_size, -1, num_v_heads))[:, :num_tokens]
    if cu_seqlens is not None:
        dk = pack(dk, cu_seqlens)
        dv = pack(dv, cu_seqlens)
        db = pack(db, cu_seqlens)
        dg = pack(dg, cu_seqlens)
    return dk, dv, db, dg


def _chunk_vn_from_history(
    w: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None:
        w = unpack(w, cu_seqlens)
        u = unpack(u, cu_seqlens)
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        h = unpack(h, chunk_offsets)

    batch_size, num_tokens = w.shape[:2]
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)
    u = pad_and_reshape(u, dim=1, chunk_size=chunk_size)
    vn = u - torch.einsum(
        "bnchk, bnhkv -> bnchv", w, h.to(w.dtype)
    )
    vn = vn.reshape(batch_size, -1, *vn.shape[3:])[:, :num_tokens]
    if cu_seqlens is not None:
        vn = pack(vn, cu_seqlens)
    return vn.contiguous()


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
    h: torch.Tensor = None,
    reduce_grouped_heads: bool = True,
    reverse_cumsum: bool = True,
    prepared_w_u: tuple[torch.Tensor, torch.Tensor] | None = None,
    prepared_vn: torch.Tensor | None = None,
):
    native_candidate = (
        cu_seqlens is None
        and chunk_size == 64
        and k.shape[-1] == v.shape[-1] == 128
        and k.shape[1] % chunk_size == 0
    )
    if native_candidate:

        batch, tokens, value_heads = g.shape
        g_chunks = g.view(batch, tokens // chunk_size, chunk_size, value_heads)
        g_chunks = g_chunks.permute(0, 3, 1, 2).contiguous()
    fused_prepare_h = (
        prepared_w_u is None
        and h is None
        and native_candidate
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
        if prepared_w_u is None:
            w, u = torch_w_u_fwd(
                k=k,
                v=v,
                beta=beta,
                A=A,
                g=g,
                cu_seqlens=cu_seqlens,
            )
        else:
            w, u = prepared_w_u
        native_state = (
            prepared_w_u is None
            and h is None
            and native_candidate
            and os.getenv("FLASHQLA_PPU_NATIVE_BWD_STATE", "1") != "0"
            and native.is_flash_qla_chunk_state_available()
        )
        if prepared_vn is not None:
            vn = prepared_vn
        elif h is not None:
            vn = _chunk_vn_from_history(
                w, u, h, cu_seqlens=cu_seqlens, chunk_size=chunk_size
            )
        elif native_state:
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
        reduce_dq_gqa=reduce_grouped_heads,
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
    if reduce_grouped_heads and Hg < H:
        dq = _group_reduce_vector(dq, Hg)
        dk = _group_reduce_vector(dk, Hg)
    if reverse_cumsum:
        dg = torch_cumsum(
            dg,
            chunk_size=chunk_size,
            reverse=True,
            cu_seqlens=cu_seqlens,
        )
    return dq, dk, dv, db, dg, dh0


def _check_chunk_size(chunk_size: int) -> None:
    if chunk_size != CHUNK_SIZE:
        raise ValueError(f"PPU gated delta rule requires chunk_size={CHUNK_SIZE}")


def _normal_state(state: torch.Tensor | None, state_v_first: bool):
    if state is None:
        return None
    if state_v_first:
        state = state.transpose(-1, -2)
    return state.float().contiguous()


def _fixed_length_cu_seqlens(
    cu_seqlens: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Treat a single packed sequence as dense so native paths remain enabled."""
    if cu_seqlens is None:
        return None
    if reference.shape[0] == 1 and cu_seqlens.numel() == 2:
        return None
    return cu_seqlens


def _uniform_packed_layout(
    cu_seqlens: torch.Tensor | None,
    reference: torch.Tensor,
) -> tuple[int, int] | None:
    if cu_seqlens is None or reference.shape[0] != 1 or cu_seqlens.numel() <= 2:
        return None
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    lengths = [right - left for left, right in zip(bounds[:-1], bounds[1:])]
    if (
        not lengths
        or lengths[0] <= 0
        or any(length != lengths[0] for length in lengths[1:])
        or bounds[0] != 0
        or bounds[-1] != reference.shape[1]
    ):
        return None
    return len(lengths), lengths[0]


def _reshape_packed_tokens(x: torch.Tensor, layout: tuple[int, int]):
    segments, tokens = layout
    return x.reshape(segments, tokens, *x.shape[2:])


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    cu_seqlens: torch.Tensor | None = None,
    reverse: bool = False,
):
    _check_chunk_size(chunk_size)
    cu_seqlens = _fixed_length_cu_seqlens(cu_seqlens, g)
    if cu_seqlens is not None and g.is_cuda:
        return torch_cumsum(
            g.float(),
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            reverse=reverse,
        ).to(g.dtype)
    if cu_seqlens is not None:
        if g.shape[0] != 1:
            raise ValueError("variable-length input requires batch size 1")
        bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
        parts = [
            torch_cumsum(
                g[:, left:right].float(),
                chunk_size=chunk_size,
                reverse=reverse,
            )
            for left, right in zip(bounds[:-1], bounds[1:])
        ]
        result = torch.cat(parts, dim=1) if parts else g.float()
    else:
        result = torch_cumsum(
            g.float(), chunk_size=chunk_size, reverse=reverse
        )
    return result.to(g.dtype)


def group_reduce_vector(buffer: torch.Tensor, Hg: int):
    return _group_reduce_vector(buffer, Hg)


def _kkt_fixed(k: torch.Tensor, b: torch.Tensor, chunk_size: int):
    batch_size, num_tokens, _, key_dim = k.shape
    num_heads = b.shape[2]
    expanded_k = _expand_gva(k, num_heads)
    k_chunks = pad_and_reshape(expanded_k, dim=1, chunk_size=chunk_size)
    b_chunks = pad_and_reshape(b, dim=1, chunk_size=chunk_size)
    k_chunks = k_chunks.permute(0, 3, 1, 2, 4).contiguous()
    b_chunks = b_chunks.permute(0, 3, 1, 2).float().contiguous()
    inverse = _kkt_inverse(k_chunks, b_chunks)
    inverse = inverse.permute(0, 2, 3, 1, 4).reshape(
        batch_size, -1, num_heads, chunk_size
    )
    return inverse[:, :num_tokens].to(k.dtype).contiguous()


def kkt_solve(
    k: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    cu_seqlens: torch.Tensor | None = None,
):
    _check_chunk_size(chunk_size)
    if k.ndim != 4 or b.ndim != 3 or k.shape[:2] != b.shape[:2]:
        raise ValueError("k and b must have shapes [B,T,Hg,K] and [B,T,Hv]")
    if k.shape[-1] != 128:
        raise ValueError("PPU gated delta rule requires key dimension 128")
    cu_seqlens = _fixed_length_cu_seqlens(cu_seqlens, k)
    if cu_seqlens is None:
        return _kkt_fixed(k, b, chunk_size)
    if k.shape[0] != 1:
        raise ValueError("variable-length input requires batch size 1")
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    parts = [
        _kkt_fixed(k[:, left:right], b[:, left:right], chunk_size)
        for left, right in zip(bounds[:-1], bounds[1:])
    ]
    return torch.cat(parts, dim=1) if parts else k.new_empty((1, 0, b.shape[2], chunk_size))


def _gate_chunks(g: torch.Tensor, chunk_size: int):
    """Chunk-major FP32 view [B, Hv, C, chunk] of the chunk-local gate cumsum."""
    return pad_and_reshape(g, dim=1, chunk_size=chunk_size).permute(0, 3, 1, 2).float()


def _weight_kkt_chunks(a: torch.Tensor, g: torch.Tensor, chunk_size: int):
    """Gated, chunk-major A = tril(exp(G_row - G_col)) * A0 in FP32."""
    a_chunks = pad_and_reshape(a, dim=1, chunk_size=chunk_size)
    a_chunks = a_chunks.permute(0, 3, 1, 2, 4).float()
    g_chunks = _gate_chunks(g, chunk_size)
    decay = torch.exp(g_chunks[..., :, None] - g_chunks[..., None, :])
    return torch.tril(decay) * a_chunks


def _weight_kkt_fixed(a: torch.Tensor, g: torch.Tensor, chunk_size: int):
    batch_size, num_tokens, num_heads, _ = a.shape
    weighted = _weight_kkt_chunks(a, g, chunk_size)
    weighted = weighted.permute(0, 2, 3, 1, 4).reshape(
        batch_size, -1, num_heads, chunk_size
    )
    # Kept in FP32: every consumer needs FP32 anyway, so narrowing to the input
    # dtype here only lost precision and added two extra casts.
    return weighted[:, :num_tokens]


def _weight_kkt(
    a: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int = CHUNK_SIZE,
):
    # PPU decomposition consumes D A D^-1, while the public primitive exposes A.
    if cu_seqlens is None:
        return _weight_kkt_fixed(a, g, chunk_size)
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    parts = [
        _weight_kkt_fixed(a[:, left:right], g[:, left:right], chunk_size)
        for left, right in zip(bounds[:-1], bounds[1:])
    ]
    return torch.cat(parts, dim=1) if parts else a.float()


def _prepare_h(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    output_h: bool = True,
    output_vn: bool = True,
    prepared: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] | None = None,
):
    uniform_layout = _uniform_packed_layout(cu_seqlens, k)
    if uniform_layout is not None:
        h, vn, final = _prepare_h(
            _reshape_packed_tokens(k, uniform_layout),
            _reshape_packed_tokens(v, uniform_layout),
            _reshape_packed_tokens(a, uniform_layout),
            _reshape_packed_tokens(g, uniform_layout),
            _reshape_packed_tokens(b, uniform_layout),
            initial_state,
            None,
            output_h=output_h,
            output_vn=output_vn,
            prepared=prepared,
        )
        return (
            None if h is None else h.reshape(1, -1, *h.shape[2:]),
            None if vn is None else vn.reshape(1, -1, *vn.shape[2:]),
            final,
        )

    if cu_seqlens is not None and not k.is_cuda:
        bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
        results = [
            _prepare_h(
                k[:, left:right],
                v[:, left:right],
                a[:, left:right],
                g[:, left:right],
                b[:, left:right],
                None if initial_state is None else initial_state[index : index + 1],
                None,
                output_h=output_h,
                output_vn=output_vn,
            )
            for index, (left, right) in enumerate(zip(bounds[:-1], bounds[1:]))
        ]
        return (
            torch.cat([result[0] for result in results], dim=1)
            if output_h else None,
            torch.cat([result[1] for result in results], dim=1)
            if output_vn else None,
            torch.cat([result[2] for result in results], dim=0),
        )

    weighted_a = _weight_kkt(a, g, cu_seqlens)
    k_work = k.float().contiguous()
    v_work = v.float().contiguous()
    g_work = g.float().contiguous()
    b_work = b.float().contiguous()
    w, u = torch_w_u_fwd(
        k=k_work,
        v=v_work,
        g=g_work,
        beta=b_work,
        A=weighted_a.float().contiguous(),
        cu_seqlens=cu_seqlens,
    )
    native_state = (
        k_work.is_cuda
        and cu_seqlens is None
        and k_work.shape[-1] == v_work.shape[-1] == 128
        and k_work.shape[1] % CHUNK_SIZE == 0
        and os.getenv("FLASHQLA_PPU_NATIVE_BWD_STATE", "1") != "0"
        and native.is_flash_qla_chunk_state_available()
    )
    if native_state:
        batch, tokens, value_heads = g_work.shape
        g_chunks = g_work.view(
            batch, tokens // CHUNK_SIZE, CHUNK_SIZE, value_heads
        ).permute(0, 3, 1, 2).contiguous()
        h, vn, final = native.flash_qla_chunk_state_forward_bf16_128(
            k_work, w.contiguous(), u.contiguous(), g_chunks, initial_state
        )
    else:
        h, vn, final = torch_chunk_gdr_fwd(
            k=k_work,
            w=w,
            u=u,
            g=g_work,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_size=CHUNK_SIZE,
            output_h=output_h,
            output_vn=output_vn or prepared is not None,
        )
    if prepared is not None:
        prepared.append((weighted_a, w, u, h, vn))
    return h if output_h else None, vn if output_vn else None, final


def _fixed_output(
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    scale: float,
):
    batch_size, num_tokens = q.shape[:2]
    num_heads = vn.shape[2]
    q_chunks = pad_and_reshape(_expand_gva(q, num_heads).float(), 1, CHUNK_SIZE)
    k_chunks = pad_and_reshape(_expand_gva(k, num_heads).float(), 1, CHUNK_SIZE)
    v_chunks = pad_and_reshape(vn.float(), 1, CHUNK_SIZE)
    g_chunks = pad_and_reshape(g.float(), 1, CHUNK_SIZE)
    g_chunks = fill_last_chunk_of_g(g_chunks, num_tokens, None, CHUNK_SIZE)
    mask = torch.triu(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    decay = _masked_chunk_decay(g_chunks, mask)
    attention = torch.einsum("bnchk,bndhk->bncdh", q_chunks, k_chunks) * decay
    output = torch.einsum("bncdh,bndhv->bnchv", attention, v_chunks)
    output += torch.einsum(
        "bnchk,bnhkv->bnchv", q_chunks, h.float()
    ) * g_chunks.exp().unsqueeze(-1)
    return (output * scale).reshape(
        batch_size, -1, num_heads, vn.shape[-1]
    )[:, :num_tokens]


def _output(
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
):
    uniform_layout = _uniform_packed_layout(cu_seqlens, q)
    if uniform_layout is not None:
        segments, tokens = uniform_layout
        chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
        output = _fixed_output(
            _reshape_packed_tokens(q, uniform_layout),
            _reshape_packed_tokens(k, uniform_layout),
            _reshape_packed_tokens(vn, uniform_layout),
            _reshape_packed_tokens(g, uniform_layout),
            h.reshape(segments, chunks, *h.shape[2:]),
            scale,
        )
        return output.reshape(1, -1, *output.shape[2:])
    if cu_seqlens is None:
        return _fixed_output(q, k, vn, g, h, scale)
    if q.is_cuda:
        q_pad, k_pad, vn_pad, g_pad = (
            unpack(x, cu_seqlens) for x in (q, k, vn, g)
        )
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)
        h_pad = unpack(h, chunk_offsets)
        return pack(
            _fixed_output(q_pad, k_pad, vn_pad, g_pad, h_pad, scale),
            cu_seqlens,
        )
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    chunk_offset = 0
    parts = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        num_chunks = (right - left + CHUNK_SIZE - 1) // CHUNK_SIZE
        parts.append(
            _fixed_output(
                q[:, left:right],
                k[:, left:right],
                vn[:, left:right],
                g[:, left:right],
                h[:, chunk_offset : chunk_offset + num_chunks],
                scale,
            )
        )
        chunk_offset += num_chunks
    return torch.cat(parts, dim=1)


def _native_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    output_h: bool,
):
    """Single-launch AIU forward, restoring the HEAD ``_run_equal_length`` path.

    ``output_h`` disables it because the kernel never persists the per-chunk
    history. The kernel applies ``A[row, col] * beta[col]`` under a causal mask
    and no decay of its own (see ``ppu_flash_qla_fwd.cu``), so it must receive
    the gated ``D A0 D^-1`` form rather than the ungated public ``A``.
    """
    if output_h or cu_seqlens is not None:
        return None
    if os.getenv("FLASHQLA_PPU_AIU_FUSED", "1") != "1":
        return None
    if not (
        native.is_flash_qla_fused_available()
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and k.shape[-1] == v.shape[-1] == 128
        and k.shape[1] % CHUNK_SIZE == 0
        and q.is_cuda
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and not any(
            tensor.requires_grad
            for tensor in (q, k, v, a, g, b, initial_state)
            if tensor is not None
        )
    ):
        return None
    return native.flash_qla_fused_forward_bf16_128(
        q,
        k,
        v,
        _weight_kkt_chunks(a, g, CHUNK_SIZE).to(torch.bfloat16).contiguous(),
        _gate_chunks(g, CHUNK_SIZE).contiguous(),
        _gate_chunks(b, CHUNK_SIZE).contiguous(),
        None if initial_state is None else initial_state.contiguous(),
        scale,
    )


def fused_gdr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    output_o: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    cp_seq_map: torch.Tensor | None = None,
    raw_cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = False,
):
    output_o = True if output_o is None else output_o
    if not output_o and not output_h and not output_final_state:
        return None, None, None
    scale = float(k.shape[-1] ** -0.5 if scale is None else scale)
    if cp_seq_map is None:
        cu_seqlens = _fixed_length_cu_seqlens(cu_seqlens, k)
    initial = _normal_state(initial_state, state_v_first)
    fused = _native_fused_forward(
        q, k, v, a, g, b, initial, scale, cu_seqlens, output_h or not output_o
    )
    if fused is not None:
        # Inference route: o plus the final state without ever persisting a
        # per-chunk FP32 history.
        output, final = fused
        h_work = None
    else:
        h_work, vn, final = _prepare_h(
            k, v, a, g, b, initial, cu_seqlens,
            output_h=output_h or output_o,
            output_vn=output_o,
        )
        output = (
            _output(q, k, vn, g, h_work, scale, cu_seqlens) if output_o else None
        )

    if cp_seq_map is not None and output_final_state:
        if cu_seqlens is None or raw_cu_seqlens is None:
            raise ValueError(
                "cp_seq_map requires cu_seqlens and raw_cu_seqlens"
            )
        if cp_seq_map.numel() != final.shape[0]:
            raise ValueError("cp_seq_map must contain one entry per packed sequence")
        raw_indices = cp_seq_map.long()
        raw_count = raw_cu_seqlens.shape[0] - 1
        if raw_indices.numel() and (
            int(raw_indices.min().item()) < 0
            or int(raw_indices.max().item()) >= raw_count
        ):
            raise ValueError("cp_seq_map contains an invalid raw sequence index")
        segment_ends = cu_seqlens[1:].to(raw_cu_seqlens.device)
        raw_ends = raw_cu_seqlens[raw_indices.to(raw_cu_seqlens.device) + 1]
        last_segment = (segment_ends == raw_ends).to(final.device)
        selected = final.new_zeros((raw_count, *final.shape[1:]))
        selected.index_copy_(
            0, raw_indices.to(final.device)[last_segment], final[last_segment]
        )
        final = selected

    h_out = h_work.to(k.dtype) if output_h else None
    final_out = final.float() if output_final_state else None
    if state_v_first:
        if h_out is not None:
            h_out = h_out.transpose(-1, -2).contiguous()
        if final_out is not None:
            final_out = final_out.transpose(-1, -2).contiguous()
    return None if output is None else output.to(v.dtype), h_out, final_out


def _chunk_ranges(length: int):
    return [(left, min(left + CHUNK_SIZE, length)) for left in range(0, length, CHUNK_SIZE)]


def _segment_chunks(
    x: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    max_chunks: int,
    chunk_size: int = CHUNK_SIZE,
):
    """Gather packed segments into chunk-major ``[S, H, C, chunk, ...]`` form.

    Segments must be chunk aligned. Trailing pad chunks are zero filled, which
    leaves the affine recurrence a no-op there (gamma == 1, k == v == x == 0),
    so ragged segment counts need no extra bookkeeping.
    """
    segments = starts.numel()
    offsets = (
        torch.arange(max_chunks, device=x.device, dtype=torch.int64)[:, None]
        * chunk_size
        + torch.arange(chunk_size, device=x.device, dtype=torch.int64)[None, :]
    )
    tokens = starts[:, None, None] + offsets[None]
    valid = tokens < ends[:, None, None]
    gathered = x[0, torch.where(valid, tokens, starts[:, None, None]).reshape(-1)]
    gathered = gathered.reshape(segments, max_chunks, chunk_size, *x.shape[2:])
    trailing = (1,) * (x.ndim - 2)
    gathered = gathered * valid.reshape(segments, max_chunks, chunk_size, *trailing)
    return gathered.permute(0, 3, 1, 2, *range(4, gathered.ndim))


def _segment_layout(bounds: list[int], device: torch.device):
    """Per-segment offsets and complete-chunk counts, or None when ragged."""
    lengths = [right - left for left, right in zip(bounds[:-1], bounds[1:])]
    if not lengths or any(length % CHUNK_SIZE or length == 0 for length in lengths):
        return None
    limits = [length // CHUNK_SIZE for length in lengths]
    return (
        torch.tensor(bounds[:-1], device=device, dtype=torch.int64),
        torch.tensor(bounds[1:], device=device, dtype=torch.int64),
        torch.tensor(limits, device=device, dtype=torch.int64)[:, None],
        max(limits),
        len(set(lengths)) == 1,
    )


def _native_affine_state(
    k_chunks: torch.Tensor,
    v_chunks: torch.Tensor,
    x_chunks: torch.Tensor,
    gamma_last: torch.Tensor,
    reverse_decay: torch.Tensor,
    counts: torch.Tensor,
    full: torch.Tensor,
    uniform: bool,
):
    """One-launch AutoCP S*/M recurrence, mirroring the HEAD dispatch."""
    if not (
        uniform
        and k_chunks.is_cuda
        and os.getenv("FLASHQLA_PPU_NATIVE_AFFINE", "1") != "0"
        and native.is_flash_qla_affine_available()
        and k_chunks.shape[-1] == v_chunks.shape[-1] == 128
    ):
        return None
    arguments = (
        k_chunks.to(torch.bfloat16).contiguous(),
        v_chunks.to(torch.bfloat16).contiguous(),
        x_chunks.to(torch.bfloat16).contiguous(),
        gamma_last.float().contiguous(),
        reverse_decay.float().contiguous(),
        counts.to(torch.int32).contiguous(),
        full.contiguous(),
    )
    return (
        native.flash_qla_affine_state_bf16_128(*arguments, matrix_mode=False),
        native.flash_qla_affine_state_bf16_128(*arguments, matrix_mode=True),
    )


def _batched_warmup_forward(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None,
    bounds: list[int],
    num_warmup_chunks: torch.Tensor,
):
    """Batched AutoCP warmup: S* = gamma S + X^T Y, M = gamma (I + X^T K) M."""
    layout = _segment_layout(bounds, k.device)
    if layout is None:
        return None
    starts, ends, limits, max_chunks, uniform = layout
    num_heads = v.shape[2]
    key_dim, value_dim = k.shape[-1], v.shape[-1]
    counts = torch.minimum(
        num_warmup_chunks.to(device=k.device, dtype=torch.int64), limits
    )
    k_chunks, v_chunks, a_chunks, b_chunks, g_chunks = (
        _segment_chunks(tensor, starts, ends, max_chunks)
        for tensor in (
            _expand_gva(k, num_heads).float(),
            v.float(),
            a.float(),
            b.float(),
            g.float(),
        )
    )
    gamma_last = g_chunks[..., -1].exp()
    reverse_decay = torch.exp(g_chunks[..., -1:] - g_chunks)
    # X = -diag(beta) A0^T K, i.e. the ungated KKT solve folded with beta. The
    # gated form D A0 D^-1 cancels against the reverse decay, so no exp() of a
    # full 64x64 block is needed here.
    x_chunks = -b_chunks[..., None] * (a_chunks.transpose(-1, -2) @ k_chunks)
    full = counts == limits

    if initial_state is None:
        native_result = _native_affine_state(
            k_chunks, v_chunks, x_chunks, gamma_last, reverse_decay,
            counts, full, uniform,
        )
        if native_result is not None:
            state, matrix = native_result
            return None, state, matrix

    state = (
        torch.zeros(
            (starts.numel(), num_heads, key_dim, value_dim),
            dtype=torch.float32,
            device=k.device,
        )
        if initial_state is None
        else initial_state.float()
    )
    matrix = torch.eye(key_dim, dtype=torch.float32, device=k.device).expand(
        starts.numel(), num_heads, key_dim, key_dim
    ).contiguous()
    first_chunk = max(int((limits - counts).min().item()), 0)
    for chunk in range(first_chunk, max_chunks):
        active = (chunk >= limits - counts) & (chunk < limits)
        gamma = gamma_last[:, :, chunk, None, None]
        kc = k_chunks[:, :, chunk]
        xt = x_chunks[:, :, chunk].transpose(-1, -2)
        y = gamma * (kc @ state) - reverse_decay[:, :, chunk, :, None] * v_chunks[:, :, chunk]
        mask = active[..., None, None]
        state = torch.where(mask, gamma * state + xt @ y, state)
        matrix = torch.where(
            mask, gamma * (matrix + xt @ (kc @ matrix)), matrix
        )
    return None, state, matrix * full[..., None, None]


def _warmup_forward(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    num_warmup_chunks: torch.Tensor,
    output_h: bool,
):
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    if not output_h:
        batched = _batched_warmup_forward(
            k, v, a, g, b, initial_state, bounds, num_warmup_chunks
        )
        if batched is not None:
            return batched
    num_heads, key_dim, value_dim = v.shape[2], k.shape[-1], v.shape[-1]
    expanded_k = _expand_gva(k, num_heads).float()
    total_chunks = sum(
        (right - left + CHUNK_SIZE - 1) // CHUNK_SIZE
        for left, right in zip(bounds[:-1], bounds[1:])
    )
    history = (
        torch.zeros(
            (1, total_chunks, num_heads, key_dim, value_dim),
            dtype=torch.float32,
            device=k.device,
        )
        if output_h
        else None
    )
    final = torch.zeros(
        (len(bounds) - 1, num_heads, key_dim, value_dim),
        dtype=torch.float32,
        device=k.device,
    )
    correction = torch.zeros(
        (len(bounds) - 1, num_heads, key_dim, key_dim),
        dtype=torch.float32,
        device=k.device,
    )
    counts = num_warmup_chunks.to(device="cpu", dtype=torch.int64)
    chunk_base = 0
    for sequence, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        num_chunks = (right - left + CHUNK_SIZE - 1) // CHUNK_SIZE
        for head in range(num_heads):
            count = min(int(counts[sequence, head]), num_chunks)
            state = (
                torch.zeros((key_dim, value_dim), dtype=torch.float32, device=k.device)
                if initial_state is None
                else initial_state[sequence, head].float()
            )
            matrix = torch.eye(key_dim, dtype=torch.float32, device=k.device)
            # Anchor the warmup window to the chunk grid, not to the raw token
            # count: a partial tail must not shift the window off a boundary.
            for chunk_index in range(num_chunks - count, num_chunks):
                begin = left + chunk_index * CHUNK_SIZE
                end = min(begin + CHUNK_SIZE, right)
                if history is not None:
                    history[0, chunk_base + chunk_index, head] = state
                kc = expanded_k[0, begin:end, head]
                vc = v[0, begin:end, head].float()
                ac = a[0, begin:end, head, : end - begin].float()
                gc = g[0, begin:end, head].float()
                bc = b[0, begin:end, head].float()
                weighted = torch.tril(
                    torch.exp(gc[:, None] - gc[None, :])
                ) * ac
                w = weighted @ (bc[:, None] * gc.exp()[:, None] * kc)
                u = weighted @ (bc[:, None] * vc)
                vn = u - w @ state
                reverse_decay = torch.exp(gc[-1] - gc)
                decayed_k = kc * reverse_decay[:, None]
                transition = torch.exp(gc[-1]) * torch.eye(
                    key_dim, dtype=torch.float32, device=k.device
                ) - decayed_k.transpose(0, 1) @ w
                state = (
                    torch.exp(gc[-1]) * state
                    + decayed_k.transpose(0, 1) @ vn
                )
                matrix = transition @ matrix
            final[sequence, head] = state
            if count == num_chunks:
                correction[sequence, head] = matrix
        chunk_base += num_chunks
    return history, final, correction


def fused_gdr_h(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    num_warmup_chunks: torch.Tensor | None = None,
    state_v_first: bool = False,
):
    if not output_h and not output_final_state:
        return None, None, None
    if num_warmup_chunks is None:
        cu_seqlens = _fixed_length_cu_seqlens(cu_seqlens, k)
    initial = _normal_state(initial_state, state_v_first)
    prepared = [] if output_h and num_warmup_chunks is None else None
    if num_warmup_chunks is not None:
        if cu_seqlens is None or k.shape[0] != 1:
            raise ValueError("warmup chunks require packed variable-length input")
        h, final, correction = _warmup_forward(
            k,
            v,
            a,
            g,
            b,
            initial,
            cu_seqlens,
            num_warmup_chunks,
            output_h,
        )
        final_dtype = k.dtype
    else:
        h, _, final = _prepare_h(
            k, v, a, g, b, initial, cu_seqlens,
            output_h=output_h,
            output_vn=False,
            prepared=prepared,
        )
        correction = (
            torch.zeros(
                final.shape[0], final.shape[1], k.shape[-1], k.shape[-1],
                dtype=torch.float32, device=k.device,
            )
            if output_final_state else None
        )
        final_dtype = torch.float32

    h_out = h if output_h and prepared else h.to(k.dtype) if output_h else None
    final_out = final.to(final_dtype) if output_final_state else None
    correction_out = correction.to(final_dtype) if output_final_state else None
    if state_v_first:
        if h_out is not None:
            h_out = h_out.transpose(-1, -2).contiguous()
        if final_out is not None:
            final_out = final_out.transpose(-1, -2).contiguous()
    if h_out is not None and prepared:
        h_out._flashqla_ppu_prepared = prepared[0]
    return h_out, final_out, correction_out


def _history_initial_state(
    h: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
):
    if h is None:
        return None
    if cu_seqlens is None:
        return h[:, 0].float().contiguous()
    if h.is_cuda:
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)
        return h[0, chunk_offsets[:-1].long()].float().contiguous()
    lengths = torch.diff(cu_seqlens).to(device="cpu", dtype=torch.int64).tolist()
    offsets = []
    current = 0
    for length in lengths:
        offsets.append(current)
        current += (length + CHUNK_SIZE - 1) // CHUNK_SIZE
    return h[0, torch.tensor(offsets, device=h.device)].float().contiguous()


def fused_gdr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    h: torch.Tensor | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
    state_v_first: bool = False,
):
    _check_chunk_size(chunk_size)
    scale = float(k.shape[-1] ** -0.5 if scale is None else scale)
    prepared = None if h is None else getattr(h, "_flashqla_ppu_prepared", None)
    prepared_backward_inputs = (
        None
        if dht is None
        else getattr(dht, "_flashqla_ppu_backward_inputs", None)
    )
    uniform_layout = _uniform_packed_layout(cu_seqlens, k)
    if uniform_layout is not None:
        segments, tokens = uniform_layout
        chunks = (tokens + chunk_size - 1) // chunk_size
        dense_h = None if h is None else h.reshape(segments, chunks, *h.shape[2:])
        if dense_h is not None and prepared is not None:
            dense_h._flashqla_ppu_prepared = prepared
        result = fused_gdr_bwd(
            _reshape_packed_tokens(q, uniform_layout),
            _reshape_packed_tokens(k, uniform_layout),
            _reshape_packed_tokens(v, uniform_layout),
            _reshape_packed_tokens(a, uniform_layout),
            _reshape_packed_tokens(g, uniform_layout),
            _reshape_packed_tokens(b, uniform_layout),
            _reshape_packed_tokens(do, uniform_layout),
            dht,
            dense_h,
            scale=scale,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
        )
        token_grads = tuple(
            tensor.reshape(1, -1, *tensor.shape[2:]) for tensor in result[:5]
        )
        return (*token_grads, result[5])

    cu_seqlens = _fixed_length_cu_seqlens(cu_seqlens, k)
    if cu_seqlens is not None and not q.is_cuda:
        bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
        chunk_offset = 0
        results = []
        for sequence, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
            num_chunks = (right - left + chunk_size - 1) // chunk_size
            results.append(
                fused_gdr_bwd(
                    q[:, left:right], k[:, left:right], v[:, left:right],
                    a[:, left:right], g[:, left:right], b[:, left:right],
                    do[:, left:right],
                    None if dht is None else dht[sequence : sequence + 1],
                    None if h is None else h[:, chunk_offset : chunk_offset + num_chunks],
                    scale=scale, chunk_size=chunk_size,
                    state_v_first=state_v_first,
                )
            )
            chunk_offset += num_chunks
        return (
            *(torch.cat([result[index] for result in results], dim=1) for index in range(5)),
            torch.cat([result[5] for result in results], dim=0),
        )
    if prepared is None:
        if h is None:
            history = None
        elif state_v_first:
            history = h.transpose(-1, -2).contiguous()
        else:
            history = h
        weighted_a = _weight_kkt(a, g, cu_seqlens, chunk_size)
        prepared_w_u = None
        prepared_vn = None
    else:
        weighted_a, w, u, history, prepared_vn = prepared
        prepared_w_u = (w, u)
    terminal = _normal_state(dht, state_v_first)
    initial = _history_initial_state(history, cu_seqlens)
    if initial is None:
        real_batch_size = (
            q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
        )
        initial = torch.zeros(
            real_batch_size,
            v.shape[2],
            k.shape[-1],
            v.shape[-1],
            dtype=torch.float32,
            device=q.device,
        )
    fused_cast = (
        prepared_backward_inputs is None
        and q.is_cuda
        and q.shape[-1] == v.shape[-1] == 128
        and all(
            tensor.dtype == torch.bfloat16 and tensor.is_contiguous()
            for tensor in (q, k, v, a, do)
        )
        and native.is_flash_qla_backward_cast_available()
    )
    if prepared_backward_inputs is not None:
        q_work, k_work, g_work, b_work, do_work = prepared_backward_inputs
        v_work = v.float().contiguous()
    elif fused_cast:
        q_work, k_work, v_work, _, do_work = (
            native.prepare_backward_inputs_bf16_128(q, k, v, a, do)
        )
        g_work = g.float().contiguous()
        b_work = b.float().contiguous()
    else:
        q_work = q.float().contiguous()
        k_work = k.float().contiguous()
        v_work = v.float().contiguous()
        g_work = g.float().contiguous()
        b_work = b.float().contiguous()
        do_work = do.float().contiguous()
    dq, dk, dv, db, dg, dh0 = decomposed_backward(
        q_work,
        k_work,
        v_work,
        g_work,
        b_work,
        weighted_a.float().contiguous(),
        scale,
        initial,
        do_work,
        terminal,
        cu_seqlens,
        chunk_size=chunk_size,
        h=None if history is None else history.float().contiguous(),
        reduce_grouped_heads=False,
        reverse_cumsum=False,
        prepared_w_u=prepared_w_u,
        prepared_vn=prepared_vn,
    )
    if state_v_first:
        dh0 = dh0.transpose(-1, -2).contiguous()
    return dq, dk, dv.to(v.dtype), dg.float(), db.to(b.dtype), dh0


def _batched_warmup_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    bounds: list[int],
    num_warmup_chunks: torch.Tensor,
    scale: float,
):
    """Batched AutoCP local dh recurrence over the leading warmup chunks."""
    layout = _segment_layout(bounds, k.device)
    if layout is None:
        return None
    starts, ends, limits, max_chunks, uniform = layout
    num_heads = do.shape[2]
    key_dim, value_dim = k.shape[-1], do.shape[-1]
    counts = torch.minimum(
        num_warmup_chunks.to(device=k.device, dtype=torch.int64), limits
    )
    if uniform:
        segments, length = starts.numel(), bounds[1] - bounds[0]

        def segment_view(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.float().contiguous().reshape(
                segments, length, *tensor.shape[2:]
            )

        if dht is None and native.is_flash_qla_cp_dh_backward_available():
            q_work = segment_view(q)
            k_work = segment_view(k)
            g_work = segment_view(g)
            b_work = segment_view(b)
            do_work = segment_view(do)
            dh0 = _official_cp_local_dh(
                q_work,
                k_work,
                segment_view(a),
                g_work,
                b_work,
                do_work,
                scale,
            )
            dh0._flashqla_ppu_backward_inputs = (
                q_work,
                k_work,
                g_work,
                b_work,
                do_work,
            )
            return None, dh0

        def chunk_heads(tensor: torch.Tensor) -> torch.Tensor:
            return segment_view(tensor).reshape(
                segments, max_chunks, CHUNK_SIZE, *tensor.shape[2:]
            ).permute(0, 3, 1, 2, *range(4, tensor.ndim + 1)).contiguous()

        q_chunks = chunk_heads(_expand_gva(q, num_heads))
        k_chunks = chunk_heads(_expand_gva(k, num_heads))
        do_chunks = chunk_heads(do)
        a_chunks = segment_view(a).reshape(
            segments, max_chunks, CHUNK_SIZE, num_heads, CHUNK_SIZE
        ).permute(0, 3, 1, 2, 4).contiguous()
        b_chunks = segment_view(b).reshape(
            segments, max_chunks, CHUNK_SIZE, num_heads
        ).permute(0, 3, 1, 2).contiguous()
        g_chunks = segment_view(g).reshape(
            segments, max_chunks, CHUNK_SIZE, num_heads
        ).permute(0, 3, 1, 2).contiguous()

        groups = segments * num_heads * max_chunks
        q_matrix = q_chunks.reshape(groups, CHUNK_SIZE, key_dim)
        k_matrix = k_chunks.reshape(groups, CHUNK_SIZE, key_dim)
        x_matrix = torch.bmm(
            (a_chunks * b_chunks[..., None, :]).reshape(
                groups, CHUNK_SIZE, CHUNK_SIZE
            ),
            k_matrix,
        )
        p_matrix = torch.bmm(q_matrix, k_matrix.transpose(-1, -2))
        r_matrix = q_matrix - torch.bmm(torch.tril(p_matrix), x_matrix)
        x_chunks = x_matrix.reshape(
            segments, num_heads, max_chunks, CHUNK_SIZE, key_dim
        )
        r_chunks = r_matrix.reshape_as(x_chunks)
    else:
        q_chunks, k_chunks, a_chunks, b_chunks, g_chunks, do_chunks = (
            _segment_chunks(tensor, starts, ends, max_chunks)
            for tensor in (
                _expand_gva(q, num_heads).float(),
                _expand_gva(k, num_heads).float(),
                a.float(),
                b.float(),
                g.float(),
                do.float(),
            )
        )
        x_chunks = None
        r_chunks = None

    state = (
        torch.zeros(
            (starts.numel(), num_heads, key_dim, value_dim),
            dtype=torch.float32,
            device=k.device,
        )
        if dht is None
        else dht.float()
    )
    for chunk in range(int(counts.max().item()) - 1, -1, -1):
        active = (chunk < counts)[..., None, None]
        kc = k_chunks[:, :, chunk]
        exp_g = g_chunks[:, :, chunk].exp()
        gamma = exp_g[..., -1, None, None]
        if x_chunks is None:
            qc = q_chunks[:, :, chunk]
            x = (a_chunks[:, :, chunk] * b_chunks[:, :, chunk][..., None, :]) @ kc
            r = qc - torch.tril(qc @ kc.transpose(-1, -2)) @ x
        else:
            x = x_chunks[:, :, chunk]
            r = r_chunks[:, :, chunk]
        y = kc @ state
        updated = (
            gamma * state
            + (scale * exp_g[..., None] * r).transpose(-1, -2) @ do_chunks[:, :, chunk]
            + x.transpose(-1, -2) @ (-gamma * y)
        )
        state = torch.where(active, updated, state)
    return None, state


def _warmup_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    num_warmup_chunks: torch.Tensor,
    scale: float,
    output_dh: bool,
):
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    if not output_dh:
        batched = _batched_warmup_dh(
            q, k, a, g, b, do, dht, bounds, num_warmup_chunks, scale
        )
        if batched is not None:
            return batched
    num_heads, key_dim, value_dim = do.shape[2], k.shape[-1], do.shape[-1]
    q_expanded = _expand_gva(q, num_heads).float()
    k_expanded = _expand_gva(k, num_heads).float()
    total_chunks = sum(
        (right - left + CHUNK_SIZE - 1) // CHUNK_SIZE
        for left, right in zip(bounds[:-1], bounds[1:])
    )
    history = (
        torch.zeros(
            (1, total_chunks, num_heads, key_dim, value_dim),
            dtype=torch.float32,
            device=k.device,
        )
        if output_dh
        else None
    )
    initial_grad = torch.zeros(
        (len(bounds) - 1, num_heads, key_dim, value_dim),
        dtype=torch.float32,
        device=k.device,
    )
    counts = num_warmup_chunks.to(device="cpu", dtype=torch.int64)
    chunk_base = 0
    for sequence, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        ranges = _chunk_ranges(right - left)
        for head in range(num_heads):
            count = min(int(counts[sequence, head]), len(ranges))
            state = (
                torch.zeros((key_dim, value_dim), dtype=torch.float32, device=k.device)
                if dht is None
                else dht[sequence, head].float()
            )
            for chunk_index in range(count - 1, -1, -1):
                begin, end = ranges[chunk_index]
                begin += left
                end += left
                if history is not None:
                    history[0, chunk_base + chunk_index, head] = state
                qc = q_expanded[0, begin:end, head]
                kc = k_expanded[0, begin:end, head]
                ac = a[0, begin:end, head, : end - begin].float()
                gc = g[0, begin:end, head].float()
                bc = b[0, begin:end, head].float()
                doc = do[0, begin:end, head].float()
                x = (ac * bc[None, :]) @ kc
                r = qc - torch.tril(qc @ kc.transpose(0, 1)) @ x
                y = kc @ state
                decay = torch.exp(gc[-1])
                state = (
                    decay * state
                    + (scale * gc.exp()[:, None] * r).transpose(0, 1) @ doc
                    + x.transpose(0, 1) @ (-decay * y)
                )
            initial_grad[sequence, head] = state
        chunk_base += len(ranges)
    return history, initial_grad


def fused_gdr_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    output_dh0: bool = True,
    output_dh: bool = True,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    num_warmup_chunks: torch.Tensor | None = None,
    state_v_first: bool = False,
):
    if not output_dh and not output_dh0:
        return None, None
    scale = float(k.shape[-1] ** -0.5 if scale is None else scale)
    if num_warmup_chunks is None:
        cu_seqlens = _fixed_length_cu_seqlens(cu_seqlens, k)
    if cu_seqlens is not None and num_warmup_chunks is None and not q.is_cuda:
        bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
        results = [
            fused_gdr_dh(
                q[:, left:right], k[:, left:right], a[:, left:right],
                g[:, left:right], b[:, left:right], do[:, left:right],
                None if dht is None else dht[index : index + 1],
                output_dh0=output_dh0, output_dh=output_dh, scale=scale,
                state_v_first=state_v_first,
            )
            for index, (left, right) in enumerate(zip(bounds[:-1], bounds[1:]))
        ]
        return (
            torch.cat([result[0] for result in results], dim=1) if output_dh else None,
            torch.cat([result[1] for result in results], dim=0) if output_dh0 else None,
        )
    terminal = _normal_state(dht, state_v_first)
    if num_warmup_chunks is not None:
        if cu_seqlens is None or k.shape[0] != 1:
            raise ValueError("warmup chunks require packed variable-length input")
        dh, dh0 = _warmup_dh(
            q,
            k,
            a,
            g,
            b,
            do,
            terminal,
            cu_seqlens,
            num_warmup_chunks,
            scale,
            output_dh,
        )
    else:
        weighted_a = _weight_kkt(a, g, cu_seqlens)
        zero_v = torch.zeros_like(do, dtype=torch.float32)
        w, _ = torch_w_u_fwd(
            k=k.float(),
            v=zero_v,
            g=g.float(),
            beta=b.float(),
            A=weighted_a.float(),
            cu_seqlens=cu_seqlens,
        )
        dv = torch_chunk_dv_bwd(
            q=q.float(),
            k=k.float(),
            g=g.float(),
            do=do.float(),
            cu_seqlens=cu_seqlens,
            scale=scale,
            chunk_size=CHUNK_SIZE,
        )
        if cu_seqlens is None:
            batch_size = k.shape[0]
        else:
            batch_size = cu_seqlens.shape[0] - 1
        dummy_h0 = torch.zeros(
            batch_size, do.shape[2], k.shape[-1], do.shape[-1],
            dtype=torch.float32, device=k.device,
        )
        dh, dh0, _ = torch_chunk_gdr_bwd(
            q=q.float(),
            k=k.float(),
            w=w,
            g=g.float(),
            do=do.float(),
            dv=dv,
            h0=dummy_h0,
            dht=terminal,
            cu_seqlens=cu_seqlens,
            scale=scale,
            chunk_size=CHUNK_SIZE,
            output_dh=output_dh,
        )
    dh_out = dh.to(k.dtype) if output_dh else None
    dh0_out = dh0.float() if output_dh0 else None
    if state_v_first:
        if dh_out is not None:
            dh_out = dh_out.transpose(-1, -2).contiguous()
        if dh0_out is not None:
            dh0_out = dh0_out.transpose(-1, -2).contiguous()
    return dh_out, dh0_out


def _chunk_last_gates(
    g: torch.Tensor,
    bounds: list[int],
    chunk_counts: list[int],
    chunk_size: int,
    max_chunks: int,
    reverse: bool,
):
    """Gather per-chunk total decay as ``[segments, max_chunks, heads]``.

    ``g`` holds the chunk-local gate cumsum, so the last token of a chunk
    already carries that chunk's total decay. Positions past a segment's chunk
    count are filled with zero, which can never trigger the negative threshold.
    """
    device = g.device
    index = torch.arange(max_chunks, device=device, dtype=torch.int64)
    starts = torch.tensor(bounds[:-1], device=device, dtype=torch.int64)
    ends = torch.tensor(bounds[1:], device=device, dtype=torch.int64)
    limits = torch.tensor(chunk_counts, device=device, dtype=torch.int64)
    if reverse:
        tokens = torch.minimum(
            starts[:, None] + (index[None, :] + 1) * chunk_size - 1,
            ends[:, None] - 1,
        )
    else:
        tokens = ends[:, None] - 1 - index[None, :] * chunk_size
    valid = index[None, :] < limits[:, None]
    tokens = torch.where(valid, tokens, starts[:, None])
    gates = g[0, tokens.reshape(-1)].float().reshape(
        len(chunk_counts), max_chunks, g.shape[2]
    )
    return torch.where(valid[..., None], gates, torch.zeros_like(gates)), limits


def _warmup_scan(
    chunk_gates: torch.Tensor,
    limits: torch.Tensor,
    threshold: float,
):
    """Vectorized replacement for the per-chunk warmup scan."""
    totals = torch.cumsum(chunk_gates, dim=1)
    hit = totals < threshold
    needs_fallback = ~hit.any(dim=1)
    first = hit.to(torch.int32).argmax(dim=1).long() + 1
    counts = torch.where(needs_fallback, limits[:, None].expand_as(first), first)
    return counts, needs_fallback


def _native_warmup_counts(
    g: torch.Tensor,
    bounds: list[int],
    chunk_counts: list[int],
    chunk_size: int,
    threshold: float,
    reverse: bool,
):
    """Single-launch warmup counts for chunk-aligned, equal-length segments."""
    if not g.is_cuda or os.getenv("FLASHQLA_PPU_NATIVE_WARMUP", "1") != "1":
        return None
    if not native.is_available():
        return None
    lengths = [right - left for left, right in zip(bounds[:-1], bounds[1:])]
    if not lengths or any(
        length != lengths[0] or length % chunk_size for length in lengths
    ):
        return None
    if any(count != lengths[0] // chunk_size for count in chunk_counts):
        return None
    segments = len(lengths)
    gates = g[0].reshape(segments, lengths[0], g.shape[2])[
        :, chunk_size - 1 :: chunk_size
    ]
    if reverse:
        gates = torch.flip(gates, dims=(1,))
    # The native kernel sums raw gates per chunk and then accumulates from the
    # tail; feeding one already-accumulated value per chunk is equivalent.
    counts, fallback = native.warmup_counts(
        gates.float().contiguous(), 1, threshold
    )
    return counts.long(), fallback


def _warmup_counts(
    g: torch.Tensor,
    bounds: list[int],
    chunk_counts: list[int],
    chunk_size: int,
    threshold: float,
    reverse: bool,
):
    segments = len(chunk_counts)
    num_heads = g.shape[2]
    max_chunks = max(chunk_counts) if chunk_counts else 0
    if max_chunks == 0:
        return (
            torch.zeros((segments, num_heads), dtype=torch.int64, device=g.device),
            torch.ones((segments, num_heads), dtype=torch.bool, device=g.device),
        )
    native_result = _native_warmup_counts(
        g, bounds, chunk_counts, chunk_size, threshold, reverse
    )
    if native_result is not None:
        return native_result
    gates, limits = _chunk_last_gates(
        g, bounds, chunk_counts, chunk_size, max_chunks, reverse
    )
    return _warmup_scan(gates, limits, threshold)


def get_warmup_chunks(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ht_mask: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = -10.0,
    reverse: bool = False,
):
    _check_chunk_size(chunk_size)
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    # Match the unidirectional Hopper kernel: only complete chunks take part in
    # this warmup scan. CP forward segments are normally aligned.
    chunk_counts = [
        (right - left) // chunk_size for left, right in zip(bounds[:-1], bounds[1:])
    ]
    counts, fallback = _warmup_counts(
        g, bounds, chunk_counts, chunk_size, threshold, reverse
    )
    skip = ht_mask.to(device=counts.device, dtype=torch.bool)[:, None]
    counts = torch.where(skip, torch.zeros_like(counts), counts)
    fallback = fallback & ~skip
    return counts.to(cu_seqlens.dtype), fallback.to(ht_mask.dtype)


def get_warmup_chunks_bidi(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ht_mask_fwd: torch.Tensor,
    ht_mask_bwd: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = -10.0,
):
    _check_chunk_size(chunk_size)
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    chunk_counts = [
        (right - left + chunk_size - 1) // chunk_size
        for left, right in zip(bounds[:-1], bounds[1:])
    ]
    count_fwd, fb_fwd = _warmup_counts(
        g, bounds, chunk_counts, chunk_size, threshold, False
    )
    count_bwd, fb_bwd = _warmup_counts(
        g, bounds, chunk_counts, chunk_size, threshold, True
    )
    skip_fwd = ht_mask_fwd.to(device=count_fwd.device, dtype=torch.bool)[:, None]
    skip_bwd = ht_mask_bwd.to(device=count_bwd.device, dtype=torch.bool)[:, None]
    count_fwd = torch.where(skip_fwd, torch.zeros_like(count_fwd), count_fwd)
    count_bwd = torch.where(skip_bwd, torch.zeros_like(count_bwd), count_bwd)
    fb_fwd = fb_fwd & ~skip_fwd
    fb_bwd = fb_bwd & ~skip_bwd
    return (
        torch.maximum(count_fwd, count_bwd).to(cu_seqlens.dtype),
        count_bwd.to(cu_seqlens.dtype),
        fb_fwd.to(ht_mask_fwd.dtype),
        fb_bwd.to(ht_mask_bwd.dtype),
    )


def _correct_states(
    raw_state: torch.Tensor | None,
    state_buffer: torch.Tensor,
    matrix_buffer: torch.Tensor,
    fallback_mask: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    reverse: bool,
    state_v_first: bool,
):
    prepared_backward_inputs = getattr(
        state_buffer, "_flashqla_ppu_backward_inputs", None
    )
    state_values = _normal_state(state_buffer, state_v_first)
    raw_values = _normal_state(raw_state, state_v_first)
    output_dtype = torch.float32 if raw_state is None else raw_state.dtype
    output_parts = []
    bounds = seq_map_r2c.to(device="cpu", dtype=torch.int64).tolist()
    for raw_index, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        current = (
            torch.zeros_like(state_values[0])
            if raw_values is None
            else raw_values[raw_index]
        )
        indices = range(right - 1, left - 1, -1) if reverse else range(left, right)
        sequence_parts = []
        for offset, index in enumerate(indices):
            sequence_parts.append(current.to(output_dtype))
            is_last = offset + 1 == right - left
            if not is_last:
                matrix = matrix_buffer[index].float()
                if reverse:
                    matrix = matrix.transpose(-1, -2)
                propagated = matrix @ current
                current = state_values[index] + torch.where(
                    fallback_mask[index, :, None, None],
                    propagated,
                    torch.zeros_like(propagated),
                )
        output_parts.extend(reversed(sequence_parts) if reverse else sequence_parts)
    output = torch.stack(output_parts)
    if state_v_first:
        output = output.transpose(-1, -2).contiguous()
    if prepared_backward_inputs is not None:
        output._flashqla_ppu_backward_inputs = prepared_backward_inputs
    return output


def correct_initial_states(
    raw_h0: torch.Tensor | None,
    ht_buffer: torch.Tensor,
    mt_buffer: torch.Tensor,
    fallback_mask: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    state_v_first: bool = False,
):
    return _correct_states(
        raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c, False, state_v_first
    )


def correct_terminal_states(
    raw_dht: torch.Tensor | None,
    dht_buffer: torch.Tensor,
    mt_buffer: torch.Tensor,
    fallback_mask: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    state_v_first: bool = False,
):
    return _correct_states(
        raw_dht, dht_buffer, mt_buffer, fallback_mask, seq_map_r2c, True, state_v_first
    )


__all__ = [
    "CHUNK_SIZE",
    "chunk_local_cumsum",
    "group_reduce_vector",
    "kkt_solve",
    "fused_gdr_fwd",
    "fused_gdr_h",
    "fused_gdr_bwd",
    "fused_gdr_dh",
    "get_warmup_chunks",
    "get_warmup_chunks_bidi",
    "correct_initial_states",
    "correct_terminal_states",
]
