"""PPU implementation of the official FlashQLA chunk algebra.

The implementation combines native PPU kernels with framework operators while
preserving the public FlashQLA contract.
"""

from __future__ import annotations

import os
import torch

from . import native
from .backward import decomposed_backward


CHUNK_SIZE = 64


def _expand_gva(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    num_qk_heads = x.shape[2]
    if num_qk_heads == num_v_heads:
        return x
    if num_v_heads % num_qk_heads:
        raise ValueError("num_v_heads must be divisible by num_qk_heads")
    return x.repeat_interleave(num_v_heads // num_qk_heads, dim=2)


def _relative_chunk_decay(g_cumsum: torch.Tensor) -> torch.Tensor:
    # [B,H,C,C], row t / column s contains exp(G_t-G_s).
    return torch.exp(g_cumsum[..., :, None] - g_cumsum[..., None, :])


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
    if k.is_cuda and k.dtype == torch.bfloat16:
        batch_shape = k.shape[:-2]
        flat_k = k.reshape(-1, length, k.shape[-1])
        gram = torch.bmm(
            flat_k,
            flat_k.transpose(-1, -2),
            out_dtype=torch.float32,
        ).reshape(*batch_shape, length, length)
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


def _run_equal_length(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    output_h: bool,
):
    batch_size, num_tokens, _, key_dim = q.shape
    num_v_heads, value_dim = v.shape[2:]

    aiu_fused_candidate = (
        native.is_flash_qla_fused_available()
        and native.is_gated_strided_kkt_available()
        and os.getenv("FLASHQLA_PPU_AIU_FUSED", "1") == "1"
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and key_dim == value_dim == 128
        and num_tokens % CHUNK_SIZE == 0
        and not output_h
        and q.is_cuda
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and not any(
            tensor.requires_grad
            for tensor in (q, k, v, g, beta, initial_state)
        )
    )
    aiu_kkt_candidate = (
        native.is_aiu_available()
        and q.dtype == k.dtype == torch.bfloat16
        and key_dim == 128
        and num_tokens % CHUNK_SIZE == 0
        and q.is_cuda
        and not any(tensor.requires_grad for tensor in (k, beta))
    )
    strided_kkt_candidate = (
        aiu_fused_candidate
        and hasattr(native, "aiu_kkt_solve_bf16_128_strided")
    )
    native_gate_candidate = (
        aiu_fused_candidate
        and native.is_available()
        and hasattr(native, "prepare_gate_beta")
    )
    matrix_dtype = torch.bfloat16 if aiu_fused_candidate else torch.float32
    qh = (
        None
        if aiu_fused_candidate
        else _expand_gva(q, num_v_heads).to(matrix_dtype).transpose(1, 2).contiguous()
    )
    expanded_k = None if strided_kkt_candidate else _expand_gva(k, num_v_heads)
    if strided_kkt_candidate:
        kh_bf16 = None
        kh = None
    elif aiu_kkt_candidate:
        kh_bf16 = expanded_k.transpose(1, 2).contiguous()
        kh = kh_bf16 if aiu_fused_candidate else kh_bf16.float()
    else:
        kh_bf16 = None
        kh = expanded_k.to(matrix_dtype).transpose(1, 2).contiguous()
    vh = None if aiu_fused_candidate else v.float().transpose(1, 2).contiguous()
    if native_gate_candidate:
        gh = None
        bh = None
    else:
        gh = g.float().transpose(1, 2).contiguous()
        bh = beta.float().transpose(1, 2).contiguous()

    num_chunks = (num_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    padded_tokens = num_chunks * CHUNK_SIZE
    token_padding = padded_tokens - num_tokens
    if token_padding:
        assert qh is not None and kh is not None and vh is not None
        qh = torch.nn.functional.pad(qh, (0, 0, 0, token_padding))
        kh = torch.nn.functional.pad(kh, (0, 0, 0, token_padding))
        vh = torch.nn.functional.pad(vh, (0, 0, 0, token_padding))
        assert gh is not None and bh is not None
        gh = torch.nn.functional.pad(gh, (0, token_padding))
        bh = torch.nn.functional.pad(bh, (0, token_padding))

    q_chunks = (
        None
        if qh is None
        else qh.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE, key_dim
        )
    )
    k_chunks = (
        None
        if kh is None
        else kh.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE, key_dim
        )
    )
    v_chunks = (
        None
        if vh is None
        else vh.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE, value_dim
        )
    )
    if native_gate_candidate:
        g_cumsum_chunks, b_chunks = native.prepare_gate_beta(
            g.float().contiguous(), beta.float().contiguous()
        )
    else:
        assert gh is not None and bh is not None
        g_chunks = gh.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE
        )
        b_chunks = bh.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE
        )
        # Batch all chunk-local work so cumsum, KKT and QK each launch once.
        g_cumsum_chunks = g_chunks.cumsum(dim=-1)
    if strided_kkt_candidate:
        kkt_input = None
        kkt_chunks = native.aiu_kkt_solve_bf16_128_strided_gated(
            k, b_chunks.contiguous(), g_cumsum_chunks.contiguous()
        )
    elif kh_bf16 is not None:
        kkt_input = kh_bf16.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE, key_dim
        )
        kkt_chunks = _kkt_inverse(kkt_input, b_chunks)
    else:
        assert k_chunks is not None
        kkt_input = k_chunks
        kkt_chunks = _kkt_inverse(kkt_input, b_chunks)
    if aiu_fused_candidate:
        output, state = native.flash_qla_fused_forward_bf16_128(
            q,
            k,
            v,
            kkt_chunks.to(torch.bfloat16).contiguous(),
            g_cumsum_chunks.contiguous(),
            b_chunks.contiguous(),
            initial_state.float().contiguous(),
            scale,
        )
        gate_cumsum = g_cumsum_chunks.permute(0, 2, 3, 1).reshape(
            batch_size, num_tokens, num_v_heads
        )
        kkt = kkt_chunks.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_tokens, num_v_heads, CHUNK_SIZE
        ).to(k.dtype)
        return gate_cumsum, kkt, output, None, state

    assert q_chunks is not None and k_chunks is not None and v_chunks is not None

    qk_chunks = (
        q_chunks @ k_chunks.transpose(-1, -2)
    ) * scale
    use_native_coefficients = (
        native.is_available()
        and q.is_cuda
        and not (
            g_cumsum_chunks.requires_grad
            or kkt_chunks.requires_grad
            or qk_chunks.requires_grad
        )
    )
    if use_native_coefficients:
        (
            weighted_kkt,
            attention_chunks,
            gamma_chunks,
            reverse_decay_chunks,
        ) = native.chunk_coefficients(
            g_cumsum_chunks.contiguous(),
            kkt_chunks.contiguous(),
            qk_chunks.contiguous(),
        )
    else:
        decay_chunks = _relative_chunk_decay(g_cumsum_chunks)
        weighted_kkt = torch.tril(decay_chunks) * kkt_chunks
        attention_chunks = torch.tril(decay_chunks * qk_chunks)
        gamma_chunks = torch.exp(g_cumsum_chunks)
        reverse_decay_chunks = torch.exp(
            g_cumsum_chunks[..., -1:, None]
            - g_cumsum_chunks[..., :, None]
        ).squeeze(-1)
    gamma_last_chunks = gamma_chunks[..., -1].contiguous()

    use_fused_chunks = (
        use_native_coefficients
        and not output_h
        and os.getenv("FLASHQLA_PPU_FUSED_CHUNKS", "0") == "1"
    )
    if use_fused_chunks:
        output_chunks, state = native.fused_chunks_fp32(
            q_chunks.contiguous(),
            k_chunks.contiguous(),
            v_chunks.contiguous(),
            b_chunks.contiguous(),
            weighted_kkt.contiguous(),
            attention_chunks.contiguous(),
            gamma_chunks.contiguous(),
            reverse_decay_chunks.contiguous(),
            initial_state.float().contiguous(),
            scale,
        )
        output = output_chunks.permute(0, 2, 3, 1, 4)
        output = output.reshape(
            batch_size, padded_tokens, num_v_heads, value_dim
        )[:, :num_tokens].to(q.dtype)
        gate_cumsum = g_cumsum_chunks.permute(0, 2, 3, 1)
        gate_cumsum = gate_cumsum.reshape(
            batch_size, padded_tokens, num_v_heads
        )[:, :num_tokens]
        kkt = weighted_kkt.permute(0, 2, 3, 1, 4)
        kkt = kkt.reshape(
            batch_size, padded_tokens, num_v_heads, CHUNK_SIZE
        )[:, :num_tokens].to(k.dtype)
        return gate_cumsum, kkt, output, None, state

    state = initial_state.float()
    output_chunks = []
    state_chunks = []

    for chunk in range(num_chunks):
        qc = q_chunks[:, :, chunk]
        kc = k_chunks[:, :, chunk]
        vc = v_chunks[:, :, chunk]
        bc = b_chunks[:, :, chunk]
        if output_h:
            state_chunks.append(state)

        # V_delta = V - diag(gamma) K S
        u = kc @ state
        if use_native_coefficients:
            w = native.prepare_w(
                u, v_chunks, b_chunks, gamma_chunks, chunk
            )
        else:
            w = bc[..., :, None] * (
                vc - gamma_chunks[:, :, chunk, :, None] * u
            )

        # V' = (Gamma o A) diag(beta) V_delta
        vp = weighted_kkt[:, :, chunk] @ w

        # O = diag(gamma) Q S + (Lower(Gamma) o QK^T) V'
        q_state = qc @ state
        attended = attention_chunks[:, :, chunk] @ vp
        if use_native_coefficients:
            output = native.combine_output(
                q_state,
                attended,
                gamma_chunks,
                num_chunks,
                chunk,
                scale,
            )
            decayed_values = native.decay_values(
                vp, reverse_decay_chunks, num_chunks, chunk
            )
        else:
            output = (
                gamma_chunks[:, :, chunk, :, None] * q_state * scale
                + attended
            )
            decayed_values = (
                reverse_decay_chunks[:, :, chunk, :, None] * vp
            )

        # S_next = gamma_last S + K^T diag(gamma_last/gamma) V'
        state_delta = kc.transpose(-1, -2) @ decayed_values
        if use_native_coefficients:
            state = native.update_state(
                state, state_delta, gamma_last_chunks, num_chunks, chunk
            )
        else:
            state = (
                gamma_last_chunks[:, :, chunk, None, None] * state
                + state_delta
            )

        output_chunks.append(output)

    output = torch.stack(output_chunks, dim=2).permute(0, 2, 3, 1, 4)
    output = output.reshape(batch_size, padded_tokens, num_v_heads, value_dim)
    output = output[:, :num_tokens].to(q.dtype)
    gate_cumsum = g_cumsum_chunks.permute(0, 2, 3, 1)
    gate_cumsum = gate_cumsum.reshape(batch_size, padded_tokens, num_v_heads)
    gate_cumsum = gate_cumsum[:, :num_tokens]
    kkt = weighted_kkt.permute(0, 2, 3, 1, 4)
    kkt = kkt.reshape(
        batch_size, padded_tokens, num_v_heads, CHUNK_SIZE
    )[:, :num_tokens].to(k.dtype)
    history = (
        torch.stack(state_chunks, dim=1)
        if output_h
        else None
    )
    return gate_cumsum, kkt, output, history, state


def _segment_warmup_affine(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    warmup_threshold: float = -10.0,
):
    """Return official gate-warmup S*, M, and exact-fallback head mask."""
    batch_size, num_tokens, _, key_dim = k.shape
    num_v_heads, value_dim = v.shape[2:]
    if num_tokens % CHUNK_SIZE:
        raise ValueError("AutoCP affine segments must contain complete chunks")
    expanded_k = _expand_gva(k, num_v_heads)
    if k.dtype == torch.bfloat16 and native.is_aiu_available() and k.is_cuda:
        kh_bf16 = expanded_k.transpose(1, 2).contiguous()
        kh = kh_bf16.float()
    else:
        kh_bf16 = None
        kh = expanded_k.float().transpose(1, 2).contiguous()
    affine_aiu_candidate = (
        native.is_flash_qla_affine_available()
        and k.dtype == v.dtype == torch.bfloat16
        and key_dim == value_dim == 128
        and k.is_cuda
        and not any(tensor.requires_grad for tensor in (k, v, g, beta))
    )
    if affine_aiu_candidate:
        vh_bf16 = v.transpose(1, 2).contiguous()
        vh = vh_bf16.float()
    else:
        vh_bf16 = None
        vh = v.float().transpose(1, 2).contiguous()
    gh = g.float().transpose(1, 2).contiguous()
    bh = beta.float().transpose(1, 2).contiguous()

    num_chunks = num_tokens // CHUNK_SIZE
    if native.is_available() and g.is_cuda:
        warmup_chunks, fallback = native.warmup_counts(
            g.float().contiguous(), CHUNK_SIZE, warmup_threshold
        )
    else:
        chunk_decay = gh.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE
        ).sum(dim=-1)
        reverse_decay = torch.cumsum(
            torch.flip(chunk_decay, dims=(-1,)), dim=-1
        )
        threshold_hit = reverse_decay < warmup_threshold
        fallback = ~threshold_hit.any(dim=-1)
        first_hit = threshold_hit.to(torch.int32).argmax(dim=-1) + 1
        warmup_chunks = torch.where(
            fallback,
            torch.full_like(first_hit, num_chunks),
            first_hit,
        )
    max_warmup = int(warmup_chunks.max().item())
    first_chunk = num_chunks - max_warmup

    k_chunks = kh.reshape(
        batch_size, num_v_heads, num_chunks, CHUNK_SIZE, key_dim
    )[:, :, first_chunk:].contiguous()
    kkt_input = (
        kh_bf16.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE, key_dim
        )[:, :, first_chunk:].contiguous()
        if kh_bf16 is not None
        else k_chunks
    )
    v_chunks = vh.reshape(
        batch_size, num_v_heads, num_chunks, CHUNK_SIZE, value_dim
    )[:, :, first_chunk:].contiguous()
    v_chunks_bf16 = (
        vh_bf16.reshape(
            batch_size, num_v_heads, num_chunks, CHUNK_SIZE, value_dim
        )[:, :, first_chunk:].contiguous()
        if vh_bf16 is not None
        else None
    )
    g_chunks = gh.reshape(
        batch_size, num_v_heads, num_chunks, CHUNK_SIZE
    )[:, :, first_chunk:].contiguous()
    b_chunks = bh.reshape(
        batch_size, num_v_heads, num_chunks, CHUNK_SIZE
    )[:, :, first_chunk:].contiguous()
    g_cumsum_chunks = g_chunks.cumsum(dim=-1)
    kkt_chunks = _kkt_inverse(kkt_input, b_chunks)
    x_unscaled = kkt_chunks.transpose(-1, -2) @ k_chunks
    use_native_affine = (
        native.is_available()
        and k.is_cuda
        and not (
            x_unscaled.requires_grad
            or b_chunks.requires_grad
            or g_cumsum_chunks.requires_grad
        )
    )
    x_chunks = (
        native.scale_rows_negative(
            x_unscaled.contiguous(), b_chunks.contiguous()
        )
        if use_native_affine
        else -b_chunks[..., :, None] * x_unscaled
    )
    gamma_last_chunks = torch.exp(
        g_cumsum_chunks[..., -1]
    ).contiguous()
    reverse_decay_chunks = torch.exp(
        g_cumsum_chunks[..., -1:, None] - g_cumsum_chunks[..., :, None]
    ).contiguous()

    state = torch.zeros(
        batch_size,
        num_v_heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=k.device,
    )
    eye = torch.eye(key_dim, dtype=torch.float32, device=k.device)
    matrix = eye.expand(batch_size, num_v_heads, key_dim, key_dim) * fallback[
        ..., None, None
    ]
    calculate_matrix = bool(fallback.any().item())
    all_fallback = bool(fallback.all().item())

    if affine_aiu_candidate and kh_bf16 is not None:
        x_chunks_bf16 = x_chunks.to(torch.bfloat16).contiguous()
        reverse_decay_flat = reverse_decay_chunks.squeeze(-1).contiguous()
        state = native.flash_qla_affine_state_bf16_128(
            kkt_input,
            v_chunks_bf16,
            x_chunks_bf16,
            gamma_last_chunks,
            reverse_decay_flat,
            warmup_chunks.contiguous(),
            fallback.contiguous(),
            matrix_mode=False,
        )
        if calculate_matrix:
            matrix = native.flash_qla_affine_state_bf16_128(
                kkt_input,
                v_chunks_bf16,
                x_chunks_bf16,
                gamma_last_chunks,
                reverse_decay_flat,
                warmup_chunks.contiguous(),
                fallback.contiguous(),
                matrix_mode=True,
            )
        return state, matrix, fallback, all_fallback

    for local_chunk, chunk in enumerate(range(first_chunk, num_chunks)):
        kc = k_chunks[:, :, local_chunk]
        vc = v_chunks[:, :, local_chunk]

        # Official CP preprocess:
        # X=-diag(beta) A^T K
        # Y=gamma_last K S* - diag(gamma_last/gamma) V
        # S*_next=gamma_last S* + X^T Y
        # M_next=gamma_last (M + X^T K M)
        x = x_chunks[:, :, local_chunk]
        gamma_last = gamma_last_chunks[:, :, local_chunk]
        reverse_decay = reverse_decay_chunks[:, :, local_chunk]
        u = kc @ state
        y = (
            native.affine_y(
                u,
                v_chunks,
                gamma_last_chunks,
                reverse_decay_chunks,
                local_chunk,
            )
            if use_native_affine
            else gamma_last[..., None, None] * u - reverse_decay * vc
        )
        state_delta = x.transpose(-1, -2) @ y
        next_state = (
            native.update_state(
                state,
                state_delta,
                gamma_last_chunks,
                max_warmup,
                local_chunk,
            )
            if use_native_affine
            else gamma_last[..., None, None] * state + state_delta
        )
        active = chunk >= num_chunks - warmup_chunks
        state = (
            next_state
            if all_fallback
            else torch.where(active[..., None, None], next_state, state)
        )
        if calculate_matrix:
            z = kc @ matrix
            matrix_delta = x.transpose(-1, -2) @ z
            next_matrix = (
                native.update_state(
                    matrix,
                    matrix_delta,
                    gamma_last_chunks,
                    max_warmup,
                    local_chunk,
                    scale_delta=True,
                )
                if use_native_affine
                else gamma_last[..., None, None] * (matrix + matrix_delta)
            )
            matrix = (
                next_matrix
                if all_fallback
                else torch.where(
                    fallback[..., None, None], next_matrix, matrix
                )
            )
    return state, matrix, fallback, all_fallback


def _auto_cp_max_local_chunks(num_chunks: int, num_heads: int, device) -> int:
    """Return the PPU-tuned CP segment size, with an override for profiling."""
    del num_heads, device
    override = os.getenv("FLASHQLA_PPU_CP_LOCAL_CHUNKS")
    if override is not None:
        local_chunks = int(override)
        if local_chunks < 1:
            raise ValueError("FLASHQLA_PPU_CP_LOCAL_CHUNKS must be positive")
        return local_chunks
    # The NVIDIA implementation derives this from SM count. PPU sweeps show a
    # piecewise optimum because batched ACBLAS efficiency grows with segment
    # size while Python/operator launch cost grows with the number of stages.
    # The native fused recurrence wins through 28 chunks; AutoCP crosses over
    # at 29--32 chunks depending on gate strength and is decisively faster by
    # 32 chunks. Keep short sequences in one launch to avoid affine setup.
    if num_chunks <= 28:
        return num_chunks
    if num_chunks <= 56:
        return 4
    # With the native affine recurrence, eight segments consistently balance
    # parallel fused work against KKT/affine setup and warmup fallback cost.
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
    # Fixed-card sweeps show a deliberate crossover to 1024-token segments:
    # the batched-GEMM path amortizes layout work better than the <=512-token
    # native micro-kernels once enough chunks or value heads are available.
    # Keep small workloads on the existing four-chunk minimum, and preserve
    # the original scalable rule above 16K tokens.
    crossover = 64 if num_heads >= 8 else 80
    if crossover <= num_chunks <= 256:
        return 16
    return max(4, (num_chunks + 15) // 16)


def _pad_cp_tail(
    x: torch.Tensor, padded_tokens: int, repeat_last: bool = False
) -> torch.Tensor:
    """Pad a CP tail to a complete chunk without changing its recurrence."""
    padding = padded_tokens - x.shape[1]
    if padding <= 0:
        return x
    if repeat_last:
        values = x[:, -1:].expand(-1, padding, *x.shape[2:])
    else:
        values = x.new_zeros(x.shape[0], padding, *x.shape[2:])
    return torch.cat((x, values), dim=1)


def _run_auto_cp_equal_length(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    output_h: bool,
    return_cp_cache: bool = False,
):
    """Exact intra-card CP using the official S*/M correction equations."""
    _, num_tokens, _, _ = q.shape
    num_v_heads = v.shape[2]
    num_chunks = (num_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    selector = (
        _auto_cp_backward_local_chunks
        if return_cp_cache
        else _auto_cp_max_local_chunks
    )
    local_chunks = selector(num_chunks, num_v_heads, q.device)
    if num_chunks <= local_chunks:
        return (
            *_run_equal_length(
                q, k, v, g, beta, initial_state, scale, output_h
            ),
            None,
        )

    segment_tokens = local_chunks * CHUNK_SIZE
    full_segments = num_tokens // segment_tokens
    full_tokens = full_segments * segment_tokens
    has_tail = full_tokens < num_tokens
    total_segments = full_segments + int(has_tail)

    def split_full(x: torch.Tensor, count: int = full_segments) -> torch.Tensor:
        tokens = count * segment_tokens
        return x[:, :tokens].reshape(count, segment_tokens, *x.shape[2:])

    # Initial states need every transition except the last. A cache also keeps
    # the last M/fallback pair for the reverse terminal-state correction.
    affine_full_segments = (
        full_segments
        if return_cp_cache or has_tail
        else full_segments - 1
    )
    affine_parts = []
    if affine_full_segments:
        affine_parts.append(
            _segment_warmup_affine(
                split_full(k, affine_full_segments),
                split_full(v, affine_full_segments),
                split_full(g, affine_full_segments),
                split_full(beta, affine_full_segments),
            )[:3]
        )
    if return_cp_cache and has_tail:
        tail_tokens = num_tokens - full_tokens
        padded_tail_tokens = (
            (tail_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
        )
        affine_parts.append(
            _segment_warmup_affine(
                _pad_cp_tail(k[:, full_tokens:], padded_tail_tokens),
                _pad_cp_tail(v[:, full_tokens:], padded_tail_tokens),
                _pad_cp_tail(g[:, full_tokens:], padded_tail_tokens),
                _pad_cp_tail(beta[:, full_tokens:], padded_tail_tokens),
            )[:3]
        )
    stars = torch.cat([part[0] for part in affine_parts], dim=0)
    matrices = torch.cat([part[1] for part in affine_parts], dim=0)
    fallback = torch.cat([part[2] for part in affine_parts], dim=0)

    corrected = initial_state[0]
    initial_parts = []
    for rank in range(total_segments):
        initial_parts.append(corrected)
        if rank + 1 < total_segments:
            propagated = matrices[rank] @ corrected
            corrected = stars[rank] + torch.where(
                fallback[rank, ..., None, None],
                propagated,
                torch.zeros_like(propagated),
            )
    cp_initial = torch.stack(initial_parts, dim=0)
    cp_cache = (
        (
            "ppu_auto_cp_v1",
            num_tokens,
            local_chunks,
            cp_initial,
            matrices,
            fallback,
        )
        if return_cp_cache
        else None
    )

    gcum_parts = []
    kkt_parts = []
    output_parts = []
    history_parts = []
    final_state = None
    if full_segments:
        gc, aa, oo, hh, ss = _run_equal_length(
            split_full(q),
            split_full(k),
            split_full(v),
            split_full(g),
            split_full(beta),
            cp_initial[:full_segments],
            scale,
            output_h,
        )
        gcum_parts.append(gc.reshape(1, -1, *gc.shape[2:]))
        kkt_parts.append(aa.reshape(1, -1, *aa.shape[2:]))
        output_parts.append(oo.reshape(1, -1, *oo.shape[2:]))
        if output_h:
            history_parts.append(hh.reshape(1, -1, *hh.shape[2:]))
        final_state = ss[-1:]
    if has_tail:
        gc, aa, oo, hh, ss = _run_equal_length(
            q[:, full_tokens:],
            k[:, full_tokens:],
            v[:, full_tokens:],
            g[:, full_tokens:],
            beta[:, full_tokens:],
            cp_initial[-1:],
            scale,
            output_h,
        )
        gcum_parts.append(gc)
        kkt_parts.append(aa)
        output_parts.append(oo)
        if output_h:
            history_parts.append(hh)
        final_state = ss

    return (
        torch.cat(gcum_parts, dim=1),
        torch.cat(kkt_parts, dim=1),
        torch.cat(output_parts, dim=1),
        torch.cat(history_parts, dim=1) if output_h else None,
        final_state,
        cp_cache,
    )


def official_chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    auto_cp: bool = True,
    state_v_first: bool = False,
    enable_fwd_cp_cache: bool = False,
):
    """Execute the official chunk/KKT FlashQLA forward dataflow on PPU."""
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q/k/v must be FP16 or BF16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q/k/v dtypes must match")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must have shape [B,T,H,D]")

    batch_size, num_tokens, _, key_dim = q.shape
    num_v_heads, value_dim = v.shape[2:]
    scale = float(key_dim**-0.5 if scale is None else scale)

    if cu_seqlens is None:
        num_sequences = batch_size
        if initial_state is None:
            state = torch.zeros(
                batch_size,
                num_v_heads,
                key_dim,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            state = (
                initial_state.transpose(-1, -2)
                if state_v_first
                else initial_state
            ).float().contiguous()
        if auto_cp and batch_size == 1:
            gcum, a, output, history, final_state, cp_cache = (
                _run_auto_cp_equal_length(
                    q, k, v, g, beta, state, scale, output_h,
                    return_cp_cache=enable_fwd_cp_cache,
                )
            )
        else:
            gcum, a, output, history, final_state = _run_equal_length(
                q, k, v, g, beta, state, scale, output_h
            )
            cp_cache = None
    else:
        if batch_size != 1:
            raise ValueError("varlen input requires batch size 1")
        bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
        num_sequences = len(bounds) - 1
        if initial_state is None:
            states = torch.zeros(
                num_sequences,
                num_v_heads,
                key_dim,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            states = (
                initial_state.transpose(-1, -2)
                if state_v_first
                else initial_state
            ).float().contiguous()

        gcum_parts = []
        a_parts = []
        output_parts = []
        history_parts = []
        final_parts = []
        sequence_caches = []
        for sequence, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
            if auto_cp:
                gc, aa, oo, hh, ss, sequence_cache = (
                    _run_auto_cp_equal_length(
                        q[:, left:right],
                        k[:, left:right],
                        v[:, left:right],
                        g[:, left:right],
                        beta[:, left:right],
                        states[sequence : sequence + 1],
                        scale,
                        output_h,
                        return_cp_cache=enable_fwd_cp_cache,
                    )
                )
                sequence_caches.append(sequence_cache)
            else:
                gc, aa, oo, hh, ss = _run_equal_length(
                    q[:, left:right],
                    k[:, left:right],
                    v[:, left:right],
                    g[:, left:right],
                    beta[:, left:right],
                    states[sequence : sequence + 1],
                    scale,
                    output_h,
                )
            gcum_parts.append(gc)
            a_parts.append(aa)
            output_parts.append(oo)
            if output_h:
                history_parts.append(hh)
            final_parts.append(ss)

        gcum = torch.cat(gcum_parts, dim=1)
        a = torch.cat(a_parts, dim=1)
        output = torch.cat(output_parts, dim=1)
        history = (
            torch.cat(history_parts, dim=1)
            if output_h
            else None
        )
        final_state = torch.cat(final_parts, dim=0)
        cp_cache = (
            ("ppu_auto_cp_varlen_v1", tuple(bounds), tuple(sequence_caches))
            if enable_fwd_cp_cache and auto_cp
            else None
        )

    if state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
        if history is not None:
            history = history.transpose(-1, -2).contiguous()

    return (
        gcum,
        a,
        output,
        history,
        final_state if output_final_state else None,
        cp_cache,
    )


def _uncumsum_gate(
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    def uncumsum_sequence(sequence: torch.Tensor) -> torch.Tensor:
        # The cumulative gate resets at every chunk boundary.  Recover every
        # non-boundary token with one vector subtraction, then restore the
        # first token of each chunk.  This is the exact inverse of the chunked
        # cumsum while avoiding one torch.cat launch per 64-token chunk.
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


def _recurrent_backward_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    initial_state: torch.Tensor,
    terminal_state_grad: torch.Tensor,
    scale: float,
):
    """Differentiate the official token recurrence without PPU autograd.

    Only chunk-boundary states are retained globally. States inside one chunk
    are reconstructed just before its reverse pass, keeping scratch bounded by
    ``CHUNK_SIZE`` instead of sequence length.
    """
    batch_size, num_tokens, q_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if value_heads % q_heads:
        raise ValueError("num_v_heads must be divisible by num_qk_heads")
    repeat = value_heads // q_heads
    q_expanded = q.float().repeat_interleave(repeat, dim=2)
    k_expanded = k.float().repeat_interleave(repeat, dim=2)
    values = v.float()
    gates = raw_g.float()
    betas = beta.float()
    output_grads = do[:, :num_tokens].float()

    def row_mm(row: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.bmm(
            row.reshape(-1, 1, key_dim),
            matrix.reshape(-1, key_dim, value_dim),
        ).reshape(batch_size, value_heads, value_dim)

    def matrix_mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return torch.bmm(
            matrix.reshape(-1, key_dim, value_dim),
            vector.reshape(-1, value_dim, 1),
        ).reshape(batch_size, value_heads, key_dim)

    state = initial_state.float().contiguous()
    boundaries = []
    for chunk_start in range(0, num_tokens, CHUNK_SIZE):
        boundaries.append(state)
        chunk_end = min(chunk_start + CHUNK_SIZE, num_tokens)
        for token in range(chunk_start, chunk_end):
            kt = k_expanded[:, token].contiguous()
            decay = torch.exp(gates[:, token])
            prediction = row_mm(kt, state)
            residual = values[:, token] - decay[..., None] * prediction
            update = betas[:, token, :, None] * residual
            state = (
                decay[..., None, None] * state
                + kt[..., :, None] * update[..., None, :]
            )

    dq_expanded = torch.empty_like(q_expanded, dtype=torch.float32)
    dk_expanded = torch.empty_like(k_expanded, dtype=torch.float32)
    dv = torch.empty_like(values, dtype=torch.float32)
    dg = torch.empty_like(gates, dtype=torch.float32)
    db = torch.empty_like(betas, dtype=torch.float32)
    dstate = terminal_state_grad.float().contiguous()

    for chunk_index in range(len(boundaries) - 1, -1, -1):
        chunk_start = chunk_index * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, num_tokens)
        chunk_states = [boundaries[chunk_index]]
        state = boundaries[chunk_index]
        for token in range(chunk_start, chunk_end):
            kt = k_expanded[:, token].contiguous()
            decay = torch.exp(gates[:, token])
            prediction = row_mm(kt, state)
            residual = values[:, token] - decay[..., None] * prediction
            update = betas[:, token, :, None] * residual
            state = (
                decay[..., None, None] * state
                + kt[..., :, None] * update[..., None, :]
            )
            chunk_states.append(state)

        for token in range(chunk_end - 1, chunk_start - 1, -1):
            state_before = chunk_states[token - chunk_start]
            state_after = chunk_states[token - chunk_start + 1]
            qt = q_expanded[:, token]
            kt = k_expanded[:, token]
            vt = values[:, token]
            gt = gates[:, token]
            bt = betas[:, token]
            dot = output_grads[:, token]
            decay = torch.exp(gt)

            dq_expanded[:, token] = matrix_mv(
                state_after, dot
            ) * scale
            dstate = (
                dstate
                + (qt * scale)[..., :, None] * dot[..., None, :]
            )

            prediction = row_mm(kt, state_before)
            residual = vt - decay[..., None] * prediction
            update = bt[..., None] * residual
            du = row_mm(kt, dstate)
            dk_outer = matrix_mv(dstate, update)
            db[:, token] = (du * residual).sum(dim=-1)
            dr = bt[..., None] * du
            dv[:, token] = dr
            de = (dstate * state_before).sum(dim=(-1, -2))
            de = de - (dr * prediction).sum(dim=-1)
            dp = -decay[..., None] * dr
            dk_prediction = matrix_mv(state_before, dp)
            dk_expanded[:, token] = dk_outer + dk_prediction
            dstate = (
                decay[..., None, None] * dstate
                + kt[..., :, None] * dp[..., None, :]
            )
            dg[:, token] = de * decay

    if repeat > 1:
        dq = dq_expanded.reshape(
            batch_size, num_tokens, q_heads, repeat, key_dim
        ).sum(dim=3)
        dk = dk_expanded.reshape(
            batch_size, num_tokens, q_heads, repeat, key_dim
        ).sum(dim=3)
    else:
        dq = dq_expanded
        dk = dk_expanded
    return dq, dk, dv, db, dg, dstate


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
        # Official cp_bwd:
        # X=(A diag(beta))K, R=Q-Lower(QK^T)X,
        # dH=e^g_last dH + (scale diag(e^g)R)^T dO
        #    + X^T(-e^g_last K dH).
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


def _run_auto_cp_backward_equal_length(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    cp_cache=None,
):
    """Segment backward using the official AutoCP terminal correction."""
    batch_size, num_tokens, _, key_dim = q.shape
    num_v_heads, value_dim = v.shape[2:]
    num_chunks = (num_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    cache_valid = (
        isinstance(cp_cache, tuple)
        and len(cp_cache) == 6
        and cp_cache[0] == "ppu_auto_cp_v1"
        and cp_cache[1] == num_tokens
    )
    local_chunks = (
        cp_cache[2]
        if cache_valid
        else _auto_cp_backward_local_chunks(
            num_chunks, num_v_heads, q.device
        )
    )
    segment_tokens = local_chunks * CHUNK_SIZE
    if (
        batch_size != 1
        or num_chunks <= local_chunks
    ):
        return decomposed_backward(
            q, k, v, g, beta, A, scale, initial_state, do, dht, None,
            chunk_size=CHUNK_SIZE,
        )

    full_segments = num_tokens // segment_tokens
    full_tokens = full_segments * segment_tokens
    has_tail = full_tokens < num_tokens
    total_segments = full_segments + int(has_tail)

    def split_full(x: torch.Tensor) -> torch.Tensor:
        return x[:, :full_tokens].reshape(
            full_segments, segment_tokens, *x.shape[2:]
        )

    if cache_valid:
        cp_initial, matrices, fallback = cp_cache[3:]
    else:
        raw_g = _uncumsum_gate(g, None)
        affine_parts = []
        if full_segments:
            affine_parts.append(
                _segment_warmup_affine(
                    split_full(k),
                    split_full(v),
                    split_full(raw_g),
                    split_full(beta),
                )[:3]
            )
        if has_tail:
            tail_tokens = num_tokens - full_tokens
            padded_tail_tokens = (
                (tail_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
            )
            affine_parts.append(
                _segment_warmup_affine(
                    _pad_cp_tail(k[:, full_tokens:], padded_tail_tokens),
                    _pad_cp_tail(v[:, full_tokens:], padded_tail_tokens),
                    _pad_cp_tail(
                        raw_g[:, full_tokens:], padded_tail_tokens
                    ),
                    _pad_cp_tail(
                        beta[:, full_tokens:], padded_tail_tokens
                    ),
                )[:3]
            )
        if len(affine_parts) == 1:
            stars, matrices, fallback = affine_parts[0]
        else:
            stars = torch.cat([part[0] for part in affine_parts], dim=0)
            matrices = torch.cat([part[1] for part in affine_parts], dim=0)
            fallback = torch.cat([part[2] for part in affine_parts], dim=0)

    raw_h0 = (
        torch.zeros(
            1,
            num_v_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        )
        if initial_state is None
        else initial_state.float()
    )
    if not cache_valid:
        corrected = raw_h0[0]
        initial_parts = []
        for rank in range(total_segments):
            initial_parts.append(corrected)
            if rank + 1 < total_segments:
                propagated = matrices[rank] @ corrected
                corrected = stars[rank] + torch.where(
                    fallback[rank, ..., None, None],
                    propagated,
                    torch.zeros_like(propagated),
                )
        cp_initial = torch.stack(initial_parts)

    local_dh_parts = []
    if full_segments:
        local_dh_parts.append(
            _official_cp_local_dh(
                split_full(q),
                split_full(k),
                split_full(A),
                split_full(g),
                split_full(beta),
                split_full(do),
                scale,
            )
        )
    if has_tail:
        tail_tokens = num_tokens - full_tokens
        padded_tail_tokens = (
            (tail_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE * CHUNK_SIZE
        )
        local_dh_parts.append(
            _official_cp_local_dh(
                _pad_cp_tail(q[:, full_tokens:], padded_tail_tokens),
                _pad_cp_tail(k[:, full_tokens:], padded_tail_tokens),
                _pad_cp_tail(A[:, full_tokens:], padded_tail_tokens),
                _pad_cp_tail(
                    g[:, full_tokens:], padded_tail_tokens, repeat_last=True
                ),
                _pad_cp_tail(beta[:, full_tokens:], padded_tail_tokens),
                _pad_cp_tail(do[:, full_tokens:], padded_tail_tokens),
                scale,
            )
        )
    # A full-length AutoCP request normally has exactly one local segment
    # tensor.  Returning it directly avoids materializing an identical copy;
    # only a partial tail actually needs concatenation.
    local_dh0 = (
        local_dh_parts[0]
        if len(local_dh_parts) == 1
        else torch.cat(local_dh_parts, dim=0)
    )

    current = (
        torch.zeros_like(raw_h0[0])
        if dht is None
        else dht.float()[0]
    )
    terminal_parts = []
    for rank in range(total_segments - 1, -1, -1):
        terminal_parts.append(current)
        if rank:
            propagated = matrices[rank].transpose(-1, -2) @ current
            current = local_dh0[rank] + torch.where(
                fallback[rank, ..., None, None],
                propagated,
                torch.zeros_like(propagated),
            )
    cp_dht = torch.stack(list(reversed(terminal_parts)))

    full_result = decomposed_backward(
        split_full(q),
        split_full(k),
        split_full(v),
        split_full(g),
        split_full(beta),
        split_full(A),
        scale,
        cp_initial[:full_segments],
        split_full(do),
        cp_dht[:full_segments],
        None,
        chunk_size=CHUNK_SIZE,
    )
    result_parts = [full_result]
    if has_tail:
        result_parts.append(
            decomposed_backward(
                q[:, full_tokens:],
                k[:, full_tokens:],
                v[:, full_tokens:],
                g[:, full_tokens:],
                beta[:, full_tokens:],
                A[:, full_tokens:],
                scale,
                cp_initial[-1:],
                do[:, full_tokens:],
                cp_dht[-1:],
                None,
                chunk_size=CHUNK_SIZE,
            )
        )

    if len(result_parts) == 1:
        joined = [
            output.reshape(1, -1, *output.shape[2:])
            for output in full_result[:5]
        ]
    else:
        joined = []
        for output in range(5):
            pieces = []
            for part in result_parts:
                pieces.append(
                    part[output].reshape(1, -1, *part[output].shape[2:])
                )
            joined.append(torch.cat(pieces, dim=1))
    return (
        *joined,
        None if initial_state is None else full_result[5][:1],
    )


def official_chunk_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = False,
    auto_cp: bool = True,
    cp_cache=None,
    **_,
):
    """Official backward contract with an explicit recurrence derivative."""
    scale = float(q.shape[-1]**-0.5 if scale is None else scale)
    if (
        os.getenv("FLASHQLA_PPU_TOKEN_BACKWARD", "0") != "1"
        and q.shape[-1] == v.shape[-1] == 128
    ):
        h0 = initial_state
        terminal_grad = dht
        if state_v_first:
            h0 = None if h0 is None else h0.transpose(-1, -2)
            terminal_grad = (
                None
                if terminal_grad is None
                else terminal_grad.transpose(-1, -2)
            )
        h0 = None if h0 is None else h0.contiguous()
        terminal_grad = (
            None if terminal_grad is None else terminal_grad.contiguous()
        )
        if auto_cp and cu_seqlens is not None:
            bounds = cu_seqlens.to(
                device="cpu", dtype=torch.int64
            ).tolist()
            sequence_results = []
            dh0_parts = []
            varlen_cache_valid = (
                isinstance(cp_cache, tuple)
                and len(cp_cache) == 3
                and cp_cache[0] == "ppu_auto_cp_varlen_v1"
                and tuple(bounds) == cp_cache[1]
            )
            for sequence, (left, right) in enumerate(
                zip(bounds[:-1], bounds[1:])
            ):
                sequence_h0 = (
                    None if h0 is None else h0[sequence : sequence + 1]
                )
                sequence_dht = (
                    None
                    if terminal_grad is None
                    else terminal_grad[sequence : sequence + 1]
                )
                result = _run_auto_cp_backward_equal_length(
                    q[:, left:right].float(),
                    k[:, left:right].float(),
                    v[:, left:right].float(),
                    g[:, left:right].float(),
                    beta[:, left:right].float(),
                    A[:, left:right].float(),
                    do[:, left:right].float(),
                    None if sequence_dht is None else sequence_dht.float(),
                    scale,
                    None if sequence_h0 is None else sequence_h0.float(),
                    (
                        cp_cache[2][sequence]
                        if varlen_cache_valid
                        else None
                    ),
                )
                sequence_results.append(result)
                if result[5] is not None:
                    dh0_parts.append(result[5])
            dq, dk, dv, db, dg = (
                torch.cat(
                    [result[index] for result in sequence_results], dim=1
                )
                for index in range(5)
            )
            dh0 = (
                torch.cat(dh0_parts, dim=0)
                if h0 is not None
                else None
            )
            if state_v_first and dh0 is not None:
                dh0 = dh0.transpose(-1, -2).contiguous()
            return (
                dq.to(q.dtype),
                dk.to(k.dtype),
                dv.to(v.dtype),
                db,
                dg,
                dh0,
            )

        backward_runner = (
            _run_auto_cp_backward_equal_length
            if auto_cp
            else None
        )
        fused_cast = (
            os.getenv("FLASHQLA_PPU_FUSED_BWD_CAST", "1") == "1"
            and native.is_flash_qla_backward_cast_available()
            and all(
                tensor.dtype == torch.bfloat16 and tensor.is_contiguous()
                for tensor in (q, k, v, A, do)
            )
        )
        if fused_cast:
            q_work, k_work, v_work, A_work, do_work = (
                native.prepare_backward_inputs_bf16_128(q, k, v, A, do)
            )
        else:
            q_work, k_work, v_work, A_work, do_work = (
                q.float(), k.float(), v.float(), A.float(), do.float()
            )
        if backward_runner is None:
            dq, dk, dv, db, dg, dh0 = decomposed_backward(
                q_work,
                k_work,
                v_work,
                g.float(),
                beta.float(),
                A_work,
                scale,
                None if h0 is None else h0.float(),
                do_work,
                None if terminal_grad is None else terminal_grad.float(),
                cu_seqlens,
                chunk_size=CHUNK_SIZE,
            )
        else:
            dq, dk, dv, db, dg, dh0 = backward_runner(
                q_work,
                k_work,
                v_work,
                g.float(),
                beta.float(),
                A_work,
                do_work,
                None if terminal_grad is None else terminal_grad.float(),
                scale,
                None if h0 is None else h0.float(),
                cp_cache,
            )
        if state_v_first and dh0 is not None:
            dh0 = dh0.transpose(-1, -2).contiguous()
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            db,
            dg,
            dh0,
        )

    raw_g = _uncumsum_gate(g.float(), cu_seqlens)
    del auto_cp
    key_dim = q.shape[-1]
    value_dim = v.shape[-1]
    value_heads = v.shape[2]

    if cu_seqlens is None:
        ranges = [(batch, 0, q.shape[1]) for batch in range(q.shape[0])]
    else:
        bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
        ranges = [(0, left, right) for left, right in zip(bounds[:-1], bounds[1:])]

    results = []
    dh0_parts = []
    for sequence, (batch, left, right) in enumerate(ranges):
        length = right - left
        if initial_state is None:
            h0 = torch.zeros(
                1, value_heads, key_dim, value_dim,
                dtype=torch.float32, device=q.device,
            )
        else:
            h0 = initial_state[sequence : sequence + 1]
            if state_v_first:
                h0 = h0.transpose(-1, -2)
            h0 = h0.float().contiguous()
        if dht is None:
            terminal_grad = torch.zeros_like(h0)
        else:
            terminal_grad = dht[sequence : sequence + 1]
            if state_v_first:
                terminal_grad = terminal_grad.transpose(-1, -2)
            terminal_grad = terminal_grad.float().contiguous()
        results.append(_recurrent_backward_sequence(
            q[batch : batch + 1, left:right],
            k[batch : batch + 1, left:right],
            v[batch : batch + 1, left:right],
            raw_g[batch : batch + 1, left:right],
            beta[batch : batch + 1, left:right],
            do[batch : batch + 1, left : left + length],
            h0,
            terminal_grad,
            scale,
        ))
        dh0_part = results[-1][5]
        if state_v_first:
            dh0_part = dh0_part.transpose(-1, -2).contiguous()
        dh0_parts.append(dh0_part)

    concat_dim = 0 if cu_seqlens is None else 1
    dq, dk, dv, db, dg = (
        torch.cat([result[index] for result in results], dim=concat_dim)
        for index in range(5)
    )
    dh0 = torch.cat(dh0_parts, dim=0) if initial_state is not None else None
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db, dg, dh0


def _l2norm_forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv_norm = torch.rsqrt(
        x.float().square().sum(dim=-1, keepdim=True) + 1e-6
    )
    return (x.float() * inv_norm).to(x.dtype), inv_norm


def _l2norm_backward(
    normalized: torch.Tensor,
    inv_norm: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    normalized_float = normalized.float()
    grad_float = grad.float()
    projection = (normalized_float * grad_float).sum(
        dim=-1, keepdim=True
    )
    return (grad_float - normalized_float * projection) * inv_norm


class _OfficialChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """PPU autograd boundary matching the official FlashQLA wrapper."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
        state_v_first: bool,
        auto_cp: bool,
        enable_fwd_cp_cache: bool,
    ):
        # A custom Function executes its forward without recording operators.
        # Detach explicitly so the native PPU fast paths can be selected; the
        # official backward below owns all input gradients.
        q_work = q.detach()
        k_work = k.detach()
        v_work = v.detach()
        g_work = g.detach()
        beta_work = beta.detach()
        initial_work = (
            None if initial_state is None else initial_state.detach()
        )
        cu_work = None if cu_seqlens is None else cu_seqlens.detach()

        q_rstd = None
        k_rstd = None
        if use_qk_l2norm_in_kernel:
            q_work, q_rstd = _l2norm_forward(q_work)
            k_work, k_rstd = _l2norm_forward(k_work)

        needs_backward = any(
            ctx.needs_input_grad[index] for index in (0, 1, 2, 3, 4, 6)
        )
        g_cumsum, A, output, _, final_state, cp_cache = (
            official_chunk_forward(
                q=q_work,
                k=k_work,
                v=v_work,
                g=g_work,
                beta=beta_work,
                scale=scale,
                initial_state=initial_work,
                cu_seqlens=cu_work,
                output_final_state=output_final_state,
                output_h=False,
                state_v_first=state_v_first,
                auto_cp=auto_cp,
                enable_fwd_cp_cache=(
                    enable_fwd_cp_cache and needs_backward
                ),
            )
        )
        ctx.save_for_backward(
            q_work,
            k_work,
            v_work,
            g_cumsum,
            beta_work,
            A,
            initial_work,
            cu_work,
            q_rstd,
            k_rstd,
        )
        # PPU AutoCP cache metadata is nested for varlen input, unlike the
        # fixed four-tensor CUDA cache. Its tensor lifetime is still bounded
        # by this autograd context.
        ctx.cp_cache = cp_cache
        ctx.scale = scale
        ctx.state_v_first = state_v_first
        ctx.auto_cp = auto_cp
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.initial_state_dtype = (
            None if initial_state is None else initial_state.dtype
        )
        return output.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor | None, dht: torch.Tensor | None):
        (
            q,
            k,
            v,
            g,
            beta,
            A,
            initial_state,
            cu_seqlens,
            q_rstd,
            k_rstd,
        ) = ctx.saved_tensors
        if do is None:
            do = v.new_zeros(v.shape)
        dq, dk, dv, db, dg, dh0 = official_chunk_backward(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            state_v_first=ctx.state_v_first,
            auto_cp=ctx.auto_cp,
            cp_cache=ctx.cp_cache,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = _l2norm_backward(q, q_rstd, dq)
            dk = _l2norm_backward(k, k_rstd, dk)
        if dh0 is not None:
            dh0 = dh0.to(ctx.initial_state_dtype)
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg.to(g.dtype),
            db.to(beta.dtype),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def official_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    state_v_first: bool = False,
    auto_cp: bool = True,
    enable_fwd_cp_cache: bool = True,
):
    if head_first:
        raise ValueError("head_first=True is not supported")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q/k/v must be FP16 or BF16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q/k/v dtypes must match")
    data_tensors = (q, k, v, g, beta) + (
        () if initial_state is None else (initial_state,)
    )
    if any(tensor.device != q.device for tensor in data_tensors[1:]):
        raise ValueError("q/k/v/g/beta/initial_state must use the same device")
    if v.shape[2] % k.shape[2]:
        raise ValueError("num_v_heads must be divisible by num_qk_heads")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("varlen input requires batch size 1")
        if (
            initial_state is not None
            and initial_state.shape[0] != cu_seqlens.numel() - 1
        ):
            raise ValueError(
                "initial_state count must match the number of sequences"
            )
    scale = float(q.shape[-1] ** -0.5 if scale is None else scale)
    return _OfficialChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        state_v_first,
        auto_cp,
        enable_fwd_cp_cache,
    )
