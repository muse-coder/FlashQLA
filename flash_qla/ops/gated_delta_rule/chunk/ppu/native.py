"""ctypes bindings for optional HGGC kernels used by the PPU backend."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch


_LIBRARY_PATH = Path(__file__).with_name("libflash_qla_ppu.so")
_LIBRARY = ctypes.CDLL(str(_LIBRARY_PATH)) if _LIBRARY_PATH.exists() else None


def is_available() -> bool:
    return _LIBRARY is not None


def is_aiu_available() -> bool:
    return _LIBRARY is not None and hasattr(
        _LIBRARY, "launch_aiu_kkt_inverse_bf16_64x128"
    )


def is_flash_qla_fused_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_fused_forward_bf16_128"
    )


def is_flash_qla_affine_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_affine_state_bf16_128"
    )


def is_flash_qla_chunk_state_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_chunk_state_forward_bf16_128"
    )


def is_flash_qla_fused_prepare_h_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_fused_prepare_h_bf16_128"
    )


def is_flash_qla_backward_cast_available() -> bool:
    return _LIBRARY is not None and hasattr(
        _LIBRARY, "launch_flash_qla_prepare_backward_inputs_bf16_128_v1"
    )


def is_flash_qla_wu_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_wu_forward_bf16_128"
    )


def is_flash_qla_cp_dh_backward_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_cp_dh_backward_bf16_128"
    )


def is_flash_qla_chunk_state_backward_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_chunk_state_backward_bf16_128"
    )


def is_flash_qla_chunk_state_backward_step_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_chunk_state_backward_step_bf16_128"
    )


def is_flash_qla_dqkwg_layout_available() -> bool:
    return _LIBRARY is not None and hasattr(
        _LIBRARY, "launch_prepare_dqkwg_head_major"
    )


def is_grouped_state_dv_update_available() -> bool:
    return _LIBRARY is not None and hasattr(
        _LIBRARY, "launch_grouped_state_dv_update"
    )


def is_flash_qla_fused_state_dqkwg_backward_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_fused_state_dqkwg_backward_bf16_128_v2"
    )


def is_flash_qla_chunk_dv_backward_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_chunk_dv_backward_bf16_128"
    )


def is_flash_qla_chunk_dqkw_backward_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_chunk_dqkw_backward_bf16_128"
    )


def is_flash_qla_chunk_dqkwg_backward_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_chunk_dqkwg_backward_bf16_128"
    )


def is_flash_qla_wy_backward_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_wy_backward_preprocess_128"
    ) and hasattr(
        _LIBRARY, "launch_flash_qla_wy_backward_postprocess_128"
    )


def is_flash_qla_fused_wy_backward_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_flash_qla_fused_wy_backward_128_v1"
    )


def is_gated_strided_kkt_available() -> bool:
    return is_aiu_available() and hasattr(
        _LIBRARY, "launch_aiu_kkt_solve_bf16_128_strided_gated"
    )


def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def prepare_gate_beta(
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare chunk-major gate prefix sums and beta in one PPU launch."""
    if _LIBRARY is None or not hasattr(_LIBRARY, "launch_prepare_gate_beta"):
        raise RuntimeError("PPU gate/beta preparation kernel is not built")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("g and beta must be FP32")
    if not (g.is_contiguous() and beta.is_contiguous()):
        raise ValueError("g and beta must be contiguous")
    if g.ndim != 3 or beta.shape != g.shape or g.shape[1] % 64:
        raise ValueError("g and beta must have shape [B, 64-aligned T, H]")

    batch_size, tokens, heads = g.shape
    chunks = tokens // 64
    output_shape = (batch_size, heads, chunks, 64)
    g_cumsum = torch.empty(output_shape, dtype=torch.float32, device=g.device)
    beta_chunks = torch.empty_like(g_cumsum)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(g.device).cuda_stream)
    _LIBRARY.launch_prepare_gate_beta(
        _pointer(g),
        _pointer(beta),
        _pointer(g_cumsum),
        _pointer(beta_chunks),
        ctypes.c_int(batch_size),
        ctypes.c_int(tokens),
        ctypes.c_int(heads),
        stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU gate/beta preparation launch failed: {result.value}")
    return g_cumsum, beta_chunks


def prepare_dqkwg_head_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Fuse the token-major to chunk/head-major dQKWg layout copies."""
    tensors = (q, k, v, do, dv, g)
    if not is_flash_qla_dqkwg_layout_available():
        raise RuntimeError("PPU dQKWg layout kernel is not built")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("dQKWg layout inputs must be contiguous FP32")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Q/K must match [B,T,Hq,128]")
    batch, tokens, q_heads, _ = q.shape
    if tokens % 64 or not (v.shape == do.shape == dv.shape):
        raise ValueError("V/dO/dV must match with a 64-aligned token axis")
    value_heads = v.shape[2]
    if v.shape != (batch, tokens, value_heads, 128):
        raise ValueError("V/dO/dV must use [B,T,Hv,128]")
    if value_heads % q_heads or g.shape != (batch, tokens, value_heads):
        raise ValueError("incompatible GVA head or gate shape")
    chunks = tokens // 64
    vector_shape = (batch, chunks, value_heads, 64, 128)
    q_head = torch.empty(vector_shape, dtype=torch.float32, device=q.device)
    k_head = torch.empty_like(q_head)
    v_head = torch.empty_like(q_head)
    do_head = torch.empty_like(q_head)
    dv_head = torch.empty_like(q_head)
    g_head = torch.empty(
        batch, chunks, value_heads, 64,
        dtype=torch.float32, device=q.device,
    )
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    _LIBRARY.launch_prepare_dqkwg_head_major(
        _pointer(q), _pointer(k), _pointer(v), _pointer(do), _pointer(dv),
        _pointer(g), _pointer(q_head), _pointer(k_head), _pointer(v_head),
        _pointer(do_head), _pointer(dv_head), _pointer(g_head),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU dQKWg layout launch failed: {result.value}")
    return q_head, k_head, v_head, do_head, dv_head, g_head


def grouped_state_dv_update(
    dv: torch.Tensor,
    update: torch.Tensor,
    decay: torch.Tensor,
    chunk: int,
) -> None:
    """Fuse the official grouped dV decay and accumulation for one chunk."""
    tensors = (dv, update, decay)
    if not is_grouped_state_dv_update_available():
        raise RuntimeError("PPU grouped state-dV update kernel is not built")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("grouped state-dV inputs must be contiguous FP32")
    if dv.ndim != 5 or dv.shape[2] != 64 or dv.shape[-1] != 128:
        raise ValueError("dV must use [B,N,64,Hv,128]")
    batch, chunks, _, value_heads, _ = dv.shape
    if update.ndim != 5 or update.shape[0] != batch:
        raise ValueError("update must use [B,Hq,R,64,128]")
    _, q_heads, head_repeat, rows, dim = update.shape
    if (rows, dim) != (64, 128) or q_heads * head_repeat != value_heads:
        raise ValueError("incompatible grouped dV update shape")
    if decay.shape != (batch, chunks, 64, value_heads, 1):
        raise ValueError("decay must use [B,N,64,Hv,1]")
    if chunk < 0 or chunk >= chunks:
        raise ValueError("chunk index is out of range")
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dv.device).cuda_stream)
    _LIBRARY.launch_grouped_state_dv_update(
        _pointer(dv), _pointer(update), _pointer(decay),
        ctypes.c_int(batch), ctypes.c_int(chunks), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), ctypes.c_int(chunk), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU grouped state-dV launch failed: {result.value}")


def reverse_chunk_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Run the official inclusive reverse scan within each 64-token chunk."""
    if _LIBRARY is None or not hasattr(
        _LIBRARY, "launch_reverse_chunk_cumsum"
    ):
        raise RuntimeError("PPU reverse chunk-cumsum kernel is not built")
    if x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError("reverse chunk-cumsum input must be contiguous FP32")
    if x.ndim != 3 or x.shape[1] % 64:
        raise ValueError("reverse chunk-cumsum input must use [B,64-aligned T,H]")
    batch, tokens, heads = x.shape
    output = torch.empty_like(x)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream)
    _LIBRARY.launch_reverse_chunk_cumsum(
        _pointer(x), _pointer(output), ctypes.c_int(batch),
        ctypes.c_int(tokens), ctypes.c_int(heads), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU reverse chunk-cumsum launch failed: {result.value}")
    return output


def warmup_counts(
    g: torch.Tensor,
    chunk_size: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute official CP warmup counts and fallback mask in one launch."""
    if _LIBRARY is None:
        raise RuntimeError("PPU native library is not built")
    if g.dtype != torch.float32 or not g.is_contiguous():
        raise ValueError("g must be contiguous FP32")
    if g.ndim != 3 or g.shape[1] % chunk_size:
        raise ValueError("g must have shape [segments, chunk-aligned tokens, heads]")

    segments, tokens, heads = g.shape
    counts = torch.empty(
        segments, heads, dtype=torch.int32, device=g.device
    )
    fallback = torch.empty(
        segments, heads, dtype=torch.bool, device=g.device
    )
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(g.device).cuda_stream)
    _LIBRARY.launch_warmup_counts(
        _pointer(g),
        _pointer(counts),
        _pointer(fallback),
        ctypes.c_int(segments),
        ctypes.c_int(tokens),
        ctypes.c_int(heads),
        ctypes.c_int(chunk_size),
        ctypes.c_float(threshold),
        stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU warmup-count launch failed: {result.value}")
    return counts, fallback


def chunk_coefficients(
    g_cumsum: torch.Tensor,
    kkt: torch.Tensor,
    qk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse causal decay, KKT/QK weighting, gamma and reverse decay."""
    if _LIBRARY is None:
        raise RuntimeError("PPU native library is not built")
    if not (g_cumsum.dtype == kkt.dtype == qk.dtype == torch.float32):
        raise ValueError("coefficient inputs must be FP32")
    if not (g_cumsum.is_contiguous() and kkt.is_contiguous() and qk.is_contiguous()):
        raise ValueError("coefficient inputs must be contiguous")
    chunk_size = g_cumsum.shape[-1]
    if kkt.shape[-2:] != (chunk_size, chunk_size) or qk.shape != kkt.shape:
        raise ValueError("KKT/QK shapes must end in [chunk_size, chunk_size]")

    weighted_kkt = torch.empty_like(kkt)
    attention = torch.empty_like(qk)
    gamma = torch.empty_like(g_cumsum)
    reverse_decay = torch.empty_like(g_cumsum)
    groups = g_cumsum.numel() // chunk_size
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(
        torch.cuda.current_stream(g_cumsum.device).cuda_stream
    )
    _LIBRARY.launch_chunk_coefficients(
        _pointer(g_cumsum),
        _pointer(kkt),
        _pointer(qk),
        _pointer(weighted_kkt),
        _pointer(attention),
        _pointer(gamma),
        _pointer(reverse_decay),
        ctypes.c_int(groups),
        ctypes.c_int(chunk_size),
        stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU coefficient launch failed: {result.value}")
    return weighted_kkt, attention, gamma, reverse_decay


def _chunk_shape(chunks: torch.Tensor) -> tuple[int, int, int, int, int]:
    if chunks.ndim != 5 or not chunks.is_contiguous():
        raise ValueError("chunk tensor must be contiguous [B,H,C,L,D]")
    return chunks.shape


def prepare_w(
    u: torch.Tensor,
    v_chunks: torch.Tensor,
    beta_chunks: torch.Tensor,
    gamma_chunks: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    batch, heads, chunks, chunk_size, value_dim = _chunk_shape(v_chunks)
    output = torch.empty_like(u)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(u.device).cuda_stream)
    _LIBRARY.launch_prepare_w(
        _pointer(u), _pointer(v_chunks), _pointer(beta_chunks),
        _pointer(gamma_chunks), _pointer(output), ctypes.c_int(batch),
        ctypes.c_int(heads), ctypes.c_int(chunks), ctypes.c_int(chunk_size),
        ctypes.c_int(value_dim), ctypes.c_int(chunk), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU prepare-W launch failed: {result.value}")
    return output


def combine_output(
    q_state: torch.Tensor,
    attended: torch.Tensor,
    gamma_chunks: torch.Tensor,
    chunks: int,
    chunk: int,
    scale: float,
) -> torch.Tensor:
    batch, heads, chunk_size, value_dim = q_state.shape
    output = torch.empty_like(q_state)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q_state.device).cuda_stream)
    _LIBRARY.launch_combine_output(
        _pointer(q_state), _pointer(attended), _pointer(gamma_chunks),
        _pointer(output), ctypes.c_int(batch), ctypes.c_int(heads),
        ctypes.c_int(chunks), ctypes.c_int(chunk_size), ctypes.c_int(value_dim),
        ctypes.c_int(chunk), ctypes.c_float(scale), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU output-combine launch failed: {result.value}")
    return output


def decay_values(
    values: torch.Tensor,
    reverse_decay_chunks: torch.Tensor,
    chunks: int,
    chunk: int,
) -> torch.Tensor:
    batch, heads, chunk_size, value_dim = values.shape
    output = torch.empty_like(values)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(values.device).cuda_stream)
    _LIBRARY.launch_decay_values(
        _pointer(values), _pointer(reverse_decay_chunks), _pointer(output),
        ctypes.c_int(batch), ctypes.c_int(heads), ctypes.c_int(chunks),
        ctypes.c_int(chunk_size), ctypes.c_int(value_dim), ctypes.c_int(chunk),
        stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU value-decay launch failed: {result.value}")
    return output


def update_state(
    state: torch.Tensor,
    delta: torch.Tensor,
    gamma_last_chunks: torch.Tensor,
    chunks: int,
    chunk: int,
    scale_delta: bool = False,
) -> torch.Tensor:
    batch, heads, key_dim, value_dim = state.shape
    output = torch.empty_like(state)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(state.device).cuda_stream)
    _LIBRARY.launch_update_state(
        _pointer(state), _pointer(delta), _pointer(gamma_last_chunks),
        _pointer(output), ctypes.c_int(batch), ctypes.c_int(heads),
        ctypes.c_int(chunks), ctypes.c_int(key_dim), ctypes.c_int(value_dim),
        ctypes.c_int(chunk), ctypes.c_int(scale_delta), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU state-update launch failed: {result.value}")
    return output


def reverse_state_update(
    state: torch.Tensor,
    inter: torch.Tensor,
    product: torch.Tensor,
    decay: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Fuse the official reverse recurrence epilogue for one chunk."""
    if _LIBRARY is None or not hasattr(_LIBRARY, "launch_reverse_state_update"):
        raise RuntimeError("PPU reverse state-update kernel is not built")
    if state.dtype != torch.float32 or product.dtype != torch.float32:
        raise ValueError("reverse state-update inputs must be FP32")
    if state.shape != product.shape or state.ndim != 4:
        raise ValueError("state/product must match [B,H,K,V]")
    batch, heads, key_dim, value_dim = state.shape
    if inter.ndim != 5 or inter.shape[0] != batch or inter.shape[2:] != state.shape[1:]:
        raise ValueError("inter must use [B,C,H,K,V]")
    chunks = inter.shape[1]
    if decay.shape != (batch, chunks, heads):
        raise ValueError("decay must use [B,C,H]")
    if not all(t.is_contiguous() for t in (state, inter, product, decay)):
        raise ValueError("reverse state-update inputs must be contiguous")
    output = torch.empty_like(state)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(state.device).cuda_stream)
    _LIBRARY.launch_reverse_state_update(
        _pointer(state), _pointer(inter), _pointer(product), _pointer(decay),
        _pointer(output), ctypes.c_int(batch), ctypes.c_int(heads),
        ctypes.c_int(chunks), ctypes.c_int(key_dim), ctypes.c_int(value_dim),
        ctypes.c_int(chunk), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU reverse state-update launch failed: {result.value}")
    return output


def reverse_state_update_store_inplace(
    state: torch.Tensor,
    inter: torch.Tensor,
    product: torch.Tensor,
    decay: torch.Tensor,
    history: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Publish current dH and advance the official reverse state in place."""
    symbol = "launch_reverse_state_update_store_inplace"
    if _LIBRARY is None or not hasattr(_LIBRARY, symbol):
        raise RuntimeError("PPU reverse state/history kernel is not built")
    if state.dtype != torch.float32 or product.dtype != torch.float32:
        raise ValueError("reverse state/history inputs must be FP32")
    if state.shape != product.shape or state.ndim != 4:
        raise ValueError("state/product must match [B,H,K,V]")
    batch, heads, key_dim, value_dim = state.shape
    if inter.ndim != 5 or inter.shape[0] != batch or inter.shape[2:] != state.shape[1:]:
        raise ValueError("inter must use [B,C,H,K,V]")
    chunks = inter.shape[1]
    if history.shape != inter.shape or history.dtype != torch.float32:
        raise ValueError("history must match contiguous FP32 inter storage")
    if decay.shape != (batch, chunks, heads):
        raise ValueError("decay must use [B,C,H]")
    if not 0 <= chunk < chunks:
        raise ValueError("chunk index is out of range")
    if not all(
        tensor.is_contiguous()
        for tensor in (state, inter, product, decay, history)
    ):
        raise ValueError("reverse state/history inputs must be contiguous")
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(state.device).cuda_stream)
    _LIBRARY.launch_reverse_state_update_store_inplace(
        _pointer(state), _pointer(inter), _pointer(product), _pointer(decay),
        _pointer(history), ctypes.c_int(batch), ctypes.c_int(heads),
        ctypes.c_int(chunks), ctypes.c_int(key_dim), ctypes.c_int(value_dim),
        ctypes.c_int(chunk), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(
            f"PPU reverse state/history launch failed: {result.value}"
        )
    return state


def wy_apply_decay_64(
    dA: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Apply the official strict-lower WY decay to dA in place."""
    if _LIBRARY is None or not hasattr(_LIBRARY, "launch_wy_decay_da"):
        raise RuntimeError("PPU WY dA-decay kernel is not built")
    if dA.dtype != torch.float32 or g.dtype != torch.float32:
        raise ValueError("WY dA/g inputs must be FP32")
    if dA.ndim != 5 or dA.shape[-2:] != (64, 64):
        raise ValueError("WY dA must use [B,N,H,64,64]")
    if g.shape != dA.shape[:-1]:
        raise ValueError("WY gate must use [B,N,H,64]")
    if not (dA.is_contiguous() and g.is_contiguous()):
        raise ValueError("WY dA/g inputs must be contiguous")
    groups = dA.numel() // (64 * 64)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dA.device).cuda_stream)
    _LIBRARY.launch_wy_decay_da(
        _pointer(dA), _pointer(g), ctypes.c_int(groups), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU WY dA-decay launch failed: {result.value}")
    return dA


def dqkwg_apply_decay_64(
    dp: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Apply the official causal dP gate and scale in place."""
    if _LIBRARY is None or not hasattr(_LIBRARY, "launch_dqkwg_decay_dp"):
        raise RuntimeError("PPU dQKWg dP-decay kernel is not built")
    if dp.dtype != torch.float32 or g.dtype != torch.float32:
        raise ValueError("dQKWg dP/g inputs must be FP32")
    if dp.ndim != 3 or dp.shape[-2:] != (64, 64):
        raise ValueError("dQKWg dP must use [groups,64,64]")
    if g.numel() != dp.shape[0] * 64:
        raise ValueError("dQKWg gate must contain [groups,64]")
    if not (dp.is_contiguous() and g.is_contiguous()):
        raise ValueError("dQKWg dP/g inputs must be contiguous")
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dp.device).cuda_stream)
    _LIBRARY.launch_dqkwg_decay_dp(
        _pointer(dp), _pointer(g), ctypes.c_int(dp.shape[0]),
        ctypes.c_float(scale), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU dQKWg dP-decay launch failed: {result.value}")
    return dp


def dqkwg_accumulate_gate_product_64(
    dp: torch.Tensor,
    p: torch.Tensor,
    dg: torch.Tensor,
) -> torch.Tensor:
    """Accumulate official row/column sums of dP*P into dg in place."""
    if _LIBRARY is None or not hasattr(
        _LIBRARY, "launch_dqkwg_gate_product"
    ):
        raise RuntimeError("PPU dQKWg gate-product kernel is not built")
    if not (dp.shape == p.shape and dp.ndim == 3):
        raise ValueError("dQKWg dP/P tensors must match [groups,64,64]")
    if dp.shape[-2:] != (64, 64) or dg.shape != (dp.shape[0], 64):
        raise ValueError("dQKWg gate-product shapes are incompatible")
    if not all(
        tensor.dtype == torch.float32 and tensor.is_contiguous()
        for tensor in (dp, p, dg)
    ):
        raise ValueError("dQKWg gate-product inputs must be contiguous FP32")
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dp.device).cuda_stream)
    _LIBRARY.launch_dqkwg_gate_product(
        _pointer(dp), _pointer(p), _pointer(dg),
        ctypes.c_int(dp.shape[0]), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(
            f"PPU dQKWg gate-product launch failed: {result.value}"
        )
    return dg


def kkt_system(
    gram: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build unit-lower KKT systems and identity RHS in one launch."""
    if gram.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("KKT inputs must be FP32")
    if not (gram.is_contiguous() and beta.is_contiguous()):
        raise ValueError("KKT inputs must be contiguous")
    chunk_size = gram.shape[-1]
    if gram.shape[-2] != chunk_size or beta.shape != gram.shape[:-1]:
        raise ValueError("incompatible gram/beta shapes")
    lower = torch.empty_like(gram)
    identity = torch.empty_like(gram)
    groups = gram.numel() // (chunk_size * chunk_size)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(gram.device).cuda_stream)
    _LIBRARY.launch_kkt_system(
        _pointer(gram), _pointer(beta), _pointer(lower), _pointer(identity),
        ctypes.c_int(groups), ctypes.c_int(chunk_size), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU KKT-system launch failed: {result.value}")
    return lower, identity


def kkt_inverse(
    gram: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Solve the official chunk-64 unit-lower KKT system in one PPU launch."""
    if gram.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("KKT inputs must be FP32")
    if not (gram.is_contiguous() and beta.is_contiguous()):
        raise ValueError("KKT inputs must be contiguous")
    if gram.shape[-2:] != (64, 64) or beta.shape != gram.shape[:-1]:
        raise ValueError("native KKT inverse requires chunk size 64")
    output = torch.empty_like(gram)
    groups = gram.numel() // (64 * 64)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(gram.device).cuda_stream)
    _LIBRARY.launch_kkt_inverse(
        _pointer(gram), _pointer(beta), _pointer(output), ctypes.c_int(groups),
        stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU KKT-inverse launch failed: {result.value}")
    return output


def aiu_kkt_inverse_bf16_64x128(
    k: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Compute the official chunk-64 KKT inverse from BF16 K in one CTA."""
    if not is_aiu_available() or not hasattr(
        _LIBRARY, "launch_aiu_kkt_inverse_bf16_64x128"
    ):
        raise RuntimeError("PPU AIU KKT kernel is not built")
    if k.dtype != torch.bfloat16 or beta.dtype != torch.float32:
        raise ValueError("KKT inputs must be BF16 K and FP32 beta")
    if not (k.is_contiguous() and beta.is_contiguous()):
        raise ValueError("KKT inputs must be contiguous")
    if k.shape[-2:] != (64, 128) or beta.shape != k.shape[:-1]:
        raise ValueError("expected K [...,64,128] and beta [...,64]")
    output = torch.empty(
        (*k.shape[:-1], 64), dtype=torch.float32, device=k.device
    )
    groups = k.numel() // (64 * 128)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_aiu_kkt_inverse_bf16_64x128(
        _pointer(k), _pointer(beta), _pointer(output), ctypes.c_int(groups),
        stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU AIU KKT launch failed: {result.value}")
    return output


def aiu_kkt_solve_bf16_128_strided(
    k: torch.Tensor,
    beta_chunks: torch.Tensor,
) -> torch.Tensor:
    """Solve KKT directly from [B,T,Hq,128], returning official BF16 A."""
    if not is_aiu_available() or not hasattr(
        _LIBRARY, "launch_aiu_kkt_solve_bf16_128_strided"
    ):
        raise RuntimeError("PPU strided AIU KKT kernel is not built")
    if k.dtype != torch.bfloat16 or beta_chunks.dtype != torch.float32:
        raise ValueError("strided KKT requires BF16 K and FP32 beta")
    if k.ndim != 4 or k.shape[-1] != 128 or k.shape[1] % 64:
        raise ValueError("K must use chunk-aligned [B,T,Hq,128] layout")
    batch, tokens, q_heads, _ = k.shape
    if beta_chunks.ndim != 4 or beta_chunks.shape[0] != batch:
        raise ValueError("beta must use [B,Hv,C,64] layout")
    value_heads, chunks, chunk_size = beta_chunks.shape[1:]
    if chunks != tokens // 64 or chunk_size != 64 or value_heads % q_heads:
        raise ValueError("incompatible K/beta head or chunk dimensions")
    if not (k.is_contiguous() and beta_chunks.is_contiguous()):
        raise ValueError("strided KKT inputs must be contiguous")
    output = torch.empty(
        batch, value_heads, chunks, 64, 64,
        dtype=torch.bfloat16, device=k.device,
    )
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_aiu_kkt_solve_bf16_128_strided(
        _pointer(k), _pointer(beta_chunks), _pointer(output),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU strided AIU KKT launch failed: {result.value}")
    return output


def aiu_kkt_solve_bf16_128_strided_gated(
    k: torch.Tensor,
    beta_chunks: torch.Tensor,
    g_cumsum_chunks: torch.Tensor,
) -> torch.Tensor:
    """Solve KKT and apply the official ``D A0 D^-1`` epilogue."""
    if not is_gated_strided_kkt_available():
        raise RuntimeError("PPU gated strided AIU KKT kernel is not built")
    if k.dtype != torch.bfloat16:
        raise ValueError("gated strided KKT requires BF16 K")
    if beta_chunks.dtype != torch.float32 or g_cumsum_chunks.dtype != torch.float32:
        raise ValueError("gated strided KKT requires FP32 beta/g")
    if beta_chunks.shape != g_cumsum_chunks.shape:
        raise ValueError("beta and cumulative gate chunk layouts must match")
    if k.ndim != 4 or k.shape[-1] != 128 or k.shape[1] % 64:
        raise ValueError("K must use chunk-aligned [B,T,Hq,128] layout")
    batch, tokens, q_heads, _ = k.shape
    if beta_chunks.ndim != 4 or beta_chunks.shape[0] != batch:
        raise ValueError("beta/g must use [B,Hv,C,64] layout")
    value_heads, chunks, chunk_size = beta_chunks.shape[1:]
    if chunks != tokens // 64 or chunk_size != 64 or value_heads % q_heads:
        raise ValueError("incompatible K/beta head or chunk dimensions")
    if not (
        k.is_contiguous()
        and beta_chunks.is_contiguous()
        and g_cumsum_chunks.is_contiguous()
    ):
        raise ValueError("gated strided KKT inputs must be contiguous")
    output = torch.empty(
        batch, value_heads, chunks, 64, 64,
        dtype=torch.bfloat16, device=k.device,
    )
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_aiu_kkt_solve_bf16_128_strided_gated(
        _pointer(k), _pointer(beta_chunks), _pointer(g_cumsum_chunks),
        _pointer(output), ctypes.c_int(batch), ctypes.c_int(tokens),
        ctypes.c_int(q_heads), ctypes.c_int(value_heads), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU gated strided AIU KKT launch failed: {result.value}")
    return output


def scale_rows_negative(
    values: torch.Tensor,
    scale_rows: torch.Tensor,
) -> torch.Tensor:
    """Compute ``-diag(scale_rows) @ values`` for packed row matrices."""
    if not (values.is_contiguous() and scale_rows.is_contiguous()):
        raise ValueError("row-scale inputs must be contiguous")
    output = torch.empty_like(values)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(values.device).cuda_stream)
    _LIBRARY.launch_scale_rows_negative(
        _pointer(values), _pointer(scale_rows), _pointer(output),
        ctypes.c_int(values.numel()), ctypes.c_int(values.shape[-1]),
        stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU row-scale launch failed: {result.value}")
    return output


def affine_y(
    u: torch.Tensor,
    v_chunks: torch.Tensor,
    gamma_last_chunks: torch.Tensor,
    reverse_decay_chunks: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Fuse ``gamma_last * (K @ S) - reverse_decay * V``."""
    batch, heads, chunks, chunk_size, value_dim = _chunk_shape(v_chunks)
    output = torch.empty_like(u)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(u.device).cuda_stream)
    _LIBRARY.launch_affine_y(
        _pointer(u), _pointer(v_chunks), _pointer(gamma_last_chunks),
        _pointer(reverse_decay_chunks), _pointer(output), ctypes.c_int(batch),
        ctypes.c_int(heads), ctypes.c_int(chunks), ctypes.c_int(chunk_size),
        ctypes.c_int(value_dim), ctypes.c_int(chunk), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU affine-Y launch failed: {result.value}")
    return output


def fused_chunks_fp32(
    q_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    v_chunks: torch.Tensor,
    beta_chunks: torch.Tensor,
    weighted_kkt: torch.Tensor,
    attention: torch.Tensor,
    gamma: torch.Tensor,
    reverse_decay: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete sequential chunk recurrence in one PPU launch."""
    batch, heads, chunks, chunk_size, key_dim = _chunk_shape(q_chunks)
    vb, vh, vc, vl, value_dim = _chunk_shape(v_chunks)
    if (vb, vh, vc, vl) != (batch, heads, chunks, chunk_size):
        raise ValueError("q/v chunk shapes are incompatible")
    tensors = (
        q_chunks, k_chunks, v_chunks, beta_chunks, weighted_kkt,
        attention, gamma, reverse_decay, initial_state,
    )
    if not all(tensor.dtype == torch.float32 for tensor in tensors):
        raise ValueError("fused FP32 chunk inputs must all be FP32")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused FP32 chunk inputs must be contiguous")

    state = initial_state.clone()
    output = torch.empty(
        batch, heads, chunks, chunk_size, value_dim,
        dtype=torch.float32, device=q_chunks.device,
    )
    scratch_w = torch.empty(
        batch, heads, chunk_size, value_dim,
        dtype=torch.float32, device=q_chunks.device,
    )
    scratch_vp = torch.empty_like(scratch_w)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(
        torch.cuda.current_stream(q_chunks.device).cuda_stream
    )
    _LIBRARY.launch_fused_chunks_fp32(
        _pointer(q_chunks), _pointer(k_chunks), _pointer(v_chunks),
        _pointer(beta_chunks), _pointer(weighted_kkt), _pointer(attention),
        _pointer(gamma), _pointer(reverse_decay), _pointer(state),
        _pointer(output), _pointer(scratch_w), _pointer(scratch_vp),
        ctypes.c_int(batch), ctypes.c_int(heads), ctypes.c_int(chunks),
        ctypes.c_int(chunk_size), ctypes.c_int(key_dim),
        ctypes.c_int(value_dim), ctypes.c_float(scale), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU fused-chunk launch failed: {result.value}")
    return output, state


def flash_qla_fused_forward_bf16_128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a_chunks: torch.Tensor,
    g_chunks: torch.Tensor,
    beta_chunks: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official chunk-64 forward recurrence in one PPU AIU launch."""
    if not is_flash_qla_fused_available():
        raise RuntimeError("fused PPU FlashQLA kernel is not built")
    if not (q.dtype == k.dtype == v.dtype == a_chunks.dtype == torch.bfloat16):
        raise ValueError("fused PPU FlashQLA inputs Q/K/V/A must be BF16")
    if g_chunks.dtype != torch.float32 or beta_chunks.dtype != torch.float32:
        raise ValueError("fused PPU FlashQLA gate and beta must be FP32")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Q/K/V must use [B,T,H,D] layout")
    batch, tokens, q_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if key_dim != 128 or value_dim != 128 or tokens % 64:
        raise ValueError("fused PPU FlashQLA requires D=128 and chunk-aligned T")
    if value_heads % q_heads:
        raise ValueError("value heads must be divisible by Q/K heads")
    chunks = tokens // 64
    if a_chunks.shape != (batch, value_heads, chunks, 64, 64):
        raise ValueError("A must use [B,H,C,64,64] layout")
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("g must use [B,H,C,64] layout")
    if beta_chunks.shape != g_chunks.shape:
        raise ValueError("beta shape must match g")
    tensors = [q, k, v, a_chunks, g_chunks, beta_chunks]
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused PPU FlashQLA inputs must be contiguous")

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            batch, value_heads, 128, 128,
            dtype=torch.float32, device=q.device,
        )
    elif (
        initial_state.dtype != torch.float32
        or initial_state.shape != (batch, value_heads, 128, 128)
        or not initial_state.is_contiguous()
    ):
        raise ValueError("initial state must be contiguous FP32 [B,H,128,128]")
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    _LIBRARY.launch_flash_qla_fused_forward_bf16_128(
        _pointer(q), _pointer(k), _pointer(v), _pointer(a_chunks),
        _pointer(g_chunks), _pointer(beta_chunks), _pointer(initial_state),
        _pointer(output), _pointer(final_state), ctypes.c_int(batch),
        ctypes.c_int(tokens), ctypes.c_int(q_heads), ctypes.c_int(value_heads),
        ctypes.c_float(scale), ctypes.c_bool(use_initial_state), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"fused PPU FlashQLA launch failed: {result.value}")
    return output, final_state


def flash_qla_affine_state_bf16_128(
    k_chunks: torch.Tensor,
    v_chunks: torch.Tensor,
    x_chunks: torch.Tensor,
    gamma_last: torch.Tensor,
    reverse_decay: torch.Tensor,
    warmup_chunks: torch.Tensor,
    fallback: torch.Tensor,
    *,
    matrix_mode: bool,
) -> torch.Tensor:
    """Run the official AutoCP S*/M affine recurrence in one PPU launch."""
    if not is_flash_qla_affine_available():
        raise RuntimeError("PPU AutoCP affine kernel is not built")
    if not (
        k_chunks.dtype == v_chunks.dtype == x_chunks.dtype == torch.bfloat16
    ):
        raise ValueError("AutoCP K/V/X chunks must be BF16")
    if k_chunks.ndim != 5 or k_chunks.shape[-2:] != (64, 128):
        raise ValueError("K chunks must use [S,H,C,64,128] layout")
    if v_chunks.shape != k_chunks.shape or x_chunks.shape != k_chunks.shape:
        raise ValueError("AutoCP V/X shapes must match K")
    segments, heads, chunks = k_chunks.shape[:3]
    if gamma_last.shape != (segments, heads, chunks):
        raise ValueError("gamma_last must use [S,H,C] layout")
    if reverse_decay.shape != (segments, heads, chunks, 64):
        raise ValueError("reverse_decay must use [S,H,C,64] layout")
    if warmup_chunks.shape != (segments, heads) or warmup_chunks.dtype != torch.int32:
        raise ValueError("warmup chunks must be int32 [S,H]")
    if fallback.shape != (segments, heads) or fallback.dtype != torch.bool:
        raise ValueError("fallback must be bool [S,H]")
    tensors = (
        k_chunks, v_chunks, x_chunks, gamma_last, reverse_decay,
        warmup_chunks, fallback,
    )
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("AutoCP affine inputs must be contiguous")

    output = torch.empty(
        segments, heads, 128, 128,
        dtype=torch.float32, device=k_chunks.device,
    )
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(
        torch.cuda.current_stream(k_chunks.device).cuda_stream
    )
    _LIBRARY.launch_flash_qla_affine_state_bf16_128(
        _pointer(k_chunks), _pointer(v_chunks), _pointer(x_chunks),
        _pointer(gamma_last), _pointer(reverse_decay),
        _pointer(warmup_chunks), _pointer(fallback), _pointer(output),
        ctypes.c_int(segments * heads), ctypes.c_int(chunks),
        ctypes.c_bool(matrix_mode), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU AutoCP affine launch failed: {result.value}")
    return output


def flash_qla_chunk_state_forward_bf16_128(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_chunks: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare official chunk histories and corrected values in one launch."""
    if not is_flash_qla_chunk_state_available():
        raise RuntimeError("PPU chunk-state kernel is not built")
    if not (k.dtype == w.dtype == u.dtype == torch.float32):
        raise ValueError("PPU chunk-state K/W/U inputs must be FP32")
    if k.ndim != 4 or k.shape[-1] != 128 or not k.is_contiguous():
        raise ValueError("K must be contiguous [B,T,Hq,128]")
    batch, tokens, q_heads, _ = k.shape
    if tokens % 64 or w.shape[:2] != (batch, tokens) or w.shape[-1] != 128:
        raise ValueError("chunk-state inputs require a 64-aligned token axis")
    if u.shape != w.shape or not (w.is_contiguous() and u.is_contiguous()):
        raise ValueError("W/U must be matching contiguous [B,T,Hv,128] tensors")
    value_heads = w.shape[2]
    chunks = tokens // 64
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use contiguous [B,Hv,C,64] layout")
    if g_chunks.dtype != torch.float32 or not g_chunks.is_contiguous():
        raise ValueError("gate must be contiguous FP32")
    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            batch, value_heads, 128, 128,
            dtype=torch.float32, device=k.device,
        )
    elif (
        initial_state.dtype != torch.float32
        or initial_state.shape != (batch, value_heads, 128, 128)
        or not initial_state.is_contiguous()
    ):
        raise ValueError("initial state must be contiguous FP32 [B,Hv,128,128]")
    history = torch.empty(
        batch, chunks, value_heads, 128, 128,
        dtype=torch.float32, device=k.device,
    )
    vn = torch.empty_like(u)
    final_state = torch.empty_like(initial_state)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_flash_qla_chunk_state_forward_bf16_128(
        _pointer(k), _pointer(w), _pointer(u), _pointer(g_chunks),
        _pointer(initial_state), _pointer(history), _pointer(vn),
        _pointer(final_state), ctypes.c_int(batch), ctypes.c_int(tokens),
        ctypes.c_int(q_heads), ctypes.c_int(value_heads),
        ctypes.c_bool(use_initial_state), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU chunk-state launch failed: {result.value}")
    return history, vn, final_state


def prepare_backward_inputs_bf16_128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast the five retained BF16 backward inputs to FP32 in one launch."""
    if not is_flash_qla_backward_cast_available():
        raise RuntimeError("PPU fused backward-input cast kernel is not built")
    tensors = (q, k, v, A, do)
    if not all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in tensors):
        raise ValueError("fused backward-input cast requires contiguous BF16 inputs")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Q/K must match [B,T,Hq,128]")
    batch, tokens, q_heads, _ = q.shape
    if v.shape != do.shape or v.ndim != 4:
        raise ValueError("V/dO must have matching shapes")
    value_heads = v.shape[2]
    if v.shape != (batch, tokens, value_heads, 128):
        raise ValueError("V/dO must use [B,T,Hv,128]")
    if value_heads % q_heads:
        raise ValueError("value heads must be divisible by Q/K heads")
    if A.shape != (batch, tokens, value_heads, 64):
        raise ValueError("A must use [B,T,Hv,64]")
    outputs = tuple(
        torch.empty_like(tensor, dtype=torch.float32) for tensor in tensors
    )
    q_float, k_float, v_float, A_float, do_float = outputs
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    _LIBRARY.launch_flash_qla_prepare_backward_inputs_bf16_128_v1(
        _pointer(q), _pointer(k), _pointer(v), _pointer(A), _pointer(do),
        _pointer(q_float), _pointer(k_float), _pointer(v_float),
        _pointer(A_float), _pointer(do_float), ctypes.c_int(batch),
        ctypes.c_int(tokens), ctypes.c_int(q_heads), ctypes.c_int(value_heads),
        stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU fused backward-input cast failed: {result.value}")
    return q_float, k_float, v_float, A_float, do_float


def flash_qla_fused_prepare_h_bf16_128(
    k: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    g_chunks: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse official W/U construction and prepare-H in one persistent CTA."""
    if not is_flash_qla_fused_prepare_h_available():
        raise RuntimeError("PPU fused prepare-H kernel is not built")
    tensors = (k, v, A, g_chunks, beta)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("PPU fused prepare-H inputs must be contiguous FP32")
    if k.ndim != 4 or k.shape[-1] != 128:
        raise ValueError("K must use [B,T,Hq,128]")
    batch, tokens, q_heads, _ = k.shape
    if tokens % 64 or v.shape[:2] != (batch, tokens) or v.shape[-1] != 128:
        raise ValueError("fused prepare-H requires a 64-aligned token axis")
    value_heads = v.shape[2]
    chunks = tokens // 64
    if value_heads % q_heads:
        raise ValueError("value heads must be divisible by Q/K heads")
    if A.shape != (batch, tokens, value_heads, 64):
        raise ValueError("A must use [B,T,Hv,64]")
    if beta.shape != (batch, tokens, value_heads):
        raise ValueError("beta must use [B,T,Hv]")
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use [B,Hv,C,64]")
    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            batch, value_heads, 128, 128,
            dtype=torch.float32, device=k.device,
        )
    elif (
        initial_state.dtype != torch.float32
        or initial_state.shape != (batch, value_heads, 128, 128)
        or not initial_state.is_contiguous()
    ):
        raise ValueError("initial state must be contiguous FP32 [B,Hv,128,128]")
    w = torch.empty_like(v)
    history = torch.empty(
        batch, chunks, value_heads, 128, 128,
        dtype=torch.float32, device=k.device,
    )
    vn = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_flash_qla_fused_prepare_h_bf16_128(
        _pointer(k), _pointer(v), _pointer(A), _pointer(g_chunks),
        _pointer(beta), _pointer(initial_state), _pointer(w),
        _pointer(history), _pointer(vn), _pointer(final_state),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), ctypes.c_bool(use_initial_state),
        stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU fused prepare-H launch failed: {result.value}")
    return w, history, vn, final_state


def flash_qla_wu_forward_bf16_128(
    k: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the official W/U preparation directly in token-major layout."""
    if not is_flash_qla_wu_available():
        raise RuntimeError("PPU W/U kernel is not built")
    tensors = (k, v, A, g, beta)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("PPU W/U inputs must be contiguous FP32")
    if k.ndim != 4 or k.shape[-1] != 128:
        raise ValueError("K must use [B,T,Hq,128]")
    batch, tokens, q_heads, _ = k.shape
    if tokens % 64 or v.shape[:2] != (batch, tokens) or v.shape[-1] != 128:
        raise ValueError("W/U inputs require a 64-aligned token axis")
    value_heads = v.shape[2]
    if value_heads % q_heads:
        raise ValueError("value heads must be divisible by Q/K heads")
    if A.shape != (batch, tokens, value_heads, 64):
        raise ValueError("A must use [B,T,Hv,64]")
    if g.shape != (batch, tokens, value_heads) or beta.shape != g.shape:
        raise ValueError("gate/beta must use [B,T,Hv]")
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_flash_qla_wu_forward_bf16_128(
        _pointer(k), _pointer(v), _pointer(A), _pointer(g), _pointer(beta),
        _pointer(w), _pointer(u), ctypes.c_int(batch), ctypes.c_int(tokens),
        ctypes.c_int(q_heads), ctypes.c_int(value_heads), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU W/U launch failed: {result.value}")
    return w, u


def flash_qla_cp_dh_backward_bf16_128(
    q: torch.Tensor,
    k: torch.Tensor,
    A: torch.Tensor,
    g_chunks: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compute official AutoCP local dH0 for independent segments."""
    if not is_flash_qla_cp_dh_backward_available():
        raise RuntimeError("PPU AutoCP dH kernel is not built")
    tensors = (q, k, A, g_chunks, beta, do)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("PPU AutoCP dH inputs must be contiguous FP32")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Q/K must match [B,T,Hq,128]")
    batch, tokens, q_heads, _ = q.shape
    if tokens % 64 or do.shape[:2] != (batch, tokens) or do.shape[-1] != 128:
        raise ValueError("dO must use a 64-aligned [B,T,Hv,128] shape")
    value_heads = do.shape[2]
    chunks = tokens // 64
    if A.shape != (batch, tokens, value_heads, 64):
        raise ValueError("A must use [B,T,Hv,64]")
    if beta.shape != (batch, tokens, value_heads):
        raise ValueError("beta must use [B,T,Hv]")
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use [B,Hv,C,64]")
    dh0 = torch.empty(
        batch, value_heads, 128, 128,
        dtype=torch.float32, device=q.device,
    )
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    _LIBRARY.launch_flash_qla_cp_dh_backward_bf16_128(
        _pointer(q), _pointer(k), _pointer(A), _pointer(g_chunks),
        _pointer(beta), _pointer(do), _pointer(dh0),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), ctypes.c_float(scale), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU AutoCP dH launch failed: {result.value}")
    return dh0


def flash_qla_chunk_state_backward_bf16_128(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g_chunks: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    terminal_state_grad: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the official reverse state/dV recurrence in one PPU launch."""
    if not is_flash_qla_chunk_state_backward_available():
        raise RuntimeError("PPU reverse chunk-state kernel is not built")
    tensors = (q, k, w, do, dv)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("PPU reverse chunk-state inputs must be contiguous FP32")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Q/K must match [B,T,Hq,128]")
    batch, tokens, q_heads, _ = q.shape
    if tokens % 64 or w.shape != do.shape or do.shape != dv.shape:
        raise ValueError("W/dO/dV must match with a 64-aligned token axis")
    value_heads = w.shape[2]
    chunks = tokens // 64
    if w.shape != (batch, tokens, value_heads, 128):
        raise ValueError("W/dO/dV must use [B,T,Hv,128]")
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use [B,Hv,C,64]")
    if g_chunks.dtype != torch.float32 or not g_chunks.is_contiguous():
        raise ValueError("gate must be contiguous FP32")
    use_terminal = terminal_state_grad is not None
    if terminal_state_grad is None:
        terminal_state_grad = torch.empty(
            batch, value_heads, 128, 128,
            dtype=torch.float32, device=q.device,
        )
    elif (
        terminal_state_grad.dtype != torch.float32
        or terminal_state_grad.shape != (batch, value_heads, 128, 128)
        or not terminal_state_grad.is_contiguous()
    ):
        raise ValueError("terminal state grad must be contiguous FP32")
    dh = torch.empty(
        batch, chunks, value_heads, 128, 128,
        dtype=torch.float32, device=q.device,
    )
    dv_output = torch.empty_like(dv)
    dh0 = torch.empty_like(terminal_state_grad)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    _LIBRARY.launch_flash_qla_chunk_state_backward_bf16_128(
        _pointer(q), _pointer(k), _pointer(w), _pointer(g_chunks),
        _pointer(do), _pointer(dv), _pointer(terminal_state_grad),
        _pointer(dh), _pointer(dv_output), _pointer(dh0),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), ctypes.c_float(scale),
        ctypes.c_bool(use_terminal), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU reverse chunk-state launch failed: {result.value}")
    return dh, dh0, dv_output


def flash_qla_chunk_state_backward_step_bf16_128(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g_chunks: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    dstate: torch.Tensor,
    dh: torch.Tensor,
    chunk: int,
    scale: float,
) -> None:
    """Apply one official reverse-state chunk step in place."""
    if not is_flash_qla_chunk_state_backward_step_available():
        raise RuntimeError("PPU reverse chunk-step kernel is not built")
    tensors = (q, k, w, g_chunks, do, dv, dstate, dh)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("PPU reverse chunk-step inputs must be contiguous FP32")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Q/K must match [B,T,Hq,128]")
    batch, tokens, q_heads, _ = q.shape
    if tokens % 64 or not (w.shape == do.shape == dv.shape):
        raise ValueError("W/dO/dV must match with a 64-aligned token axis")
    value_heads = w.shape[2]
    chunks = tokens // 64
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use [B,Hv,C,64]")
    state_shape = (batch, value_heads, 128, 128)
    if dstate.shape != state_shape:
        raise ValueError("dState must use [B,Hv,128,128]")
    if dh.shape != (batch, chunks, value_heads, 128, 128):
        raise ValueError("dH must use [B,C,Hv,128,128]")
    if chunk < 0 or chunk >= chunks:
        raise ValueError("chunk index is out of range")
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    _LIBRARY.launch_flash_qla_chunk_state_backward_step_bf16_128(
        _pointer(q), _pointer(k), _pointer(w), _pointer(g_chunks),
        _pointer(do), _pointer(dv), _pointer(dstate), _pointer(dh),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), ctypes.c_int(chunk), ctypes.c_float(scale),
        stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU reverse chunk-step launch failed: {result.value}")


def flash_qla_fused_state_dqkwg_backward_bf16_128(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g_chunks: torch.Tensor,
    do: torch.Tensor,
    terminal_state_grad: torch.Tensor | None,
    v_corrected: torch.Tensor,
    history: torch.Tensor,
    scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run official persistent S/K/A reverse consumers in one PPU CTA."""
    if not is_flash_qla_fused_state_dqkwg_backward_available():
        raise RuntimeError("PPU fused S/K/A backward kernel is not built")
    vectors = (q, k, w, do, v_corrected)
    if not all(
        tensor.dtype == torch.float32 and tensor.is_contiguous()
        for tensor in vectors
    ):
        raise ValueError("PPU fused S/K/A vectors must be contiguous FP32")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Q/K must match [B,T,Hq,128]")
    batch, tokens, q_heads, _ = q.shape
    if tokens % 64 or not (w.shape == do.shape == v_corrected.shape):
        raise ValueError("W/dO/V' must match on a 64-aligned token axis")
    value_heads = w.shape[2]
    chunks = tokens // 64
    if w.shape != (batch, tokens, value_heads, 128):
        raise ValueError("W/dO/V' must use [B,T,Hv,128]")
    if history.shape != (batch, chunks, value_heads, 128, 128):
        raise ValueError("history must use [B,C,Hv,128,128]")
    if history.dtype != torch.float32 or not history.is_contiguous():
        raise ValueError("history must be contiguous FP32")
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use [B,Hv,C,64]")
    if g_chunks.dtype != torch.float32 or not g_chunks.is_contiguous():
        raise ValueError("gate must be contiguous FP32")
    use_terminal = terminal_state_grad is not None
    if terminal_state_grad is None:
        terminal_state_grad = torch.empty(
            batch, value_heads, 128, 128,
            dtype=torch.float32, device=q.device,
        )
    elif (
        terminal_state_grad.dtype != torch.float32
        or terminal_state_grad.shape != (batch, value_heads, 128, 128)
        or not terminal_state_grad.is_contiguous()
    ):
        raise ValueError("terminal state grad must be contiguous FP32")
    dq = torch.empty(
        batch, tokens, value_heads, 128,
        dtype=torch.float32, device=q.device,
    )
    dk = torch.empty_like(dq)
    dw = torch.empty_like(dq)
    dg = torch.empty(
        batch, tokens, value_heads,
        dtype=torch.float32, device=q.device,
    )
    dv_output = torch.empty_like(w)
    dh0 = torch.empty_like(terminal_state_grad)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    _LIBRARY.launch_flash_qla_fused_state_dqkwg_backward_bf16_128_v2(
        _pointer(q), _pointer(k), _pointer(w), _pointer(g_chunks),
        _pointer(do), _pointer(terminal_state_grad),
        _pointer(v_corrected), _pointer(history),
        _pointer(dq), _pointer(dk), _pointer(dw), _pointer(dg),
        _pointer(dv_output), _pointer(dh0),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), ctypes.c_float(scale),
        ctypes.c_bool(use_terminal), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU fused S/K/A launch failed: {result.value}")
    return dq, dk, dw, dg, dv_output, dh0


def flash_qla_chunk_dv_backward_bf16_128(
    q: torch.Tensor,
    k: torch.Tensor,
    g_chunks: torch.Tensor,
    do: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compute the official causal intra-chunk dV seed in one PPU launch."""
    tensors = (q, k, do)
    if not is_flash_qla_chunk_dv_backward_available():
        raise RuntimeError("PPU chunk-dV backward kernel is not built")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("PPU chunk-dV inputs must be contiguous FP32")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Q/K must match [B,T,Hq,128]")
    batch, tokens, q_heads, _ = q.shape
    if tokens % 64 or do.shape[:2] != (batch, tokens) or do.shape[-1] != 128:
        raise ValueError("dO must use a 64-aligned [B,T,Hv,128] shape")
    value_heads = do.shape[2]
    chunks = tokens // 64
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use [B,Hv,C,64]")
    if g_chunks.dtype != torch.float32 or not g_chunks.is_contiguous():
        raise ValueError("gate must be contiguous FP32")
    dv = torch.empty_like(do)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    _LIBRARY.launch_flash_qla_chunk_dv_backward_bf16_128(
        _pointer(q), _pointer(k), _pointer(g_chunks), _pointer(do),
        _pointer(dv), ctypes.c_int(batch), ctypes.c_int(tokens),
        ctypes.c_int(q_heads), ctypes.c_int(value_heads),
        ctypes.c_float(scale), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU chunk-dV backward launch failed: {result.value}")
    return dv


def flash_qla_chunk_dqkw_backward_bf16_128(
    do: torch.Tensor,
    v_corrected: torch.Tensor,
    dv: torch.Tensor,
    history: torch.Tensor,
    dh: torch.Tensor,
    g_chunks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the official H/dH contributions to dQ, dK and dW."""
    if not is_flash_qla_chunk_dqkw_backward_available():
        raise RuntimeError("PPU chunk-dQKW backward kernel is not built")
    vectors = (do, v_corrected, dv)
    states = (history, dh)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in vectors):
        raise ValueError("dO/V'/dV must be contiguous FP32")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in states):
        raise ValueError("H/dH must be contiguous FP32")
    if not (do.shape == v_corrected.shape == dv.shape):
        raise ValueError("dO/V'/dV shapes must match")
    if do.ndim != 4 or do.shape[-1] != 128 or do.shape[1] % 64:
        raise ValueError("vectors must use [B,64-aligned T,H,128]")
    batch, tokens, value_heads, _ = do.shape
    chunks = tokens // 64
    state_shape = (batch, chunks, value_heads, 128, 128)
    if history.shape != state_shape or dh.shape != state_shape:
        raise ValueError("H/dH must use [B,C,H,128,128]")
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use [B,H,C,64]")
    if g_chunks.dtype != torch.float32 or not g_chunks.is_contiguous():
        raise ValueError("gate must be contiguous FP32")
    dq = torch.empty_like(do)
    dk = torch.empty_like(do)
    dw = torch.empty_like(do)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(do.device).cuda_stream)
    _LIBRARY.launch_flash_qla_chunk_dqkw_backward_bf16_128(
        _pointer(do), _pointer(v_corrected), _pointer(dv),
        _pointer(history), _pointer(dh), _pointer(g_chunks),
        _pointer(dq), _pointer(dk), _pointer(dw),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(value_heads),
        ctypes.c_float(scale), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU chunk-dQKW backward launch failed: {result.value}")
    return dq, dk, dw


def flash_qla_chunk_dqkwg_backward_bf16_128(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    v_corrected: torch.Tensor,
    dv: torch.Tensor,
    history: torch.Tensor,
    dh: torch.Tensor,
    g_chunks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete official per-chunk dQ/dK/dW/dg stage."""
    if not is_flash_qla_chunk_dqkwg_backward_available():
        raise RuntimeError("PPU chunk-dQKWg backward kernel is not built")
    vectors = (q, k, do, v_corrected, dv)
    states = (history, dh)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in vectors):
        raise ValueError("Q/K/dO/V'/dV must be contiguous FP32")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in states):
        raise ValueError("H/dH must be contiguous FP32")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Q/K must match [B,T,Hq,128]")
    batch, tokens, q_heads, _ = q.shape
    if not (do.shape == v_corrected.shape == dv.shape):
        raise ValueError("dO/V'/dV shapes must match")
    if do.ndim != 4 or do.shape[:2] != (batch, tokens) or do.shape[-1] != 128:
        raise ValueError("dO/V'/dV must use [B,T,Hv,128]")
    if tokens % 64:
        raise ValueError("token count must be divisible by 64")
    value_heads = do.shape[2]
    if value_heads % q_heads:
        raise ValueError("value heads must be divisible by Q/K heads")
    chunks = tokens // 64
    state_shape = (batch, chunks, value_heads, 128, 128)
    if history.shape != state_shape or dh.shape != state_shape:
        raise ValueError("H/dH must use [B,C,Hv,128,128]")
    if g_chunks.shape != (batch, value_heads, chunks, 64):
        raise ValueError("gate must use [B,Hv,C,64]")
    if g_chunks.dtype != torch.float32 or not g_chunks.is_contiguous():
        raise ValueError("gate must be contiguous FP32")
    dq = torch.empty_like(do)
    dk = torch.empty_like(do)
    dw = torch.empty_like(do)
    dg = torch.empty(
        batch, tokens, value_heads, dtype=torch.float32, device=do.device
    )
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(do.device).cuda_stream)
    _LIBRARY.launch_flash_qla_chunk_dqkwg_backward_bf16_128(
        _pointer(q), _pointer(k), _pointer(do), _pointer(v_corrected),
        _pointer(dv), _pointer(history), _pointer(dh), _pointer(g_chunks),
        _pointer(dq), _pointer(dk), _pointer(dw), _pointer(dg),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), ctypes.c_float(scale), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU chunk-dQKWg backward launch failed: {result.value}")
    return dq, dk, dw, dg


def flash_qla_fused_wy_backward_128(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dk1: torch.Tensor,
    dg1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the complete official WY/KKT gradient chain in one PPU launch."""
    if not is_flash_qla_fused_wy_backward_available():
        raise RuntimeError("PPU fused WY-backward kernel is not built")
    vectors = (k, v, dw, du, dk1)
    scalars = (beta, g, dg1)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in vectors):
        raise ValueError("fused WY vector inputs must be contiguous FP32")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in scalars):
        raise ValueError("fused WY scalar inputs must be contiguous FP32")
    if k.ndim != 4 or k.shape[-1] != 128:
        raise ValueError("K must use [B,T,Hq,128]")
    batch, tokens, q_heads, _ = k.shape
    if tokens % 64 or v.shape[:2] != (batch, tokens) or v.shape[-1] != 128:
        raise ValueError("fused WY requires a 64-aligned token axis")
    value_heads = v.shape[2]
    value_shape = (batch, tokens, value_heads, 128)
    scalar_shape = (batch, tokens, value_heads)
    if value_heads % q_heads or not (
        v.shape == dw.shape == du.shape == dk1.shape == value_shape
    ):
        raise ValueError(
            "V/dW/dU/dK1 must match [B,T,Hv,128]; "
            f"expected {value_shape}, got V={tuple(v.shape)}, "
            f"dW={tuple(dw.shape)}, dU={tuple(du.shape)}, "
            f"dK1={tuple(dk1.shape)}"
        )
    if A.shape != (batch, tokens, value_heads, 64):
        raise ValueError("A must use [B,T,Hv,64]")
    if not A.is_contiguous() or A.dtype != torch.float32:
        raise ValueError("A must be contiguous FP32")
    if not (beta.shape == g.shape == dg1.shape == scalar_shape):
        raise ValueError("beta/g/dg1 must match [B,T,Hv]")
    dk = torch.empty(value_shape, dtype=torch.float32, device=k.device)
    dv = torch.empty_like(dk)
    db = torch.empty(scalar_shape, dtype=torch.float32, device=k.device)
    dg = torch.empty_like(db)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_flash_qla_fused_wy_backward_128_v1(
        _pointer(k), _pointer(v), _pointer(beta), _pointer(A), _pointer(g),
        _pointer(dw), _pointer(du), _pointer(dk1), _pointer(dg1),
        _pointer(dk), _pointer(dv), _pointer(db), _pointer(dg),
        ctypes.c_int(batch), ctypes.c_int(tokens), ctypes.c_int(q_heads),
        ctypes.c_int(value_heads), stream, ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU fused WY-backward launch failed: {result.value}")
    return dk, dv, db, dg


def flash_qla_wy_backward_preprocess_128(
    dk_beta_g: torch.Tensor,
    dv_beta: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the first official WY-backward vector/reduction stage."""
    if not is_flash_qla_wy_backward_available():
        raise RuntimeError("PPU WY-backward kernels are not built")
    vectors = (dk_beta_g, dv_beta, k, v)
    scalars = (beta, g)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in vectors):
        raise ValueError("WY vector inputs must be contiguous FP32")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in scalars):
        raise ValueError("WY scalar inputs must be contiguous FP32")
    if not (dk_beta_g.shape == dv_beta.shape == k.shape == v.shape):
        raise ValueError("WY vector shapes must match")
    if k.ndim != 5 or k.shape[-2:] != (64, 128):
        raise ValueError("WY inputs must use [B,N,H,64,128] layout")
    batch, chunks, value_heads = k.shape[:3]
    if beta.shape != (batch, chunks, value_heads, 64):
        raise ValueError("WY beta must use [B,N,H,64] layout")
    if g.shape != beta.shape:
        raise ValueError("WY beta/g shapes must match")
    tokens = chunks * 64
    output_shape = (batch, tokens, value_heads)
    dk = torch.empty(*output_shape, 128, dtype=torch.float32, device=k.device)
    dv = torch.empty_like(dk)
    db = torch.empty(output_shape, dtype=torch.float32, device=k.device)
    dg = torch.empty_like(db)
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_flash_qla_wy_backward_preprocess_128(
        _pointer(dk_beta_g), _pointer(dv_beta), _pointer(k), _pointer(v),
        _pointer(beta), _pointer(g), _pointer(dk), _pointer(dv),
        _pointer(db), _pointer(dg), ctypes.c_int(batch), ctypes.c_int(tokens),
        ctypes.c_int(value_heads), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU WY preprocess launch failed: {result.value}")
    return dk, dv, db, dg


def flash_qla_wy_backward_postprocess_128(
    dk_da: torch.Tensor,
    dk_beta: torch.Tensor,
    dk1: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    dA: torch.Tensor,
    A: torch.Tensor,
    dg1: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the final official WY-backward vector/reduction stage."""
    if not is_flash_qla_wy_backward_available():
        raise RuntimeError("PPU WY-backward kernels are not built")
    vectors = (dk_da, dk_beta, dk1, k, dk)
    matrices = (dA, A)
    scalars = (beta, dg1, db, dg)
    tensors = vectors + matrices + scalars
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError("WY postprocess inputs must be contiguous FP32")
    if not all(t.shape == k.shape for t in vectors[:-1]):
        raise ValueError("WY postprocess vector shapes must match")
    if k.ndim != 5 or k.shape[-2:] != (64, 128):
        raise ValueError("WY vectors must use [B,N,H,64,128] layout")
    batch, chunks, value_heads = k.shape[:3]
    if dA.shape != (batch, chunks, value_heads, 64, 64):
        raise ValueError("WY matrices must use [B,N,H,64,64] layout")
    if A.shape != dA.shape or beta.shape != (batch, chunks, value_heads, 64):
        raise ValueError("WY matrix/scalar shapes are incompatible")
    if dg1.shape != beta.shape:
        raise ValueError("WY head-major scalar shapes must match")
    token_shape = (batch, chunks * 64, value_heads)
    if dk.shape != (*token_shape, 128) or db.shape != token_shape:
        raise ValueError("WY output tensors must use token-major layout")
    if dg.shape != token_shape:
        raise ValueError("WY scalar output shapes must match")
    result = ctypes.c_int(-1)
    stream = ctypes.c_void_p(torch.cuda.current_stream(k.device).cuda_stream)
    _LIBRARY.launch_flash_qla_wy_backward_postprocess_128(
        _pointer(dk_da), _pointer(dk_beta), _pointer(dk1), _pointer(k),
        _pointer(beta), _pointer(dA), _pointer(A), _pointer(dg1),
        _pointer(dk), _pointer(db), _pointer(dg), ctypes.c_int(batch),
        ctypes.c_int(chunks * 64), ctypes.c_int(value_heads), stream,
        ctypes.byref(result),
    )
    if result.value:
        raise RuntimeError(f"PPU WY postprocess launch failed: {result.value}")
    return dk, db, dg
